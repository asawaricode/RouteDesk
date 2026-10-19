"""
RouteDesk ML Service — FastAPI application entry point.

Provides:
1. Support-ticket classification (TF-IDF + Logistic Regression).
2. Semantic duplicate ticket detection (all-MiniLM-L6-v2 sentence transformer).
3. Health and diagnostics endpoints.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

load_dotenv()

# ── Paths relative to application directory (not current working dir) ─────────
APP_DIR = Path(__file__).resolve().parent
ML_SERVICE_DIR = APP_DIR.parent
ARTIFACTS_DIR = ML_SERVICE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "tfidf_lr_pipeline.joblib"
CLASSES_PATH = ARTIFACTS_DIR / "classes.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global state for loaded model artifacts
model_state: dict[str, Any] = {
    "model": None,
    "classes": [],
    "model_path": str(MODEL_PATH),
    "loaded_at": None,
    "error": None,
}

# Global state for sentence embedding model
embedding_state: dict[str, Any] = {
    "model": None,
    "model_name": EMBEDDING_MODEL_NAME,
    "loaded_at": None,
    "error": None,
}

_embedding_lock: Optional[asyncio.Lock] = None


def get_embedding_lock() -> asyncio.Lock:
    """Returns or lazily creates the asyncio lock to prevent concurrent model loads."""
    global _embedding_lock
    if _embedding_lock is None:
        _embedding_lock = asyncio.Lock()
    return _embedding_lock


async def get_embedding_model():
    """Lazily load SentenceTransformer model on first duplicate detection call and cache it."""
    if embedding_state.get("model") is not None:
        return embedding_state["model"]

    lock = get_embedding_lock()
    async with lock:
        # Double-check inside lock
        if embedding_state.get("model") is not None:
            return embedding_state["model"]

        print(f"[*] Lazily loading sentence-transformer model: {EMBEDDING_MODEL_NAME}")
        try:
            from sentence_transformers import SentenceTransformer

            embed_model = await asyncio.to_thread(SentenceTransformer, EMBEDDING_MODEL_NAME)
            embedding_state["model"] = embed_model
            embedding_state["loaded_at"] = datetime.now(timezone.utc).isoformat()
            embedding_state["error"] = None
            print(f"[OK] SentenceTransformer '{EMBEDDING_MODEL_NAME}' loaded successfully (lazy)")
            return embed_model
        except Exception as exc:
            print(f"[ERROR] Failed to load SentenceTransformer: {exc}")
            embedding_state["model"] = None
            embedding_state["error"] = str(exc)
            raise exc


# ── Lifespan handler ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook — loads classification model at startup."""
    # 1. Load classification pipeline
    print(f"[*] Loading classification model from: {MODEL_PATH}")
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Classification model file not found at: {MODEL_PATH}")

        model = joblib.load(MODEL_PATH)
        model_state["model"] = model

        if CLASSES_PATH.exists():
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                classes = json.load(f)
        elif hasattr(model, "classes_"):
            classes = [str(c) for c in model.classes_]
        else:
            classes = []

        model_state["classes"] = classes
        model_state["loaded_at"] = datetime.now(timezone.utc).isoformat()
        model_state["error"] = None
        print(f"[OK] Classifier loaded ({len(classes)} target classes)")
    except Exception as exc:
        print(f"[ERROR] Failed to load classifier model: {exc}")
        model_state["model"] = None
        model_state["error"] = str(exc)

    yield

    # Teardown
    model_state["model"] = None
    embedding_state["model"] = None
    print("[--] ML service shut down")


# ── App definition ────────────────────────────────────────────────────────────

app = FastAPI(
    title="RouteDesk ML Service",
    description="Support-ticket classification and semantic duplicate detection API.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    subject: Optional[str] = Field(
        default=None,
        description="Ticket subject line or headline.",
        examples=["Cannot reset my password or access email"],
    )
    body: Optional[str] = Field(
        default=None,
        description="Ticket message or detailed body description.",
        examples=[
            "I have been locked out of my account since this morning. The reset link is expired."
        ],
    )

    @model_validator(mode="after")
    def validate_non_empty(self) -> PredictRequest:
        s = (self.subject or "").strip()
        b = (self.body or "").strip()
        if not s and not b:
            raise ValueError(
                "Invalid request: at least one of 'subject' or 'body' must be non-empty."
            )
        return self


class PredictResponse(BaseModel):
    queue: str = Field(description="Predicted target queue.")
    confidence: float = Field(
        description="Confidence score for the predicted queue (between 0.0 and 1.0)."
    )
    probabilities: Optional[dict[str, float]] = Field(
        default=None,
        description="Probability distribution across all candidate queues.",
    )


class DuplicateCandidate(BaseModel):
    id: Any = Field(description="Unique ticket ID or reference.")
    subject: Optional[str] = Field(default="", description="Candidate ticket subject.")
    body: Optional[str] = Field(default="", description="Candidate ticket body.")
    predicted_queue: Optional[str] = Field(
        default=None, description="Previous predicted queue if known."
    )
    confidence: Optional[float] = Field(default=None, description="Model confidence.")
    created_at: Optional[str] = Field(
        default=None, description="ISO timestamp when candidate ticket was logged."
    )


class DuplicateDetectionRequest(BaseModel):
    subject: Optional[str] = Field(
        default=None, description="Subject of the incoming ticket to check."
    )
    body: Optional[str] = Field(
        default=None, description="Body description of the incoming ticket."
    )
    candidates: list[DuplicateCandidate] = Field(
        default_factory=list,
        description="List of previously saved tickets to compare against.",
    )
    threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Similarity score threshold (0.0 to 1.0) above which a ticket is flagged as duplicate.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Maximum number of most similar candidates to return.",
    )

    @model_validator(mode="after")
    def validate_incoming_ticket(self) -> DuplicateDetectionRequest:
        s = (self.subject or "").strip()
        b = (self.body or "").strip()
        if not s and not b:
            raise ValueError(
                "Invalid request: at least one of 'subject' or 'body' must be non-empty."
            )
        return self


class SimilarTicket(BaseModel):
    id: Any
    subject: str
    body: str
    predicted_queue: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[str] = None
    similarity_score: float = Field(
        description="Cosine similarity score (0.0 to 1.0) between incoming ticket and this candidate."
    )


class DuplicateDetectionResponse(BaseModel):
    has_duplicates: bool = Field(
        description="True if the top candidate's similarity score is greater than or equal to threshold."
    )
    threshold: float = Field(description="Threshold used for duplicate qualification.")
    candidate_count: int = Field(
        description="Total number of candidates evaluated from ticket history."
    )
    duplicates: list[SimilarTicket] = Field(
        default_factory=list,
        description="Ranked list of top candidate tickets with highest semantic similarity.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health_check():
    """Health-check endpoint — reports status of classifier and sentence-transformer models."""
    is_classifier_loaded = model_state.get("model") is not None
    is_embed_loaded = embedding_state.get("model") is not None
    classes = model_state.get("classes", [])

    # Embedding model is ready if already loaded in memory or available for lazy loading without error
    is_embed_ready = is_embed_loaded or (embedding_state.get("error") is None)

    payload: dict[str, Any] = {
        "status": "ok" if (is_classifier_loaded and is_embed_ready) else "degraded",
        "service": "routedesk-ml-service",
        "model_loaded": is_classifier_loaded,
        "embedding_model_loaded": is_embed_ready,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "model_path": model_state.get("model_path"),
        "classes_count": len(classes),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if is_classifier_loaded:
        payload["classes"] = classes
        payload["loaded_at"] = model_state.get("loaded_at")
    else:
        payload["error"] = model_state.get("error", "Classification model artifact not loaded")

    if is_embed_loaded:
        payload["embedding_loaded_at"] = embedding_state.get("loaded_at")
    elif embedding_state.get("error") is not None:
        payload["embedding_error"] = embedding_state.get("error")

    return payload


@app.get("/metrics", tags=["Meta"])
async def get_model_metrics():
    """Returns saved model evaluation metrics from metrics.json."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation metrics file not found at: {METRICS_PATH}",
        )
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read metrics artifact: {exc}",
        )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_ticket(request: PredictRequest):
    """
    Predict ticket destination queue and return model confidence score.

    Accepts ticket 'subject' and/or 'body'. Combines non-empty parts into a
    single text input matching the training preprocessing pipeline.
    """
    model = model_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Classification model is not loaded. Details: {model_state.get('error')}",
        )

    parts: list[str] = []
    if request.subject and request.subject.strip():
        parts.append(request.subject.strip())
    if request.body and request.body.strip():
        parts.append(request.body.strip())

    combined_text = " ".join(parts).strip()
    if not combined_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Ticket text cannot be empty. Please provide a non-empty subject or body.",
        )

    try:
        classes = model.classes_
        probabilities = model.predict_proba([combined_text])[0]
        best_idx = int(np.argmax(probabilities))

        predicted_queue = str(classes[best_idx])
        confidence = float(probabilities[best_idx])

        prob_dict = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }

        return PredictResponse(
            queue=predicted_queue,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )


@app.post(
    "/duplicates",
    response_model=DuplicateDetectionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Semantic Similarity"],
)
async def detect_duplicates(request: DuplicateDetectionRequest):
    """
    Compare incoming ticket subject/body against candidate tickets using all-MiniLM-L6-v2 embeddings.

    Computes cosine similarity between incoming ticket vector and candidate vectors.
    Returns ranked most similar tickets with similarity scores.
    """
    # If candidates list is empty (e.g. no prior tickets in DB), return empty results immediately
    if not request.candidates:
        return DuplicateDetectionResponse(
            has_duplicates=False,
            threshold=request.threshold,
            candidate_count=0,
            duplicates=[],
        )

    try:
        embed_model = await get_embedding_model()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"SentenceTransformer embedding model is not loaded. Details: {exc}",
        )

    # 1. Format incoming ticket text
    incoming_parts: list[str] = []
    if request.subject and request.subject.strip():
        incoming_parts.append(request.subject.strip())
    if request.body and request.body.strip():
        incoming_parts.append(request.body.strip())

    incoming_text = " ".join(incoming_parts).strip()
    if not incoming_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Incoming ticket text cannot be empty.",
        )

    # 2. Format candidate texts
    candidate_texts: list[str] = []
    valid_candidates: list[DuplicateCandidate] = []

    for c in request.candidates:
        c_parts: list[str] = []
        if c.subject and str(c.subject).strip():
            c_parts.append(str(c.subject).strip())
        if c.body and str(c.body).strip():
            c_parts.append(str(c.body).strip())
        c_text = " ".join(c_parts).strip()

        # Only evaluate candidates with some text content
        if c_text:
            candidate_texts.append(c_text)
            valid_candidates.append(c)

    if not candidate_texts:
        return DuplicateDetectionResponse(
            has_duplicates=False,
            threshold=request.threshold,
            candidate_count=len(request.candidates),
            duplicates=[],
        )

    try:
        from sentence_transformers import util

        # 3. Compute normalized sentence embeddings
        target_embedding = embed_model.encode(
            incoming_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        candidate_embeddings = embed_model.encode(
            candidate_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=32,
        )

        # 4. Compute cosine similarity (dot product of normalized tensors)
        cosine_scores = util.cos_sim(target_embedding, candidate_embeddings)[0]

        scored_matches: list[SimilarTicket] = []
        for cand, score in zip(valid_candidates, cosine_scores):
            score_float = round(float(score.item()), 4)
            scored_matches.append(
                SimilarTicket(
                    id=cand.id,
                    subject=cand.subject or "",
                    body=cand.body or "",
                    predicted_queue=cand.predicted_queue,
                    confidence=cand.confidence,
                    created_at=cand.created_at,
                    similarity_score=score_float,
                )
            )

        # Sort descending by similarity score
        scored_matches.sort(key=lambda item: item.similarity_score, reverse=True)

        top_candidates = scored_matches[: request.top_k]
        has_duplicates = any(
            cand.similarity_score >= request.threshold for cand in top_candidates
        )

        return DuplicateDetectionResponse(
            has_duplicates=has_duplicates,
            threshold=request.threshold,
            candidate_count=len(request.candidates),
            duplicates=top_candidates,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic duplicate detection error: {exc}",
        )
