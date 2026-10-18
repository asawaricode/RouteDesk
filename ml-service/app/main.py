"""
RouteDesk ML Service — FastAPI application entry point.

Provides ticket classification model inference and health endpoints.
Loads the trained TF-IDF + LogisticRegression baseline pipeline on startup.
"""

from __future__ import annotations

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

# Global state for loaded model artifacts
model_state: dict[str, Any] = {
    "model": None,
    "classes": [],
    "model_path": str(MODEL_PATH),
    "loaded_at": None,
    "error": None,
}


# ── Lifespan handler ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook — load saved model and class labels at startup."""
    print(f"[*] Loading ML model from: {MODEL_PATH}")
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        # Load pipeline
        model = joblib.load(MODEL_PATH)
        model_state["model"] = model

        # Load classes
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

        print(
            f"[OK] Model pipeline loaded successfully ({len(classes)} target classes)"
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load ML model: {exc}")
        model_state["model"] = None
        model_state["error"] = str(exc)

    yield

    # Teardown
    model_state["model"] = None
    print("[--] ML service shut down")


# ── App definition ────────────────────────────────────────────────────────────

app = FastAPI(
    title="RouteDesk ML Service",
    description="Support-ticket classification and routing API.",
    version="0.2.0",
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health_check():
    """Health-check endpoint — reports service health and whether the model is loaded."""
    is_loaded = model_state.get("model") is not None
    classes = model_state.get("classes", [])

    payload: dict[str, Any] = {
        "status": "ok" if is_loaded else "degraded",
        "service": "routedesk-ml-service",
        "model_loaded": is_loaded,
        "model_path": model_state.get("model_path"),
        "classes_count": len(classes),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if is_loaded:
        payload["classes"] = classes
        payload["loaded_at"] = model_state.get("loaded_at")
    else:
        payload["error"] = model_state.get("error", "Model artifact not loaded")

    return payload


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

    # Combine subject and body exactly as done during training
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
        # Run inference via scikit-learn pipeline (TF-IDF + LogisticRegression)
        classes = model.classes_
        probabilities = model.predict_proba([combined_text])[0]
        best_idx = int(np.argmax(probabilities))

        predicted_queue = str(classes[best_idx])
        confidence = float(probabilities[best_idx])

        # Distribution across all classes
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
