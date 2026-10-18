"""
TriageIQ ML Service — FastAPI application entry point.

This module wires up the FastAPI app, middleware, and routers.
No ML models are loaded here yet; this is the base scaffold.
"""

from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Startup / shutdown hook — load models here in a later phase."""
    print("[OK] ML service started")
    yield
    print("[--] ML service shutting down")


# ── App factory ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="TriageIQ ML Service",
    description="Support-ticket classification and routing API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health_check():
    """Health-check endpoint — returns service status."""
    from datetime import datetime, timezone

    return {
        "status": "ok",
        "service": "triageiq-ml-service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
