"""FastAPI application entry point.

Endpoints
---------
POST /chat              → {answer, sources, intent, guardrail_flags}
GET  /health            → {status, version}
GET  /lots              → live parking-lot availability from the DB
POST /reset-session     → clear all in-memory state for a session
GET  /reservations      → list reservations (from routes/reservations.py)
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.routes.admin import router as admin_router
from app.api.routes.chat import router as chat_router
from app.api.routes.reservations import router as reservations_router
from app.api.routes.reservations import sessions_router
from app.chat.conversation import clear_session
from app.chat.reservation import reset as reset_reservation
from app.core.config import settings
from app.core.logging_config import logger
from app.db.models import ParkingLot
from app.db.session import create_tables, get_db


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "CityPark RAG Chatbot starting — %s:%s", settings.app_host, settings.app_port
    )
    create_tables()
    yield
    logger.info("CityPark RAG Chatbot shutting down")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CityPark RAG Chatbot",
    version="1.0.0",
    description=(
        "RAG-powered parking assistant with guardrails, live SQL availability, "
        "and a multi-step reservation flow."
    ),
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"] is incompatible with allow_credentials=True (browsers
    # reject it).  Use explicit origins in production; the wildcard is safe only
    # for fully public, credential-free APIs.  Admin routes are protected by the
    # X-Admin-Key header, not cookies, so credentials=False is correct here.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ── Request logging ───────────────────────────────────────────────────────────


@app.middleware("http")
async def log_requests(request: Request, call_next) -> JSONResponse:
    """Assign a unique request ID for log correlation and measure latency."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled error req_id=%s %s %s",
            request_id, request.method, request.url.path,
        )
        raise
    ms = (time.perf_counter() - start) * 1000
    logger.info(
        "http req_id=%s method=%s path=%s status=%d duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ── Global error handler ──────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception: %s %s — %s", request.method, request.url.path, exc
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(chat_router)
app.include_router(reservations_router)
app.include_router(sessions_router)
app.include_router(admin_router)


# ---------------------------------------------------------------------------
# Ops endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"], summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok", "version": app.version}


# ---------------------------------------------------------------------------
# Lots endpoint — live availability from the DB
# ---------------------------------------------------------------------------


@app.get("/lots", tags=["lots"], summary="Live parking-lot availability")
def get_lots(db: Session = Depends(get_db)) -> list[dict]:
    """Return all parking lots with real-time space counts and pricing."""
    lots = db.query(ParkingLot).order_by(ParkingLot.id).all()
    return [
        {
            "id": lot.id,
            "name": lot.name,
            "address": lot.address,
            "total_spaces": lot.total_spaces,
            "available_spaces": lot.available_spaces,
            "hourly_price": lot.hourly_price,
            "daily_price": lot.daily_price,
            "opens_at": lot.opens_at.strftime("%H:%M"),
            "closes_at": lot.closes_at.strftime("%H:%M"),
        }
        for lot in lots
    ]


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    session_id: str


@app.post("/reset-session", tags=["ops"], summary="Clear all session state")
def reset_session_endpoint(req: ResetRequest) -> dict:
    """Wipe conversation history and reservation state for *session_id*."""
    clear_session(req.session_id)
    reset_reservation(req.session_id)
    logger.info("Session reset: %s", req.session_id)
    return {"status": "ok", "session_id": req.session_id}
