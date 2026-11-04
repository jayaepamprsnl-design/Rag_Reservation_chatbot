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

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agents.mcp_client import export_configured, warn_if_insecure_url
from app.agents.reservation_writer import retry_pending_writes
from app.api.routes.admin import router as admin_router
from app.api.routes.chat import router as chat_router
from app.api.routes.graph import router as graph_router
from app.api.routes.reservations import router as reservations_router
from app.api.routes.reservations import sessions_router
from app.agents.channels.tokens import secret_is_configured
from app.core.config import settings
from app.core.logging_config import logger
from app.db.models import ParkingLot
from app.db.session import SessionLocal, create_tables, get_db
from app.orchestration import graph as orchestration_graph
from app.orchestration.service import expire_stale, reconcile_decided, reset_session


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


async def _retry_worker() -> None:
    """Periodically write CONFIRMED reservations whose MCP write failed earlier."""
    while True:
        try:
            await asyncio.to_thread(retry_pending_writes, SessionLocal)
        except Exception as exc:  # noqa: BLE001 - the worker must survive any sweep
            logger.error("retry_worker|sweep_failed error=%s", type(exc).__name__)
        await asyncio.sleep(settings.mcp_retry_interval_seconds)


def _expire_sweep() -> int:
    """Maintenance sweep: expire unanswered requests, then resume threads whose decision was
    committed but never delivered to the graph (e.g. a crash in between)."""
    db = SessionLocal()
    try:
        n = expire_stale(db)
        reconcile_decided(db)
        return n
    finally:
        db.close()


async def _expiry_worker() -> None:
    """EXPIRE requests no admin answered within APPROVAL_TTL_HOURS and resume their graphs."""
    while True:
        try:
            await asyncio.to_thread(_expire_sweep)
        except Exception as exc:  # noqa: BLE001 - the worker must survive any sweep
            logger.error("expiry_worker|sweep_failed error=%s", type(exc).__name__)
        await asyncio.sleep(settings.approval_expiry_check_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "CityPark RAG Chatbot starting — %s:%s", settings.app_host, settings.app_port
    )
    create_tables()
    if not secret_is_configured():
        logger.warning(
            "email_links|disabled: EMAIL_TOKEN_SECRET is unset/placeholder, one-click approve/refuse "
            "links are neither sent nor accepted"
        )
    worker = None
    if export_configured():
        warn_if_insecure_url()
        worker = asyncio.create_task(_retry_worker())
        logger.info(
            "retry_worker|started interval=%ds", settings.mcp_retry_interval_seconds
        )
    else:
        logger.warning("retry_worker|disabled (MCP_API_KEY not set or export off)")
    expiry = asyncio.create_task(_expiry_worker()) if settings.approval_expiry_enabled else None
    yield
    for task in (worker, expiry):
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    orchestration_graph.reset_graph()
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
    except Exception as exc:
        # Class name only: exception messages can carry customer data (names, plates).
        logger.error(
            "Unhandled error req_id=%s %s %s error=%s",
            request_id, request.method, request.url.path, type(exc).__name__,
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
    logger.error(
        "Unhandled exception: %s %s error=%s", request.method, request.url.path, type(exc).__name__,
        exc_info=settings.log_level.upper() == "DEBUG",   # tracebacks quote exception text
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
app.include_router(graph_router)


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
    reset_session(req.session_id)   # checkpoints + in-memory booking state
    logger.info("Session reset: %s", req.session_id)
    return {"status": "ok", "session_id": req.session_id}
