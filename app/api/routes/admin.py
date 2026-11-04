"""
Admin API routes — protected by X-Admin-Key header.

Endpoints
---------
GET  /admin/reservations/pending           list all PENDING reservations
GET  /admin/reservations/{id}              single reservation detail
POST /admin/reservations/{id}/approve      approve with optional note
POST /admin/reservations/{id}/refuse       refuse with required reason
POST /admin/chat                           natural-language admin commands
GET  /admin/approve?token=...              one-click approve (email link)
GET  /admin/refuse?token=...               one-click refuse  (email link)

Authentication
--------------
Every endpoint except /admin/approve and /admin/refuse requires the
X-Admin-Key header matched against ADMIN_API_KEY from .env using
constant-time comparison (hmac.compare_digest) to prevent timing attacks.

The email-link endpoints are authenticated by the signed, expiring,
single-use token embedded in the URL — no API key needed.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agents.admin_agent import AdminAgent
from app.agents.reservation_writer import export_note
from app.agents.channels.tokens import TokenExpired, TokenInvalid, TokenUsed, consume, verify
from app.agents.contracts import MessageType
from app.core.config import settings
from app.core.logging_config import logger
from app.db.models import LotReservation, ReservationStatus
from app.db.repository import (
    ValidationError,
    get_lot,
    get_reservation,
    list_pending,
)
from app.db.session import get_db
from app.orchestration.service import decide, sync_decision
from app.rag.generator import BaseLLMClient, get_llm_client

router = APIRouter(prefix="/admin", tags=["admin"])

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def require_admin(raw_key: Optional[str] = Security(_key_header)) -> str:
    """
    Validate the X-Admin-Key header using constant-time comparison.

    Returns the key on success.  Raises:
        401 — header absent (or key not configured on server)
        403 — header present but value is wrong
    """
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API key not configured")
    if not raw_key:
        raise HTTPException(status_code=401, detail="X-Admin-Key header required")
    if not hmac.compare_digest(
        raw_key.encode("utf-8"),
        settings.admin_api_key.encode("utf-8"),
    ):
        # Do NOT log any part of the submitted key — even a prefix prefix leaks
        # information that can be used to enumerate the key character by character.
        logger.warning("admin_api|auth_failed")
        raise HTTPException(status_code=403, detail="Invalid admin key")
    return raw_key


def get_admin_llm() -> BaseLLMClient:
    """FastAPI dependency — injectable in tests."""
    return get_llm_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _admin_id(admin_key: str) -> str:
    """Stable, non-reversible admin identity for audit columns.

    Previously "api:" + the first 8 characters of ADMIN_API_KEY, which put part of the
    secret into the database and into API responses.
    """
    return "api:" + hashlib.sha256(admin_key.encode("utf-8")).hexdigest()[:8]


def _mask_plate(plate: str) -> str:
    if len(plate) <= 4:
        return plate[0] + "***"
    return plate[:3] + "****" + plate[-2:]


def _repo_http_status(exc: ValidationError) -> int:
    """Map repository ValidationError messages to HTTP status codes."""
    msg = str(exc).lower()
    if "not found" in msg:
        return 404
    if "cannot be approved" in msg or "cannot be refused" in msg or "already" in msg:
        return 409
    return 422


def _reservation_to_dict(res: "LotReservation", lot_name: str) -> dict:  # type: ignore[name-defined]
    return {
        "id":                   res.id,
        "customer_name":        res.customer_name,
        "car_plate_masked":     _mask_plate(res.car_plate),
        "lot_id":               res.lot_id,
        "lot_name":             lot_name,
        "start_time":           res.start_time.isoformat(),
        "end_time":             res.end_time.isoformat(),
        "status":               res.status.value if hasattr(res.status, "value") else str(res.status),
        "session_id":           res.session_id,
        "admin_decision_reason": res.admin_decision_reason,
        "decided_at":           res.decided_at.isoformat() if res.decided_at else None,
        "decided_by":           res.decided_by,
        "created_at":           res.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ReservationOut(BaseModel):
    id: int
    customer_name: str
    car_plate_masked: str
    lot_id: int
    lot_name: str
    start_time: str
    end_time: str
    status: str
    session_id: Optional[str] = None
    admin_decision_reason: Optional[str] = None
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None
    created_at: str


class DecisionOut(BaseModel):
    reservation_id: int
    status: str
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    reason: Optional[str] = None
    # Approve only: was the line appended to the confirmed-reservations file?
    file_written: Optional[bool] = None
    file_note: Optional[str] = None


class ApproveIn(BaseModel):
    note: str = ""


class RefuseIn(BaseModel):
    reason: str = Field(..., min_length=1, description="Reason for refusal (required, non-empty)")


class AdminChatIn(BaseModel):
    message: str
    admin_id: str = "api-admin"


class AdminChatOut(BaseModel):
    response: str
    decision_made: bool = False
    reservation_id: Optional[int] = None
    file_written: Optional[bool] = None   # admin-side note, approve only
    file_note: Optional[str] = None


class ActionResultOut(BaseModel):
    message: str
    reservation_id: int
    status: str
    reason: Optional[str] = None
    file_written: Optional[bool] = None
    file_note: Optional[str] = None


# ---------------------------------------------------------------------------
# GET /admin/verify  — lightweight key-validation endpoint used by the login form
# ---------------------------------------------------------------------------


@router.get(
    "/verify",
    summary="Verify admin key",
    responses={
        200: {"description": "Key is valid"},
        401: {"description": "No key supplied"},
        403: {"description": "Key is wrong"},
        503: {"description": "ADMIN_API_KEY not configured on server"},
    },
)
def verify_key(_: str = Depends(require_admin)) -> dict:
    """Returns 200 if the X-Admin-Key header matches the server's ADMIN_API_KEY."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /admin/reservations/pending
# ---------------------------------------------------------------------------


@router.get(
    "/reservations/pending",
    response_model=list[ReservationOut],
    summary="List pending reservations",
)
def list_pending_reservations(
    db: Session = Depends(get_db),
    _: str = Depends(require_admin),
) -> list[dict]:
    rows = list_pending(db)
    result = []
    for r in rows:
        try:
            lot = get_lot(db, r.lot_id)
            lot_name = lot.name
        except ValidationError:
            lot_name = f"lot#{r.lot_id}"
        result.append(_reservation_to_dict(r, lot_name))
    logger.info("admin_api|list_pending count=%d", len(result))
    return result


# ---------------------------------------------------------------------------
# GET /admin/reservations/history
# ---------------------------------------------------------------------------


@router.get(
    "/reservations/history",
    response_model=list[ReservationOut],
    summary="List decided reservations (history)",
)
def list_reservation_history(
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
    _: str = Depends(require_admin),
) -> list[dict]:
    """Return the most recent decided (non-PENDING) reservations."""
    rows = (
        db.query(LotReservation)
        .filter(LotReservation.status != ReservationStatus.PENDING)
        .order_by(LotReservation.created_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for r in rows:
        try:
            lot = get_lot(db, r.lot_id)
            lot_name = lot.name
        except ValidationError:
            lot_name = f"lot#{r.lot_id}"
        result.append(_reservation_to_dict(r, lot_name))
    return result


# ---------------------------------------------------------------------------
# GET /admin/reservations/{id}
# ---------------------------------------------------------------------------


@router.get(
    "/reservations/{reservation_id}",
    response_model=ReservationOut,
    summary="Get reservation details",
)
def get_reservation_detail(
    reservation_id: int,
    db: Session = Depends(get_db),
    _: str = Depends(require_admin),
) -> dict:
    try:
        res = get_reservation(db, reservation_id)
        lot = get_lot(db, res.lot_id)
    except ValidationError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _reservation_to_dict(res, lot.name)


# ---------------------------------------------------------------------------
# POST /admin/reservations/{id}/approve
# ---------------------------------------------------------------------------


@router.post(
    "/reservations/{reservation_id}/approve",
    response_model=DecisionOut,
    summary="Approve a pending reservation",
)
def approve(
    reservation_id: int,
    body: ApproveIn = ApproveIn(),
    db: Session = Depends(get_db),
    admin_key: str = Depends(require_admin),
) -> dict:
    try:
        # Commits the decision, then resumes the paused graph, whose record node does
        # the best-effort MCP write (it never rolls the approval back).
        outcome = decide(
            db, reservation_id, "APPROVE", admin=_admin_id(admin_key), note=body.note
        )
    except ValidationError as exc:
        raise HTTPException(status_code=_repo_http_status(exc), detail=str(exc))

    res, file_written = outcome.reservation, outcome.file_written
    logger.info(
        "admin_api|approved reservation_id=%d file_written=%s", reservation_id, file_written
    )
    return {
        "reservation_id": res.id,
        "status":         res.status.value,
        "decided_by":     res.decided_by,
        "decided_at":     res.decided_at.isoformat() if res.decided_at else None,
        "reason":         res.admin_decision_reason,
        "file_written":   file_written,
        "file_note":      export_note(file_written),
    }


# ---------------------------------------------------------------------------
# POST /admin/reservations/{id}/refuse
# ---------------------------------------------------------------------------


@router.post(
    "/reservations/{reservation_id}/refuse",
    response_model=DecisionOut,
    summary="Refuse a pending reservation",
)
def refuse(
    reservation_id: int,
    body: RefuseIn,
    db: Session = Depends(get_db),
    admin_key: str = Depends(require_admin),
) -> dict:
    try:
        res = decide(
            db, reservation_id, "REFUSE", admin=_admin_id(admin_key), reason=body.reason
        ).reservation
    except ValidationError as exc:
        raise HTTPException(status_code=_repo_http_status(exc), detail=str(exc))

    logger.info("admin_api|refused reservation_id=%d", reservation_id)
    return {
        "reservation_id": res.id,
        "status":         res.status.value,
        "decided_by":     res.decided_by,
        "decided_at":     res.decided_at.isoformat() if res.decided_at else None,
        "reason":         res.admin_decision_reason,
    }


# ---------------------------------------------------------------------------
# POST /admin/chat  (natural-language commands)
# ---------------------------------------------------------------------------


@router.post(
    "/chat",
    response_model=AdminChatOut,
    summary="Natural-language admin command",
)
def admin_chat(
    request: AdminChatIn,
    db: Session = Depends(get_db),
    admin_key: str = Depends(require_admin),
    llm: BaseLLMClient = Depends(get_admin_llm),
) -> AdminChatOut:
    """
    Parse and execute a natural-language admin command.

    Examples: "approve 12", "refuse 12 because closed for maintenance",
    "show pending" (returns clarification).

    The LLM only parses text — it never executes a decision autonomously
    (see AdminAgent safety invariant).
    """
    admin_id = _admin_id(admin_key)
    agent = AdminAgent(lambda: db, llm=llm)

    # Phase 2: parse decision (safety gate: no LLM call unless clear keyword)
    result = agent.parse_decision(request.message, reservation_id=0, admin_id=admin_id)

    if isinstance(result, str):
        # Clarification needed — no decision was extracted
        return AdminChatOut(response=result, decision_made=False)

    # Phase 3: execute the explicit decision
    msg = agent.execute_decision(result, db)

    if msg.type == MessageType.ERROR:
        err_detail = msg.payload.get("error", "Unknown error")
        raise HTTPException(status_code=422, detail=err_detail)

    # The agent committed the decision (and already tried the file write); resume the
    # paused conversation so it records/notifies.  Idempotent: no second line is written.
    sync_decision(db, result.reservation_id)

    notification_text = msg.payload.get("notification_text", "Decision recorded.")
    logger.info(
        "admin_api|chat_decision reservation_id=%d decision=%s",
        result.reservation_id,
        result.decision.value,
    )
    return AdminChatOut(
        response=notification_text,
        decision_made=True,
        reservation_id=result.reservation_id,
        file_written=msg.payload.get("file_written"),
        file_note=msg.payload.get("file_note"),
    )


# ---------------------------------------------------------------------------
# GET /admin/approve?token=...   (one-click email link — no API key)
# ---------------------------------------------------------------------------


@router.get(
    "/approve",
    response_model=ActionResultOut,
    summary="One-click approve via email link",
)
def email_approve(
    token: str = Query(..., description="Signed action token from the approval email"),
    db: Session = Depends(get_db),
) -> dict:
    """
    Consume the signed token, verify it encodes an APPROVE decision, and
    approve the reservation.  No X-Admin-Key required — the token IS the
    authentication.
    """
    # Verify signature and expiry first (does NOT consume the token yet).
    try:
        data = verify(token)
    except TokenExpired:
        raise HTTPException(status_code=410, detail="This link has expired")
    except TokenInvalid as exc:
        raise HTTPException(status_code=400, detail=f"Invalid token: {exc}")

    # Check decision type BEFORE consuming so a mis-routed REFUSE token is
    # not permanently burned against this endpoint.
    if data.get("dec") != "APPROVE":
        raise HTTPException(status_code=400, detail="This link is for APPROVE actions only")

    # Now consume (single-use enforcement).
    try:
        consume(token, db)
    except TokenUsed:
        raise HTTPException(status_code=409, detail="This link has already been used")
    except (TokenExpired, TokenInvalid) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    reservation_id: int = data["rid"]
    try:
        outcome = decide(db, reservation_id, "APPROVE", admin="email-link")
    except ValidationError as exc:
        raise HTTPException(status_code=_repo_http_status(exc), detail=str(exc))

    file_written = outcome.file_written
    logger.info(
        "admin_api|email_approve reservation_id=%d file_written=%s", reservation_id, file_written
    )
    return {
        "message":        f"Reservation #{reservation_id} has been approved",
        "reservation_id": reservation_id,
        "status":         "confirmed",
        "file_written":   file_written,
        "file_note":      export_note(file_written),
    }


# ---------------------------------------------------------------------------
# GET /admin/refuse?token=...   (one-click email link — no API key)
# ---------------------------------------------------------------------------


@router.get(
    "/refuse",
    response_model=ActionResultOut,
    summary="One-click refuse via email link",
)
def email_refuse(
    token: str = Query(..., description="Signed action token from the approval email"),
    db: Session = Depends(get_db),
) -> dict:
    """
    Consume the signed token, verify it encodes a REFUSE decision, and
    refuse the reservation using the reason embedded in the token.
    """
    # Verify before consuming — same pattern as email_approve.
    try:
        data = verify(token)
    except TokenExpired:
        raise HTTPException(status_code=410, detail="This link has expired")
    except TokenInvalid as exc:
        raise HTTPException(status_code=400, detail=f"Invalid token: {exc}")

    if data.get("dec") != "REFUSE":
        raise HTTPException(status_code=400, detail="This link is for REFUSE actions only")

    try:
        consume(token, db)
    except TokenUsed:
        raise HTTPException(status_code=409, detail="This link has already been used")
    except (TokenExpired, TokenInvalid) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    reservation_id: int = data["rid"]
    reason: str = (data.get("rsn") or "").strip() or "Refused via email link"

    try:
        decide(db, reservation_id, "REFUSE", admin="email-link", reason=reason)
    except ValidationError as exc:
        raise HTTPException(status_code=_repo_http_status(exc), detail=str(exc))

    logger.info("admin_api|email_refuse reservation_id=%d", reservation_id)
    return {
        "message":        f"Reservation #{reservation_id} has been refused",
        "reservation_id": reservation_id,
        "status":         "refused",
        "reason":         reason,
    }
