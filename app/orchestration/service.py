"""The API-facing surface of the orchestration layer.

run_turn(...)            one customer message  -> graph.invoke on thread = session_id
decide(...)              admin approve / refuse -> commit the decision, then resume the graph
sync_decision(...)       resume a paused thread from a decision already committed in the DB
                         (admin chat / agent path, expiry, recovery)
expire_stale(...)        EXPIRE PENDING rows older than the TTL and resume their threads
reset_session(...)       forget a conversation (checkpoints)
get_state(...)           debug view of a thread

Order of operations for a decision is deliberate: the database commit comes FIRST
(approve_reservation / refuse_reservation enforce PENDING-only, so a double click
or a second channel fails with the same ValidationError the routes already map to
409), and only then is the graph resumed with that committed decision.  If there is
no paused thread (reservation created before Stage 4, thread reset, graph error)
decide() falls back to the previous direct behaviour, so a decision is never lost.
"""
from __future__ import annotations

import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional

from langgraph.types import Command
from sqlalchemy.orm import Session

from app.agents.reservation_writer import export_approved
from app.chat.reservation import _mask_plate, reset as reset_booking
from app.core.config import settings
from app.core.logging_config import logger
from app.db.models import LotReservation, ReservationStatus
from app.db.repository import approve_reservation, expire_stale_pending, refuse_reservation
from app.orchestration import graph as graph_mod
from app.orchestration.nodes import TurnContext

_STATUS_TO_DECISION = {
    ReservationStatus.CONFIRMED: "APPROVE",
    ReservationStatus.REFUSED: "REFUSE",
    ReservationStatus.EXPIRED: "EXPIRE",
}


class _SessionLock:
    __slots__ = ("lock", "__weakref__")

    def __init__(self) -> None:
        self.lock = threading.RLock()


_locks: "weakref.WeakValueDictionary[str, _SessionLock]" = weakref.WeakValueDictionary()
_locks_guard = threading.Lock()


@contextmanager
def session_lock(session_id: str) -> Iterator[None]:
    """One writer per conversation thread: a customer turn and an admin resume for the same
    session must not interleave on the same checkpoint.  Lock order is always
    session lock -> repository.decision_lock, never the reverse."""
    with _locks_guard:
        holder = _locks.get(session_id)
        if holder is None:
            holder = _locks[session_id] = _SessionLock()
    with holder.lock:
        yield


@dataclass
class TurnResult:
    answer: str
    sources: list[str]
    intent: str
    guardrail_flags: list[str]
    reservation_id: Optional[int]


@dataclass
class DecisionOutcome:
    reservation: LotReservation
    file_written: bool
    via_graph: bool


def _cfg(session_id: str, db: Session, raw_text: str = "") -> dict:
    return {"configurable": {"thread_id": session_id, "turn": TurnContext(db=db, raw_text=raw_text)}}


# ---------------------------------------------------------------------------
# Customer turn
# ---------------------------------------------------------------------------


def run_turn(session_id: str, message: str, db: Session) -> TurnResult:
    graph = graph_mod.get_graph()
    cfg = _cfg(session_id, db, raw_text=message)
    # Per-turn outputs are reset so values from the previous turn never leak into this reply.
    with session_lock(session_id):
        graph.invoke(
                {
                "session_id": session_id,
                "confirmed": False,
                "answer": "",
                "intent": "UNKNOWN",
                "sources": [],
                "guardrail_flags": [],
                "active_reservation_id": None,
                "errors": [],
                "notification": "",
            },
            cfg,
        )
        v = graph.get_state(cfg).values
    return TurnResult(
        answer=v.get("answer", ""),
        sources=v.get("sources", []),
        intent=v.get("intent", "UNKNOWN"),
        guardrail_flags=v.get("guardrail_flags", []),
        reservation_id=v.get("active_reservation_id"),
    )


# ---------------------------------------------------------------------------
# Admin decisions
# ---------------------------------------------------------------------------


def sync_decision(db: Session, reservation_id: int) -> Optional[dict[str, Any]]:
    """Resume the thread paused on *reservation_id* with the decision committed in the DB.

    Returns the final graph state, or None when there is nothing to resume (no thread,
    not paused at admin_approval, or not decided).  Never raises.
    """
    try:
        res = db.get(LotReservation, reservation_id)
        decision = _STATUS_TO_DECISION.get(res.status) if res is not None else None
        if res is None or decision is None or not res.session_id:
            return None
        graph = graph_mod.get_graph()
        cfg = _cfg(res.session_id, db)
        with session_lock(res.session_id):
            snap = graph.get_state(cfg)
            if "admin_approval" not in snap.next or snap.values.get("reservation_id") != reservation_id:
                return None
            graph.invoke(
                Command(resume={
                    "decision": decision,
                    "admin_id": res.decided_by,
                    "reason": res.admin_decision_reason,
                }),
                cfg,
            )
            return graph.get_state(cfg).values
    except Exception as exc:  # noqa: BLE001 - a decision is already committed; never undo it
        logger.error(
            "orchestration|resume_failed reservation_id=%s error=%s", reservation_id, type(exc).__name__
        )
        return None


def decide(
    db: Session,
    reservation_id: int,
    decision: str,
    admin: str,
    reason: str = "",
    note: str = "",
) -> DecisionOutcome:
    """Apply an admin APPROVE / REFUSE and resume the graph.

    Raises repository.ValidationError unchanged (not found / not PENDING / bad reason),
    so callers keep their existing HTTP mapping.
    """
    if decision == "APPROVE":
        res = approve_reservation(db, reservation_id, admin=admin, note=note)
    elif decision == "REFUSE":
        res = refuse_reservation(db, reservation_id, admin=admin, reason=reason)
    else:
        raise ValueError(f"decision must be APPROVE or REFUSE, got {decision!r}")

    final = sync_decision(db, res.id)
    if final is not None and final.get("status") in ("approved", "refused", "expired"):
        file_written = bool(final.get("file_written"))
        via_graph = True
    else:
        # No paused thread, or the run failed after the commit: keep the pre-Stage-4
        # behaviour so an approved reservation is still exported (idempotent).
        file_written = export_approved(db, res.id) if decision == "APPROVE" else False
        via_graph = False
    return DecisionOutcome(reservation=res, file_written=file_written, via_graph=via_graph)


def expire_stale(db: Session, hours: Optional[int] = None) -> int:
    """EXPIRE PENDING requests nobody decided within *hours*; resume their threads."""
    expired = expire_stale_pending(db, hours if hours is not None else settings.approval_ttl_hours)
    for r in expired:
        sync_decision(db, r.id)
    if expired:
        logger.info("orchestration|expired count=%d", len(expired))
    return len(expired)


# ---------------------------------------------------------------------------
# Session management / debug
# ---------------------------------------------------------------------------


def reconcile_decided(db: Session, within_days: int = 2, limit: int = 500) -> int:
    """Resume threads that are still paused although their reservation was already decided.

    Heals the crash window "decision committed, process died before the graph was resumed"
    (and a failed resume).  Runs at startup and on every maintenance sweep.  Cheap: one query
    plus one checkpoint read per recently decided reservation; sync_decision() is a no-op for
    threads that are not paused.
    """
    since = datetime.utcnow() - timedelta(days=within_days)
    rows = (
        db.query(LotReservation.id)
        .filter(
            LotReservation.session_id.isnot(None),
            LotReservation.status.in_(list(_STATUS_TO_DECISION)),
            LotReservation.decided_at.isnot(None),
            LotReservation.decided_at >= since,
        )
        .order_by(LotReservation.id.desc())
        .limit(limit)
        .all()
    )
    healed = sum(1 for (rid,) in rows if sync_decision(db, rid) is not None)
    if healed:
        logger.info("orchestration|reconciled count=%d", healed)
    return healed


def reset_session(session_id: str) -> None:
    with session_lock(session_id):
        graph_mod.delete_thread(session_id)
        reset_booking(session_id)


def get_state(session_id: str, db: Session) -> dict[str, Any]:
    """Sanitised view of a thread for the admin debug endpoint (no names, masked plate)."""
    snap = graph_mod.get_graph().get_state(_cfg(session_id, db))
    v = dict(snap.values or {})
    slots = v.get("reservation") or {}
    return {
        "session_id": session_id,
        "exists": bool(snap.values),
        "next": list(snap.next),
        "interrupted": "admin_approval" in snap.next,
        "status": v.get("status"),
        "intent": v.get("intent"),
        "reservation_id": v.get("reservation_id"),
        "file_written": v.get("file_written"),
        "errors": v.get("errors", []),
        "message_count": len(v.get("messages") or []),
        "booking": {
            "state": slots.get("state"),
            "lot_id": slots.get("lot_id"),
            "lot_name": slots.get("lot_name"),
            "start": slots.get("start_dt"),
            "end": slots.get("end_dt"),
            "car_plate_masked": _mask_plate(slots["car_plate"]) if slots.get("car_plate") else None,
        },
        "checkpoint_id": (snap.config or {}).get("configurable", {}).get("checkpoint_id"),
    }
