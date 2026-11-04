"""
Glue between the approval flow and the MCP file writer.

export_approved(db, id)       called by the CODE PATH of an actual admin approval
retry_pending_writes(...)     sweeps approved rows whose write failed earlier

Invariants
----------
* The reservation is already committed as CONFIRMED before any write is tried;
  a failure here never rolls back or raises into the approval.
* Only admin-approved CONFIRMED rows are eligible (is_admin_approved), so
  PENDING / REFUSED / CANCELLED / EXPIRED rows are never written.
* Success flips file_written=True. The MCP server is idempotent per
  reservation_id, so a repeated attempt can never add a second line.
"""
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy.orm import Session

from app.agents.mcp_client import (
    OK,
    PERMANENT,
    export_configured,
    is_admin_approved,
    run_sync,
    send_reservation,
)
from app.core.logging_config import logger
from app.db.models import LotReservation, ReservationStatus

_INLINE_ATTEMPTS = 2  # keep the admin's HTTP request snappy; the worker retries more

NOTE_WRITTEN = "Reservation recorded in the confirmed-reservations file."
NOTE_QUEUED = (
    "Approval saved, but the file write is pending (MCP server unavailable or not "
    "configured). It will be retried automatically."
)


def export_note(file_written: bool) -> str:
    return NOTE_WRITTEN if file_written else NOTE_QUEUED


def _mark_written(db: Session, res: LotReservation) -> None:
    res.file_written = True
    res.file_written_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()


def export_approved(db: Session, reservation_id: int) -> bool:
    """Try to write one approved reservation. Returns the resulting file_written."""
    try:
        res = db.get(LotReservation, reservation_id)
        if res is not None:
            db.refresh(res)  # another session (a graph resume) may have written it meanwhile
        if res is None or not is_admin_approved(res):
            logger.warning("reservation_writer|not_eligible reservation_id=%s", reservation_id)
            return False
        if res.file_written:
            return True
        if not export_configured():
            return False
        if run_sync(send_reservation(res, max_attempts=_INLINE_ATTEMPTS)) == OK:
            _mark_written(db, res)
            return True
        return False
    except Exception as exc:  # noqa: BLE001 - must never break the approval
        db.rollback()
        logger.error(
            "reservation_writer|export_error reservation_id=%s error=%s",
            reservation_id, type(exc).__name__,
        )
        return False


def pending_ids(db: Session, limit: int = 100) -> list[int]:
    rows = (
        db.query(LotReservation.id)
        .filter(
            LotReservation.status == ReservationStatus.CONFIRMED,
            LotReservation.file_written.is_(False),
            LotReservation.decided_by.isnot(None),
            LotReservation.decided_at.isnot(None),
        )
        .order_by(LotReservation.id)
        .limit(limit)
        .all()
    )
    return [r[0] for r in rows]


def retry_pending_writes(session_factory: Callable[[], Session], limit: int = 100) -> dict:
    """Write every approved-but-unwritten reservation.

    A transient failure (server down) stops the sweep: no point burning
    timeout x backoff on every row; the next sweep resumes.  A PERMANENT rejection
    of one row is counted as failed but the sweep carries on, so a single bad row
    can never block the reservations queued behind it.
    """
    db = session_factory()
    summary = {"attempted": 0, "written": 0, "failed": 0}
    try:
        for rid in pending_ids(db, limit):
            res = db.get(LotReservation, rid)
            if res is None or res.file_written or not is_admin_approved(res):
                continue
            summary["attempted"] += 1
            if not export_configured():
                summary["failed"] += 1
                break
            try:
                outcome = run_sync(send_reservation(res))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "reservation_writer|retry_error reservation_id=%s error=%s",
                    rid, type(exc).__name__,
                )
                outcome = "transient"
            if outcome == OK:
                _mark_written(db, res)
                summary["written"] += 1
            elif outcome == PERMANENT:
                summary["failed"] += 1  # skip this row, keep draining the queue
            else:
                summary["failed"] += 1
                break
    finally:
        db.close()
    if summary["attempted"]:
        logger.info("reservation_writer|retry_sweep %s", summary)
    return summary
