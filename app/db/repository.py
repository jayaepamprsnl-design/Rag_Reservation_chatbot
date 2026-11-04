"""Repository layer: read helpers, validation, and the full approval workflow."""
from __future__ import annotations

import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import update
from sqlalchemy.orm import Session

from app.db.models import LotReservation, ParkingLot, ReservationStatus

_PLATE_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,8}[A-Z0-9]$", re.IGNORECASE)


class ValidationError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_lot(db: Session, lot_id: int) -> ParkingLot:
    lot = db.get(ParkingLot, lot_id)
    if lot is None:
        raise ValidationError(f"Parking lot {lot_id} not found.")
    return lot


def get_availability(db: Session, lot_id: int) -> dict:
    lot = get_lot(db, lot_id)
    return {
        "lot_id": lot.id,
        "name": lot.name,
        "total_spaces": lot.total_spaces,
        "available_spaces": lot.available_spaces,
        "is_open": lot.available_spaces > 0,
    }


def get_prices(db: Session, lot_id: int) -> dict:
    lot = get_lot(db, lot_id)
    return {
        "lot_id": lot.id,
        "name": lot.name,
        "hourly_price": lot.hourly_price,
        "daily_price": lot.daily_price,
        "currency": "USD",
    }


def get_working_hours(db: Session, lot_id: int) -> dict:
    lot = get_lot(db, lot_id)
    return {
        "lot_id": lot.id,
        "name": lot.name,
        "opens_at": lot.opens_at.strftime("%H:%M"),
        "closes_at": lot.closes_at.strftime("%H:%M"),
    }


# ---------------------------------------------------------------------------
# Validation helpers (private)
# ---------------------------------------------------------------------------

def _validate_plate(plate: str) -> str:
    plate = plate.strip().upper()
    if not _PLATE_RE.match(plate):
        raise ValidationError(
            f"Invalid plate format '{plate}'. Use letters, digits, and hyphens (e.g. ABC-1234)."
        )
    return plate


def _validate_times(start_time: datetime, end_time: datetime) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if start_time <= now:
        raise ValidationError("start_time must be in the future.")
    if end_time <= start_time:
        raise ValidationError("end_time must be after start_time.")


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def check_conflicts(
    db: Session,
    lot_id: int,
    start_time: datetime,
    end_time: datetime,
    exclude_id: Optional[int] = None,
) -> list[LotReservation]:
    """Return CONFIRMED reservations at *lot_id* that overlap the given window."""
    q = (
        db.query(LotReservation)
        .filter(
            LotReservation.lot_id == lot_id,
            LotReservation.status == ReservationStatus.CONFIRMED,
            LotReservation.start_time < end_time,
            LotReservation.end_time > start_time,
        )
    )
    if exclude_id is not None:
        q = q.filter(LotReservation.id != exclude_id)
    return q.all()


# ---------------------------------------------------------------------------
# Space management (private)
# ---------------------------------------------------------------------------

def _decrement_available_spaces(db: Session, lot: ParkingLot) -> None:  # noqa: D401
    """Take one space, atomically: `available_spaces = available_spaces - 1 WHERE > 0`.

    A Python-side `lot.available_spaces -= 1` is a read-modify-write that loses updates
    when two requests run at once (measured: 12 concurrent approvals consumed 2 spaces).
    """
    taken = db.execute(
        update(ParkingLot)
        .where(ParkingLot.id == lot.id, ParkingLot.available_spaces > 0)
        .values(available_spaces=ParkingLot.available_spaces - 1)
        .execution_options(synchronize_session=False)
    ).rowcount
    if taken != 1:
        raise ValidationError("Cannot decrement: no available spaces left.")
    db.refresh(lot)


# ---------------------------------------------------------------------------
# Stage-1 path: create a reservation that is immediately CONFIRMED
# ---------------------------------------------------------------------------

def create_reservation(
    db: Session,
    lot_id: int,
    customer_name: str,
    car_plate: str,
    start_time: datetime,
    end_time: datetime,
) -> LotReservation:
    """Create and immediately confirm a reservation (Stage-1 direct-confirm path)."""
    car_plate = _validate_plate(car_plate)
    _validate_times(start_time, end_time)

    lot = get_lot(db, lot_id)
    if lot.available_spaces <= 0:
        raise ValidationError(f"No available spaces at '{lot.name}'.")

    conflicts = check_conflicts(db, lot_id, start_time, end_time)
    if len(conflicts) >= lot.total_spaces:
        raise ValidationError(
            f"All spaces at '{lot.name}' are booked for the requested window."
        )

    reservation = LotReservation(
        lot_id=lot_id,
        customer_name=customer_name.strip(),
        car_plate=car_plate,
        start_time=start_time,
        end_time=end_time,
        status=ReservationStatus.CONFIRMED,
    )
    db.add(reservation)
    _decrement_available_spaces(db, lot)
    db.commit()
    db.refresh(reservation)
    return reservation


def cancel_reservation(db: Session, reservation_id: int) -> LotReservation:
    res = db.get(LotReservation, reservation_id)
    if res is None:
        raise ValidationError(f"Reservation {reservation_id} not found.")
    if res.status != ReservationStatus.CONFIRMED:
        raise ValidationError(f"Reservation {reservation_id} is already '{res.status}'.")
    res.status = ReservationStatus.CANCELLED
    lot = get_lot(db, res.lot_id)
    lot.available_spaces = min(lot.available_spaces + 1, lot.total_spaces)
    db.commit()
    db.refresh(res)
    return res


# ---------------------------------------------------------------------------
# Stage-2 path: approval workflow
# ---------------------------------------------------------------------------

def create_pending_reservation(
    db: Session,
    lot_id: int,
    customer_name: str,
    car_plate: str,
    start_time: datetime,
    end_time: datetime,
    session_id: Optional[str] = None,
) -> LotReservation:
    """
    Create a reservation in PENDING state.

    Does NOT decrement available_spaces; that only happens on approval.
    Validates plate format and times but does NOT check availability —
    availability is re-checked at approval time to avoid false rejections.
    """
    car_plate = _validate_plate(car_plate)
    _validate_times(start_time, end_time)
    get_lot(db, lot_id)  # ensure lot exists

    reservation = LotReservation(
        lot_id=lot_id,
        customer_name=customer_name.strip(),
        car_plate=car_plate,
        start_time=start_time,
        end_time=end_time,
        status=ReservationStatus.PENDING,
        session_id=session_id,
    )
    db.add(reservation)
    db.commit()
    db.refresh(reservation)
    return reservation


def get_reservation(db: Session, reservation_id: int) -> LotReservation:
    """Return a reservation by ID, or raise ValidationError if not found."""
    res = db.get(LotReservation, reservation_id)
    if res is None:
        raise ValidationError(f"Reservation {reservation_id} not found.")
    return res


def list_pending(db: Session) -> list[LotReservation]:
    """Return all PENDING reservations ordered by creation time (oldest first)."""
    return (
        db.query(LotReservation)
        .filter(LotReservation.status == ReservationStatus.PENDING)
        .order_by(LotReservation.created_at)
        .all()
    )


# Serialises every decision on a reservation (approve / refuse / expire / customer
# cancel) inside this process, so the read -> check -> write sequences below cannot
# interleave.  It is the *only* protection for the overlap check; the status change and
# the space counter are additionally atomic in SQL (see _claim_pending / _decrement...),
# so two processes can never both win the same reservation or the last space either.
# The app keeps booking state in process memory, so it is deployed as ONE worker anyway.
decision_lock = threading.RLock()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _claim_pending(db: Session, reservation_id: int, **values) -> bool:
    """Atomically move a PENDING row to a decided state; False if it was no longer PENDING."""
    claimed = db.execute(
        update(LotReservation)
        .where(
            LotReservation.id == reservation_id,
            LotReservation.status == ReservationStatus.PENDING,
        )
        .values(**values)
        .execution_options(synchronize_session=False)
    ).rowcount
    return claimed == 1


def _not_pending(db: Session, res: LotReservation, verb: str) -> ValidationError:
    db.rollback()
    db.refresh(res)
    return ValidationError(
        f"Reservation {res.id} cannot be {verb}: "
        f"current status is '{res.status}' (only PENDING reservations can be {verb})."
    )


def _auto_refuse(
    db: Session, res: LotReservation, admin: str, reason: str, message: str
) -> ValidationError:
    """Record an automatic refusal (no capacity) and return the error to raise."""
    db.rollback()
    if not _claim_pending(
        db, res.id,
        status=ReservationStatus.REFUSED, admin_decision_reason=reason,
        decided_at=_now(), decided_by=admin,
    ):
        return _not_pending(db, res, "approved")
    db.commit()
    db.refresh(res)
    return ValidationError(message)


def approve_reservation(
    db: Session,
    reservation_id: int,
    admin: str,
    note: str = "",
) -> LotReservation:
    """
    Approve a PENDING reservation.

    Re-checks availability and conflicts at approval time so that concurrent
    approvals of overlapping slots stay consistent.  If the lot is full at
    approval time, the reservation is auto-refused (status=REFUSED) and a
    ValidationError is raised.  Only a PENDING row can be claimed, atomically,
    so a double click or two channels deciding at once yields exactly one winner.
    """
    with decision_lock:
        res = get_reservation(db, reservation_id)
        db.refresh(res)  # never trust a cached row: another request may have decided it
        if res.status != ReservationStatus.PENDING:
            raise _not_pending(db, res, "approved")

        lot = get_lot(db, res.lot_id)
        db.refresh(lot)

        if lot.available_spaces <= 0:
            raise _auto_refuse(
                db, res, admin, "No available spaces at time of approval.",
                f"No available spaces at '{lot.name}': reservation {reservation_id} auto-refused.",
            )

        conflicts = check_conflicts(db, res.lot_id, res.start_time, res.end_time)
        if len(conflicts) >= lot.total_spaces:
            raise _auto_refuse(
                db, res, admin,
                "All spaces are booked for the requested time window at approval time.",
                f"All spaces at '{lot.name}' are fully booked for the requested window: "
                f"reservation {reservation_id} auto-refused.",
            )

        # Claim the row and take the space in ONE transaction: both or neither.
        if not _claim_pending(
            db, res.id,
            status=ReservationStatus.CONFIRMED, admin_decision_reason=note or None,
            decided_at=_now(), decided_by=admin,
        ):
            raise _not_pending(db, res, "approved")
        try:
            _decrement_available_spaces(db, lot)
        except ValidationError:
            raise _auto_refuse(
                db, res, admin, "No available spaces at time of approval.",
                f"No available spaces at '{lot.name}': reservation {reservation_id} auto-refused.",
            ) from None
        db.commit()
        db.refresh(res)
        return res


def refuse_reservation(
    db: Session,
    reservation_id: int,
    admin: str,
    reason: str = "",
) -> LotReservation:
    """
    Refuse a PENDING reservation.  Does not touch available_spaces.
    Raises ValidationError if the reservation is not in PENDING state.
    """
    with decision_lock:
        res = get_reservation(db, reservation_id)
        db.refresh(res)
        if res.status != ReservationStatus.PENDING:
            raise _not_pending(db, res, "refused")
        if not _claim_pending(
            db, res.id,
            status=ReservationStatus.REFUSED, admin_decision_reason=reason or None,
            decided_at=_now(), decided_by=admin,
        ):
            raise _not_pending(db, res, "refused")
        db.commit()
        db.refresh(res)
        return res


def cancel_pending_reservation(
    db: Session, reservation_id: int, reason: str = "Cancelled by customer"
) -> bool:
    """The customer withdraws a request.  True only if it was still PENDING (atomic)."""
    with decision_lock:
        claimed = _claim_pending(
            db, reservation_id,
            status=ReservationStatus.CANCELLED, admin_decision_reason=reason, decided_at=_now(),
        )
        db.commit()
        return claimed


def expire_stale_pending(db: Session, hours: int = 24) -> list[LotReservation]:
    """
    Mark PENDING reservations older than *hours* as EXPIRED.

    Returns the list of newly expired reservations.  Does not touch spaces
    (PENDING reservations never held a space).  Each row is claimed atomically, so a
    request approved a moment earlier is never overwritten.
    """
    cutoff = _now() - timedelta(hours=hours)
    with decision_lock:
        stale = (
            db.query(LotReservation)
            .filter(
                LotReservation.status == ReservationStatus.PENDING,
                LotReservation.created_at < cutoff,
            )
            .all()
        )
        now = _now()
        expired = [
            res for res in stale
            if _claim_pending(
                db, res.id,
                status=ReservationStatus.EXPIRED,
                admin_decision_reason=f"Auto-expired after {hours} hours with no admin decision.",
                decided_at=now,
            )
        ]
        db.commit()
        for res in expired:
            db.refresh(res)
        return expired


def get_reservations_by_session(db: Session, session_id: str) -> list[LotReservation]:
    """Return all reservations linked to a chat session, newest first."""
    return (
        db.query(LotReservation)
        .filter(LotReservation.session_id == session_id)
        .order_by(LotReservation.created_at.desc())
        .all()
    )
