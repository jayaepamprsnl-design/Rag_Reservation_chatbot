"""Repository layer: read helpers, validation, and the full approval workflow."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

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
    if lot.available_spaces <= 0:
        raise ValidationError("Cannot decrement: no available spaces left.")
    lot.available_spaces -= 1


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
    ValidationError is raised — no double-decrement is possible because only
    PENDING reservations can enter this path.
    """
    res = get_reservation(db, reservation_id)

    if res.status != ReservationStatus.PENDING:
        raise ValidationError(
            f"Reservation {reservation_id} cannot be approved: "
            f"current status is '{res.status}' (only PENDING reservations can be approved)."
        )

    lot = get_lot(db, res.lot_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Re-check space availability
    if lot.available_spaces <= 0:
        res.status = ReservationStatus.REFUSED
        res.admin_decision_reason = "No available spaces at time of approval."
        res.decided_at = now
        res.decided_by = admin
        db.commit()
        db.refresh(res)
        raise ValidationError(
            f"No available spaces at '{lot.name}': reservation {reservation_id} auto-refused."
        )

    # Re-check time-slot conflicts (against CONFIRMED reservations only)
    conflicts = check_conflicts(db, res.lot_id, res.start_time, res.end_time)
    if len(conflicts) >= lot.total_spaces:
        res.status = ReservationStatus.REFUSED
        res.admin_decision_reason = (
            "All spaces are booked for the requested time window at approval time."
        )
        res.decided_at = now
        res.decided_by = admin
        db.commit()
        db.refresh(res)
        raise ValidationError(
            f"All spaces at '{lot.name}' are fully booked for the requested window: "
            f"reservation {reservation_id} auto-refused."
        )

    res.status = ReservationStatus.CONFIRMED
    res.admin_decision_reason = note or None
    res.decided_at = now
    res.decided_by = admin
    _decrement_available_spaces(db, lot)
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
    res = get_reservation(db, reservation_id)

    if res.status != ReservationStatus.PENDING:
        raise ValidationError(
            f"Reservation {reservation_id} cannot be refused: "
            f"current status is '{res.status}' (only PENDING reservations can be refused)."
        )

    res.status = ReservationStatus.REFUSED
    res.admin_decision_reason = reason or None
    res.decided_at = datetime.now(timezone.utc).replace(tzinfo=None)
    res.decided_by = admin
    db.commit()
    db.refresh(res)
    return res


def expire_stale_pending(db: Session, hours: int = 24) -> list[LotReservation]:
    """
    Mark PENDING reservations older than *hours* as EXPIRED.

    Returns the list of newly expired reservations.  Does not touch spaces
    (PENDING reservations never held a space).
    """
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    stale = (
        db.query(LotReservation)
        .filter(
            LotReservation.status == ReservationStatus.PENDING,
            LotReservation.created_at < cutoff,
        )
        .all()
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for res in stale:
        res.status = ReservationStatus.EXPIRED
        res.admin_decision_reason = (
            f"Auto-expired after {hours} hours with no admin decision."
        )
        res.decided_at = now
    db.commit()
    return stale


def get_reservations_by_session(db: Session, session_id: str) -> list[LotReservation]:
    """Return all reservations linked to a chat session, newest first."""
    return (
        db.query(LotReservation)
        .filter(LotReservation.session_id == session_id)
        .order_by(LotReservation.created_at.desc())
        .all()
    )
