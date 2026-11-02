"""Repository layer for ParkingLot and LotReservation operations."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import LotReservation, ParkingLot

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
# Validation helpers
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
    """Return active reservations at *lot_id* that overlap the given window."""
    q = (
        db.query(LotReservation)
        .filter(
            LotReservation.lot_id == lot_id,
            LotReservation.status == "confirmed",
            LotReservation.start_time < end_time,
            LotReservation.end_time > start_time,
        )
    )
    if exclude_id is not None:
        q = q.filter(LotReservation.id != exclude_id)
    return q.all()


# ---------------------------------------------------------------------------
# Create reservation
# ---------------------------------------------------------------------------

def create_reservation(
    db: Session,
    lot_id: int,
    customer_name: str,
    car_plate: str,
    start_time: datetime,
    end_time: datetime,
) -> LotReservation:
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
        status="confirmed",
    )
    db.add(reservation)
    _decrement_available_spaces(db, lot)
    db.commit()
    db.refresh(reservation)
    return reservation


# ---------------------------------------------------------------------------
# Space management
# ---------------------------------------------------------------------------

def _decrement_available_spaces(db: Session, lot: ParkingLot) -> None:
    if lot.available_spaces <= 0:
        raise ValidationError("Cannot decrement: no available spaces left.")
    lot.available_spaces -= 1


def cancel_reservation(db: Session, reservation_id: int) -> LotReservation:
    res = db.get(LotReservation, reservation_id)
    if res is None:
        raise ValidationError(f"Reservation {reservation_id} not found.")
    if res.status != "confirmed":
        raise ValidationError(f"Reservation {reservation_id} is already '{res.status}'.")
    res.status = "cancelled"
    lot = get_lot(db, res.lot_id)
    lot.available_spaces = min(lot.available_spaces + 1, lot.total_spaces)
    db.commit()
    db.refresh(res)
    return res
