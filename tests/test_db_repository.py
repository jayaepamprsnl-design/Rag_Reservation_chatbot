"""Tests for app/db/repository.py — CRUD, validation, conflicts, cancel."""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, ParkingLot
from app.db.repository import (
    ValidationError,
    cancel_reservation,
    check_conflicts,
    create_reservation,
    get_availability,
    get_lot,
    get_prices,
    get_working_hours,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add_all([
        ParkingLot(
            name="TestLot A",
            address="1 Test Street",
            total_spaces=20,
            available_spaces=10,
            hourly_price=5.00,
            daily_price=30.00,
            opens_at=time(6, 0),
            closes_at=time(22, 0),
        ),
        ParkingLot(
            name="FullLot B",
            address="2 Full Street",
            total_spaces=5,
            available_spaces=0,
            hourly_price=8.00,
            daily_price=40.00,
            opens_at=time(0, 0),
            closes_at=time(23, 59),
        ),
    ])
    session.commit()
    yield session
    session.close()
    engine.dispose()


def _future(hours: int = 2) -> datetime:
    return datetime.utcnow() + timedelta(hours=hours)


# ---------------------------------------------------------------------------
# get_lot
# ---------------------------------------------------------------------------


def test_get_lot_returns_correct_lot(db):
    lot = get_lot(db, 1)
    assert lot.name == "TestLot A"


def test_get_lot_raises_for_missing_id(db):
    with pytest.raises(ValidationError, match="not found"):
        get_lot(db, 9999)


# ---------------------------------------------------------------------------
# get_availability
# ---------------------------------------------------------------------------


def test_get_availability_returns_counts(db):
    av = get_availability(db, 1)
    assert av["total_spaces"] == 20
    assert av["available_spaces"] == 10
    assert av["is_open"] is True


def test_get_availability_full_lot_is_not_open(db):
    av = get_availability(db, 2)
    assert av["available_spaces"] == 0
    assert av["is_open"] is False


def test_get_availability_includes_lot_id(db):
    av = get_availability(db, 1)
    assert "lot_id" in av
    assert av["lot_id"] == 1


# ---------------------------------------------------------------------------
# get_prices
# ---------------------------------------------------------------------------


def test_get_prices_returns_rates(db):
    pr = get_prices(db, 1)
    assert pr["hourly_price"] == 5.00
    assert pr["daily_price"] == 30.00


def test_get_prices_has_currency(db):
    pr = get_prices(db, 1)
    assert pr["currency"] == "USD"


def test_get_prices_lot_name_present(db):
    pr = get_prices(db, 1)
    assert "name" in pr
    assert pr["name"] == "TestLot A"


# ---------------------------------------------------------------------------
# get_working_hours
# ---------------------------------------------------------------------------


def test_get_working_hours_string_format(db):
    wh = get_working_hours(db, 1)
    assert wh["opens_at"] == "06:00"
    assert wh["closes_at"] == "22:00"


def test_get_working_hours_airport_24h(db):
    wh = get_working_hours(db, 2)
    assert wh["opens_at"] == "00:00"


# ---------------------------------------------------------------------------
# create_reservation
# ---------------------------------------------------------------------------


def test_create_reservation_success(db):
    res = create_reservation(
        db, lot_id=1, customer_name="Alice",
        car_plate="AB-1234",
        start_time=_future(3),
        end_time=_future(5),
    )
    assert res.id is not None
    assert res.status == "confirmed"
    assert res.car_plate == "AB-1234"


def test_create_reservation_decrements_available_spaces(db):
    before = get_availability(db, 1)["available_spaces"]
    create_reservation(
        db, lot_id=1, customer_name="Bob",
        car_plate="CD-5678",
        start_time=_future(6),
        end_time=_future(8),
    )
    after = get_availability(db, 1)["available_spaces"]
    assert after == before - 1


def test_create_reservation_invalid_plate(db):
    with pytest.raises(ValidationError, match="[Ii]nvalid"):
        create_reservation(
            db, lot_id=1, customer_name="Bad Plate",
            car_plate="!!!INVALID!!!",
            start_time=_future(10),
            end_time=_future(12),
        )


def test_create_reservation_past_start_rejected(db):
    with pytest.raises(ValidationError, match="future"):
        create_reservation(
            db, lot_id=1, customer_name="Past User",
            car_plate="EF-9012",
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=_future(1),
        )


def test_create_reservation_end_before_start_rejected(db):
    start = _future(15)
    end   = start - timedelta(hours=1)
    with pytest.raises(ValidationError):
        create_reservation(
            db, lot_id=1, customer_name="Bad Times",
            car_plate="GH-3456",
            start_time=start,
            end_time=end,
        )


def test_create_reservation_no_spaces_raises(db):
    with pytest.raises(ValidationError, match="[Nn]o available"):
        create_reservation(
            db, lot_id=2, customer_name="Full Lot User",
            car_plate="IJ-7890",
            start_time=_future(20),
            end_time=_future(22),
        )


# ---------------------------------------------------------------------------
# check_conflicts
# ---------------------------------------------------------------------------


def test_check_conflicts_no_conflicts_for_far_future(db):
    conflicts = check_conflicts(
        db, lot_id=1,
        start_time=_future(100),
        end_time=_future(102),
    )
    assert isinstance(conflicts, list)
    assert len(conflicts) == 0


def test_check_conflicts_detects_overlap(db):
    start = _future(50)
    end   = start + timedelta(hours=3)
    res = create_reservation(
        db, lot_id=1, customer_name="Overlap User",
        car_plate="KL-1122",
        start_time=start,
        end_time=end,
    )
    # Query window that overlaps
    overlap_start = start + timedelta(hours=1)
    overlap_end   = end   + timedelta(hours=1)
    conflicts = check_conflicts(db, lot_id=1, start_time=overlap_start, end_time=overlap_end)
    assert any(c.id == res.id for c in conflicts)


def test_check_conflicts_exclude_id(db):
    start = _future(60)
    end   = start + timedelta(hours=2)
    res = create_reservation(
        db, lot_id=1, customer_name="Exclude User",
        car_plate="MN-3344",
        start_time=start,
        end_time=end,
    )
    # Should be excluded from conflicts
    conflicts = check_conflicts(
        db, lot_id=1, start_time=start, end_time=end, exclude_id=res.id
    )
    assert not any(c.id == res.id for c in conflicts)


# ---------------------------------------------------------------------------
# cancel_reservation
# ---------------------------------------------------------------------------


def test_cancel_reservation_sets_cancelled(db):
    res = create_reservation(
        db, lot_id=1, customer_name="Cancel User",
        car_plate="OP-5566",
        start_time=_future(70),
        end_time=_future(72),
    )
    cancelled = cancel_reservation(db, res.id)
    assert cancelled.status == "cancelled"


def test_cancel_reservation_restores_available_spaces(db):
    before_avail = get_availability(db, 1)["available_spaces"]
    res = create_reservation(
        db, lot_id=1, customer_name="Restore User",
        car_plate="QR-7788",
        start_time=_future(80),
        end_time=_future(82),
    )
    cancel_reservation(db, res.id)
    after_avail = get_availability(db, 1)["available_spaces"]
    assert after_avail == before_avail


def test_cancel_already_cancelled_raises(db):
    res = create_reservation(
        db, lot_id=1, customer_name="Double Cancel",
        car_plate="ST-9900",
        start_time=_future(90),
        end_time=_future(92),
    )
    cancel_reservation(db, res.id)
    with pytest.raises(ValidationError, match="already"):
        cancel_reservation(db, res.id)


def test_cancel_nonexistent_raises(db):
    with pytest.raises(ValidationError, match="not found"):
        cancel_reservation(db, 99999)
