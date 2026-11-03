"""Tests for the Stage-2 approval-workflow repository functions.

Each test function gets a fresh in-memory database via the function-scoped
`db` fixture, so state mutations never bleed between tests.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, LotReservation, ParkingLot, ReservationStatus
from app.db.repository import (
    ValidationError,
    approve_reservation,
    create_pending_reservation,
    create_reservation,
    expire_stale_pending,
    get_reservation,
    get_reservations_by_session,
    list_pending,
    refuse_reservation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    """Fresh in-memory SQLite with two lots; disposed after each test."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add_all([
        ParkingLot(
            name="MainLot",
            address="1 Main Street",
            total_spaces=10,
            available_spaces=10,
            hourly_price=5.0,
            daily_price=25.0,
            opens_at=time(0, 0),
            closes_at=time(23, 59),
        ),
        ParkingLot(
            name="TinyLot",
            address="1 Tiny Street",
            total_spaces=1,
            available_spaces=1,
            hourly_price=5.0,
            daily_price=25.0,
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


def _lot_id(db, name: str) -> int:
    return db.query(ParkingLot).filter_by(name=name).one().id


# ---------------------------------------------------------------------------
# create_pending_reservation
# ---------------------------------------------------------------------------


def test_create_pending_status_is_pending(db):
    res = create_pending_reservation(
        db, lot_id=_lot_id(db, "MainLot"),
        customer_name="Alice", car_plate="AB-1234",
        start_time=_future(3), end_time=_future(5),
    )
    assert res.id is not None
    assert res.status == ReservationStatus.PENDING


def test_create_pending_does_not_decrement_spaces(db):
    lot = db.query(ParkingLot).filter_by(name="MainLot").one()
    before = lot.available_spaces
    create_pending_reservation(
        db, lot_id=lot.id,
        customer_name="Bob", car_plate="CD-5678",
        start_time=_future(6), end_time=_future(8),
    )
    db.refresh(lot)
    assert lot.available_spaces == before


def test_create_pending_stores_session_id(db):
    res = create_pending_reservation(
        db, lot_id=_lot_id(db, "MainLot"),
        customer_name="Carol", car_plate="EF-9012",
        start_time=_future(10), end_time=_future(12),
        session_id="sess-abc123",
    )
    assert res.session_id == "sess-abc123"


def test_create_pending_invalid_plate_raises(db):
    with pytest.raises(ValidationError, match="[Ii]nvalid"):
        create_pending_reservation(
            db, lot_id=_lot_id(db, "MainLot"),
            customer_name="Dave", car_plate="!!!BAD!!!",
            start_time=_future(3), end_time=_future(5),
        )


def test_create_pending_past_start_raises(db):
    with pytest.raises(ValidationError, match="future"):
        create_pending_reservation(
            db, lot_id=_lot_id(db, "MainLot"),
            customer_name="Eve", car_plate="GH-3456",
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=_future(1),
        )


# ---------------------------------------------------------------------------
# list_pending
# ---------------------------------------------------------------------------


def test_list_pending_returns_only_pending(db):
    lid = _lot_id(db, "MainLot")
    p = create_pending_reservation(
        db, lot_id=lid, customer_name="Frank", car_plate="IJ-7890",
        start_time=_future(20), end_time=_future(22),
    )
    # A directly confirmed reservation should NOT appear
    create_reservation(
        db, lot_id=lid, customer_name="Grace", car_plate="KL-1122",
        start_time=_future(25), end_time=_future(27),
    )
    pending = list_pending(db)
    ids = [r.id for r in pending]
    assert p.id in ids
    assert all(r.status == ReservationStatus.PENDING for r in pending)


def test_list_pending_empty_when_none_pending(db):
    lid = _lot_id(db, "MainLot")
    create_reservation(
        db, lot_id=lid, customer_name="Henry", car_plate="MN-3344",
        start_time=_future(30), end_time=_future(32),
    )
    assert list_pending(db) == []


def test_list_pending_ordered_oldest_first(db):
    lid = _lot_id(db, "MainLot")
    r1 = create_pending_reservation(
        db, lot_id=lid, customer_name="Irene", car_plate="OP-5566",
        start_time=_future(40), end_time=_future(42),
    )
    r2 = create_pending_reservation(
        db, lot_id=lid, customer_name="Jack", car_plate="QR-7788",
        start_time=_future(45), end_time=_future(47),
    )
    result = list_pending(db)
    ids = [r.id for r in result]
    assert ids.index(r1.id) < ids.index(r2.id)


# ---------------------------------------------------------------------------
# get_reservation
# ---------------------------------------------------------------------------


def test_get_reservation_returns_correct(db):
    r = create_pending_reservation(
        db, lot_id=_lot_id(db, "MainLot"),
        customer_name="Karen", car_plate="ST-9900",
        start_time=_future(50), end_time=_future(52),
    )
    fetched = get_reservation(db, r.id)
    assert fetched.id == r.id
    assert fetched.customer_name == "Karen"


def test_get_reservation_raises_for_missing(db):
    with pytest.raises(ValidationError, match="not found"):
        get_reservation(db, 99999)


# ---------------------------------------------------------------------------
# approve_reservation — happy path
# ---------------------------------------------------------------------------


def test_approve_sets_confirmed_and_decrements_spaces(db):
    lot = db.query(ParkingLot).filter_by(name="MainLot").one()
    before = lot.available_spaces
    res = create_pending_reservation(
        db, lot_id=lot.id,
        customer_name="Leo", car_plate="UV-1234",
        start_time=_future(60), end_time=_future(62),
    )
    approved = approve_reservation(db, res.id, admin="admin@example.com", note="Looks good")
    db.refresh(lot)
    assert approved.status == ReservationStatus.CONFIRMED
    assert approved.decided_by == "admin@example.com"
    assert approved.decided_at is not None
    assert lot.available_spaces == before - 1


def test_approve_sets_note_on_reservation(db):
    res = create_pending_reservation(
        db, lot_id=_lot_id(db, "MainLot"),
        customer_name="Mia", car_plate="WX-5678",
        start_time=_future(65), end_time=_future(67),
    )
    approved = approve_reservation(db, res.id, admin="admin", note="VIP customer")
    assert approved.admin_decision_reason == "VIP customer"


# ---------------------------------------------------------------------------
# approve_reservation — double-approval guard
# ---------------------------------------------------------------------------


def test_approve_double_approval_raises(db):
    """Approving an already-CONFIRMED reservation must raise ValidationError."""
    res = create_pending_reservation(
        db, lot_id=_lot_id(db, "MainLot"),
        customer_name="Nina", car_plate="YZ-9012",
        start_time=_future(70), end_time=_future(72),
    )
    approve_reservation(db, res.id, admin="admin")

    with pytest.raises(ValidationError, match="cannot be approved"):
        approve_reservation(db, res.id, admin="admin")


def test_approve_double_does_not_double_decrement(db):
    """Even if approve is called twice, spaces must only be decremented once."""
    lot = db.query(ParkingLot).filter_by(name="MainLot").one()
    before = lot.available_spaces
    res = create_pending_reservation(
        db, lot_id=lot.id,
        customer_name="Omar", car_plate="AB-3456",
        start_time=_future(75), end_time=_future(77),
    )
    approve_reservation(db, res.id, admin="admin")
    with pytest.raises(ValidationError):
        approve_reservation(db, res.id, admin="admin")
    db.refresh(lot)
    assert lot.available_spaces == before - 1   # decremented exactly once


# ---------------------------------------------------------------------------
# approve_reservation — no space at approval time
# ---------------------------------------------------------------------------


def test_approve_no_space_auto_refuses(db):
    """
    When the lot is full at approval time the reservation is auto-refused
    and a ValidationError is raised.
    """
    tiny_id = _lot_id(db, "TinyLot")
    r1 = create_pending_reservation(
        db, lot_id=tiny_id, customer_name="Pam", car_plate="CD-7890",
        start_time=_future(80), end_time=_future(82),
    )
    r2 = create_pending_reservation(
        db, lot_id=tiny_id, customer_name="Quinn", car_plate="EF-1234",
        start_time=_future(83), end_time=_future(85),
    )
    # Approve r1 — this occupies the only space
    approve_reservation(db, r1.id, admin="admin")

    # Approving r2 must fail because no space is left
    with pytest.raises(ValidationError, match="auto-refused"):
        approve_reservation(db, r2.id, admin="admin")

    db.refresh(r2)
    assert r2.status == ReservationStatus.REFUSED
    assert r2.admin_decision_reason is not None


def test_approve_no_space_preserves_lot_spaces(db):
    """After an auto-refuse, available_spaces must not go below zero."""
    tiny_id = _lot_id(db, "TinyLot")
    r1 = create_pending_reservation(
        db, lot_id=tiny_id, customer_name="Rachel", car_plate="GH-5678",
        start_time=_future(90), end_time=_future(92),
    )
    r2 = create_pending_reservation(
        db, lot_id=tiny_id, customer_name="Steve", car_plate="IJ-9012",
        start_time=_future(95), end_time=_future(97),
    )
    approve_reservation(db, r1.id, admin="admin")
    with pytest.raises(ValidationError):
        approve_reservation(db, r2.id, admin="admin")

    lot = db.query(ParkingLot).filter_by(name="TinyLot").one()
    db.refresh(lot)
    assert lot.available_spaces == 0


# ---------------------------------------------------------------------------
# refuse_reservation
# ---------------------------------------------------------------------------


def test_refuse_sets_refused_status(db):
    res = create_pending_reservation(
        db, lot_id=_lot_id(db, "MainLot"),
        customer_name="Tina", car_plate="KL-3456",
        start_time=_future(100), end_time=_future(102),
    )
    refused = refuse_reservation(db, res.id, admin="admin", reason="Policy violation")
    assert refused.status == ReservationStatus.REFUSED
    assert refused.admin_decision_reason == "Policy violation"
    assert refused.decided_by == "admin"
    assert refused.decided_at is not None


def test_refuse_does_not_change_spaces(db):
    lot = db.query(ParkingLot).filter_by(name="MainLot").one()
    before = lot.available_spaces
    res = create_pending_reservation(
        db, lot_id=lot.id,
        customer_name="Uma", car_plate="MN-7890",
        start_time=_future(105), end_time=_future(107),
    )
    refuse_reservation(db, res.id, admin="admin")
    db.refresh(lot)
    assert lot.available_spaces == before


def test_refuse_then_approve_raises(db):
    """A REFUSED reservation cannot be approved afterward."""
    res = create_pending_reservation(
        db, lot_id=_lot_id(db, "MainLot"),
        customer_name="Victor", car_plate="OP-1234",
        start_time=_future(110), end_time=_future(112),
    )
    refuse_reservation(db, res.id, admin="admin", reason="Not eligible")

    with pytest.raises(ValidationError, match="cannot be approved"):
        approve_reservation(db, res.id, admin="admin")


def test_refuse_already_refused_raises(db):
    """Refusing an already-REFUSED reservation must raise ValidationError."""
    res = create_pending_reservation(
        db, lot_id=_lot_id(db, "MainLot"),
        customer_name="Wendy", car_plate="QR-5678",
        start_time=_future(115), end_time=_future(117),
    )
    refuse_reservation(db, res.id, admin="admin")

    with pytest.raises(ValidationError, match="cannot be refused"):
        refuse_reservation(db, res.id, admin="admin")


# ---------------------------------------------------------------------------
# expire_stale_pending
# ---------------------------------------------------------------------------


def test_expire_stale_marks_old_pending(db):
    lid = _lot_id(db, "MainLot")
    old = create_pending_reservation(
        db, lot_id=lid, customer_name="Xena", car_plate="ST-9012",
        start_time=_future(120), end_time=_future(122),
    )
    # Force created_at to be 25 hours in the past
    old.created_at = datetime.utcnow() - timedelta(hours=25)
    db.commit()

    expired = expire_stale_pending(db, hours=24)
    assert any(r.id == old.id for r in expired)
    db.refresh(old)
    assert old.status == ReservationStatus.EXPIRED
    assert old.admin_decision_reason is not None


def test_expire_stale_ignores_recent_pending(db):
    lid = _lot_id(db, "MainLot")
    recent = create_pending_reservation(
        db, lot_id=lid, customer_name="Yves", car_plate="UV-3456",
        start_time=_future(125), end_time=_future(127),
    )
    expired = expire_stale_pending(db, hours=24)
    assert not any(r.id == recent.id for r in expired)
    db.refresh(recent)
    assert recent.status == ReservationStatus.PENDING


def test_expire_stale_does_not_affect_confirmed(db):
    lid = _lot_id(db, "MainLot")
    confirmed = create_reservation(
        db, lot_id=lid, customer_name="Zara", car_plate="WX-7890",
        start_time=_future(130), end_time=_future(132),
    )
    confirmed.created_at = datetime.utcnow() - timedelta(hours=48)
    db.commit()

    expire_stale_pending(db, hours=24)
    db.refresh(confirmed)
    assert confirmed.status == ReservationStatus.CONFIRMED


# ---------------------------------------------------------------------------
# get_reservations_by_session
# ---------------------------------------------------------------------------


def test_get_by_session_returns_all_for_session(db):
    lid = _lot_id(db, "MainLot")
    r1 = create_pending_reservation(
        db, lot_id=lid, customer_name="Alan", car_plate="YZ-1234",
        start_time=_future(140), end_time=_future(142),
        session_id="sess-xyz",
    )
    r2 = create_pending_reservation(
        db, lot_id=lid, customer_name="Alan", car_plate="YZ-1234",
        start_time=_future(145), end_time=_future(147),
        session_id="sess-xyz",
    )
    # A different session — must not appear
    create_pending_reservation(
        db, lot_id=lid, customer_name="Other", car_plate="AB-5678",
        start_time=_future(150), end_time=_future(152),
        session_id="sess-other",
    )
    results = get_reservations_by_session(db, "sess-xyz")
    ids = {r.id for r in results}
    assert r1.id in ids
    assert r2.id in ids
    assert len(results) == 2


def test_get_by_session_returns_empty_for_unknown(db):
    results = get_reservations_by_session(db, "no-such-session")
    assert results == []


def test_get_by_session_newest_first(db):
    lid = _lot_id(db, "MainLot")
    r1 = create_pending_reservation(
        db, lot_id=lid, customer_name="Beth", car_plate="CD-9012",
        start_time=_future(160), end_time=_future(162),
        session_id="sess-order",
    )
    # Backdate r1 so r2 is definitely newer regardless of clock resolution
    r1.created_at = datetime.utcnow() - timedelta(hours=1)
    db.commit()

    r2 = create_pending_reservation(
        db, lot_id=lid, customer_name="Beth", car_plate="CD-9012",
        start_time=_future(165), end_time=_future(167),
        session_id="sess-order",
    )
    results = get_reservations_by_session(db, "sess-order")
    # newest first → r2 before r1
    assert results[0].id == r2.id
    assert results[1].id == r1.id
