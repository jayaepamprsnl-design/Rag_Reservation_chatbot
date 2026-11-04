"""Tests for app/chat/reservation.py state machine.

Uses an in-memory SQLite database seeded with 3 parking lots.
Each test gets a unique session_id so sessions never bleed across tests.
"""
from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.reservation import (
    ReplyResult,
    ReservationState,
    _mask_plate,
    advance,
    get_state,
    reset,
    start,
)
from app.db.models import Base, ParkingLot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db():
    """In-memory SQLite with 3 seeded lots; shared across all tests in this module."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add_all([
        ParkingLot(
            name="CityPark Central",
            address="123 Central Ave",
            total_spaces=350,
            available_spaces=287,
            hourly_price=8.00,
            daily_price=45.00,
            opens_at=time(6, 0),
            closes_at=time(22, 0),
        ),
        ParkingLot(
            name="CityPark Eastside",
            address="45 Eastside Blvd",
            total_spaces=220,
            available_spaces=194,
            hourly_price=5.00,
            daily_price=28.00,
            opens_at=time(6, 0),
            closes_at=time(22, 0),
        ),
        ParkingLot(
            name="CityPark Airport",
            address="1 Airport Connector Rd",
            total_spaces=480,
            available_spaces=431,
            hourly_price=6.00,
            daily_price=35.00,
            opens_at=time(0, 0),
            closes_at=time(23, 59),
        ),
    ])
    session.commit()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sid():
    """Unique session ID per test."""
    return f"test-{uuid.uuid4().hex[:8]}"


# Convenience datetimes far enough in the future and inside working hours.
def _future(hours_ahead: int = 25, hour: int = 10) -> str:
    dt = datetime.now() + timedelta(hours=hours_ahead)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")


def _future_end(hours_ahead: int = 27, hour: int = 14) -> str:
    dt = datetime.now() + timedelta(hours=hours_ahead)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_to_confirm(sid: str, db) -> ReplyResult:
    """Drive the machine from IDLE → CONFIRM and return the confirm-step reply."""
    start(sid)
    advance(sid, "Jane Smith", db)
    advance(sid, "ABC-1234", db)
    advance(sid, "1", db)
    advance(sid, _future(), db)
    return advance(sid, _future_end(), db)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_start_returns_greeting(sid):
    msg = start(sid)
    assert "name" in msg.lower()
    assert get_state(sid) == ReservationState.ASK_NAME


def test_happy_path_full_flow(sid, db):
    start(sid)

    r = advance(sid, "Jane Smith", db)
    assert r.state == ReservationState.ASK_CAR_PLATE
    assert "plate" in r.message.lower() or "licence" in r.message.lower()

    r = advance(sid, "ABC-1234", db)
    assert r.state == ReservationState.ASK_LOT
    assert "facility" in r.message.lower() or "parking" in r.message.lower()

    r = advance(sid, "1", db)
    assert r.state == ReservationState.ASK_START
    assert "CityPark Central" in r.message

    r = advance(sid, _future(), db)
    assert r.state == ReservationState.ASK_END
    assert "depart" in r.message.lower() or "end" in r.message.lower()

    r = advance(sid, _future_end(), db)
    assert r.state == ReservationState.CONFIRM
    # Summary must show masked plate and customer name
    assert "Jane Smith" in r.message
    assert "ABC****34" in r.message
    assert "CityPark Central" in r.message
    assert "confirm" in r.message.lower()

    r = advance(sid, "confirm", db)
    # Stage-2: confirmation escalates to admin; final state is AWAITING_APPROVAL
    assert r.state == ReservationState.AWAITING_APPROVAL
    assert r.reservation_id is not None
    assert "CPK-" in r.message
    assert r.cancelled is False


def test_plate_masking():
    assert _mask_plate("ABC-1234") == "ABC****34"
    assert _mask_plate("XY12") == "X***"
    assert _mask_plate("LONGPLATE99") == "LON****99"


def test_lot_selection_by_name(sid, db):
    start(sid)
    advance(sid, "Bob Jones", db)
    advance(sid, "XY-9900", db)
    r = advance(sid, "airport", db)   # keyword alias
    assert r.state == ReservationState.ASK_START
    assert "CityPark Airport" in r.message


def test_lot_selection_by_id_2(sid, db):
    start(sid)
    advance(sid, "Alice Lee", db)
    advance(sid, "EF-5678", db)
    r = advance(sid, "2", db)
    assert r.state == ReservationState.ASK_START
    assert "CityPark Eastside" in r.message


def test_confirm_transitions_to_awaiting_approval(sid, db):
    """After confirm, state is AWAITING_APPROVAL (pending admin decision)."""
    _run_to_confirm(sid, db)
    r = advance(sid, "confirm", db)
    assert r.state == ReservationState.AWAITING_APPROVAL
    assert r.reservation_id is not None


def test_awaiting_approval_preserves_reservation_id_on_next_message(sid, db):
    """Messages in AWAITING_APPROVAL return the same reservation_id."""
    _run_to_confirm(sid, db)
    r1 = advance(sid, "confirm", db)
    r2 = advance(sid, "hello", db)
    assert r2.reservation_id == r1.reservation_id
    assert r2.state == ReservationState.AWAITING_APPROVAL


# ---------------------------------------------------------------------------
# Invalid name
# ---------------------------------------------------------------------------


def test_invalid_name_too_short_re_asks(sid, db):
    start(sid)
    r = advance(sid, "X", db)
    assert r.state == ReservationState.ASK_NAME
    assert "name" in r.message.lower()


def test_invalid_name_digits_only_re_asks(sid, db):
    start(sid)
    r = advance(sid, "12345", db)
    assert r.state == ReservationState.ASK_NAME
    assert "letters" in r.message.lower() or "name" in r.message.lower()


def test_valid_name_after_invalid_proceeds(sid, db):
    start(sid)
    advance(sid, "Z", db)  # invalid
    r = advance(sid, "Mary Jones", db)  # valid
    assert r.state == ReservationState.ASK_CAR_PLATE


# ---------------------------------------------------------------------------
# Invalid plate
# ---------------------------------------------------------------------------


def test_invalid_plate_re_asks(sid, db):
    start(sid)
    advance(sid, "Tom Brown", db)
    r = advance(sid, "!!!!", db)
    assert r.state == ReservationState.ASK_CAR_PLATE
    assert "plate" in r.message.lower() or "valid" in r.message.lower()


def test_valid_plate_after_invalid_proceeds(sid, db):
    start(sid)
    advance(sid, "Tom Brown", db)
    advance(sid, "BAD PLATE", db)      # invalid
    r = advance(sid, "GH-7890", db)    # valid
    assert r.state == ReservationState.ASK_LOT


# ---------------------------------------------------------------------------
# Invalid lot
# ---------------------------------------------------------------------------


def test_invalid_lot_re_asks(sid, db):
    start(sid)
    advance(sid, "Sam Green", db)
    advance(sid, "MN-1111", db)
    r = advance(sid, "Narnia", db)
    assert r.state == ReservationState.ASK_LOT
    assert "recognise" in r.message.lower() or "number" in r.message.lower()


def test_valid_lot_after_invalid_proceeds(sid, db):
    start(sid)
    advance(sid, "Sam Green", db)
    advance(sid, "MN-1111", db)
    advance(sid, "UnknownPlace", db)   # invalid
    r = advance(sid, "central", db)   # valid
    assert r.state == ReservationState.ASK_START


# ---------------------------------------------------------------------------
# Invalid start time
# ---------------------------------------------------------------------------


def test_start_in_past_re_asks(sid, db):
    start(sid)
    advance(sid, "Pat Blue", db)
    advance(sid, "OP-2222", db)
    advance(sid, "1", db)
    r = advance(sid, "2020-01-01 09:00", db)   # past
    assert r.state == ReservationState.ASK_START
    assert "future" in r.message.lower() or "minutes" in r.message.lower()


def test_start_outside_working_hours_re_asks(sid, db):
    # CityPark Central closes at 22:00; pick 23:00 far in the future.
    dt = (datetime.now() + timedelta(days=10)).replace(hour=23, minute=0)
    far_future_night = dt.strftime("%Y-%m-%d %H:%M")

    start(sid)
    advance(sid, "Pat Blue", db)
    advance(sid, "QR-3333", db)
    advance(sid, "1", db)   # Central, closes 22:00
    r = advance(sid, far_future_night, db)
    assert r.state == ReservationState.ASK_START
    assert "open" in r.message.lower() or "hours" in r.message.lower()


def test_unparseable_start_re_asks(sid, db):
    start(sid)
    advance(sid, "Alex Red", db)
    advance(sid, "ST-4444", db)
    advance(sid, "2", db)
    r = advance(sid, "not a date at all %%%", db)
    assert r.state == ReservationState.ASK_START
    assert "parse" in r.message.lower() or "date" in r.message.lower()


# ---------------------------------------------------------------------------
# Invalid end time
# ---------------------------------------------------------------------------


def test_end_before_start_re_asks(sid, db):
    start(sid)
    advance(sid, "Ella White", db)
    advance(sid, "UV-5555", db)
    advance(sid, "3", db)   # Airport — open 00:00–23:59
    advance(sid, _future(hours_ahead=25, hour=10), db)
    # End time is the same as start — should fail
    r = advance(sid, _future(hours_ahead=25, hour=10), db)
    assert r.state == ReservationState.ASK_END
    assert "after" in r.message.lower() or "departure" in r.message.lower()


def test_end_too_short_duration_re_asks(sid, db):
    start(sid)
    advance(sid, "Chris Grey", db)
    advance(sid, "WX-6666", db)
    advance(sid, "3", db)
    start_str = _future(hours_ahead=25, hour=10)
    advance(sid, start_str, db)
    # Only 10 minutes later — below minimum
    base = datetime.now() + timedelta(hours=25)
    end_10m = base.replace(hour=10, minute=10).strftime("%Y-%m-%d %H:%M")
    r = advance(sid, end_10m, db)
    assert r.state == ReservationState.ASK_END
    assert "minimum" in r.message.lower() or "30" in r.message


# ---------------------------------------------------------------------------
# Cancel at various steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cancel_word", ["cancel", "quit", "stop", "exit"])
def test_cancel_at_ask_name(sid, db, cancel_word):
    start(sid)
    r = advance(sid, cancel_word, db)
    assert r.cancelled is True
    assert r.state == ReservationState.IDLE


def test_cancel_at_ask_plate(sid, db):
    start(sid)
    advance(sid, "Dana Fox", db)
    r = advance(sid, "cancel", db)
    assert r.cancelled is True
    assert r.state == ReservationState.IDLE


def test_cancel_at_confirm(sid, db):
    r_confirm = _run_to_confirm(sid, db)
    assert r_confirm.state == ReservationState.CONFIRM
    r = advance(sid, "cancel", db)
    assert r.cancelled is True
    assert r.state == ReservationState.IDLE


def test_cancel_clears_session(sid, db):
    start(sid)
    advance(sid, "Dana Fox", db)
    advance(sid, "cancel", db)
    assert get_state(sid) == ReservationState.IDLE


# ---------------------------------------------------------------------------
# No availability / conflict
# ---------------------------------------------------------------------------


def test_no_availability_redirects_to_lot_selection(db):
    """Set lot 1 to 0 available spaces; machine should reject and ask for a new lot."""
    sid = f"noavail-{uuid.uuid4().hex[:6]}"

    # Save and override available_spaces for lot 1
    lot1 = db.query(ParkingLot).filter_by(id=1).one()
    original = lot1.available_spaces
    lot1.available_spaces = 0
    db.commit()

    try:
        start(sid)
        advance(sid, "No Space User", db)
        advance(sid, "NS-0000", db)
        advance(sid, "1", db)
        advance(sid, _future(), db)
        r = advance(sid, _future_end(), db)

        assert r.state == ReservationState.ASK_LOT
        assert "fully booked" in r.message.lower() or "no spaces" in r.message.lower()
        assert "CityPark Central" in r.message
    finally:
        lot1.available_spaces = original
        db.commit()


# ---------------------------------------------------------------------------
# Confirm step validation
# ---------------------------------------------------------------------------


def test_confirm_garbage_input_re_asks(sid, db):
    _run_to_confirm(sid, db)
    r = advance(sid, "maybe", db)
    assert r.state == ReservationState.CONFIRM
    assert "confirm" in r.message.lower()


def test_confirm_aliases_accepted(sid, db):
    _run_to_confirm(sid, db)
    r = advance(sid, "yes", db)
    # Stage-2: confirmation goes to AWAITING_APPROVAL, not DONE
    assert r.state == ReservationState.AWAITING_APPROVAL
    assert r.reservation_id is not None


# ---------------------------------------------------------------------------
# Summary content
# ---------------------------------------------------------------------------


def test_summary_contains_masked_plate(sid, db):
    r = _run_to_confirm(sid, db)
    assert "ABC****34" in r.message


def test_summary_contains_lot_name(sid, db):
    r = _run_to_confirm(sid, db)
    assert "CityPark Central" in r.message


def test_summary_contains_cost_estimate(sid, db):
    r = _run_to_confirm(sid, db)
    assert "$" in r.message


# ---------------------------------------------------------------------------
# Auto-start via advance()
# ---------------------------------------------------------------------------


def test_advance_autostart_idle_session(sid, db):
    # No call to start() — advance() should kick off the flow.
    r = advance(sid, "Auto Start", db)
    # Either it starts and processes the name, or asks for the name.
    assert r.state in (ReservationState.ASK_NAME, ReservationState.ASK_CAR_PLATE)


def test_reset_returns_to_idle(sid, db):
    start(sid)
    advance(sid, "Jane Doe", db)
    assert get_state(sid) == ReservationState.ASK_CAR_PLATE
    reset(sid)
    assert get_state(sid) == ReservationState.IDLE


# ---------------------------------------------------------------------------
# AWAITING_APPROVAL — dedicated tests
# ---------------------------------------------------------------------------

from app.db.repository import approve_reservation, refuse_reservation


def _confirm(sid, db):
    """Drive to AWAITING_APPROVAL and return the ReplyResult."""
    _run_to_confirm(sid, db)
    return advance(sid, "confirm", db)


def test_awaiting_approval_status_query(sid, db):
    _confirm(sid, db)
    r = advance(sid, "what's the status?", db)
    assert r.state == ReservationState.AWAITING_APPROVAL
    assert "PENDING" in r.message.upper() or "status" in r.message.lower()


def test_awaiting_approval_status_query_shows_reference(sid, db):
    rc = _confirm(sid, db)
    r = advance(sid, "status", db)
    assert f"CPK-{rc.reservation_id:06d}" in r.message


def test_awaiting_approval_cancel_returns_idle(sid, db):
    _confirm(sid, db)
    r = advance(sid, "cancel", db)
    assert r.state == ReservationState.IDLE
    assert r.cancelled is True


def test_awaiting_approval_cancel_marks_cancelled_in_db(sid, db):
    from app.db.models import LotReservation, ReservationStatus
    rc = _confirm(sid, db)
    advance(sid, "cancel my request", db)
    res = db.get(LotReservation, rc.reservation_id)
    assert res.status == ReservationStatus.CANCELLED


def test_awaiting_approval_new_booking_attempt_blocked(sid, db):
    _confirm(sid, db)
    r = advance(sid, "I want to book again", db)
    assert r.state == ReservationState.AWAITING_APPROVAL
    assert "pending" in r.message.lower() or "awaiting" in r.message.lower()


def test_awaiting_approval_delivers_approved_notification(sid, db):
    """When admin approves, user's next message receives the APPROVED notice."""
    rc = _confirm(sid, db)
    res_id = rc.reservation_id
    # Simulate admin approval
    approve_reservation(db, res_id, admin="test-admin", note="")
    r = advance(sid, "hi", db)
    assert r.state == ReservationState.DONE
    assert "APPROVED" in r.message.upper()
    assert f"CPK-{res_id:06d}" in r.message


def test_awaiting_approval_delivers_refused_notification(sid, db):
    """When admin refuses, user's next message receives the REFUSED notice."""
    rc = _confirm(sid, db)
    res_id = rc.reservation_id
    refuse_reservation(db, res_id, admin="test-admin", reason="lot closed")
    r = advance(sid, "hi", db)
    assert r.state == ReservationState.IDLE
    assert "lot closed" in r.message.lower()
    assert r.cancelled is True


def test_awaiting_approval_notification_delivered_only_once(sid, db):
    """Notification must be marked as delivered and not repeated."""
    rc = _confirm(sid, db)
    approve_reservation(db, rc.reservation_id, admin="test-admin")
    r1 = advance(sid, "hi", db)
    assert r1.state == ReservationState.DONE
    # Second message — now DONE, no repeat notification
    r2 = advance(sid, "hi again", db)
    assert r2.state == ReservationState.DONE
    assert "APPROVED" not in r2.message.upper()


def test_server_restart_recovery_from_db(sid, db):
    """After clearing in-memory state, advance() recovers from the DB."""
    from app.chat.reservation import _sessions
    rc = _confirm(sid, db)
    # Simulate server restart by clearing in-memory state
    _sessions.pop(sid, None)
    # Next message should recover to AWAITING_APPROVAL
    r = advance(sid, "what happened?", db)
    assert r.state == ReservationState.AWAITING_APPROVAL
    assert r.reservation_id == rc.reservation_id


def test_server_restart_delivers_decision_after_recovery(sid, db):
    """After restart, a pending-but-now-decided reservation is delivered."""
    from app.chat.reservation import _sessions
    rc = _confirm(sid, db)
    approve_reservation(db, rc.reservation_id, admin="test-admin")
    _sessions.pop(sid, None)  # simulate restart
    r = advance(sid, "hello", db)
    assert r.state == ReservationState.DONE
    assert "APPROVED" in r.message.upper()


def test_start_blocked_while_awaiting_approval(sid, db):
    """Calling start() while AWAITING_APPROVAL returns reminder, not reset."""
    _confirm(sid, db)
    msg = start(sid)
    assert get_state(sid) == ReservationState.AWAITING_APPROVAL
    assert "pending" in msg.lower() or "awaiting" in msg.lower() or "CPK-" in msg
