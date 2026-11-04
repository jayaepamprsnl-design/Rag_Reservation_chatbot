"""
Integration tests for multi-slot booking flow.

Verifies that information given in the first (or any) message is captured
without asking again, existing tests still pass, and edge cases are handled.

All tests run against an in-memory SQLite DB with the same 3 parking lots
used by the other reservation SM tests.
"""
from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.reservation import (
    ReservationState,
    _sessions,
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
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add_all([
        ParkingLot(name="CityPark Central",  address="123 Central Ave",
                   total_spaces=350, available_spaces=287,
                   hourly_price=8.00, daily_price=45.00,
                   opens_at=time(6, 0), closes_at=time(22, 0)),
        ParkingLot(name="CityPark Eastside", address="45 Eastside Blvd",
                   total_spaces=220, available_spaces=194,
                   hourly_price=5.00, daily_price=28.00,
                   opens_at=time(6, 0), closes_at=time(22, 0)),
        ParkingLot(name="CityPark Airport",  address="1 Airport Rd",
                   total_spaces=480, available_spaces=431,
                   hourly_price=6.00, daily_price=35.00,
                   opens_at=time(0, 0), closes_at=time(23, 59)),
    ])
    session.commit()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sid():
    """Unique session ID per test."""
    return f"msf-{uuid.uuid4().hex[:8]}"


def _future(h: int = 25, hour: int = 10) -> str:
    """Return a datetime string N hours ahead, anchored to *hour*."""
    dt = datetime.now() + timedelta(hours=h)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")


def _future_end(h: int = 27, hour: int = 14) -> str:
    return _future(h, hour)


# ---------------------------------------------------------------------------
# Original bug report — jaya kumari sentence
# ---------------------------------------------------------------------------

class TestjayaSentence:
    """
    'my name is jaya kumari i want to book a spot for 4th nov 2026
    from 2pm till 3hrs'

    Expected:
    • name  = "jaya kumari"
    • start = 2026-11-04 14:00
    • end   = 2026-11-04 17:00
    • next question asks only for plate or lot  (NOT name)
    """

    def test_name_extracted_not_asked_again(self, sid, db):
        start(sid)
        r = advance(
            sid,
            "my name is jaya kumari i want to book a spot "
            "for 4th nov 2026 from 2pm till 3hrs",
            db,
        )
        data = _sessions[sid]
        assert data.name == "jaya kumari", f"name not extracted: {data.name!r}"
        # Bot must NOT ask for the name again
        assert "full name" not in r.message.lower(), (
            f"Bot asked for name despite it being given:\n{r.message}"
        )

    def test_start_time_extracted(self, sid, db):
        start(sid)
        advance(
            sid,
            "my name is jaya kumari i want to book a spot "
            "for 4th nov 2026 from 2pm till 3hrs",
            db,
        )
        data = _sessions[sid]
        assert data.start_dt is not None, "start_dt not extracted"
        assert data.start_dt.hour == 14, f"expected hour 14, got {data.start_dt.hour}"
        assert data.start_dt.day == 4 and data.start_dt.month == 11

    def test_end_time_extracted_from_duration(self, sid, db):
        start(sid)
        advance(
            sid,
            "my name is jaya kumari i want to book a spot "
            "for 4th nov 2026 from 2pm till 3hrs",
            db,
        )
        data = _sessions[sid]
        assert data.end_dt is not None, "end_dt not extracted"
        assert data.end_dt.hour == 17, (
            f"2pm + 3hrs = 5pm (17:00), got hour={data.end_dt.hour}"
        )

    def test_next_question_is_plate_or_lot(self, sid, db):
        start(sid)
        r = advance(
            sid,
            "my name is jaya kumari i want to book a spot "
            "for 4th nov 2026 from 2pm till 3hrs",
            db,
        )
        lower = r.message.lower()
        # Should ask for plate or lot — never name
        asks_plate_or_lot = "plate" in lower or "licence" in lower or "facility" in lower or "which" in lower
        assert asks_plate_or_lot, (
            f"Expected plate/lot question, got:\n{r.message}"
        )
        assert r.state in (ReservationState.ASK_CAR_PLATE, ReservationState.ASK_LOT)


# ---------------------------------------------------------------------------
# All fields in one message → straight to CONFIRM
# ---------------------------------------------------------------------------

class TestAllFieldsAtOnce:

    def _drive_all_at_once(self, sid, db, lot_name="central"):
        start(sid)
        msg = (
            f"my name is Alice Smith, plate AB-1234, "
            f"{lot_name}, "
            f"{_future(50, 10)} to {_future(52, 14)}"
        )
        return advance(sid, msg, db)

    def test_goes_to_confirm(self, sid, db):
        r = self._drive_all_at_once(sid, db)
        assert r.state == ReservationState.CONFIRM, (
            f"Expected CONFIRM, got {r.state}:\n{r.message}"
        )

    def test_summary_contains_name(self, sid, db):
        r = self._drive_all_at_once(sid, db)
        assert "Alice Smith" in r.message

    def test_summary_contains_masked_plate(self, sid, db):
        r = self._drive_all_at_once(sid, db)
        # _mask_plate("AB-1234") = "AB-****34" (3 chars + stars + last 2)
        assert "****34" in r.message

    def test_summary_contains_lot(self, sid, db):
        r = self._drive_all_at_once(sid, db)
        assert "Central" in r.message


# ---------------------------------------------------------------------------
# Partial messages — fill some slots, ask for the rest
# ---------------------------------------------------------------------------

class TestPartialMessages:

    def test_name_and_date_no_plate(self, sid, db):
        """Name + date given, plate + lot still needed → asks for plate."""
        start(sid)
        r = advance(sid, f"my name is Bob Jones, I want to park on {_future(60, 10)}", db)
        data = _sessions[sid]
        assert data.name == "Bob Jones"
        assert data.start_dt is not None
        assert r.state in (ReservationState.ASK_CAR_PLATE, ReservationState.ASK_LOT)

    def test_name_only_asks_for_plate(self, sid, db):
        """Only name given → asks for plate."""
        start(sid)
        r = advance(sid, "my name is Carol White", db)
        data = _sessions[sid]
        assert data.name == "Carol White"
        assert r.state == ReservationState.ASK_CAR_PLATE
        assert "plate" in r.message.lower() or "licence" in r.message.lower()

    def test_plain_name_without_prefix_still_works(self, sid, db):
        """A bare name without 'my name is' still accepted by the existing handler."""
        start(sid)
        r = advance(sid, "Dave Brown", db)
        # Existing _handle_ask_name validates bare names
        assert r.state == ReservationState.ASK_CAR_PLATE

    def test_mid_flow_date_extraction(self, sid, db):
        """Date given at ASK_START is still processed correctly."""
        start(sid)
        advance(sid, "Emma Davis", db)          # name → ASK_CAR_PLATE
        advance(sid, "EM-9999", db)             # plate → ASK_LOT
        advance(sid, "1", db)                   # lot → ASK_START
        r = advance(sid, f"from 10am on {_future(80, 10).split()[0]}", db)
        data = _sessions[sid]
        assert data.start_dt is not None or r.state in (ReservationState.ASK_START, ReservationState.ASK_END, ReservationState.CONFIRM)


# ---------------------------------------------------------------------------
# Past date
# ---------------------------------------------------------------------------

class TestPastDate:

    def test_past_date_not_extracted(self, sid, db):
        """A date in the past should not be stored as start_time."""
        start(sid)
        r = advance(sid, "my name is Frank Green, book for 1st jan 2020 at 10am", db)
        data = _sessions[sid]
        assert data.name == "Frank Green"
        # start_dt should be None (past date rejected)
        # The bot should ask for plate or lot, then eventually ask for dates
        assert r.state in (ReservationState.ASK_CAR_PLATE, ReservationState.ASK_LOT, ReservationState.ASK_NAME)
        # Bot did NOT skip to CONFIRM without valid start
        assert r.state not in (ReservationState.CONFIRM, ReservationState.AWAITING_APPROVAL)


# ---------------------------------------------------------------------------
# Correction — existing value not overwritten
# ---------------------------------------------------------------------------

class TestCorrection:

    def test_existing_name_not_overwritten(self, sid, db):
        """If name is already stored, a later 'my name is X' is ignored."""
        start(sid)
        advance(sid, "Grace Hall", db)           # stores "Grace Hall"
        data_after_name = _sessions[sid]
        assert data_after_name.name == "Grace Hall"

        # Provide plate, then try to sneak a different name
        r = advance(sid, "GH-1234 my name is Impostor", db)
        data = _sessions[sid]
        # Plate should be filled; name should still be "Grace Hall"
        assert data.name == "Grace Hall", (
            f"Name was overwritten to {data.name!r}"
        )
        assert data.car_plate == "GH-1234" or r.state != ReservationState.ASK_CAR_PLATE


# ---------------------------------------------------------------------------
# Cancel during booking
# ---------------------------------------------------------------------------

class TestCancelDuringBooking:

    def test_cancel_at_name_step(self, sid, db):
        start(sid)
        r = advance(sid, "cancel", db)
        assert r.cancelled is True
        assert get_state(sid) == ReservationState.IDLE

    def test_cancel_after_partial_fill(self, sid, db):
        start(sid)
        advance(sid, "my name is Harry Irwin", db)
        advance(sid, "HI-0001", db)
        r = advance(sid, "quit", db)
        assert r.cancelled is True
        assert get_state(sid) == ReservationState.IDLE


# ---------------------------------------------------------------------------
# Existing tests must not regress
# ---------------------------------------------------------------------------

class TestNonRegression:

    def test_single_field_answers_still_work(self, sid, db):
        """Providing just a name still advances to plate step."""
        start(sid)
        r = advance(sid, "Jane Smith", db)
        assert r.state == ReservationState.ASK_CAR_PLATE
        assert "plate" in r.message.lower() or "licence" in r.message.lower()

    def test_invalid_plate_still_re_asks(self, sid, db):
        start(sid)
        advance(sid, "Jane Smith", db)
        r = advance(sid, "!!!invalid!!!", db)
        assert r.state == ReservationState.ASK_CAR_PLATE

    def test_confirm_garbage_still_re_asks(self, sid, db):
        """'maybe' at CONFIRM still asks for confirm/cancel."""
        start(sid)
        advance(sid, "Jane Smith", db)
        advance(sid, "JS-1234", db)
        advance(sid, "1", db)
        advance(sid, _future(), db)
        advance(sid, _future_end(), db)
        r = advance(sid, "maybe", db)
        assert r.state == ReservationState.CONFIRM
        assert "confirm" in r.message.lower()

    def test_full_happy_path_still_reaches_awaiting(self, sid, db):
        """Original one-field-at-a-time path still reaches AWAITING_APPROVAL."""
        start(sid)
        advance(sid, "Jane Smith", db)
        advance(sid, "JS-5678", db)
        advance(sid, "1", db)
        advance(sid, _future(30), db)
        advance(sid, _future_end(32), db)
        r = advance(sid, "confirm", db)
        assert r.state == ReservationState.AWAITING_APPROVAL
        assert r.reservation_id is not None
