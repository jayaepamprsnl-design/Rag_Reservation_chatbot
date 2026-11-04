"""
End-to-end approval workflow scenarios.

Run with:  pytest tests/test_e2e_approval.py -v -s

The -s flag disables output capture so the full conversation is visible.
Each scenario prints the exact messages both agents and users see.

Scenarios
---------
1. User books → Admin approves → User notified (CPK reference, facility, times)
2. User books → Admin refuses with reason → User offered re-booking
"""
from __future__ import annotations

import sys
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
from app.db.repository import approve_reservation, refuse_reservation
from app.guardrails.output_guard import check_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add_all([
        ParkingLot(
            name="CityPark Central",
            address="123 Central Avenue",
            total_spaces=350,
            available_spaces=287,
            hourly_price=8.00,
            daily_price=45.00,
            opens_at=time(6, 0),
            closes_at=time(22, 0),
        ),
        ParkingLot(
            name="CityPark Eastside",
            address="45 Eastside Boulevard",
            total_spaces=220,
            available_spaces=194,
            hourly_price=5.00,
            daily_price=28.00,
            opens_at=time(6, 0),
            closes_at=time(22, 0),
        ),
    ])
    session.commit()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def sid():
    return f"e2e-{uuid.uuid4().hex[:8]}"


def _future(h: int = 25, hour: int = 10) -> str:
    dt = datetime.now() + timedelta(hours=h)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")


def _future_end(h: int = 27, hour: int = 14) -> str:
    dt = datetime.now() + timedelta(hours=h)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")


def _sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _safe(text: str) -> str:
    """Replace characters that can't be printed on the current terminal."""
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc)


def _turn(role: str, text: str) -> None:
    label = "[USER]" if role == "user" else "[BOT] "
    lines = _safe(text).strip().split("\n")
    print(f"\n  {label}  {lines[0]}")
    for line in lines[1:]:
        print(f"          {line}")


def _bot(reply, skip_entities=None) -> str:
    """Run reply through output guardrail and return safe text."""
    guard = check_output(reply.message, skip_entities=skip_entities)
    return guard.text


def _run_booking(sid, db, *, name="Alice Johnson", plate="ABC-1234", lot="1") -> int:
    """Drive user through the booking form and return the pending reservation ID."""
    start(sid)
    advance(sid, name, db)
    advance(sid, plate, db)
    advance(sid, lot, db)
    advance(sid, _future(), db)
    advance(sid, _future_end(), db)
    r = advance(sid, "confirm", db)
    assert r.state == ReservationState.AWAITING_APPROVAL, f"Expected AWAITING_APPROVAL, got {r.state}"
    return r.reservation_id


# ===========================================================================
# SCENARIO 1 — User books → Admin approves → User notified
# ===========================================================================

def test_scenario_approve(sid, db):
    _sep("SCENARIO 1: User books -> Admin approves -> User notified")

    # ── Booking flow ─────────────────────────────────────────────────────────
    msg = start(sid)
    _turn("bot", msg)

    _turn("user", "Alice Johnson")
    r = advance(sid, "Alice Johnson", db)
    _turn("bot", r.message)
    assert r.state == ReservationState.ASK_CAR_PLATE

    _turn("user", "ABC-1234")
    r = advance(sid, "ABC-1234", db)
    _turn("bot", r.message)
    assert r.state == ReservationState.ASK_LOT

    _turn("user", "1")
    r = advance(sid, "1", db)
    _turn("bot", r.message)
    assert r.state == ReservationState.ASK_START

    _turn("user", _future())
    r = advance(sid, _future(), db)
    _turn("bot", r.message)
    assert r.state == ReservationState.ASK_END

    _turn("user", _future_end())
    r = advance(sid, _future_end(), db)
    _turn("bot", r.message)
    assert r.state == ReservationState.CONFIRM
    assert "ABC****34" in r.message
    assert "CityPark Central" in r.message

    _turn("user", "confirm")
    r = advance(sid, "confirm", db)
    safe_msg = _bot(r, skip_entities=["PERSON", "LICENSE_PLATE"])
    _turn("bot", safe_msg)
    assert r.state == ReservationState.AWAITING_APPROVAL
    assert r.reservation_id is not None
    res_id = r.reservation_id
    print(f"\n  [System] Reservation #{res_id} created in DB with status=PENDING")

    # ── User asks for status while waiting ───────────────────────────────────
    _turn("user", "What's the status of my reservation?")
    r = advance(sid, "What's the status of my reservation?", db)
    safe_msg = _bot(r, skip_entities=["PERSON"])
    _turn("bot", safe_msg)
    assert r.state == ReservationState.AWAITING_APPROVAL
    assert "PENDING" in r.message.upper() or "status" in r.message.lower()

    # ── Admin approves ───────────────────────────────────────────────────────
    print(f"\n  [Admin]  Approving reservation #{res_id}...")
    approve_reservation(db, res_id, admin="admin@citypark.example.com", note="VIP customer")
    print(f"  [Admin]  Reservation #{res_id} -> status=CONFIRMED")

    # ── User's next message receives the notification ────────────────────────
    _turn("user", "Any update?")
    r = advance(sid, "Any update?", db)
    safe_msg = _bot(r, skip_entities=["PERSON"])
    _turn("bot", safe_msg)

    # ── Assertions ───────────────────────────────────────────────────────────
    assert r.state == ReservationState.DONE, (
        f"Expected DONE after approval, got {r.state}"
    )
    assert "APPROVED" in r.message.upper(), "Message must say APPROVED"
    assert f"CPK-{res_id:06d}" in r.message, "Message must include booking reference"
    assert r.reservation_id == res_id

    # ── Second message must NOT repeat the notification ───────────────────────
    r2 = advance(sid, "Thanks!", db)
    assert r2.state == ReservationState.DONE
    assert "APPROVED" not in r2.message.upper(), (
        "Notification must not be repeated after delivery"
    )
    print("\n  [OK] Notification delivered once and marked as delivered")


# ===========================================================================
# SCENARIO 2 — User books → Admin refuses with reason → Re-booking offered
# ===========================================================================

def test_scenario_refuse(sid, db):
    _sep("SCENARIO 2: User books -> Admin refuses -> Re-booking offered")

    # ── Booking form ─────────────────────────────────────────────────────────
    msg = start(sid)
    _turn("bot", msg)

    _turn("user", "Bob Smith")
    r = advance(sid, "Bob Smith", db)
    _turn("bot", r.message)

    _turn("user", "XY-9900")
    r = advance(sid, "XY-9900", db)
    _turn("bot", r.message)

    _turn("user", "1")
    r = advance(sid, "1", db)
    _turn("bot", r.message)

    _turn("user", _future(30))
    r = advance(sid, _future(30), db)
    _turn("bot", r.message)

    _turn("user", _future_end(32))
    r = advance(sid, _future_end(32), db)
    _turn("bot", r.message)

    _turn("user", "yes")
    r = advance(sid, "yes", db)
    safe_msg = _bot(r, skip_entities=["PERSON", "LICENSE_PLATE"])
    _turn("bot", safe_msg)
    assert r.state == ReservationState.AWAITING_APPROVAL
    res_id = r.reservation_id
    print(f"\n  [System] Reservation #{res_id} created with status=PENDING")

    # ── Admin refuses with reason ─────────────────────────────────────────────
    reason = "Lot closed for maintenance from 10:00–18:00 that day"
    print(f"\n  [Admin]  Refusing reservation #{res_id}: \"{reason}\"")
    refuse_reservation(db, res_id, admin="admin@citypark.example.com", reason=reason)
    print(f"  [Admin]  Reservation #{res_id} -> status=REFUSED")

    # ── User receives refusal notice ──────────────────────────────────────────
    _turn("user", "Hello?")
    r = advance(sid, "Hello?", db)
    safe_msg = _bot(r, skip_entities=["PERSON"])
    _turn("bot", safe_msg)

    assert r.state == ReservationState.IDLE, (
        f"Expected IDLE after refusal, got {r.state}"
    )
    assert r.cancelled is True
    assert reason in r.message, "Refusal reason must appear in the message"
    assert "book" in r.message.lower() or "reserv" in r.message.lower(), (
        "Message must offer re-booking"
    )

    # ── User can immediately start a new reservation ──────────────────────────
    r2 = advance(sid, "New User", db)
    assert r2.state == ReservationState.ASK_CAR_PLATE, (
        f"After refusal user should be able to restart; got {r2.state}"
    )
    print("\n  [OK] User can immediately start a new reservation after refusal")
    print(f"  [OK] Refusal reason was delivered to user")


# ===========================================================================
# SCENARIO 3 — Server restart recovery
# ===========================================================================

def test_scenario_server_restart(sid, db):
    _sep("SCENARIO 3: Server restart - session recovered from DB")

    msg = start(sid)
    _turn("bot", msg)

    res_id = _run_booking(sid, db, name="Carol White", plate="CW-1234")
    print(f"\n  [System] Reservation #{res_id} pending. Simulating server restart...")

    # Clear in-memory state (simulates server restart)
    _sessions.pop(sid, None)
    assert sid not in _sessions, "Session must be gone from memory"
    print(f"  [System] In-memory session cleared. DB still has reservation #{res_id}.")

    # User sends a message after restart
    _turn("user", "Hi, I had a pending booking")
    r = advance(sid, "Hi, I had a pending booking", db)
    safe_msg = _bot(r, skip_entities=["PERSON"])
    _turn("bot", safe_msg)

    assert r.state == ReservationState.AWAITING_APPROVAL, (
        f"Session must be recovered from DB; got {r.state}"
    )
    assert r.reservation_id == res_id

    # Admin approves post-restart
    approve_reservation(db, res_id, admin="admin")
    print(f"\n  [Admin]  Approving #{res_id} post-restart")

    _turn("user", "Did admin decide?")
    r = advance(sid, "Did admin decide?", db)
    safe_msg = _bot(r, skip_entities=["PERSON"])
    _turn("bot", safe_msg)

    assert r.state == ReservationState.DONE
    assert "APPROVED" in r.message.upper()
    print("\n  [OK] Session correctly recovered from DB after restart")
    print("  [OK] Approval notification delivered post-restart")
