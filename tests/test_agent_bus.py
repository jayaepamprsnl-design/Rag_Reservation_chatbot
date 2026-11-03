"""Tests for app/agents/bus.py — InProcessBus operations against a real DB.

Each test gets a fresh in-memory SQLite database so state never bleeds.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.bus import InProcessBus, MessageBus, get_bus, set_bus
from app.agents.contracts import (
    AdminDecision,
    AgentMessage,
    AvailabilitySnapshot,
    Decision,
    MessageType,
    ReservationRequest,
)
from app.db.models import ApprovalRequest, Base, LotReservation, ParkingLot, ReservationStatus
from app.db.repository import (
    approve_reservation,
    create_pending_reservation,
    refuse_reservation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(ParkingLot(
        name="TestLot",
        address="1 Test St",
        total_spaces=5,
        available_spaces=5,
        hourly_price=8.0,
        daily_price=40.0,
        opens_at=time(0, 0),
        closes_at=time(23, 59),
    ))
    session.commit()
    yield session
    session.close()
    engine.dispose()


def _lot_id(db) -> int:
    return db.query(ParkingLot).filter_by(name="TestLot").one().id


def _future(hours: int = 2) -> datetime:
    return datetime.utcnow() + timedelta(hours=hours)


def _make_request(reservation_id: int, session_id: str = "sess-test") -> ReservationRequest:
    return ReservationRequest(
        reservation_id=reservation_id,
        customer_name="Test User",
        car_plate_masked="TST****99",
        lot_name="TestLot",
        start_time=datetime(2027, 8, 1, 10, 0),
        end_time=datetime(2027, 8, 1, 14, 0),
        price_estimate="~$32.00",
        availability_snapshot=AvailabilitySnapshot(
            total_spaces=5, available_spaces=5, is_open=True
        ),
        session_id=session_id,
    )


bus = InProcessBus()


# ===========================================================================
# send_escalation
# ===========================================================================


def test_send_escalation_creates_approval_request_row(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Alice", car_plate="AB-1234",
        start_time=_future(3), end_time=_future(5),
    )
    bus.send_escalation(_make_request(res.id), db)

    row = db.query(ApprovalRequest).filter_by(reservation_id=res.id).first()
    assert row is not None
    assert row.status == "sent"
    assert row.sent_at is not None


def test_send_escalation_returns_agent_message(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Bob", car_plate="CD-5678",
        start_time=_future(6), end_time=_future(8),
    )
    msg = bus.send_escalation(_make_request(res.id), db)

    assert isinstance(msg, AgentMessage)
    assert msg.type == MessageType.ESCALATION


def test_send_escalation_message_correlation_id_matches_reservation(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Carol", car_plate="EF-9012",
        start_time=_future(10), end_time=_future(12),
    )
    msg = bus.send_escalation(_make_request(res.id), db)
    assert msg.correlation_id == res.id


def test_send_escalation_payload_contains_session_id(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Dave", car_plate="GH-3456",
        start_time=_future(14), end_time=_future(16),
    )
    msg = bus.send_escalation(_make_request(res.id, session_id="sess-dave"), db)
    assert msg.payload["session_id"] == "sess-dave"


def test_send_escalation_audit_row_stores_request_json(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Eve", car_plate="IJ-7890",
        start_time=_future(18), end_time=_future(20),
    )
    req = _make_request(res.id, session_id="sess-eve")
    bus.send_escalation(req, db)

    row = db.query(ApprovalRequest).filter_by(reservation_id=res.id).first()
    assert row.request_text is not None
    assert "sess-eve" in row.request_text
    assert "TST****99" in row.request_text


def test_send_escalation_multiple_calls_create_multiple_audit_rows(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Frank", car_plate="KL-1122",
        start_time=_future(22), end_time=_future(24),
    )
    req = _make_request(res.id)
    bus.send_escalation(req, db)
    bus.send_escalation(req, db)   # resend (e.g. admin didn't respond in time)

    rows = db.query(ApprovalRequest).filter_by(reservation_id=res.id).all()
    assert len(rows) == 2


# ===========================================================================
# get_decision
# ===========================================================================


def test_get_decision_none_when_reservation_still_pending(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Grace", car_plate="MN-3344",
        start_time=_future(26), end_time=_future(28),
    )
    assert bus.get_decision(res.id, db) is None


def test_get_decision_none_for_unknown_reservation_id(db):
    assert bus.get_decision(99999, db) is None


def test_get_decision_approve_after_approval(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Henry", car_plate="OP-5566",
        start_time=_future(30), end_time=_future(32),
    )
    approve_reservation(db, res.id, admin="mgr@example.com", note="Looks fine")

    decision = bus.get_decision(res.id, db)
    assert decision is not None
    assert isinstance(decision, AdminDecision)
    assert decision.decision == Decision.APPROVE
    assert decision.reservation_id == res.id
    assert decision.admin_id == "mgr@example.com"


def test_get_decision_approve_includes_admin_note(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Iris", car_plate="QR-7788",
        start_time=_future(34), end_time=_future(36),
    )
    approve_reservation(db, res.id, admin="admin", note="Priority customer")

    decision = bus.get_decision(res.id, db)
    assert decision.reason == "Priority customer"


def test_get_decision_refuse_after_refusal(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Jack", car_plate="ST-9900",
        start_time=_future(38), end_time=_future(40),
    )
    refuse_reservation(db, res.id, admin="mgr2", reason="No capacity in system")

    decision = bus.get_decision(res.id, db)
    assert decision is not None
    assert decision.decision == Decision.REFUSE
    assert decision.reason == "No capacity in system"
    assert decision.admin_id == "mgr2"


def test_get_decision_refuse_with_no_stored_reason_gets_fallback(db):
    """approve_reservation sets reason=None when note=''; bus adds a fallback."""
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Karen", car_plate="UV-1234",
        start_time=_future(42), end_time=_future(44),
    )
    # Manually set REFUSED with no reason (simulates edge-case from repository)
    db_res = db.get(LotReservation, res.id)
    db_res.status = ReservationStatus.REFUSED
    db_res.decided_at = datetime.utcnow()
    db_res.decided_by = "system"
    db_res.admin_decision_reason = None
    db.commit()

    decision = bus.get_decision(res.id, db)
    assert decision is not None
    assert decision.decision == Decision.REFUSE
    assert decision.reason  # must be non-empty — bus provides fallback


def test_get_decision_returns_none_for_expired_reservation(db):
    """EXPIRED is not an admin decision — bus must return None."""
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Leo", car_plate="WX-5678",
        start_time=_future(46), end_time=_future(48),
    )
    db_res = db.get(LotReservation, res.id)
    db_res.status = ReservationStatus.EXPIRED
    db_res.decided_at = datetime.utcnow()
    db.commit()

    assert bus.get_decision(res.id, db) is None


# ===========================================================================
# mark_notified
# ===========================================================================


def test_mark_notified_sets_flag(db):
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Mia", car_plate="YZ-9012",
        start_time=_future(50), end_time=_future(52),
    )
    approve_reservation(db, res.id, admin="admin")
    bus.mark_notified(res.id, db)

    db.refresh(res)
    assert res.notified_user is True


def test_mark_notified_idempotent(db):
    """Calling mark_notified twice must not raise and flag stays True."""
    lid = _lot_id(db)
    res = create_pending_reservation(
        db, lot_id=lid, customer_name="Nick", car_plate="AB-3456",
        start_time=_future(54), end_time=_future(56),
    )
    approve_reservation(db, res.id, admin="admin")
    bus.mark_notified(res.id, db)
    bus.mark_notified(res.id, db)

    db.refresh(res)
    assert res.notified_user is True


def test_mark_notified_unknown_id_is_silent(db):
    """Unknown reservation IDs must not raise — no-op."""
    bus.mark_notified(99999, db)   # should complete without error


# ===========================================================================
# get_bus / set_bus singleton
# ===========================================================================


def test_get_bus_returns_inprocess_bus_by_default():
    assert isinstance(get_bus(), InProcessBus)


def test_set_bus_replaces_singleton():
    class _DummyBus(MessageBus):
        def send_escalation(self, r, db): ...
        def get_decision(self, rid, db): return None
        def mark_notified(self, rid, db): ...

    original = get_bus()
    try:
        dummy = _DummyBus()
        set_bus(dummy)
        assert get_bus() is dummy
    finally:
        set_bus(original)   # always restore
