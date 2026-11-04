"""Tests for app/agents/contracts.py — pure validation and serialisation.

No database, no I/O.  Every test is isolated from the rest of the system.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from app.agents.contracts import (
    AdminDecision,
    AgentMessage,
    AvailabilitySnapshot,
    Decision,
    MessageType,
    ReservationRequest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(**kw) -> AvailabilitySnapshot:
    return AvailabilitySnapshot(total_spaces=10, available_spaces=8, is_open=True, **kw)


def _req_data(**kw) -> dict:
    base = dict(
        reservation_id=42,
        customer_name="Alice Smith",
        car_plate_masked="ABC****34",
        lot_name="CityPark Central",
        start_time=datetime(2027, 6, 15, 10, 0),
        end_time=datetime(2027, 6, 15, 14, 0),
        price_estimate="~$32.00",
        availability_snapshot=_snapshot(),
        session_id="sess-abc",
    )
    base.update(kw)
    return base


# ===========================================================================
# ReservationRequest
# ===========================================================================


def test_reservation_request_json_round_trip():
    req = ReservationRequest(**_req_data())
    req2 = ReservationRequest.model_validate_json(req.model_dump_json())
    assert req2.reservation_id == 42
    assert req2.customer_name == "Alice Smith"
    assert req2.car_plate_masked == "ABC****34"
    assert req2.lot_name == "CityPark Central"
    assert req2.price_estimate == "~$32.00"
    assert req2.session_id == "sess-abc"
    assert req2.availability_snapshot.available_spaces == 8


def test_reservation_request_dict_round_trip():
    req = ReservationRequest(**_req_data())
    req2 = ReservationRequest(**req.model_dump(mode="json"))
    assert req2.reservation_id == req.reservation_id
    assert req2.availability_snapshot.is_open is True


def test_reservation_request_created_at_auto_populated():
    req = ReservationRequest(**_req_data())
    assert isinstance(req.created_at, datetime)


def test_reservation_request_explicit_created_at_preserved():
    ts = datetime(2027, 1, 1, 9, 0)
    req = ReservationRequest(**_req_data(created_at=ts))
    assert req.created_at == ts


def test_reservation_request_availability_snapshot_nested():
    req = ReservationRequest(**_req_data())
    assert isinstance(req.availability_snapshot, AvailabilitySnapshot)
    assert req.availability_snapshot.total_spaces == 10


# ===========================================================================
# AdminDecision — valid cases
# ===========================================================================


def test_admin_decision_approve_without_reason_is_valid():
    dec = AdminDecision(
        reservation_id=42,
        decision=Decision.APPROVE,
        admin_id="admin@example.com",
        decided_at=datetime.utcnow(),
    )
    assert dec.decision == Decision.APPROVE
    assert dec.reason is None


def test_admin_decision_approve_with_optional_note():
    dec = AdminDecision(
        reservation_id=42,
        decision=Decision.APPROVE,
        reason="VIP customer — fast-track",
        admin_id="admin",
        decided_at=datetime.utcnow(),
    )
    assert dec.reason == "VIP customer — fast-track"


def test_admin_decision_refuse_with_reason_is_valid():
    dec = AdminDecision(
        reservation_id=42,
        decision=Decision.REFUSE,
        reason="Duplicate booking in the same slot",
        admin_id="admin",
        decided_at=datetime.utcnow(),
    )
    assert dec.decision == Decision.REFUSE
    assert dec.reason == "Duplicate booking in the same slot"


def test_admin_decision_json_round_trip():
    dec = AdminDecision(
        reservation_id=7,
        decision=Decision.REFUSE,
        reason="Policy violation",
        admin_id="mgr1",
        decided_at=datetime(2027, 1, 1, 12, 0),
    )
    dec2 = AdminDecision.model_validate_json(dec.model_dump_json())
    assert dec2.decision == Decision.REFUSE
    assert dec2.reason == "Policy violation"
    assert dec2.reservation_id == 7
    assert dec2.admin_id == "mgr1"


def test_admin_decision_decided_at_auto_populated():
    dec = AdminDecision(
        reservation_id=1,
        decision=Decision.APPROVE,
        admin_id="admin",
    )
    assert isinstance(dec.decided_at, datetime)


# ===========================================================================
# AdminDecision — validation failures
# ===========================================================================


def test_admin_decision_refuse_without_reason_raises():
    """REFUSE with no reason must raise ValidationError."""
    with pytest.raises(ValidationError, match="reason is required"):
        AdminDecision(
            reservation_id=42,
            decision=Decision.REFUSE,
            # reason omitted
            admin_id="admin",
            decided_at=datetime.utcnow(),
        )


def test_admin_decision_refuse_whitespace_reason_raises():
    """REFUSE with a whitespace-only reason is not acceptable."""
    with pytest.raises(ValidationError, match="reason is required"):
        AdminDecision(
            reservation_id=42,
            decision=Decision.REFUSE,
            reason="   \t  ",
            admin_id="admin",
            decided_at=datetime.utcnow(),
        )


def test_admin_decision_refuse_empty_string_raises():
    with pytest.raises(ValidationError, match="reason is required"):
        AdminDecision(
            reservation_id=42,
            decision=Decision.REFUSE,
            reason="",
            admin_id="admin",
            decided_at=datetime.utcnow(),
        )


def test_admin_decision_invalid_decision_string_raises():
    """Completely unknown decision value must raise ValidationError."""
    with pytest.raises(ValidationError):
        AdminDecision(
            reservation_id=42,
            decision="INVALID_VALUE",
            admin_id="admin",
            decided_at=datetime.utcnow(),
        )


def test_admin_decision_lowercase_approve_raises():
    """Enum values are case-sensitive; 'approve' is not a valid Decision."""
    with pytest.raises(ValidationError):
        AdminDecision(
            reservation_id=42,
            decision="approve",
            admin_id="admin",
            decided_at=datetime.utcnow(),
        )


def test_admin_decision_lowercase_refuse_raises():
    with pytest.raises(ValidationError):
        AdminDecision(
            reservation_id=42,
            decision="refuse",
            reason="some reason",
            admin_id="admin",
            decided_at=datetime.utcnow(),
        )


# ===========================================================================
# AgentMessage
# ===========================================================================


def test_agent_message_auto_generates_uuid_message_id():
    msg = AgentMessage(
        type=MessageType.ESCALATION,
        payload={"key": "value"},
        correlation_id=42,
    )
    assert msg.message_id
    # UUID4 canonical form: 8-4-4-4-12 hex chars, 36 chars total
    assert len(msg.message_id) == 36


def test_agent_message_two_instances_have_different_ids():
    m1 = AgentMessage(type=MessageType.ESCALATION, payload={}, correlation_id=1)
    m2 = AgentMessage(type=MessageType.ESCALATION, payload={}, correlation_id=1)
    assert m1.message_id != m2.message_id


def test_agent_message_timestamp_auto_populated():
    msg = AgentMessage(type=MessageType.DECISION, payload={}, correlation_id=5)
    assert isinstance(msg.timestamp, datetime)


def test_agent_message_correlation_id_equals_reservation_id():
    msg = AgentMessage(type=MessageType.ESCALATION, payload={}, correlation_id=99)
    assert msg.correlation_id == 99


def test_agent_message_invalid_type_raises():
    with pytest.raises(ValidationError):
        AgentMessage(type="UNKNOWN_TYPE", payload={}, correlation_id=1)


def test_agent_message_wraps_reservation_request_payload():
    """Payload serialised from a ReservationRequest can be round-tripped."""
    req = ReservationRequest(**_req_data())
    msg = AgentMessage(
        type=MessageType.ESCALATION,
        payload=req.model_dump(mode="json"),
        correlation_id=req.reservation_id,
    )
    # Receiver reconstructs the inner model from payload
    req2 = ReservationRequest(**msg.payload)
    assert req2.customer_name == "Alice Smith"
    assert req2.availability_snapshot.is_open is True


def test_agent_message_json_round_trip():
    req = ReservationRequest(**_req_data())
    msg = AgentMessage(
        type=MessageType.ESCALATION,
        payload=req.model_dump(mode="json"),
        correlation_id=req.reservation_id,
    )
    msg2 = AgentMessage.model_validate_json(msg.model_dump_json())
    assert msg2.type == MessageType.ESCALATION
    assert msg2.correlation_id == 42
    assert msg2.message_id == msg.message_id


def test_agent_message_all_types_accepted():
    for t in MessageType:
        msg = AgentMessage(type=t, payload={}, correlation_id=1)
        assert msg.type == t


def test_agent_message_user_notification_type():
    msg = AgentMessage(
        type=MessageType.USER_NOTIFICATION,
        payload={"text": "Your booking was approved!"},
        correlation_id=10,
    )
    assert msg.type == MessageType.USER_NOTIFICATION
