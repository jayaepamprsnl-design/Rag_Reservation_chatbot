"""
Tests for app/agents/admin_agent.py.

Coverage:
  1. Tool invocations (get_pending, get_details, check_availability,
     send_notification, record_decision)
  2. AdminAgent.generate_notification — LLM call, fact-grounding
  3. AdminAgent.parse_decision — approve, refuse+reason, ambiguous, no
     decision keyword, unknown reservation ID fallback
  4. AdminAgent.execute_decision — approve, refuse, repository error
  5. Safety case: LLM cannot approve without explicit admin decision keyword

Each test gets a fresh in-memory SQLite database via the function-scoped
`db` fixture.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.admin_agent import (
    TOOLS,
    AdminAgent,
    MockAdminLLMClient,
    _detect_decision_keyword,
    _extract_id,
    _extract_reason_regex,
    check_lot_availability,
    get_pending_reservations,
    get_reservation_details,
    record_admin_decision,
    send_admin_notification,
)
from app.agents.contracts import AdminDecision, AgentMessage, Decision, MessageType, ReservationRequest, AvailabilitySnapshot
from app.db.models import ApprovalRequest, Base, ParkingLot, ReservationStatus
from app.db.repository import (
    approve_reservation,
    create_pending_reservation,
    create_reservation,
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
        name="AdminTestLot",
        address="1 Admin St",
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
    return db.query(ParkingLot).filter_by(name="AdminTestLot").one().id


def _future(h: int = 2) -> datetime:
    return datetime.utcnow() + timedelta(hours=h)


def _pending(db, *, hours_start: int = 3, hours_end: int = 5,
             name: str = "Test User", plate: str = "AB-1234",
             session_id: str = "sess-x") -> int:
    r = create_pending_reservation(
        db, lot_id=_lot_id(db), customer_name=name, car_plate=plate,
        start_time=_future(hours_start), end_time=_future(hours_end),
        session_id=session_id,
    )
    return r.id


def _make_request(res_id: int, db) -> ReservationRequest:
    """Build a ReservationRequest from a live DB row."""
    from app.db.repository import get_reservation, get_lot
    r   = get_reservation(db, res_id)
    lot = get_lot(db, r.lot_id)
    return ReservationRequest(
        reservation_id=res_id,
        customer_name=r.customer_name,
        car_plate_masked="AB****34",
        lot_name=lot.name,
        start_time=r.start_time,
        end_time=r.end_time,
        price_estimate="~$16.00",
        availability_snapshot=AvailabilitySnapshot(
            total_spaces=lot.total_spaces,
            available_spaces=lot.available_spaces,
            is_open=True,
        ),
        session_id=r.session_id or "sess-x",
    )


# ---------------------------------------------------------------------------
# Deterministic helper tests (no DB, no LLM)
# ---------------------------------------------------------------------------


def test_detect_approve_keyword():
    assert _detect_decision_keyword("approve 5") == "APPROVE"
    assert _detect_decision_keyword("APPROVE reservation 5") == "APPROVE"
    assert _detect_decision_keyword("confirmed") == "APPROVE"


def test_detect_refuse_keyword():
    assert _detect_decision_keyword("refuse 7 because maintenance") == "REFUSE"
    assert _detect_decision_keyword("reject 7") == "REFUSE"
    assert _detect_decision_keyword("deny 7") == "REFUSE"


def test_detect_ambiguous_returns_none():
    assert _detect_decision_keyword("approve or refuse?") is None


def test_detect_no_keyword_returns_none():
    assert _detect_decision_keyword("looks fine") is None
    assert _detect_decision_keyword("hello") is None


def test_extract_id_finds_number():
    assert _extract_id("approve 42") == 42
    assert _extract_id("refuse 7 because X") == 7


def test_extract_id_returns_none_when_absent():
    assert _extract_id("approve it please") is None


def test_extract_reason_regex_because():
    r = _extract_reason_regex("refuse 5 because the lot is full")
    assert "lot is full" in r


def test_extract_reason_regex_no_match():
    assert _extract_reason_regex("refuse 5") == ""


# ---------------------------------------------------------------------------
# Tool: get_pending_reservations
# ---------------------------------------------------------------------------


def test_get_pending_returns_list(db):
    res_id = _pending(db)
    result = get_pending_reservations.invoke({"db": db})
    assert isinstance(result, list)
    ids = [r["reservation_id"] for r in result]
    assert res_id in ids


def test_get_pending_masks_plate(db):
    _pending(db, plate="XY-9900")
    result = get_pending_reservations.invoke({"db": db})
    raw = next(r["car_plate"] for r in result)
    assert "9900" not in raw     # last 4 digits masked
    assert "XY" in raw or "****" in raw


def test_get_pending_empty_when_none(db):
    # no pending reservations
    result = get_pending_reservations.invoke({"db": db})
    assert result == []


# ---------------------------------------------------------------------------
# Tool: get_reservation_details
# ---------------------------------------------------------------------------


def test_get_reservation_details_returns_dict(db):
    res_id = _pending(db, name="Alice Jones")
    result = get_reservation_details.invoke({"reservation_id": res_id, "db": db})
    assert result["reservation_id"] == res_id
    assert result["customer_name"] == "Alice Jones"
    assert result["lot_name"] == "AdminTestLot"
    assert "hourly_price" in result


def test_get_reservation_details_masks_plate(db):
    res_id = _pending(db, plate="MN-4567")
    result = get_reservation_details.invoke({"reservation_id": res_id, "db": db})
    assert "4567" not in result["car_plate"]
    assert "****" in result["car_plate"]


def test_get_reservation_details_unknown_id_returns_error(db):
    result = get_reservation_details.invoke({"reservation_id": 99999, "db": db})
    assert "error" in result


# ---------------------------------------------------------------------------
# Tool: check_lot_availability
# ---------------------------------------------------------------------------


def test_check_lot_availability_no_conflicts(db):
    res_id = _pending(db)
    result = check_lot_availability.invoke({
        "lot_id":     _lot_id(db),
        "start_time": _future(20).isoformat(),
        "end_time":   _future(22).isoformat(),
        "db":         db,
    })
    assert result["available_spaces"] == 5
    assert result["conflicts_in_window"] == 0
    assert result["can_accommodate"] is True


def test_check_lot_availability_with_confirmed_conflict(db):
    # Create a confirmed reservation that overlaps the query window
    create_reservation(
        db, lot_id=_lot_id(db), customer_name="Blocker", car_plate="BL-0001",
        start_time=_future(30), end_time=_future(32),
    )
    result = check_lot_availability.invoke({
        "lot_id":     _lot_id(db),
        "start_time": _future(30).isoformat(),
        "end_time":   _future(32).isoformat(),
        "db":         db,
    })
    assert result["conflicts_in_window"] >= 1


def test_check_lot_availability_unknown_lot_returns_error(db):
    result = check_lot_availability.invoke({
        "lot_id": 99999,
        "start_time": _future(5).isoformat(),
        "end_time":   _future(7).isoformat(),
        "db": db,
    })
    assert "error" in result


# ---------------------------------------------------------------------------
# Tool: send_admin_notification
# ---------------------------------------------------------------------------


def test_send_admin_notification_creates_audit_row(db):
    res_id = _pending(db)
    result = send_admin_notification.invoke({
        "reservation_id":   res_id,
        "notification_text": "Please approve reservation #1.",
        "db": db,
    })
    assert result["status"] == "sent"
    row = db.query(ApprovalRequest).filter_by(reservation_id=res_id).first()
    assert row is not None
    assert row.request_text == "Please approve reservation #1."


def test_send_admin_notification_returns_receipt(db):
    res_id = _pending(db)
    result = send_admin_notification.invoke({
        "reservation_id":   res_id,
        "notification_text": "Notification text here.",
        "db": db,
    })
    assert "audit_id" in result
    assert "preview" in result


# ---------------------------------------------------------------------------
# Tool: record_admin_decision
# ---------------------------------------------------------------------------


def test_record_admin_decision_approve(db):
    res_id = _pending(db)
    result = record_admin_decision.invoke({
        "reservation_id": res_id,
        "decision": "APPROVE",
        "admin_id": "mgr@test.com",
        "db": db,
    })
    assert result["status"] == "approved"
    assert result["reservation_id"] == res_id


def test_record_admin_decision_refuse(db):
    res_id = _pending(db)
    result = record_admin_decision.invoke({
        "reservation_id": res_id,
        "decision": "REFUSE",
        "reason": "Maintenance scheduled",
        "admin_id": "mgr@test.com",
        "db": db,
    })
    assert result["status"] == "refused"
    assert "Maintenance" in result.get("reason", "")


def test_record_admin_decision_unknown_id_returns_error(db):
    result = record_admin_decision.invoke({
        "reservation_id": 99999,
        "decision": "APPROVE",
        "db": db,
    })
    assert "error" in result


def test_record_admin_decision_invalid_decision_returns_error(db):
    res_id = _pending(db)
    result = record_admin_decision.invoke({
        "reservation_id": res_id,
        "decision": "MAYBE",
        "db": db,
    })
    assert "error" in result


def test_record_admin_decision_double_approve_returns_error(db):
    res_id = _pending(db)
    record_admin_decision.invoke({"reservation_id": res_id, "decision": "APPROVE", "db": db})
    result = record_admin_decision.invoke({"reservation_id": res_id, "decision": "APPROVE", "db": db})
    assert "error" in result


# ---------------------------------------------------------------------------
# AdminAgent.generate_notification
# ---------------------------------------------------------------------------


def test_generate_notification_calls_llm_once(db):
    res_id = _pending(db)
    mock = MockAdminLLMClient(["ADMIN NOTIFICATION: Booking #1 needs approval."])
    agent = AdminAgent(lambda: db, llm=mock)
    text = agent.generate_notification(_make_request(res_id, db))
    assert mock.call_count == 1
    assert "ADMIN NOTIFICATION" in text


def test_generate_notification_passes_facts_to_llm(db):
    """The LLM prompt must include real tool results (lot name, times)."""
    res_id = _pending(db, name="Bob Builder", hours_start=10, hours_end=12)

    captured: list[list[dict]] = []

    class _CaptureMock(MockAdminLLMClient):
        def complete(self, messages):
            captured.append(messages)
            return super().complete(messages)

    mock = _CaptureMock(["Notification text"])
    agent = AdminAgent(lambda: db, llm=mock)
    agent.generate_notification(_make_request(res_id, db))

    # LLM must receive the tool results in the user message
    user_content = captured[0][-1]["content"]
    assert "AdminTestLot" in user_content
    assert "Bob Builder" in user_content


def test_generate_notification_error_on_unknown_reservation(db):
    bad_request = ReservationRequest(
        reservation_id=99999, customer_name="Ghost", car_plate_masked="X***",
        lot_name="N/A", start_time=_future(3), end_time=_future(5),
        price_estimate="~$0", availability_snapshot=AvailabilitySnapshot(
            total_spaces=0, available_spaces=0, is_open=False),
        session_id="ghost",
    )
    mock = MockAdminLLMClient(["should not reach LLM"])
    agent = AdminAgent(lambda: db, llm=mock)
    result = agent.generate_notification(bad_request)
    assert "Error" in result
    assert mock.call_count == 0   # LLM not called when tool returns error


# ---------------------------------------------------------------------------
# AdminAgent.parse_decision — approve
# ---------------------------------------------------------------------------


def test_parse_decision_approve_simple(db):
    res_id = _pending(db)
    agent = AdminAgent(lambda: db, llm=MockAdminLLMClient())
    result = agent.parse_decision(f"approve {res_id}", res_id, "admin1")
    assert isinstance(result, AdminDecision)
    assert result.decision == Decision.APPROVE
    assert result.reservation_id == res_id


def test_parse_decision_approve_extracts_id_from_text(db):
    res_id = _pending(db)
    agent = AdminAgent(lambda: db, llm=MockAdminLLMClient())
    result = agent.parse_decision(f"looks good, approve {res_id}", 0, "admin1")
    assert isinstance(result, AdminDecision)
    assert result.reservation_id == res_id


def test_parse_decision_approve_uses_fallback_id_when_none_in_text(db):
    res_id = _pending(db)
    agent = AdminAgent(lambda: db, llm=MockAdminLLMClient())
    result = agent.parse_decision("approve it", res_id, "admin1")
    assert isinstance(result, AdminDecision)
    assert result.reservation_id == res_id


def test_parse_decision_approve_no_llm_call(db):
    """APPROVE decision must never invoke the LLM (no reason extraction needed)."""
    res_id = _pending(db)
    mock = MockAdminLLMClient(["should not be called"])
    agent = AdminAgent(lambda: db, llm=mock)
    agent.parse_decision(f"approve {res_id}", res_id, "admin1")
    assert mock.call_count == 0


# ---------------------------------------------------------------------------
# AdminAgent.parse_decision — refuse
# ---------------------------------------------------------------------------


def test_parse_decision_refuse_with_regex_reason(db):
    res_id = _pending(db)
    agent = AdminAgent(lambda: db, llm=MockAdminLLMClient())
    result = agent.parse_decision(
        f"refuse {res_id} because the lot is closed for maintenance",
        res_id, "admin1",
    )
    assert isinstance(result, AdminDecision)
    assert result.decision == Decision.REFUSE
    assert "maintenance" in (result.reason or "")


def test_parse_decision_refuse_reason_via_llm_fallback(db):
    """When regex finds no 'because' clause, the LLM extracts the reason."""
    res_id = _pending(db)
    mock = MockAdminLLMClient(["scheduled maintenance work"])
    agent = AdminAgent(lambda: db, llm=mock)
    result = agent.parse_decision(
        f"refuse {res_id}, there is scheduled maintenance",
        res_id, "admin1",
    )
    assert isinstance(result, AdminDecision)
    assert result.decision == Decision.REFUSE
    # LLM was called exactly once for reason extraction
    assert mock.call_count == 1


def test_parse_decision_refuse_without_reason_returns_clarification(db):
    res_id = _pending(db)
    # Mock LLM returns empty string → no reason found
    mock = MockAdminLLMClient([""])
    agent = AdminAgent(lambda: db, llm=mock)
    result = agent.parse_decision(f"refuse {res_id}", res_id, "admin1")
    assert isinstance(result, str)
    assert "reason" in result.lower()


# ---------------------------------------------------------------------------
# AdminAgent.parse_decision — ambiguous / no decision
# ---------------------------------------------------------------------------


def test_parse_decision_ambiguous_returns_clarification(db):
    res_id = _pending(db)
    agent = AdminAgent(lambda: db, llm=MockAdminLLMClient())
    result = agent.parse_decision(
        "should I approve or refuse this?", res_id, "admin1"
    )
    assert isinstance(result, str)
    assert "approve" in result.lower() or "decide" in result.lower()


def test_parse_decision_no_keyword_returns_clarification(db):
    res_id = _pending(db)
    agent = AdminAgent(lambda: db, llm=MockAdminLLMClient())
    result = agent.parse_decision("I'll look at this later", res_id, "admin1")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# SAFETY CASE: LLM cannot approve without explicit admin decision keyword
# ---------------------------------------------------------------------------


def test_llm_cannot_approve_without_admin_decision_keyword(db):
    """
    Even if the mock LLM would respond with an approval, the agent must
    return a clarification and never call the LLM when no decision keyword
    is present in the admin message.
    """
    res_id = _pending(db)
    # LLM would hand back an approval response if ever called
    aggressive_mock = MockAdminLLMClient(["APPROVE", "approve", "yes confirm it"])
    agent = AdminAgent(lambda: db, llm=aggressive_mock)

    result = agent.parse_decision(
        "I need more time to think about this reservation",
        res_id,
        "admin1",
    )

    # Must return a clarification string, not an AdminDecision
    assert isinstance(result, str), (
        "Safety violation: agent returned AdminDecision without explicit keyword"
    )
    # The LLM was never consulted — safety gate fired before any LLM call
    assert aggressive_mock.call_count == 0, (
        f"Safety violation: LLM was called {aggressive_mock.call_count} time(s) "
        "despite no decision keyword in admin input"
    )


def test_llm_cannot_approve_on_empty_input(db):
    res_id = _pending(db)
    mock = MockAdminLLMClient(["APPROVE 99"])
    agent = AdminAgent(lambda: db, llm=mock)
    result = agent.parse_decision("", res_id, "admin1")
    assert isinstance(result, str)
    assert mock.call_count == 0


# ---------------------------------------------------------------------------
# AdminAgent.execute_decision — approve
# ---------------------------------------------------------------------------


def test_execute_decision_approve_returns_user_notification(db):
    res_id = _pending(db)
    mock = MockAdminLLMClient(["Your parking reservation has been confirmed!"])
    agent = AdminAgent(lambda: db, llm=mock)

    decision = AdminDecision(
        reservation_id=res_id, decision=Decision.APPROVE,
        admin_id="admin1",
    )
    msg = agent.execute_decision(decision, db)

    assert isinstance(msg, AgentMessage)
    assert msg.type == MessageType.USER_NOTIFICATION
    assert msg.correlation_id == res_id
    assert msg.payload["decision"] == "APPROVE"
    assert "confirmed" in msg.payload["notification_text"].lower()


def test_execute_decision_refuse_returns_user_notification(db):
    res_id = _pending(db)
    mock = MockAdminLLMClient(["We're sorry, your reservation could not be confirmed."])
    agent = AdminAgent(lambda: db, llm=mock)

    decision = AdminDecision(
        reservation_id=res_id, decision=Decision.REFUSE,
        reason="lot closed for maintenance", admin_id="admin1",
    )
    msg = agent.execute_decision(decision, db)

    assert msg.type == MessageType.USER_NOTIFICATION
    assert msg.payload["decision"] == "REFUSE"
    assert msg.payload["reason"] == "lot closed for maintenance"


def test_execute_decision_unknown_reservation_returns_error_message(db):
    mock = MockAdminLLMClient(["should not reach LLM"])
    agent = AdminAgent(lambda: db, llm=mock)

    decision = AdminDecision(
        reservation_id=99999, decision=Decision.APPROVE, admin_id="admin1"
    )
    msg = agent.execute_decision(decision, db)

    assert msg.type == MessageType.ERROR
    assert "error" in msg.payload
    assert mock.call_count == 0  # LLM not called when tool errors


def test_execute_decision_calls_llm_for_notification_text(db):
    res_id = _pending(db)
    mock = MockAdminLLMClient(["Booking approved. See you at the facility!"])
    agent = AdminAgent(lambda: db, llm=mock)

    decision = AdminDecision(
        reservation_id=res_id, decision=Decision.APPROVE, admin_id="admin1"
    )
    agent.execute_decision(decision, db)
    assert mock.call_count == 1


# ---------------------------------------------------------------------------
# TOOLS list completeness
# ---------------------------------------------------------------------------


def test_tools_list_has_five_entries():
    assert len(TOOLS) == 5


def test_tools_list_contains_expected_names():
    names = {t.name for t in TOOLS}
    assert "get_pending_reservations" in names
    assert "get_reservation_details" in names
    assert "check_lot_availability" in names
    assert "send_admin_notification" in names
    assert "record_admin_decision" in names


def test_tools_db_param_hidden_from_llm_schema():
    """InjectedToolArg must exclude 'db' from the OpenAI function spec."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    for t in TOOLS:
        oai = convert_to_openai_tool(t)
        params = oai["function"]["parameters"]["properties"]
        assert "db" not in params, (
            f"Tool '{t.name}' exposes 'db' in its LLM schema — "
            "InjectedToolArg is not working correctly"
        )
