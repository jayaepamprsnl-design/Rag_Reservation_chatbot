from app.chat.conversation import ConversationContext, ConversationState
from app.chat.reservation_flow import (
    confirm_reservation,
    handle_reservation_input,
    start_reservation,
)


def _ctx() -> ConversationContext:
    return ConversationContext(session_id="test-session")


def test_start_asks_for_name():
    msg = start_reservation(_ctx())
    assert "name" in msg.lower()


def test_start_sets_collecting_state():
    ctx = _ctx()
    start_reservation(ctx)
    assert ctx.state == ConversationState.COLLECTING_RESERVATION


def test_full_flow_reaches_confirmation():
    ctx = _ctx()
    start_reservation(ctx)
    inputs = ["Jane Smith", "XYZ-999", "2026-01-20", "09:00", "11:00"]
    for inp in inputs:
        msg, complete = handle_reservation_input(ctx, inp)
        assert not complete
    assert ctx.state == ConversationState.CONFIRMING_RESERVATION
    assert "summary" in msg.lower() or "confirm" in msg.lower()


def test_confirm_yes_returns_done():
    ctx = _ctx()
    ctx.reservation_data = {
        "name": "Jane", "car_number": "XYZ",
        "date": "2026-01-20", "start_time": "09:00", "end_time": "11:00",
    }
    ctx.state = ConversationState.CONFIRMING_RESERVATION
    msg, done = confirm_reservation(ctx, "yes")
    assert done
    assert ctx.state == ConversationState.DONE


def test_confirm_no_resets_state():
    ctx = _ctx()
    ctx.reservation_data = {
        "name": "Jane", "car_number": "XYZ",
        "date": "2026-01-20", "start_time": "09:00", "end_time": "11:00",
    }
    ctx.state = ConversationState.CONFIRMING_RESERVATION
    msg, done = confirm_reservation(ctx, "no")
    assert not done
    assert ctx.state == ConversationState.GENERAL
    assert ctx.reservation_data == {}


def test_fields_stored_in_order():
    ctx = _ctx()
    start_reservation(ctx)
    handle_reservation_input(ctx, "Alice")
    assert ctx.reservation_data.get("name") == "Alice"
    handle_reservation_input(ctx, "ABC-111")
    assert ctx.reservation_data.get("car_number") == "ABC-111"
