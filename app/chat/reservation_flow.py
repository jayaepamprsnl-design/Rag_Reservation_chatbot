from typing import Dict, Optional, Tuple

from app.chat.conversation import ConversationContext, ConversationState

# Ordered list of fields to collect.
STEPS = ["name", "car_number", "date", "start_time", "end_time"]

PROMPTS: Dict[str, str] = {
    "name": "What is your full name?",
    "car_number": "What is your car license plate number?",
    "date": "What date would you like to park? (YYYY-MM-DD)",
    "start_time": "What time will you arrive? (HH:MM, 24-hour format)",
    "end_time": "What time will you leave? (HH:MM, 24-hour format)",
}


def _next_missing(data: dict) -> Optional[str]:
    for step in STEPS:
        if not data.get(step):
            return step
    return None


def start_reservation(ctx: ConversationContext) -> str:
    """Transition to reservation collection and ask for the first field."""
    ctx.state = ConversationState.COLLECTING_RESERVATION
    return PROMPTS["name"]


def handle_reservation_input(
    ctx: ConversationContext,
    user_input: str,
) -> Tuple[str, bool]:
    """
    Store *user_input* for the currently pending field, then return
    the next prompt (or confirmation summary).
    Returns (message, is_complete).
    """
    pending = _next_missing(ctx.reservation_data)
    if pending:
        ctx.reservation_data[pending] = user_input.strip()

    nxt = _next_missing(ctx.reservation_data)
    if nxt:
        return PROMPTS[nxt], False

    ctx.state = ConversationState.CONFIRMING_RESERVATION
    summary = "\n".join(f"  {k}: {v}" for k, v in ctx.reservation_data.items())
    return (
        f"Here is your reservation summary:\n{summary}\n\nShall I confirm? (yes / no)",
        False,
    )


def confirm_reservation(
    ctx: ConversationContext,
    user_input: str,
) -> Tuple[str, bool]:
    """
    Process yes/no confirmation.
    Returns (message, is_complete).  is_complete=True signals the caller
    to persist the reservation to the database.
    """
    if user_input.strip().lower() in ("yes", "y", "confirm", "ok"):
        ctx.state = ConversationState.DONE
        return "Your reservation has been confirmed! See you at the facility.", True
    ctx.reservation_data.clear()
    ctx.state = ConversationState.GENERAL
    return "Reservation cancelled. How else can I help you?", False
