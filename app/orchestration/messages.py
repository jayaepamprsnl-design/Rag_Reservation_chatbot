"""Customer-facing text for a decided reservation (shared by the graph and the notifications route)."""
from __future__ import annotations

from app.db.models import LotReservation


def build_user_message(r: LotReservation) -> str:
    """Build a user-facing notification message from a decided LotReservation."""
    status = r.status.value if hasattr(r.status, "value") else str(r.status)
    if status == "confirmed":
        return (
            f"Great news! Your parking reservation #{r.id} has been approved. "
            "We look forward to seeing you."
        )
    reason = r.admin_decision_reason or "no reason given"
    return (
        f"We're sorry — your parking reservation #{r.id} was not approved "
        f"({reason}). Please try booking again."
    )
