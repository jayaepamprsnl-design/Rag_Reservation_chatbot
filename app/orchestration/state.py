"""Graph state: everything that is checkpointed per conversation (thread_id = session_id).

Privacy: the raw user message and the database session are NOT part of the state
(they travel in the run config, see nodes.TurnContext), so the checkpoint file
never contains free text the PII guard has not masked.  The only raw personal data
is the booking-slot snapshot in ``reservation`` (needed to resume a half-finished
booking after a restart) - the same data parking.db stores for the reservation.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

_MAX_MESSAGES = 20

# status values
IDLE = "idle"
COLLECTING = "collecting"            # booking questions in progress
AWAITING_ADMIN = "awaiting_admin"    # paused at admin_approval
APPROVED = "approved"                # an admin approved (file write may still be pending)
REFUSED = "refused"
EXPIRED = "expired"                  # no admin reply within the TTL
CANCELLED = "cancelled"              # the customer withdrew the request
ERROR = "error"


def append_messages(old: list[dict[str, str]] | None, new: list[dict[str, str]] | None):
    """Reducer: append, keeping only the most recent turns (the checkpoint stays small)."""
    return ((old or []) + (new or []))[-_MAX_MESSAGES:]


class GraphState(TypedDict, total=False):
    session_id: str
    messages: Annotated[list[dict[str, str]], append_messages]   # masked text only
    intent: str
    sources: list[str]
    guardrail_flags: list[str]
    # booking
    reservation: dict        # slot snapshot (reservation.export_session)
    confirmed: bool          # user said "confirm"; escalate_node must run
    reservation_id: int | None          # the PENDING/decided LotReservation (persists)
    active_reservation_id: int | None   # set only on turns whose reply concerns a reservation
    # outcome
    status: str
    file_written: bool
    notification: str        # text for the customer once an admin decided
    answer: str
    errors: list[str]
