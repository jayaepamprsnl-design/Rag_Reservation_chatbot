"""
Message bus: the only channel through which Agent 1 and Agent 2 communicate.

Interface
---------
MessageBus (ABC)
    send_escalation(request, db) → AgentMessage
    get_decision(reservation_id, db) → AdminDecision | None
    mark_notified(reservation_id, db) → None

In-process implementation
--------------------------
InProcessBus backs onto the SQL tables already defined in Stage 2:
  • send_escalation  → writes an ApprovalRequest audit row
  • get_decision     → reads LotReservation.decided_at / .status / .decided_by
  • mark_notified    → sets LotReservation.notified_user = True

Swap for a queue-backed implementation by calling set_bus() at application
startup — all callers go through get_bus() so no import changes are needed.
"""
from __future__ import annotations

import abc
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.agents.contracts import (
    AdminDecision,
    AgentMessage,
    Decision,
    MessageType,
    ReservationRequest,
)
from app.db.models import ApprovalRequest, LotReservation, ReservationStatus


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class MessageBus(abc.ABC):
    """
    Minimal, replaceable contract for agent-to-agent communication.

    Each method accepts a SQLAlchemy Session so the caller controls the
    transaction boundary.  A queue-backed implementation would ignore the
    session entirely and use its own I/O channel instead.
    """

    @abc.abstractmethod
    def send_escalation(
        self,
        request: ReservationRequest,
        db: Session,
    ) -> AgentMessage:
        """
        Agent 1 → bus: escalate a user-confirmed reservation for admin review.

        Persists an audit row and returns an ESCALATION AgentMessage whose
        correlation_id ties it to future DECISION messages for the same
        reservation.
        """

    @abc.abstractmethod
    def get_decision(
        self,
        reservation_id: int,
        db: Session,
    ) -> Optional[AdminDecision]:
        """
        Agent 1 polls: return the admin's decision, or None if still pending.

        Returns None when the reservation does not exist, has not been decided
        yet, or ended up in a non-decision state (EXPIRED, CANCELLED, etc.).
        """

    @abc.abstractmethod
    def mark_notified(
        self,
        reservation_id: int,
        db: Session,
    ) -> None:
        """
        Agent 1 confirms it has relayed the outcome to the user.

        Sets LotReservation.notified_user = True.  Idempotent; silently
        ignores unknown reservation IDs.
        """


# ---------------------------------------------------------------------------
# In-process implementation
# ---------------------------------------------------------------------------


class InProcessBus(MessageBus):
    """
    SQL-backed bus for single-process deployments.

    send_escalation  writes an ApprovalRequest row (audit trail).
    get_decision     reads from LotReservation fields set by
                     approve_reservation() / refuse_reservation().
    mark_notified    flips LotReservation.notified_user to True.
    """

    def send_escalation(
        self,
        request: ReservationRequest,
        db: Session,
    ) -> AgentMessage:
        audit = ApprovalRequest(
            reservation_id=request.reservation_id,
            status="sent",
            sent_at=datetime.utcnow(),
            channel="api",
            request_text=request.model_dump_json(),
        )
        db.add(audit)
        db.commit()

        return AgentMessage(
            type=MessageType.ESCALATION,
            payload=request.model_dump(mode="json"),
            correlation_id=request.reservation_id,
        )

    def get_decision(
        self,
        reservation_id: int,
        db: Session,
    ) -> Optional[AdminDecision]:
        res: Optional[LotReservation] = db.get(LotReservation, reservation_id)

        # Not found, or decision not yet recorded
        if res is None or res.decided_at is None:
            return None

        # Only CONFIRMED / REFUSED count as admin decisions
        if res.status not in (ReservationStatus.CONFIRMED, ReservationStatus.REFUSED):
            return None

        admin_id = res.decided_by or "system"

        if res.status == ReservationStatus.CONFIRMED:
            return AdminDecision(
                reservation_id=reservation_id,
                decision=Decision.APPROVE,
                reason=res.admin_decision_reason or None,
                admin_id=admin_id,
                decided_at=res.decided_at,
            )

        # REFUSED — AdminDecision requires a non-empty reason
        reason = (res.admin_decision_reason or "").strip() or "No reason provided."
        return AdminDecision(
            reservation_id=reservation_id,
            decision=Decision.REFUSE,
            reason=reason,
            admin_id=admin_id,
            decided_at=res.decided_at,
        )

    def mark_notified(
        self,
        reservation_id: int,
        db: Session,
    ) -> None:
        res: Optional[LotReservation] = db.get(LotReservation, reservation_id)
        if res is not None and not res.notified_user:
            res.notified_user = True
            db.commit()


# ---------------------------------------------------------------------------
# Module-level singleton — replaceable without changing callers
# ---------------------------------------------------------------------------

_bus: MessageBus = InProcessBus()


def get_bus() -> MessageBus:
    """Return the active message bus (default: InProcessBus)."""
    return _bus


def set_bus(bus: MessageBus) -> None:
    """Replace the singleton — use in tests or to swap in a queue-backed bus."""
    global _bus
    _bus = bus
