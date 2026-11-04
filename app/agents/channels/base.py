"""
Channel interface and result type.

All channel implementations must:
  • Never raise — return ChannelResult(success=False) instead.
  • Not log secrets (SMTP passwords, token signing keys, API keys).
  • Be independently replaceable (RestChannel, EmailChannel, future SMS, etc.).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app.agents.contracts import ReservationRequest


@dataclass
class ChannelResult:
    """Outcome of a single channel's send() call."""
    channel_name:        str
    success:             bool
    detail:              str = ""
    # ID of the ApprovalRequest audit row created by this channel (if any)
    approval_request_id: Optional[int] = None


class Channel(abc.ABC):
    """
    Abstract base for notification channels.

    Implementations receive a ``ReservationRequest`` (already PII-safe) and
    a live ``Session``.  They are responsible for creating their own
    ``ApprovalRequest`` audit row if they persist anything.
    """

    #: Unique lower-case name used in logs and the channel field.
    name: str = ""

    @abc.abstractmethod
    def send(self, request: ReservationRequest, db: Session) -> ChannelResult:
        """
        Deliver the escalation request through this channel.

        Must NEVER raise.  On any failure, catch the exception, log it
        (without secrets), and return ``ChannelResult(success=False)``.
        """
