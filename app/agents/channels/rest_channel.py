"""
RestChannel — primary notification channel.

Stores the escalation request in the ``approval_requests`` table as a
``channel="rest"`` row.  No external calls; the admin API (Prompt 5) reads
these rows to surface pending reservations.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.agents.channels.base import Channel, ChannelResult
from app.agents.contracts import ReservationRequest
from app.core.logging_config import logger
from app.db.models import ApprovalRequest


class RestChannel(Channel):
    """
    Primary channel — persists the escalation so the REST admin API can
    discover and display it.  Always runs first in the ChannelRegistry.
    """

    name = "rest"

    def send(self, request: ReservationRequest, db: Session) -> ChannelResult:
        try:
            audit = ApprovalRequest(
                reservation_id=request.reservation_id,
                status="sent",
                sent_at=datetime.utcnow(),
                channel=self.name,
                request_text=request.model_dump_json(),
            )
            db.add(audit)
            db.commit()

            logger.info(
                "rest_channel|stored reservation_id=%d approval_request_id=%d",
                request.reservation_id,
                audit.id,
            )
            return ChannelResult(
                channel_name=self.name,
                success=True,
                detail="stored in approval_requests",
                approval_request_id=audit.id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "rest_channel|send_failed reservation_id=%d error_type=%s",
                request.reservation_id,
                type(exc).__name__,
            )
            return ChannelResult(
                channel_name=self.name,
                success=False,
                detail=f"DB write failed ({type(exc).__name__})",
            )
