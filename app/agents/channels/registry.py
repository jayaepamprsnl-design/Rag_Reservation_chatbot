"""
ChannelRegistry — fan-out to all enabled channels.

Each channel is called independently; a failure in one never raises to
the caller.  Results are collected and returned; failures are logged
(without secrets) so they are visible in monitoring but do not block
the reservation flow.

Usage
-----
    from app.agents.channels import build_registry

    registry = build_registry()          # reads settings
    results  = registry.send_all(request, db)
    # reservation stays PENDING if both channels fail; admin can still
    # decide via direct API call
"""
from __future__ import annotations

from typing import Sequence

from sqlalchemy.orm import Session

from app.agents.channels.base import Channel, ChannelResult
from app.agents.contracts import ReservationRequest
from app.core.config import settings
from app.core.logging_config import logger


class ChannelRegistry:
    """
    Sends a ReservationRequest to every registered channel in order.

    Channels are called sequentially so the primary RestChannel
    always completes before optional side-channels (email, etc.).
    """

    def __init__(self, channels: Sequence[Channel]) -> None:
        self._channels = list(channels)

    # ------------------------------------------------------------------

    def send_all(
        self,
        request: ReservationRequest,
        db: Session,
    ) -> list[ChannelResult]:
        """
        Deliver the escalation to all channels.

        Returns a list of ChannelResult in channel order.
        A failing channel does NOT abort the remaining channels.
        """
        results: list[ChannelResult] = []

        for channel in self._channels:
            try:
                result = channel.send(request, db)
            except Exception as exc:  # noqa: BLE001
                # Belt-and-suspenders: channel.send() should never raise,
                # but if it does we isolate the failure here.
                logger.error(
                    "channel_registry|unexpected_error channel=%s "
                    "reservation_id=%d error_type=%s",
                    channel.name,
                    request.reservation_id,
                    type(exc).__name__,
                )
                result = ChannelResult(
                    channel_name=channel.name,
                    success=False,
                    detail=f"unexpected error ({type(exc).__name__})",
                )

            results.append(result)

            if result.success:
                logger.info(
                    "channel_registry|sent channel=%s reservation_id=%d",
                    result.channel_name,
                    request.reservation_id,
                )
            else:
                logger.warning(
                    "channel_registry|failed channel=%s reservation_id=%d detail=%s",
                    result.channel_name,
                    request.reservation_id,
                    result.detail,         # detail must not contain secrets
                )

        return results

    # ------------------------------------------------------------------

    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self._channels]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_registry() -> ChannelRegistry:
    """
    Build the active ChannelRegistry from current settings.

    Always includes RestChannel.  Adds EmailChannel when
    ADMIN_EMAIL_ENABLED=true.
    """
    from app.agents.channels.rest_channel import RestChannel
    from app.agents.channels.email_channel import EmailChannel

    channels: list[Channel] = [RestChannel()]

    if settings.admin_email_enabled:
        channels.append(EmailChannel())
        logger.info("channel_registry|email_channel_enabled")

    return ChannelRegistry(channels)
