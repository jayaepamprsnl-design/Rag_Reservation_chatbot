"""
Notification channels for Agent 2.

RestChannel   – primary: persists escalation in the approval_requests table
               so the admin API can surface it.
EmailChannel  – optional: SMTP with signed, expiring, single-use links.
ChannelRegistry – fan-out to all enabled channels; failures are isolated.
"""
from app.agents.channels.base import Channel, ChannelResult
from app.agents.channels.registry import ChannelRegistry, build_registry
from app.agents.channels.rest_channel import RestChannel
from app.agents.channels.email_channel import EmailChannel
from app.agents.channels.tokens import (
    TokenExpired,
    TokenInvalid,
    TokenUsed,
    consume,
    sign,
    verify,
)

__all__ = [
    "Channel",
    "ChannelResult",
    "ChannelRegistry",
    "build_registry",
    "RestChannel",
    "EmailChannel",
    "TokenExpired",
    "TokenInvalid",
    "TokenUsed",
    "consume",
    "sign",
    "verify",
]
