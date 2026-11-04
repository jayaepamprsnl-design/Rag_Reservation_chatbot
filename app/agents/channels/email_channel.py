"""
EmailChannel — optional SMTP notification channel.

Enabled by setting ADMIN_EMAIL_ENABLED=true in .env.

Each send() creates an ApprovalRequest row (channel="email"), generates two
signed single-use tokens (APPROVE / REFUSE), embeds them as click-through
links in a plain-text email, and delivers via SMTP.

Single-use enforcement
----------------------
Both links point to the same ApprovalRequest row (approval_request_id).
When the admin clicks either link, tokens.consume() marks the row
status="responded".  The second link then raises TokenUsed regardless of
whether the reservation itself has already been decided.

Secret hygiene
--------------
SMTP credentials and the token signing key are never included in log
messages.  Errors are logged using only the exception class name.

Links
-----
Approve: {ADMIN_API_BASE_URL}/admin/approve?token=<signed-approve-token>
Refuse:  {ADMIN_API_BASE_URL}/admin/refuse?token=<signed-refuse-token>

Both endpoints live in app/api/routes/admin.py.
"""
from __future__ import annotations

import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from sqlalchemy.orm import Session

from app.agents.channels.base import Channel, ChannelResult
from app.agents.channels.tokens import sign
from app.agents.contracts import ReservationRequest
from app.core.config import settings
from app.core.logging_config import logger
from app.db.models import ApprovalRequest


# ---------------------------------------------------------------------------
# Email body builder
# ---------------------------------------------------------------------------

_DIVIDER = "═" * 52

_BODY_TEMPLATE = """\
{divider}
  CITYPARK — RESERVATION APPROVAL REQUEST
{divider}

Reservation #{rid} is waiting for your decision.

CUSTOMER
  Name    :  {customer_name}
  Plate   :  {car_plate}

BOOKING
  Facility:  {lot_name}
  Arrives :  {start_time}
  Departs :  {end_time}
  Est. cost: {price_estimate}

AVAILABILITY (at escalation time)
  Spaces  :  {available_spaces} available / {total_spaces} total

{divider}
  ACTION REQUIRED
{divider}

  ✓  APPROVE
     {approve_url}

  ✗  REFUSE  (reason: "Refused via one-click email link")
     {refuse_url}

     To refuse with a custom reason, use the admin API instead.

This link expires in {max_age_hours:.0f} hours and is single-use.
{divider}
"""


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _build_body(
    request: ReservationRequest,
    approve_url: str,
    refuse_url: str,
) -> str:
    snap = request.availability_snapshot
    return _BODY_TEMPLATE.format(
        divider=_DIVIDER,
        rid=request.reservation_id,
        customer_name=request.customer_name,
        car_plate=request.car_plate_masked,
        lot_name=request.lot_name,
        start_time=_fmt_dt(request.start_time),
        end_time=_fmt_dt(request.end_time),
        price_estimate=request.price_estimate,
        available_spaces=snap.available_spaces,
        total_spaces=snap.total_spaces,
        approve_url=approve_url,
        refuse_url=refuse_url,
        max_age_hours=settings.email_token_max_age_seconds / 3600,
    )


# ---------------------------------------------------------------------------
# EmailChannel
# ---------------------------------------------------------------------------


class EmailChannel(Channel):
    """
    SMTP-based notification channel.

    Uses the synchronous ``smtplib`` module.  Inject a pre-constructed
    SMTP connection via ``_smtp_factory`` in tests to avoid network calls.
    """

    name = "email"

    def __init__(
        self,
        _smtp_factory: Optional[object] = None,
    ) -> None:
        # _smtp_factory is the smtplib module (or a mock) so tests can
        # substitute it without patching at the module level.
        self._smtp_module = _smtp_factory or smtplib

    # ------------------------------------------------------------------

    def send(self, request: ReservationRequest, db: Session) -> ChannelResult:
        try:
            return self._send(request, db)
        except Exception as exc:  # noqa: BLE001
            # Never raise — log type only (no credentials in message)
            logger.error(
                "email_channel|send_failed reservation_id=%d error_type=%s",
                request.reservation_id,
                type(exc).__name__,
            )
            return ChannelResult(
                channel_name=self.name,
                success=False,
                detail=f"send failed ({type(exc).__name__})",
            )

    def _send(self, request: ReservationRequest, db: Session) -> ChannelResult:
        # 1. Create audit row first (before SMTP so we always have a record)
        audit = ApprovalRequest(
            reservation_id=request.reservation_id,
            status="sent",
            sent_at=datetime.utcnow(),
            channel=self.name,
            request_text=request.model_dump_json(),
        )
        db.add(audit)
        db.commit()

        # 2. Generate signed tokens (approve + refuse both anchor to this row).
        # Pass the secret explicitly so tests that patch settings here also
        # control the token secret without having to patch tokens.settings too.
        approve_token = sign(
            request.reservation_id,
            audit.id,
            "APPROVE",
            secret=settings.email_token_secret,
        )
        refuse_token = sign(
            request.reservation_id,
            audit.id,
            "REFUSE",
            reason="Refused via one-click email link",
            secret=settings.email_token_secret,
        )

        # 3. Build action URLs — must match the actual admin API endpoints
        base = settings.admin_api_base_url.rstrip("/")
        approve_url = f"{base}/admin/approve?token={approve_token}"
        refuse_url  = f"{base}/admin/refuse?token={refuse_token}"

        # 4. Compose message
        body    = _build_body(request, approve_url, refuse_url)
        subject = (
            f"[CityPark] Approval Required: "
            f"Reservation #{request.reservation_id}"
        )
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = settings.email_from
        msg["To"]      = settings.admin_email
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # 5. Send via SMTP (no credentials logged)
        with self._smtp_module.SMTP(settings.smtp_host, settings.smtp_port) as srv:
            if settings.smtp_user:
                srv.login(settings.smtp_user, settings.smtp_password)
            srv.send_message(msg)

        logger.info(
            "email_channel|sent reservation_id=%d to=%s approval_request_id=%d",
            request.reservation_id,
            settings.admin_email,
            audit.id,
        )
        return ChannelResult(
            channel_name=self.name,
            success=True,
            detail=f"sent to {settings.admin_email}",
            approval_request_id=audit.id,
        )
