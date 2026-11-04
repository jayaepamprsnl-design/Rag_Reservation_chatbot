"""
Tests for app/agents/channels/ — channel implementations, token lifecycle,
registry fan-out, and failure isolation.

Coverage targets:
  • RestChannel     — stores row, sets correct channel field
  • EmailChannel    — mocked SMTP, message content, link validity, SMTP failure
  • tokens          — sign/verify round-trip, expiry, tamper, single-use, reuse
  • ChannelRegistry — all channels called, failure isolation, log hygiene
"""
from __future__ import annotations

import smtplib
import time as _time
from datetime import datetime, timedelta
from email.message import Message
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agents.channels.base import Channel, ChannelResult
from app.agents.channels.email_channel import EmailChannel, _build_body
from app.agents.channels.registry import ChannelRegistry
from app.agents.channels.rest_channel import RestChannel
from app.agents.channels.tokens import (
    TokenExpired,
    TokenInvalid,
    TokenUsed,
    consume,
    sign,
    verify,
)
from app.agents.contracts import AvailabilitySnapshot, ReservationRequest
from app.db.models import ApprovalRequest, Base, ParkingLot
from app.db.repository import create_pending_reservation

# ---------------------------------------------------------------------------
# Shared test secret (never the production value, never logged)
# ---------------------------------------------------------------------------
_SECRET = "test-signing-secret-not-production"
_EXPIRE = 3600   # 1 h for most tests; overridden where testing expiry


# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(ParkingLot(
        name="ChannelTestLot",
        address="1 Channel St",
        total_spaces=10,
        available_spaces=10,
        hourly_price=8.0,
        daily_price=40.0,
        opens_at=datetime.min.time(),
        closes_at=datetime.max.time(),
    ))
    session.commit()
    yield session
    session.close()
    engine.dispose()


def _lot_id(db) -> int:
    return db.query(ParkingLot).filter_by(name="ChannelTestLot").one().id


def _future(h: int = 3) -> datetime:
    return datetime.utcnow() + timedelta(hours=h)


def _pending_id(db) -> int:
    r = create_pending_reservation(
        db, lot_id=_lot_id(db), customer_name="Chan User",
        car_plate="CH-0001", start_time=_future(3), end_time=_future(5),
    )
    return r.id


def _make_request(reservation_id: int = 42) -> ReservationRequest:
    return ReservationRequest(
        reservation_id=reservation_id,
        customer_name="Alice Smith",
        car_plate_masked="AL****34",
        lot_name="ChannelTestLot",
        start_time=datetime(2027, 6, 15, 10, 0),
        end_time=datetime(2027, 6, 15, 14, 0),
        price_estimate="~$32.00",
        availability_snapshot=AvailabilitySnapshot(
            total_spaces=10, available_spaces=8, is_open=True
        ),
        session_id="sess-chan",
    )


# ===========================================================================
# tokens — sign / verify
# ===========================================================================

class TestSignVerify:

    def test_round_trip(self):
        token = sign(7, 3, "APPROVE", secret=_SECRET)
        data  = verify(token, secret=_SECRET, max_age=_EXPIRE)
        assert data["rid"] == 7
        assert data["aid"] == 3
        assert data["dec"] == "APPROVE"
        assert data["rsn"] == ""

    def test_refuse_with_reason(self):
        token = sign(7, 3, "REFUSE", "maintenance", secret=_SECRET)
        data  = verify(token, secret=_SECRET, max_age=_EXPIRE)
        assert data["dec"] == "REFUSE"
        assert data["rsn"] == "maintenance"

    def test_decision_normalised_to_uppercase(self):
        token = sign(1, 1, "approve", secret=_SECRET)
        data  = verify(token, secret=_SECRET, max_age=_EXPIRE)
        assert data["dec"] == "APPROVE"

    def test_token_is_url_safe_string(self):
        token = sign(1, 1, "APPROVE", secret=_SECRET)
        assert isinstance(token, str)
        # URL-safe: no characters that would need percent-encoding
        assert " " not in token

    def test_tampered_payload_raises_invalid(self):
        token = sign(1, 1, "APPROVE", secret=_SECRET)
        # flip a character in the middle of the token
        bad = token[:10] + ("X" if token[10] != "X" else "Y") + token[11:]
        with pytest.raises(TokenInvalid):
            verify(bad, secret=_SECRET, max_age=_EXPIRE)

    def test_wrong_secret_raises_invalid(self):
        token = sign(1, 1, "APPROVE", secret=_SECRET)
        with pytest.raises(TokenInvalid):
            verify(token, secret="wrong-secret", max_age=_EXPIRE)

    def test_required_fields_present(self):
        token = sign(12, 5, "REFUSE", "reason", secret=_SECRET)
        data  = verify(token, secret=_SECRET, max_age=_EXPIRE)
        for key in ("rid", "aid", "dec", "rsn"):
            assert key in data


# ===========================================================================
# tokens — expiry
# ===========================================================================

class TestTokenExpiry:

    def test_expired_raises_token_expired(self):
        token = sign(1, 1, "APPROVE", secret=_SECRET)
        # max_age=-1 → timedelta(seconds=-1); any token age >= 0 > -1 → expired
        with pytest.raises(TokenExpired):
            verify(token, secret=_SECRET, max_age=-1)

    def test_not_yet_expired(self):
        token = sign(1, 1, "APPROVE", secret=_SECRET)
        # Large max_age — should not expire
        data = verify(token, secret=_SECRET, max_age=99999)
        assert data["rid"] == 1

    def test_token_expired_type_is_value_error(self):
        token = sign(1, 1, "APPROVE", secret=_SECRET)
        with pytest.raises(ValueError):   # TokenExpired is a ValueError
            verify(token, secret=_SECRET, max_age=-1)


# ===========================================================================
# tokens — single-use (consume)
# ===========================================================================

class TestConsume:

    def test_consume_returns_payload(self, db):
        res_id = _pending_id(db)
        # Create an ApprovalRequest row to anchor the token
        audit = ApprovalRequest(
            reservation_id=res_id, status="sent",
            sent_at=datetime.utcnow(), channel="email",
        )
        db.add(audit); db.commit()

        token = sign(res_id, audit.id, "APPROVE", secret=_SECRET)
        data  = consume(token, db, secret=_SECRET, max_age=_EXPIRE)
        assert data["rid"] == res_id
        assert data["dec"] == "APPROVE"

    def test_consume_marks_row_responded(self, db):
        res_id = _pending_id(db)
        audit = ApprovalRequest(
            reservation_id=res_id, status="sent",
            sent_at=datetime.utcnow(), channel="email",
        )
        db.add(audit); db.commit()

        token = sign(res_id, audit.id, "APPROVE", secret=_SECRET)
        consume(token, db, secret=_SECRET, max_age=_EXPIRE)

        db.refresh(audit)
        assert audit.status == "responded"
        assert audit.responded_at is not None

    def test_consume_second_use_raises_token_used(self, db):
        """Single-use enforcement: the second consume raises TokenUsed."""
        res_id = _pending_id(db)
        audit = ApprovalRequest(
            reservation_id=res_id, status="sent",
            sent_at=datetime.utcnow(), channel="email",
        )
        db.add(audit); db.commit()

        token = sign(res_id, audit.id, "APPROVE", secret=_SECRET)
        consume(token, db, secret=_SECRET, max_age=_EXPIRE)  # first use OK

        with pytest.raises(TokenUsed):
            consume(token, db, secret=_SECRET, max_age=_EXPIRE)  # second → error

    def test_consume_approve_blocks_refuse_link(self, db):
        """Clicking Approve must prevent the Refuse link from working."""
        res_id = _pending_id(db)
        audit = ApprovalRequest(
            reservation_id=res_id, status="sent",
            sent_at=datetime.utcnow(), channel="email",
        )
        db.add(audit); db.commit()

        # Both tokens point to the SAME ApprovalRequest row
        approve_tok = sign(res_id, audit.id, "APPROVE", secret=_SECRET)
        refuse_tok  = sign(res_id, audit.id, "REFUSE", "reason", secret=_SECRET)

        consume(approve_tok, db, secret=_SECRET, max_age=_EXPIRE)  # click Approve

        with pytest.raises(TokenUsed):
            consume(refuse_tok, db, secret=_SECRET, max_age=_EXPIRE)  # Refuse blocked

    def test_consume_expired_raises_token_expired(self, db):
        res_id = _pending_id(db)
        audit = ApprovalRequest(
            reservation_id=res_id, status="sent",
            sent_at=datetime.utcnow(), channel="email",
        )
        db.add(audit); db.commit()

        token = sign(res_id, audit.id, "APPROVE", secret=_SECRET)
        with pytest.raises(TokenExpired):
            consume(token, db, secret=_SECRET, max_age=-1)

    def test_consume_unknown_approval_request_raises_invalid(self, db):
        res_id = _pending_id(db)
        # aid=99999 does not exist in DB
        token = sign(res_id, 99999, "APPROVE", secret=_SECRET)
        with pytest.raises(TokenInvalid, match="not found"):
            consume(token, db, secret=_SECRET, max_age=_EXPIRE)


# ===========================================================================
# RestChannel
# ===========================================================================

class TestRestChannel:

    def test_creates_approval_request_row(self, db):
        req    = _make_request(_pending_id(db))
        result = RestChannel().send(req, db)
        row = db.query(ApprovalRequest).filter_by(
            reservation_id=req.reservation_id
        ).first()
        assert row is not None

    def test_returns_successful_result(self, db):
        result = RestChannel().send(_make_request(_pending_id(db)), db)
        assert result.success is True
        assert result.channel_name == "rest"

    def test_sets_channel_field_to_rest(self, db):
        req = _make_request(_pending_id(db))
        RestChannel().send(req, db)
        row = db.query(ApprovalRequest).filter_by(
            reservation_id=req.reservation_id
        ).first()
        assert row.channel == "rest"

    def test_stores_request_json(self, db):
        req = _make_request(_pending_id(db))
        RestChannel().send(req, db)
        row = db.query(ApprovalRequest).filter_by(
            reservation_id=req.reservation_id
        ).first()
        assert row.request_text is not None
        assert "Alice Smith" in row.request_text

    def test_approval_request_id_returned(self, db):
        result = RestChannel().send(_make_request(_pending_id(db)), db)
        assert result.approval_request_id is not None
        assert isinstance(result.approval_request_id, int)

    def test_db_failure_returns_failed_result(self, db):
        """If the DB write raises, RestChannel returns success=False (no re-raise)."""
        req = _make_request(_pending_id(db))
        # Mock db.add to raise so we get a reliable failure regardless of session state
        with patch.object(db, "add", side_effect=Exception("simulated DB error")):
            result = RestChannel().send(req, db)
        assert result.success is False
        assert result.channel_name == "rest"


# ===========================================================================
# EmailChannel — mocked SMTP
# ===========================================================================

def _mock_smtp():
    """Return a mock that behaves like smtplib when used as a context manager."""
    smtp_instance = MagicMock()
    smtp_class    = MagicMock()
    smtp_class.return_value.__enter__ = MagicMock(return_value=smtp_instance)
    smtp_class.return_value.__exit__  = MagicMock(return_value=False)
    smtp_module   = MagicMock()
    smtp_module.SMTP = smtp_class
    return smtp_module, smtp_instance


class TestEmailChannel:

    def test_send_calls_smtp_send_message(self, db):
        smtp_mod, smtp_inst = _mock_smtp()
        req    = _make_request(_pending_id(db))
        result = EmailChannel(_smtp_factory=smtp_mod).send(req, db)
        assert result.success is True
        smtp_inst.send_message.assert_called_once()

    def test_creates_approval_request_row_with_email_channel(self, db):
        smtp_mod, _ = _mock_smtp()
        req = _make_request(_pending_id(db))
        EmailChannel(_smtp_factory=smtp_mod).send(req, db)
        row = db.query(ApprovalRequest).filter_by(
            reservation_id=req.reservation_id, channel="email"
        ).first()
        assert row is not None
        assert row.status == "sent"

    def test_message_subject_contains_reservation_id(self, db):
        smtp_mod, smtp_inst = _mock_smtp()
        req    = _make_request(_pending_id(db))
        EmailChannel(_smtp_factory=smtp_mod).send(req, db)

        sent_msg: Message = smtp_inst.send_message.call_args[0][0]
        assert str(req.reservation_id) in sent_msg["Subject"]

    def test_message_to_is_admin_email(self, db):
        smtp_mod, smtp_inst = _mock_smtp()
        req = _make_request(_pending_id(db))
        with patch("app.agents.channels.email_channel.settings") as mock_settings:
            mock_settings.admin_email              = "manager@test.com"
            mock_settings.email_from               = "noreply@test.com"
            mock_settings.smtp_host                = "localhost"
            mock_settings.smtp_port                = 1025
            mock_settings.smtp_user                = ""
            mock_settings.smtp_password            = ""
            mock_settings.admin_api_base_url       = "http://test.local"
            mock_settings.email_token_secret       = _SECRET
            mock_settings.email_token_max_age_seconds = _EXPIRE
            EmailChannel(_smtp_factory=smtp_mod).send(req, db)

        sent_msg = smtp_inst.send_message.call_args[0][0]
        assert "manager@test.com" in sent_msg["To"]

    def test_body_contains_customer_name(self, db):
        smtp_mod, smtp_inst = _mock_smtp()
        req = _make_request(_pending_id(db))
        with patch("app.agents.channels.email_channel.settings") as ms:
            ms.admin_email = "admin@x.com"; ms.email_from = "nr@x.com"
            ms.smtp_host = "localhost"; ms.smtp_port = 1025
            ms.smtp_user = ""; ms.smtp_password = ""
            ms.admin_api_base_url = "http://x.local"
            ms.email_token_secret = _SECRET
            ms.email_token_max_age_seconds = _EXPIRE
            EmailChannel(_smtp_factory=smtp_mod).send(req, db)

        payload = smtp_inst.send_message.call_args[0][0].get_payload(decode=True)
        body = (payload or b"").decode() if payload else ""
        # Fallback: try string payload
        if not body:
            body = smtp_inst.send_message.call_args[0][0].get_payload()
            if isinstance(body, list):
                body = body[0].get_payload(decode=True).decode()
        assert "Alice Smith" in body

    def test_body_contains_approve_link(self, db):
        smtp_mod, smtp_inst = _mock_smtp()
        req = _make_request(_pending_id(db))
        with patch("app.agents.channels.email_channel.settings") as ms:
            ms.admin_email = "a@b.com"; ms.email_from = "n@b.com"
            ms.smtp_host = "localhost"; ms.smtp_port = 1025
            ms.smtp_user = ""; ms.smtp_password = ""
            ms.admin_api_base_url = "http://myapi.local"
            ms.email_token_secret = _SECRET
            ms.email_token_max_age_seconds = _EXPIRE
            EmailChannel(_smtp_factory=smtp_mod).send(req, db)

        msg = smtp_inst.send_message.call_args[0][0]
        # Walk MIME parts to find the text body
        body = _extract_text_body(msg)
        assert "http://myapi.local/admin/approve?token=" in body

    def test_body_contains_refuse_link(self, db):
        smtp_mod, smtp_inst = _mock_smtp()
        req = _make_request(_pending_id(db))
        with patch("app.agents.channels.email_channel.settings") as ms:
            ms.admin_email = "a@b.com"; ms.email_from = "n@b.com"
            ms.smtp_host = "localhost"; ms.smtp_port = 1025
            ms.smtp_user = ""; ms.smtp_password = ""
            ms.admin_api_base_url = "http://refuseapi.local"
            ms.email_token_secret = _SECRET
            ms.email_token_max_age_seconds = _EXPIRE
            EmailChannel(_smtp_factory=smtp_mod).send(req, db)

        msg  = smtp_inst.send_message.call_args[0][0]
        body = _extract_text_body(msg)
        # There is one approve link and one refuse link
        assert body.count("/admin/approve?token=") == 1
        assert body.count("/admin/refuse?token=") == 1

    def test_approve_and_refuse_tokens_are_different(self, db):
        smtp_mod, smtp_inst = _mock_smtp()
        req = _make_request(_pending_id(db))
        with patch("app.agents.channels.email_channel.settings") as ms:
            ms.admin_email = "a@b.com"; ms.email_from = "n@b.com"
            ms.smtp_host = "localhost"; ms.smtp_port = 1025
            ms.smtp_user = ""; ms.smtp_password = ""
            ms.admin_api_base_url = "http://tok.local"
            ms.email_token_secret = _SECRET
            ms.email_token_max_age_seconds = _EXPIRE
            EmailChannel(_smtp_factory=smtp_mod).send(req, db)

        body = _extract_text_body(smtp_inst.send_message.call_args[0][0])
        tokens = [p.split("token=", 1)[1].split()[0]
                  for p in body.split("\n")
                  if "/admin/approve?token=" in p or "/admin/refuse?token=" in p]
        assert len(tokens) == 2
        assert tokens[0] != tokens[1]

    def test_smtp_failure_returns_failed_result(self, db):
        """An SMTP error must not raise — channel returns success=False."""
        smtp_mod = MagicMock()
        smtp_mod.SMTP.side_effect = smtplib.SMTPConnectError(
            421, "Connection refused"
        )
        req    = _make_request(_pending_id(db))
        result = EmailChannel(_smtp_factory=smtp_mod).send(req, db)
        assert result.success is False
        assert result.channel_name == "email"

    def test_smtp_failure_detail_has_no_password(self, db):
        """The ChannelResult.detail must not expose SMTP credentials."""
        smtp_mod = MagicMock()
        smtp_mod.SMTP.side_effect = smtplib.SMTPAuthenticationError(
            535, b"Authentication failed"
        )
        req    = _make_request(_pending_id(db))
        result = EmailChannel(_smtp_factory=smtp_mod).send(req, db)
        # The detail contains only the exception class name, not any secret
        assert "password" not in result.detail.lower()
        assert "secret" not in result.detail.lower()

    def test_body_contains_expiry_notice(self, db):
        """Email body must mention token expiry so admin knows the link is time-limited."""
        req = _make_request(_pending_id(db))
        snap = req.availability_snapshot
        body = _build_body(
            req,
            approve_url="http://x/admin/email-action?token=AAA",
            refuse_url="http://x/admin/email-action?token=BBB",
        )
        assert "expire" in body.lower() or "single-use" in body.lower()

    def test_email_tokens_are_valid_and_decodable(self, db):
        """Tokens generated by EmailChannel must be verifiable with the same secret."""
        smtp_mod, smtp_inst = _mock_smtp()
        req = _make_request(_pending_id(db))
        with patch("app.agents.channels.email_channel.settings") as ms:
            ms.admin_email = "a@b.com"; ms.email_from = "n@b.com"
            ms.smtp_host = "localhost"; ms.smtp_port = 1025
            ms.smtp_user = ""; ms.smtp_password = ""
            ms.admin_api_base_url = "http://v.local"
            ms.email_token_secret = _SECRET
            ms.email_token_max_age_seconds = _EXPIRE
            EmailChannel(_smtp_factory=smtp_mod).send(req, db)

        body = _extract_text_body(smtp_inst.send_message.call_args[0][0])
        tokens = [p.split("token=", 1)[1].split()[0]
                  for p in body.split("\n")
                  if "/admin/approve?token=" in p or "/admin/refuse?token=" in p]
        for tok in tokens:
            data = verify(tok, secret=_SECRET, max_age=_EXPIRE)
            assert data["rid"] == req.reservation_id


# ===========================================================================
# ChannelRegistry
# ===========================================================================

class TestChannelRegistry:

    def test_send_all_calls_every_channel(self, db):
        results_a: list[ChannelResult] = []
        results_b: list[ChannelResult] = []

        class _A(Channel):
            name = "a"
            def send(self, r, db):
                results_a.append(ChannelResult("a", True))
                return results_a[-1]

        class _B(Channel):
            name = "b"
            def send(self, r, db):
                results_b.append(ChannelResult("b", True))
                return results_b[-1]

        registry = ChannelRegistry([_A(), _B()])
        all_results = registry.send_all(_make_request(_pending_id(db)), db)
        assert len(all_results) == 2
        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_failing_channel_does_not_prevent_others(self, db):
        """A channel that raises must not abort subsequent channels."""

        class _Boom(Channel):
            name = "boom"
            def send(self, r, db):
                raise RuntimeError("simulated crash")

        class _Good(Channel):
            name = "good"
            called = False
            def send(self, r, db):
                _Good.called = True
                return ChannelResult("good", True)

        registry = ChannelRegistry([_Boom(), _Good()])
        results = registry.send_all(_make_request(_pending_id(db)), db)

        assert _Good.called, "Good channel was not called after Boom failed"
        boom_result = next(r for r in results if r.channel_name == "boom")
        assert boom_result.success is False

    def test_registry_returns_results_for_all_channels(self, db):
        class _Ok(Channel):
            name = "ok"
            def send(self, r, db):
                return ChannelResult("ok", True)

        class _Fail(Channel):
            name = "fail"
            def send(self, r, db):
                return ChannelResult("fail", False, "reason")

        registry = ChannelRegistry([_Ok(), _Fail()])
        results = registry.send_all(_make_request(_pending_id(db)), db)
        assert {r.channel_name for r in results} == {"ok", "fail"}

    def test_rest_channel_first_email_second(self, db):
        """build_registry puts RestChannel first, EmailChannel second."""
        from app.agents.channels.registry import build_registry
        with patch("app.agents.channels.registry.settings") as ms:
            ms.admin_email_enabled = True
            registry = build_registry()
        assert registry.channel_names[0] == "rest"
        assert registry.channel_names[1] == "email"

    def test_build_registry_email_disabled(self):
        from app.agents.channels.registry import build_registry
        with patch("app.agents.channels.registry.settings") as ms:
            ms.admin_email_enabled = False
            registry = build_registry()
        assert registry.channel_names == ["rest"]

    def test_failure_detail_does_not_contain_secret(self, db):
        """Log/result detail must never expose signing secrets or passwords."""
        smtp_mod = MagicMock()
        smtp_mod.SMTP.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")
        email_ch = EmailChannel(_smtp_factory=smtp_mod)
        registry = ChannelRegistry([email_ch])
        results  = registry.send_all(_make_request(_pending_id(db)), db)
        failed   = next(r for r in results if not r.success)
        assert _SECRET not in failed.detail
        assert "password" not in failed.detail.lower()


# ===========================================================================
# _build_body helper (unit)
# ===========================================================================

class TestBuildBody:

    def _body(self) -> str:
        req = _make_request(42)
        return _build_body(req, "http://x/approve", "http://x/refuse")

    def test_contains_reservation_id(self):
        assert "42" in self._body()

    def test_contains_customer_name(self):
        assert "Alice Smith" in self._body()

    def test_contains_lot_name(self):
        assert "ChannelTestLot" in self._body()

    def test_contains_price_estimate(self):
        assert "~$32.00" in self._body()

    def test_contains_approve_url(self):
        assert "http://x/approve" in self._body()

    def test_contains_refuse_url(self):
        assert "http://x/refuse" in self._body()

    def test_contains_availability(self):
        body = self._body()
        assert "8" in body and "10" in body   # available / total

    def test_single_use_notice(self):
        body = self._body()
        assert "single-use" in body.lower() or "expire" in body.lower()


# ===========================================================================
# Helpers
# ===========================================================================

def _extract_text_body(msg: Any) -> str:
    """Recursively extract the first plain-text part from a MIME message."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")
        # Fallback: string payload
        payload = msg.get_payload()
        if isinstance(payload, list):
            first = payload[0].get_payload(decode=True)
            return (first or b"").decode("utf-8", errors="replace")
        return str(payload)
    payload = msg.get_payload(decode=True)
    return (payload or b"").decode("utf-8", errors="replace")
