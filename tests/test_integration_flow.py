"""
End-to-end integration tests: full escalation -> decision -> notification.

Uses:
  • TestClient with in-memory StaticPool SQLite  (no disk state)
  • MockAdminLLMClient injected via dependency override
  • RAG layer mocked (no ChromaDB / sentence-transformers)
  • SMTP mocked  (EmailChannel disabled by default; tests RestChannel path)

Scenarios
---------
1. User books  → admin approves via POST    → user gets APPROVED notification
2. User books  → admin refuses via POST     → user sees reason, can re-book
3. Admin uses  /admin/chat natural language → decision recorded
4. GET /sessions/{id}/notifications        → delivers, then clears
5. GET /reservations/{id}/status           → reflects current DB state
6. Email-link flow: sign a token, hit /admin/approve?token=... (RestChannel)
"""
from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.admin_agent import MockAdminLLMClient
from app.agents.channels.tokens import sign as tok_sign
from app.api.main import app
from app.api.routes.admin import get_admin_llm
from app.chat.reservation import ReservationState, _sessions, advance, start
from app.core.config import settings
from app.db.models import ApprovalRequest, Base, LotReservation, ParkingLot
from app.db.repository import create_pending_reservation, get_lot
from app.db.session import get_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ADMIN_KEY     = "integration-test-admin-key"
_TOKEN_SECRET  = "integration-token-secret"
_H = {"X-Admin-Key": _ADMIN_KEY}
_RAG_STUB = (
    '{"intent":"STATIC_INFO",'
    '"entities":{"lot_name":null,"sub_intent":null,"date":null}}'
)
_COUNTER = [9000]


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    _Sess = sessionmaker(bind=eng)
    db = _Sess()
    db.add(ParkingLot(
        name="IntLot",
        address="1 Integration Ave",
        total_spaces=10,
        available_spaces=10,
        hourly_price=8.0,
        daily_price=40.0,
        opens_at=time(0, 0),
        closes_at=time(23, 59),
    ))
    db.commit()
    db.close()
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def client(engine):
    """TestClient with all external dependencies stubbed."""
    orig_key   = settings.admin_api_key
    settings.admin_api_key = _ADMIN_KEY
    _Sess = sessionmaker(bind=engine)

    def _override_db():
        db = _Sess()
        try:
            yield db
        finally:
            db.close()

    mock_llm = MockAdminLLMClient(
        ["Decision recorded and user notified."] * 50
    )

    app.dependency_overrides[get_db]        = _override_db
    app.dependency_overrides[get_admin_llm] = lambda: mock_llm

    with (
        patch("app.rag.retriever.embed_query",    return_value=[0.1] * 384),
        patch("app.rag.retriever.similarity_search", return_value=[{
            "text": "CityPark is open all hours.",
            "metadata": {
                "source": "general_info",
                "section": "About",
                "chunk_id": "inttest001",
            },
            "score": 0.92,
        }]),
        patch("app.chat.router.call_llm", return_value=_RAG_STUB),
    ):
        yield TestClient(app, raise_server_exceptions=True)

    settings.admin_api_key = orig_key
    app.dependency_overrides.clear()


@pytest.fixture
def db(engine):
    _Sess = sessionmaker(bind=engine)
    session = _Sess()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fstr(h: int = 25, hr: int = 10) -> str:
    dt = datetime.now() + timedelta(hours=h)
    return dt.replace(hour=hr, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")


def _fend(h: int = 27, hr: int = 14) -> str:
    return _fstr(h, hr)


def _unique_offset() -> int:
    val = _COUNTER[0]
    _COUNTER[0] += 4
    return val


def _drive_to_awaiting(sid: str, db) -> int:
    """Drive SM from IDLE to AWAITING_APPROVAL; return reservation_id.

    The booking form has 6 steps: name → plate → lot → start → end → confirm.
    The state after providing the end time is CONFIRM (shows summary); a second
    advance with "confirm" triggers create_pending_reservation and transitions
    to AWAITING_APPROVAL.
    """
    lot = db.query(ParkingLot).filter_by(name="IntLot").one()
    start(sid)
    advance(sid, "Integ User", db)
    advance(sid, f"IT-{_unique_offset():04d}", db)
    advance(sid, str(lot.id), db)
    o = _unique_offset()
    advance(sid, _fstr(o), db)
    advance(sid, _fend(o + 2), db)        # → CONFIRM (shows summary)
    r = advance(sid, "confirm", db)        # → AWAITING_APPROVAL
    assert r.state == ReservationState.AWAITING_APPROVAL, (
        f"Expected AWAITING_APPROVAL after confirm, got {r.state}: {r.message}"
    )
    return r.reservation_id


# ===========================================================================
# Scenario 1 — User books → Admin approves → User notified in chat
# ===========================================================================


def test_full_approve_user_then_chat_notification(client, db):
    sid = f"integ-a-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    # Admin sees pending
    r = client.get("/admin/reservations/pending", headers=_H)
    assert r.status_code == 200
    assert any(row["id"] == res_id for row in r.json()), (
        f"Reservation #{res_id} must appear in pending list"
    )

    # Admin approves via structured endpoint
    r = client.post(
        f"/admin/reservations/{res_id}/approve",
        json={"note": "Looks good"},
        headers=_H,
    )
    assert r.status_code == 200, r.json()
    assert r.json()["status"] == "confirmed"
    assert r.json()["reservation_id"] == res_id

    # User's next message must deliver the APPROVED notification
    r = client.post(
        "/chat",
        json={"message": "Any news?", "session_id": sid},
        timeout=30,
    )
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert "APPROVED" in answer.upper(), f"Expected APPROVED in answer: {answer[:200]}"

    # Notification must NOT be re-sent on the next message
    r = client.post(
        "/chat",
        json={"message": "Thanks!", "session_id": sid},
        timeout=30,
    )
    answer2 = r.json()["answer"]
    assert "APPROVED" not in answer2.upper(), (
        "Notification was duplicated — notified_user flag not set"
    )

    # History endpoint must show it
    r = client.get("/admin/reservations/history", headers=_H)
    assert r.status_code == 200
    assert any(row["id"] == res_id for row in r.json())

    _sessions.pop(sid, None)


# ===========================================================================
# Scenario 2 — User books → Admin refuses with reason → User sees reason
# ===========================================================================


def test_full_refuse_user_then_chat_notification(client, db):
    sid = f"integ-r-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    r = client.post(
        f"/admin/reservations/{res_id}/refuse",
        json={"reason": "Lot under maintenance"},
        headers=_H,
    )
    assert r.status_code == 200
    assert r.json()["status"] == "refused"
    assert r.json()["reason"] == "Lot under maintenance"

    # User gets refusal with reason
    r = client.post(
        "/chat",
        json={"message": "Hello?", "session_id": sid},
        timeout=30,
    )
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert "maintenance" in answer.lower() or "refused" in answer.lower(), (
        f"Reason not surfaced: {answer[:200]}"
    )

    # Session should be back to IDLE — user can start fresh
    r = client.post(
        "/chat",
        json={"message": "New Integ User", "session_id": sid},
        timeout=30,
    )
    assert r.status_code == 200

    _sessions.pop(sid, None)


# ===========================================================================
# Scenario 3 — Admin uses /admin/chat to approve in natural language
# ===========================================================================


def test_admin_chat_natural_language_approve(client, db):
    sid = f"integ-chat-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    r = client.post(
        "/admin/chat",
        json={"message": f"approve {res_id}"},
        headers=_H,
    )
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["decision_made"] is True
    assert body["reservation_id"] == res_id

    _sessions.pop(sid, None)


def test_admin_chat_natural_language_refuse(client, db):
    sid = f"integ-chat-r-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    r = client.post(
        "/admin/chat",
        json={"message": f"refuse {res_id} because capacity exceeded"},
        headers=_H,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision_made"] is True

    _sessions.pop(sid, None)


def test_admin_chat_ambiguous_returns_clarification(client):
    r = client.post(
        "/admin/chat",
        json={"message": "I'm not sure what to do"},
        headers=_H,
    )
    assert r.status_code == 200
    assert r.json()["decision_made"] is False
    assert len(r.json()["response"]) > 0


# ===========================================================================
# Scenario 4 — GET /sessions/{id}/notifications delivers once
# ===========================================================================


def test_notifications_endpoint_delivers_and_clears(client, db):
    sid = f"integ-notif-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    # Approve
    client.post(
        f"/admin/reservations/{res_id}/approve",
        json={},
        headers=_H,
    )

    # First poll — delivers 1 notification
    r = client.get(f"/sessions/{sid}/notifications")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["notifications"][0]["decision"] == "confirmed"
    assert body["notifications"][0]["reservation_id"] == res_id

    # Second poll — nothing left
    r = client.get(f"/sessions/{sid}/notifications")
    assert r.json()["count"] == 0

    _sessions.pop(sid, None)


def test_notifications_empty_for_unknown_session(client):
    r = client.get("/sessions/no-such-session-xyz/notifications")
    assert r.status_code == 200
    assert r.json()["count"] == 0


# ===========================================================================
# Scenario 5 — GET /reservations/{id}/status reflects DB state
# ===========================================================================


def test_reservation_status_endpoint_pending(client, db):
    sid = f"integ-st-p-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    r = client.get(f"/reservations/{res_id}/status")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    assert r.json()["id"] == res_id

    _sessions.pop(sid, None)


def test_reservation_status_endpoint_after_approval(client, db):
    sid = f"integ-st-a-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    client.post(
        f"/admin/reservations/{res_id}/approve",
        json={"note": "status test"},
        headers=_H,
    )

    r = client.get(f"/reservations/{res_id}/status")
    assert r.status_code == 200
    s = r.json()
    assert s["status"] == "confirmed"
    assert s["decided_by"] is not None
    assert s["decided_at"] is not None

    _sessions.pop(sid, None)


def test_reservation_status_404_for_unknown(client):
    r = client.get("/reservations/99999/status")
    assert r.status_code == 404


# ===========================================================================
# Scenario 6 — Email-link flow (token → /admin/approve endpoint)
# ===========================================================================


def test_email_link_approve_flow(client, db):
    """Sign a token, hit /admin/approve?token=..., verify reservation confirmed."""
    sid = f"integ-email-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    # Create an ApprovalRequest row (normally done by RestChannel)
    audit = ApprovalRequest(
        reservation_id=res_id, status="sent",
        sent_at=datetime.utcnow(), channel="email",
    )
    db.add(audit); db.commit()

    # Sign the token with our test secret
    orig_secret = settings.email_token_secret
    settings.email_token_secret = _TOKEN_SECRET
    token = tok_sign(res_id, audit.id, "APPROVE", secret=_TOKEN_SECRET)

    try:
        r = client.get(f"/admin/approve?token={token}")
        assert r.status_code == 200, r.json()
        assert r.json()["status"] == "confirmed"
        assert r.json()["reservation_id"] == res_id

        # Token must be single-use: second click → 409
        r = client.get(f"/admin/approve?token={token}")
        assert r.status_code == 409
    finally:
        settings.email_token_secret = orig_secret
        _sessions.pop(sid, None)


def test_email_link_refuse_flow(client, db):
    """Token-based refuse flow: single-use, reason preserved."""
    sid = f"integ-email-r-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    audit = ApprovalRequest(
        reservation_id=res_id, status="sent",
        sent_at=datetime.utcnow(), channel="email",
    )
    db.add(audit); db.commit()

    orig_secret = settings.email_token_secret
    settings.email_token_secret = _TOKEN_SECRET
    token = tok_sign(
        res_id, audit.id, "REFUSE",
        reason="lot under maintenance",
        secret=_TOKEN_SECRET,
    )

    try:
        r = client.get(f"/admin/refuse?token={token}")
        assert r.status_code == 200
        assert r.json()["status"] == "refused"
        assert "maintenance" in r.json()["reason"]

        # Single-use
        r2 = client.get(f"/admin/refuse?token={token}")
        assert r2.status_code == 409
    finally:
        settings.email_token_secret = orig_secret
        _sessions.pop(sid, None)


# ===========================================================================
# Scenario 7 — Double-approval and double-refuse must return 409
# ===========================================================================


def test_double_approval_returns_409(client, db):
    sid = f"integ-dbl-a-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    client.post(f"/admin/reservations/{res_id}/approve", json={}, headers=_H)
    r = client.post(f"/admin/reservations/{res_id}/approve", json={}, headers=_H)
    assert r.status_code == 409

    _sessions.pop(sid, None)


def test_double_refuse_returns_409(client, db):
    sid = f"integ-dbl-r-{uuid.uuid4().hex[:6]}"
    res_id = _drive_to_awaiting(sid, db)

    client.post(
        f"/admin/reservations/{res_id}/refuse",
        json={"reason": "first"},
        headers=_H,
    )
    r = client.post(
        f"/admin/reservations/{res_id}/refuse",
        json={"reason": "second"},
        headers=_H,
    )
    assert r.status_code == 409

    _sessions.pop(sid, None)


# ===========================================================================
# Scenario 8 — SMTP login path for EmailChannel
# ===========================================================================


def test_email_channel_smtp_login_when_user_set(db):
    """
    Covers email_channel.py line 207: srv.login() called when smtp_user is set.

    Root cause of prior failure: storing a lambda as a class attribute creates
    a bound method (extra 'self' injected).  Fix: use MagicMock like the
    existing tests in test_channels.py — it handles context-manager protocol
    and call semantics correctly.
    """
    from app.agents.channels.email_channel import EmailChannel
    from app.agents.contracts import AvailabilitySnapshot, ReservationRequest

    res = create_pending_reservation(
        db, lot_id=db.query(ParkingLot).filter_by(name="IntLot").one().id,
        customer_name="SMTP Login User", car_plate="SL-0001",
        start_time=datetime.utcnow() + timedelta(hours=450),
        end_time=datetime.utcnow() + timedelta(hours=452),
    )
    db.commit()

    req = ReservationRequest(
        reservation_id=res.id,
        customer_name=res.customer_name,
        car_plate_masked="SL-****01",
        lot_name="IntLot",
        start_time=res.start_time,
        end_time=res.end_time,
        price_estimate="~$16.00",
        availability_snapshot=AvailabilitySnapshot(
            total_spaces=10, available_spaces=10, is_open=True
        ),
        session_id="smtp-login-test",
    )

    # Build a proper mock using MagicMock (context manager + method calls)
    smtp_instance = MagicMock()
    smtp_class    = MagicMock()
    smtp_class.return_value.__enter__ = MagicMock(return_value=smtp_instance)
    smtp_class.return_value.__exit__  = MagicMock(return_value=False)
    smtp_mod = MagicMock()
    smtp_mod.SMTP = smtp_class

    orig_user   = settings.smtp_user
    orig_pass   = settings.smtp_password
    orig_secret = settings.email_token_secret
    settings.smtp_user          = "admin@example.com"
    settings.smtp_password      = "test-password"   # noqa: S105
    settings.email_token_secret = _TOKEN_SECRET

    try:
        result = EmailChannel(_smtp_factory=smtp_mod).send(req, db)
        assert result.success is True
        # login() must have been called with the configured credentials
        smtp_instance.login.assert_called_once_with(
            "admin@example.com", "test-password"
        )
    finally:
        settings.smtp_user          = orig_user
        settings.smtp_password      = orig_pass
        settings.email_token_secret = orig_secret
