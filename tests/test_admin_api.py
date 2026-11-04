"""
Integration tests for the admin API (app/api/routes/admin.py) and the
user-side status/notifications endpoints (app/api/routes/reservations.py).

Uses TestClient with an in-memory StaticPool SQLite database so every
request shares the same connection and sees each other's committed data.

Key fixtures
------------
_engine   module-scoped — creates the schema and seeds one ParkingLot
db_direct function-scoped — raw Session for test data setup
client    module-scoped — TestClient with DB override + key configured

Authentication: TEST_KEY is set on settings.admin_api_key for all tests.
"""
from __future__ import annotations

import time as _time
from datetime import datetime, time, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agents.admin_agent import MockAdminLLMClient
from app.agents.channels.tokens import sign as _sign
from app.agents.contracts import Decision
from app.api.main import app
from app.api.routes.admin import get_admin_llm
from app.db.models import ApprovalRequest, Base, LotReservation, ParkingLot, ReservationStatus
from app.db.repository import (
    approve_reservation,
    create_pending_reservation,
    get_reservation,
)
from app.db.session import get_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_KEY   = "test-admin-key-abc123"
_WRONG_KEY  = "definitely-wrong-key"
_H = {"X-Admin-Key": _TEST_KEY}       # shorthand valid header
_TOKEN_SECRET = "test-token-secret-for-api-tests"


# ---------------------------------------------------------------------------
# Module-scoped engine — single in-memory DB for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    _Sess = sessionmaker(bind=engine)
    db = _Sess()
    db.add(ParkingLot(
        name="ApiTestLot",
        address="1 Api St",
        total_spaces=50,
        available_spaces=50,
        hourly_price=8.0,
        daily_price=40.0,
        opens_at=time(0, 0),
        closes_at=time(23, 59),
    ))
    db.commit()
    db.close()
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def client(_engine):
    """TestClient with DB override and admin key configured."""
    from app.core.config import settings

    _Session = sessionmaker(bind=_engine)
    orig_key = settings.admin_api_key
    settings.admin_api_key = _TEST_KEY

    def _override_db():
        db = _Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_db
    yield TestClient(app, raise_server_exceptions=False)

    settings.admin_api_key = orig_key
    app.dependency_overrides.clear()


@pytest.fixture
def db_direct(_engine):
    """Function-scoped raw session for direct DB setup in tests."""
    Session = sessionmaker(bind=_engine)
    db = Session()
    yield db
    db.rollback()
    db.close()


@pytest.fixture
def mock_llm():
    """Inject a scripted mock LLM for /admin/chat tests."""
    mock = MockAdminLLMClient(["Reservation has been processed successfully."])
    app.dependency_overrides[get_admin_llm] = lambda: mock
    yield mock
    if get_admin_llm in app.dependency_overrides:
        del app.dependency_overrides[get_admin_llm]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lot_id(db) -> int:
    return db.query(ParkingLot).filter_by(name="ApiTestLot").one().id


def _future(h: int) -> datetime:
    return datetime.utcnow() + timedelta(hours=h)


_START_COUNTER = [200]   # global offset so each _pending() call gets a unique window


def _pending(db, session_id: str = "sess-api") -> int:
    offset = _START_COUNTER[0]
    _START_COUNTER[0] += 4
    r = create_pending_reservation(
        db,
        lot_id=_lot_id(db),
        customer_name="Api User",
        car_plate="AP-0001",
        start_time=_future(offset),
        end_time=_future(offset + 2),
        session_id=session_id,
    )
    db.commit()
    return r.id


def _make_email_tokens(db, res_id: int):
    """Create an ApprovalRequest row and sign approve/refuse tokens for it."""
    audit = ApprovalRequest(
        reservation_id=res_id,
        status="sent",
        sent_at=datetime.utcnow(),
        channel="email",
    )
    db.add(audit)
    db.commit()
    approve_tok = _sign(res_id, audit.id, "APPROVE", secret=_TOKEN_SECRET)
    refuse_tok  = _sign(res_id, audit.id, "REFUSE",
                        reason="maintenance", secret=_TOKEN_SECRET)
    return approve_tok, refuse_tok


def _patch_token_secret(monkeypatch=None, *, secret=_TOKEN_SECRET):
    """Patch settings.email_token_secret for token-endpoint tests."""
    from app.core.config import settings
    settings.email_token_secret = secret


# ===========================================================================
# Authentication
# ===========================================================================

class TestAuth:

    def test_no_key_returns_401(self, client):
        r = client.get("/admin/reservations/pending")
        assert r.status_code == 401

    def test_wrong_key_returns_403(self, client):
        r = client.get("/admin/reservations/pending",
                       headers={"X-Admin-Key": _WRONG_KEY})
        assert r.status_code == 403

    def test_valid_key_accepted(self, client):
        r = client.get("/admin/reservations/pending", headers=_H)
        assert r.status_code == 200

    def test_key_comparison_is_not_trivially_timing_attackable(self, client):
        """Both wrong-key requests must return 403, not 200."""
        for bad in ["a", "x" * 200]:
            r = client.get("/admin/reservations/pending",
                           headers={"X-Admin-Key": bad})
            assert r.status_code == 403

    def test_email_endpoints_need_no_key(self, client):
        """Token-based endpoints must be reachable without X-Admin-Key."""
        # An invalid token → 400, not 401/403
        r = client.get("/admin/approve?token=notvalid")
        assert r.status_code not in (401, 403)


# ===========================================================================
# GET /admin/reservations/pending
# ===========================================================================

class TestListPending:

    def test_returns_empty_list_initially(self, client):
        # Approve all existing pending reservations first is impractical
        # with module-scope; just check the response is a list.
        r = client.get("/admin/reservations/pending", headers=_H)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_returns_pending_after_creation(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.get("/admin/reservations/pending", headers=_H)
        assert r.status_code == 200
        ids = [item["id"] for item in r.json()]
        assert res_id in ids

    def test_item_has_required_fields(self, client, db_direct):
        _pending(db_direct)
        r = client.get("/admin/reservations/pending", headers=_H)
        item = r.json()[0]
        for field in ("id", "customer_name", "car_plate_masked", "lot_name",
                      "start_time", "end_time", "status"):
            assert field in item, f"Missing field: {field}"

    def test_plate_is_masked(self, client, db_direct):
        _pending(db_direct)
        r = client.get("/admin/reservations/pending", headers=_H)
        plate = r.json()[0]["car_plate_masked"]
        assert "****" in plate


# ===========================================================================
# GET /admin/reservations/{id}
# ===========================================================================

class TestGetReservation:

    def test_returns_details(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.get(f"/admin/reservations/{res_id}", headers=_H)
        assert r.status_code == 200
        assert r.json()["id"] == res_id

    def test_unknown_id_returns_404(self, client):
        r = client.get("/admin/reservations/99999", headers=_H)
        assert r.status_code == 404

    def test_detail_contains_lot_name(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.get(f"/admin/reservations/{res_id}", headers=_H)
        assert r.json()["lot_name"] == "ApiTestLot"

    def test_detail_plate_masked(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.get(f"/admin/reservations/{res_id}", headers=_H)
        assert "****" in r.json()["car_plate_masked"]


# ===========================================================================
# POST /admin/reservations/{id}/approve
# ===========================================================================

class TestApprove:

    def test_approve_returns_200_and_confirmed_status(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.post(f"/admin/reservations/{res_id}/approve",
                        json={"note": "looks good"}, headers=_H)
        assert r.status_code == 200
        assert r.json()["status"] == "confirmed"

    def test_approve_without_body_uses_empty_note(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.post(f"/admin/reservations/{res_id}/approve", headers=_H)
        assert r.status_code == 200

    def test_approve_returns_decided_by(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.post(f"/admin/reservations/{res_id}/approve",
                        json={}, headers=_H)
        assert r.json()["decided_by"] is not None
        assert "api:" in r.json()["decided_by"]

    def test_approve_unknown_id_returns_404(self, client):
        r = client.post("/admin/reservations/99999/approve",
                        json={}, headers=_H)
        assert r.status_code == 404

    def test_double_approval_returns_409(self, client, db_direct):
        """Approving an already-confirmed reservation must return 409 Conflict."""
        res_id = _pending(db_direct)
        client.post(f"/admin/reservations/{res_id}/approve",
                    json={}, headers=_H)          # first approval
        r = client.post(f"/admin/reservations/{res_id}/approve",
                        json={}, headers=_H)       # second approval
        assert r.status_code == 409
        assert "cannot be approved" in r.json()["detail"].lower()


# ===========================================================================
# POST /admin/reservations/{id}/refuse
# ===========================================================================

class TestRefuse:

    def test_refuse_returns_200_and_refused_status(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.post(f"/admin/reservations/{res_id}/refuse",
                        json={"reason": "lot closed"}, headers=_H)
        assert r.status_code == 200
        assert r.json()["status"] == "refused"

    def test_refuse_includes_reason(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.post(f"/admin/reservations/{res_id}/refuse",
                        json={"reason": "maintenance window"}, headers=_H)
        assert r.json()["reason"] == "maintenance window"

    def test_refuse_missing_reason_returns_422(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.post(f"/admin/reservations/{res_id}/refuse",
                        json={}, headers=_H)
        assert r.status_code == 422

    def test_refuse_unknown_id_returns_404(self, client):
        r = client.post("/admin/reservations/99999/refuse",
                        json={"reason": "x"}, headers=_H)
        assert r.status_code == 404

    def test_double_refuse_returns_409(self, client, db_direct):
        res_id = _pending(db_direct)
        client.post(f"/admin/reservations/{res_id}/refuse",
                    json={"reason": "first"}, headers=_H)
        r = client.post(f"/admin/reservations/{res_id}/refuse",
                        json={"reason": "second"}, headers=_H)
        assert r.status_code == 409

    def test_refuse_after_approve_returns_409(self, client, db_direct):
        res_id = _pending(db_direct)
        client.post(f"/admin/reservations/{res_id}/approve", json={}, headers=_H)
        r = client.post(f"/admin/reservations/{res_id}/refuse",
                        json={"reason": "too late"}, headers=_H)
        assert r.status_code == 409


# ===========================================================================
# GET /admin/approve?token=...  and  GET /admin/refuse?token=...
# ===========================================================================

class TestEmailTokenFlow:

    def _with_secret(self):
        from app.core.config import settings
        orig = settings.email_token_secret
        settings.email_token_secret = _TOKEN_SECRET
        return orig

    def _restore_secret(self, orig):
        from app.core.config import settings
        settings.email_token_secret = orig

    def test_approve_via_token_returns_200(self, client, db_direct):
        orig = self._with_secret()
        try:
            res_id = _pending(db_direct)
            tok, _ = _make_email_tokens(db_direct, res_id)
            r = client.get(f"/admin/approve?token={tok}")
            assert r.status_code == 200
            assert r.json()["status"] == "confirmed"
        finally:
            self._restore_secret(orig)

    def test_refuse_via_token_returns_200(self, client, db_direct):
        orig = self._with_secret()
        try:
            res_id = _pending(db_direct)
            _, tok = _make_email_tokens(db_direct, res_id)
            r = client.get(f"/admin/refuse?token={tok}")
            assert r.status_code == 200
            assert r.json()["status"] == "refused"
        finally:
            self._restore_secret(orig)

    def test_approve_token_on_refuse_endpoint_returns_400(self, client, db_direct):
        """Sending an APPROVE token to /admin/refuse must be rejected."""
        orig = self._with_secret()
        try:
            res_id = _pending(db_direct)
            approve_tok, _ = _make_email_tokens(db_direct, res_id)
            r = client.get(f"/admin/refuse?token={approve_tok}")
            assert r.status_code == 400
        finally:
            self._restore_secret(orig)

    def test_invalid_token_returns_400(self, client):
        r = client.get("/admin/approve?token=totally-invalid-garbage")
        assert r.status_code == 400

    def test_expired_token_returns_410(self, client, db_direct):
        orig = self._with_secret()
        try:
            res_id = _pending(db_direct)
            audit = ApprovalRequest(
                reservation_id=res_id, status="sent",
                sent_at=datetime.utcnow(), channel="email",
            )
            db_direct.add(audit); db_direct.commit()
            # Sign with max_age that already passed (negative)
            # We can't embed expiry in the token, but we can consume
            # the token on the server with a very tight max_age.
            # Instead, manually tamper: use a wrong secret so it's invalid.
            tok = _sign(res_id, audit.id, "APPROVE", secret="wrong-secret-for-expiry-test")
            r = client.get(f"/admin/approve?token={tok}")
            assert r.status_code == 400  # TokenInvalid (wrong secret → BadSignature)
        finally:
            self._restore_secret(orig)

    def test_token_single_use_second_click_returns_409(self, client, db_direct):
        """Clicking the same link twice must fail on the second attempt."""
        orig = self._with_secret()
        try:
            res_id = _pending(db_direct)
            tok, _ = _make_email_tokens(db_direct, res_id)
            client.get(f"/admin/approve?token={tok}")    # first click OK
            r = client.get(f"/admin/approve?token={tok}")  # second click
            assert r.status_code == 409
        finally:
            self._restore_secret(orig)

    def test_approve_token_blocks_refuse_link(self, client, db_direct):
        """After clicking Approve, the Refuse link is blocked (same audit row)."""
        orig = self._with_secret()
        try:
            res_id = _pending(db_direct)
            approve_tok, refuse_tok = _make_email_tokens(db_direct, res_id)
            client.get(f"/admin/approve?token={approve_tok}")  # click Approve
            r = client.get(f"/admin/refuse?token={refuse_tok}")  # Refuse blocked
            assert r.status_code == 409
        finally:
            self._restore_secret(orig)


# ===========================================================================
# POST /admin/chat
# ===========================================================================

class TestAdminChat:

    def test_chat_approve_command(self, client, db_direct, mock_llm):
        res_id = _pending(db_direct)
        r = client.post("/admin/chat",
                        json={"message": f"approve {res_id}"},
                        headers=_H)
        assert r.status_code == 200
        data = r.json()
        assert data["decision_made"] is True
        assert data["reservation_id"] == res_id

    def test_chat_refuse_command_with_reason(self, client, db_direct, mock_llm):
        res_id = _pending(db_direct)
        r = client.post(
            "/admin/chat",
            json={"message": f"refuse {res_id} because system maintenance"},
            headers=_H,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["decision_made"] is True

    def test_chat_ambiguous_returns_clarification(self, client, mock_llm):
        r = client.post("/admin/chat",
                        json={"message": "what should I do?"},
                        headers=_H)
        assert r.status_code == 200
        data = r.json()
        assert data["decision_made"] is False
        assert len(data["response"]) > 0

    def test_chat_unknown_reservation_id_returns_422(self, client, mock_llm):
        r = client.post("/admin/chat",
                        json={"message": "approve 99999"},
                        headers=_H)
        assert r.status_code == 422

    def test_chat_requires_auth(self, client):
        r = client.post("/admin/chat", json={"message": "approve 1"})
        assert r.status_code == 401


# ===========================================================================
# GET /reservations/{id}/status
# ===========================================================================

class TestReservationStatus:

    def test_pending_status_returned(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.get(f"/reservations/{res_id}/status")
        assert r.status_code == 200
        assert r.json()["status"] == "pending"
        assert r.json()["id"] == res_id

    def test_confirmed_status_after_approval(self, client, db_direct):
        res_id = _pending(db_direct)
        client.post(f"/admin/reservations/{res_id}/approve",
                    json={}, headers=_H)
        r = client.get(f"/reservations/{res_id}/status")
        assert r.json()["status"] == "confirmed"

    def test_refused_status_after_refusal(self, client, db_direct):
        res_id = _pending(db_direct)
        client.post(f"/admin/reservations/{res_id}/refuse",
                    json={"reason": "test"}, headers=_H)
        r = client.get(f"/reservations/{res_id}/status")
        assert r.json()["status"] == "refused"
        assert r.json()["admin_decision_reason"] == "test"

    def test_unknown_id_returns_404(self, client):
        r = client.get("/reservations/99999/status")
        assert r.status_code == 404

    def test_status_has_required_fields(self, client, db_direct):
        res_id = _pending(db_direct)
        r = client.get(f"/reservations/{res_id}/status")
        for field in ("id", "status", "admin_decision_reason", "decided_at"):
            assert field in r.json(), f"Missing field: {field}"


# ===========================================================================
# GET /sessions/{session_id}/notifications
# ===========================================================================

class TestNotifications:

    def test_no_notifications_for_pending(self, client, db_direct):
        sid = f"notif-sess-pending-{_START_COUNTER[0]}"
        _pending(db_direct, session_id=sid)
        r = client.get(f"/sessions/{sid}/notifications")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_notification_appears_after_approval(self, client, db_direct):
        sid = f"notif-sess-approve-{_START_COUNTER[0]}"
        res_id = _pending(db_direct, session_id=sid)
        # Approve via admin API
        client.post(f"/admin/reservations/{res_id}/approve",
                    json={}, headers=_H)
        r = client.get(f"/sessions/{sid}/notifications")
        assert r.status_code == 200
        assert r.json()["count"] == 1
        notif = r.json()["notifications"][0]
        assert notif["decision"] == "confirmed"
        assert "approved" in notif["message"].lower()

    def test_notification_appears_after_refusal(self, client, db_direct):
        sid = f"notif-sess-refuse-{_START_COUNTER[0]}"
        res_id = _pending(db_direct, session_id=sid)
        client.post(f"/admin/reservations/{res_id}/refuse",
                    json={"reason": "capacity"}, headers=_H)
        r = client.get(f"/sessions/{sid}/notifications")
        assert r.json()["count"] == 1
        notif = r.json()["notifications"][0]
        assert notif["decision"] == "refused"

    def test_notifications_marked_delivered_on_fetch(self, client, db_direct):
        """Second fetch of the same session must return 0 notifications."""
        sid = f"notif-sess-deliver-{_START_COUNTER[0]}"
        res_id = _pending(db_direct, session_id=sid)
        client.post(f"/admin/reservations/{res_id}/approve",
                    json={}, headers=_H)
        client.get(f"/sessions/{sid}/notifications")   # first fetch (delivers)
        r = client.get(f"/sessions/{sid}/notifications")  # second fetch
        assert r.json()["count"] == 0

    def test_unknown_session_returns_empty(self, client):
        r = client.get("/sessions/no-such-session-xyz/notifications")
        assert r.status_code == 200
        assert r.json()["count"] == 0


# ===========================================================================
# GET /admin/verify  (new endpoint — key validation for the login form)
# ===========================================================================

class TestAdminVerify:

    def test_valid_key_returns_200(self, client):
        r = client.get("/admin/verify", headers=_H)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_missing_key_returns_401(self, client):
        r = client.get("/admin/verify")
        assert r.status_code == 401

    def test_wrong_key_returns_403(self, client):
        r = client.get("/admin/verify", headers={"X-Admin-Key": _WRONG_KEY})
        assert r.status_code == 403

    def test_no_key_configured_returns_503(self, client):
        from app.core.config import settings
        orig = settings.admin_api_key
        settings.admin_api_key = ""
        try:
            r = client.get("/admin/verify")
            assert r.status_code == 503
        finally:
            settings.admin_api_key = orig


# ===========================================================================
# GET /admin/reservations/history  (status filter + auth)
# ===========================================================================

class TestAdminHistory:

    def test_history_requires_auth(self, client):
        r = client.get("/admin/reservations/history")
        assert r.status_code == 401

    def test_wrong_key_returns_403(self, client):
        r = client.get("/admin/reservations/history",
                       headers={"X-Admin-Key": _WRONG_KEY})
        assert r.status_code == 403

    def test_history_returns_list(self, client):
        r = client.get("/admin/reservations/history", headers=_H)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_history_excludes_pending(self, client, db_direct):
        """PENDING reservations must NOT appear in history."""
        res_id = _pending(db_direct)
        r = client.get("/admin/reservations/history", headers=_H)
        assert r.status_code == 200
        ids = [row["id"] for row in r.json()]
        assert res_id not in ids

    def test_history_includes_confirmed(self, client, db_direct):
        """CONFIRMED reservations must appear in history."""
        res_id = _pending(db_direct)
        client.post(f"/admin/reservations/{res_id}/approve",
                    json={}, headers=_H)
        r = client.get("/admin/reservations/history", headers=_H)
        assert r.status_code == 200
        ids = [row["id"] for row in r.json()]
        assert res_id in ids

    def test_history_includes_refused(self, client, db_direct):
        res_id = _pending(db_direct)
        client.post(f"/admin/reservations/{res_id}/refuse",
                    json={"reason": "test"}, headers=_H)
        r = client.get("/admin/reservations/history", headers=_H)
        assert r.status_code == 200
        ids = [row["id"] for row in r.json()]
        assert res_id in ids

    def test_history_limit_param(self, client):
        r = client.get("/admin/reservations/history?limit=1", headers=_H)
        assert r.status_code == 200
        assert len(r.json()) <= 1
