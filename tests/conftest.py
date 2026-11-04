import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Base


@pytest.fixture(autouse=True)
def _no_mcp_export(monkeypatch):
    """Existing tests must never reach a real MCP server; tests/test_mcp_integration.py opts in."""
    monkeypatch.setattr(settings, "mcp_export_enabled", False)


@pytest.fixture(autouse=True)
def _real_token_secret(monkeypatch):
    """The shipped placeholder secret is refused (forgeable links); tests use a real-looking one.

    Tests that need a specific value still set settings.email_token_secret themselves.
    """
    monkeypatch.setattr(settings, "email_token_secret", "pytest-email-token-secret-0123456789")


@pytest.fixture(autouse=True)
def _isolated_graph(monkeypatch):
    """Every test gets its own in-memory LangGraph checkpointer; closed afterwards."""
    from app.orchestration import graph as graph_mod

    monkeypatch.setattr(settings, "checkpoint_db", ":memory:")
    monkeypatch.setattr(settings, "approval_expiry_enabled", False)
    graph_mod.reset_graph()
    yield
    graph_mod.reset_graph()


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# Stage 4: orchestration harness (tmp-file app DB + tmp-file checkpointer, fake
# classifier LLM, mocked MCP client).  Everything is disposed on teardown.
# ---------------------------------------------------------------------------

ORCH_ADMIN_KEY = "orch-test-admin-key"


class OrchHarness:
    """Drives the real FastAPI app through TestClient against tmp SQLite files."""

    def __init__(self, client, Session, engine, send, tmp_path):
        self.client, self.Session, self.engine = client, Session, engine
        self.send = send                      # AsyncMock replacing mcp_client.send_reservation
        self.tmp_path = tmp_path
        self.headers = {"X-Admin-Key": ORCH_ADMIN_KEY}
        self.sessions: list[str] = []

    # -- customer side ------------------------------------------------------
    def chat(self, sid, message):
        r = self.client.post("/chat", json={"message": message, "session_id": sid})
        assert r.status_code == 200, r.text
        return r.json()

    def book(self, sid, name="Jane Doe", plate="ABC-1234", confirm=True):
        """Run a whole booking up to (and by default including) the user's 'confirm'.

        Returns the reservation id (None when confirm=False: the SM waits in CONFIRM)."""
        from datetime import datetime, timedelta

        self.sessions.append(sid)
        start = (datetime.now() + timedelta(days=2)).replace(hour=10, minute=0, second=0, microsecond=0)
        end = start.replace(hour=14)
        for m in ["I want to book a parking space", name, plate, "1",
                  start.strftime("%Y-%m-%d %H:%M"), end.strftime("%Y-%m-%d %H:%M")]:
            self.chat(sid, m)
        if not confirm:
            return None
        last = self.chat(sid, "confirm")
        assert last["reservation_id"], last
        return last["reservation_id"]

    # -- admin side ---------------------------------------------------------
    def approve(self, rid):
        return self.client.post(f"/admin/reservations/{rid}/approve", headers=self.headers, json={})

    def refuse(self, rid, reason="lot closed for maintenance"):
        return self.client.post(
            f"/admin/reservations/{rid}/refuse", headers=self.headers, json={"reason": reason}
        )

    # -- inspection ---------------------------------------------------------
    def graph(self, sid):
        r = self.client.get(f"/graph/{sid}/state", headers=self.headers)
        assert r.status_code == 200, r.text
        return r.json()

    def row(self, rid):
        from app.db.models import LotReservation

        s = self.Session()
        try:
            r = s.get(LotReservation, rid)
            s.expunge(r)
            return r
        finally:
            s.close()


def _fake_llm(messages):
    """Stands in for the router's LLM: classifies by keyword, answers grounded facts."""
    if messages[0]["content"].startswith("You are a classifier"):
        last = messages[-1]["content"].lower()
        if "book" in last or "reserve" in last:
            intent, sub = "RESERVATION", None
        elif "cost" in last or "price" in last:
            intent, sub = "DYNAMIC_INFO", "prices"
        else:
            intent, sub = "STATIC_INFO", None
        import json
        return json.dumps({"intent": intent, "entities": {"lot_name": None, "sub_intent": sub, "date": None}})
    return "Parking at OrchLot costs $8.00 per hour."


@pytest.fixture
def orch(tmp_path, monkeypatch):
    from datetime import time
    from unittest.mock import AsyncMock, patch

    from fastapi.testclient import TestClient

    from app.api.main import app
    from app.chat import reservation as resv
    from app.db.models import ParkingLot
    from app.db.session import get_db
    from app.orchestration import graph as graph_mod

    engine = create_engine(
        f"sqlite:///{(tmp_path / 'orch.db').as_posix()}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    seed = Session()
    seed.add(ParkingLot(
        name="OrchLot", address="1 Test Way", total_spaces=10, available_spaces=10,
        hourly_price=8.0, daily_price=40.0, opens_at=time(0, 0), closes_at=time(23, 59),
    ))
    seed.commit()
    seed.close()

    monkeypatch.setattr(settings, "admin_api_key", ORCH_ADMIN_KEY)
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(settings, "mcp_export_enabled", True)
    monkeypatch.setattr(settings, "mcp_api_key", "m" * 24)
    graph_mod.reset_graph()

    def _override_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_db
    send = AsyncMock(return_value="ok")        # mcp_client.send_reservation -> OK
    with patch("app.chat.router.call_llm", side_effect=_fake_llm), \
         patch("app.agents.reservation_writer.send_reservation", send):
        h = OrchHarness(TestClient(app), Session, engine, send, tmp_path)
        yield h
        for sid in h.sessions:
            resv._sessions.pop(sid, None)
    app.dependency_overrides.clear()
    graph_mod.reset_graph()       # closes the checkpoint connection (Windows file lock)
    engine.dispose()
