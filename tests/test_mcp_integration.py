"""
Stage-3 integration: admin approval -> MCP server -> confirmed-reservations file.

A REAL MCP server (app/mcp_server) runs in a background thread on an ephemeral
port, the FastAPI app and AdminAgent are the real ones, and the DB is a
file-backed SQLite database in tmp_path.  Nothing is mocked on the happy path.
"""
import asyncio
import logging
import socket
import threading
import time
from datetime import datetime, time as dtime, timedelta
from unittest.mock import AsyncMock

import pytest
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import app.agents.mcp_client as mcp_client
import app.agents.reservation_writer as writer
from app.agents.admin_agent import (
    EXPORT_TOOLS,
    TOOLS,
    AdminAgent,
    MockAdminLLMClient,
    write_reservation,
)
from app.agents.contracts import AdminDecision, Decision
from app.api.main import app
from app.api.routes.admin import get_admin_llm
from app.core.config import settings
from app.agents.channels.tokens import sign
from app.db.models import ApprovalRequest, Base, LotReservation, ParkingLot, ReservationStatus
from app.db.repository import (
    approve_reservation,
    cancel_reservation,
    create_pending_reservation,
    expire_stale_pending,
    refuse_reservation,
)
from app.db.session import ensure_stage3_columns, get_db
from app.mcp_server.server import McpSettings, build_app

TOKEN = "integration-token-" + "z" * 20
ADMIN_KEY = "integration-admin-key"
HDR = {"X-Admin-Key": ADMIN_KEY}
NAME, PLATE = "Ivy Quillfeather", "ZQ77XK99"


# ── helpers ──────────────────────────────────────────────────────────────────


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    """The real MCP server in a thread; can be started late to simulate an outage."""

    def __init__(self, tmp_path, port):
        self.port = port
        self.file = tmp_path / "out" / "confirmed.txt"
        self.server = None
        self.thread = None

    def start(self):
        cfg = McpSettings(
            _env_file=None, mcp_api_key=TOKEN, mcp_port=self.port,
            reservations_file=str(self.file), mcp_rate_limit_per_minute=10_000,
        )
        self.server = uvicorn.Server(
            uvicorn.Config(build_app(cfg), host="127.0.0.1", port=self.port, log_level="error")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            assert time.time() < deadline, "MCP test server did not start"
            time.sleep(0.02)

    def stop(self):
        if self.server:
            self.server.should_exit = True
            self.thread.join(timeout=10)
            self.server = None

    def data_lines(self):
        if not self.file.exists():
            return []
        return [ln for ln in self.file.read_text(encoding="utf-8").splitlines()[1:] if ln]


@pytest.fixture
def port():
    return free_port()


@pytest.fixture
def mcp_env(monkeypatch, port):
    """Point the app's MCP client at `port`; fast timeouts/backoff for tests."""
    monkeypatch.setattr(settings, "mcp_export_enabled", True)
    monkeypatch.setattr(settings, "mcp_api_key", TOKEN)
    monkeypatch.setattr(settings, "mcp_server_url", f"http://127.0.0.1:{port}/mcp")
    monkeypatch.setattr(settings, "mcp_client_timeout", 2.0)
    monkeypatch.setattr(settings, "mcp_client_max_attempts", 2)
    monkeypatch.setattr(settings, "mcp_client_backoff_seconds", 0.01)
    monkeypatch.setattr(settings, "admin_api_key", ADMIN_KEY)


@pytest.fixture
def server(tmp_path, port, mcp_env):
    srv = LiveServer(tmp_path, port)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def down_server(tmp_path, port, mcp_env):
    """Server NOT running yet; call .start() to bring it up. Always cleaned up."""
    srv = LiveServer(tmp_path, port)
    yield srv
    srv.stop()


@pytest.fixture
def Session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    S = sessionmaker(bind=engine, autoflush=False)
    with S() as db:
        db.add(ParkingLot(
            name="IntLot", address="1 Int St", total_spaces=20, available_spaces=20,
            hourly_price=5.0, daily_price=30.0, opens_at=dtime(0, 0), closes_at=dtime(23, 59),
        ))
        db.commit()
    yield S
    engine.dispose()


@pytest.fixture
def db(Session):
    s = Session()
    yield s
    s.close()


@pytest.fixture
def client(Session):
    def _db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _db
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


_counter = [0]


def pending(db, name=NAME, plate=PLATE) -> int:
    _counter[0] += 1
    start = datetime.utcnow() + timedelta(days=2 + _counter[0])
    lot = db.query(ParkingLot).first()
    r = create_pending_reservation(
        db, lot.id, name, plate, start, start + timedelta(hours=3), session_id="s1"
    )
    return r.id


def approved_unwritten(db, **kw) -> int:
    """CONFIRMED by an admin through the repository, no export attempted (= queued)."""
    rid = pending(db, **kw)
    approve_reservation(db, rid, admin="test-admin")
    return rid


def row(Session, rid) -> LotReservation:
    with Session() as s:
        r = s.get(LotReservation, rid)
        s.expunge(r)
        return r


def approve_token(db, rid, monkeypatch) -> str:
    """Signed single-use email-link token (needs its ApprovalRequest anchor row)."""
    monkeypatch.setattr(settings, "email_token_secret", "tok-secret-for-integration")
    audit = ApprovalRequest(reservation_id=rid, status="sent", sent_at=datetime.utcnow(), channel="email")
    db.add(audit)
    db.commit()
    return sign(rid, audit.id, "APPROVE")


def approve_http(client, rid):
    return client.post(f"/admin/reservations/{rid}/approve", headers=HDR, json={})


# ── approve -> line written ──────────────────────────────────────────────────


class TestApproveWritesLine:
    def test_rest_approve_writes_exact_line_and_flags_db(self, client, db, Session, server):
        rid = pending(db, name="Shalini Singh", plate="TS09AB1234")
        r = approve_http(client, rid)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "confirmed" and body["file_written"] is True
        assert "recorded" in body["file_note"]
        line = server.data_lines()[0]
        parts = line.split(" | ")
        assert parts[0] == "Shalini Singh" and parts[1] == "TS09AB1234"
        assert parts[2].count(" - ") == 1 and len(parts) == 4
        saved = row(Session, rid)
        assert saved.file_written is True and saved.file_written_at is not None
        assert saved.status == ReservationStatus.CONFIRMED

    def test_admin_chat_approve_writes_line_and_reports_note(self, client, db, server):
        rid = pending(db)
        app.dependency_overrides[get_admin_llm] = lambda: MockAdminLLMClient(["Your booking is confirmed."])
        try:
            r = client.post("/admin/chat", headers=HDR, json={"message": f"approve {rid}"})
        finally:
            app.dependency_overrides.pop(get_admin_llm, None)
        assert r.status_code == 200
        assert r.json()["decision_made"] is True and r.json()["file_written"] is True
        assert len(server.data_lines()) == 1

    def test_email_link_approve_writes_line(self, client, db, server, monkeypatch):
        rid = pending(db)
        token = approve_token(db, rid, monkeypatch)
        r = client.get("/admin/approve", params={"token": token})
        assert r.status_code == 200, r.text
        assert r.json()["file_written"] is True
        assert len(server.data_lines()) == 1


# ── refuse / ineligible -> nothing written ───────────────────────────────────


class TestNothingWrittenWhenNotApproved:
    def test_refuse_writes_nothing(self, client, db, Session, server):
        rid = pending(db)
        r = client.post(f"/admin/reservations/{rid}/refuse", headers=HDR, json={"reason": "closed"})
        assert r.status_code == 200 and r.json()["file_written"] is None
        assert server.data_lines() == []
        assert not server.file.exists()
        assert row(Session, rid).file_written is False

    def test_pending_reservation_is_not_written_by_worker(self, db, Session, server):
        pending(db)
        summary = writer.retry_pending_writes(Session)
        assert summary == {"attempted": 0, "written": 0, "failed": 0}
        assert not server.file.exists()

    def test_refused_cancelled_expired_legacy_never_written_by_worker(self, db, Session, server):
        refused = pending(db)
        refuse_reservation(db, refused, admin="a", reason="no")
        cancelled = approved_unwritten(db)
        cancel_reservation(db, cancelled)
        expired = pending(db)
        db.query(LotReservation).filter_by(id=expired).update(
            {"created_at": datetime.utcnow() - timedelta(hours=48)}
        )
        db.commit()
        expire_stale_pending(db, hours=24)
        legacy = pending(db)  # Stage-1 style: CONFIRMED but never decided by an admin
        db.query(LotReservation).filter_by(id=legacy).update(
            {"status": ReservationStatus.CONFIRMED}
        )
        db.commit()

        summary = writer.retry_pending_writes(Session)
        assert summary["attempted"] == 0
        assert not server.file.exists()
        statuses = {rid: row(Session, rid).status for rid in (refused, cancelled, expired, legacy)}
        assert statuses[refused] == ReservationStatus.REFUSED
        assert statuses[cancelled] == ReservationStatus.CANCELLED
        assert statuses[expired] == ReservationStatus.EXPIRED


# ── MCP server down -> CONFIRMED, queued, retried ────────────────────────────


class TestServerDownThenRetry:
    def test_approval_survives_outage_and_is_queued(self, client, db, Session, down_server, caplog):
        rid = pending(db, name=NAME, plate=PLATE)
        with caplog.at_level(logging.DEBUG):
            r = approve_http(client, rid)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "confirmed"           # not rolled back
        assert body["file_written"] is False
        assert "pending" in body["file_note"] and "retried" in body["file_note"]
        saved = row(Session, rid)
        assert saved.status == ReservationStatus.CONFIRMED
        assert saved.file_written is False and saved.file_written_at is None
        assert writer.pending_ids(db) == [rid]
        assert not down_server.file.exists()
        # failure was logged, without PII
        assert f"reservation_id={rid}" in caplog.text
        for secret in ("Quillfeather", "Ivy", PLATE):
            assert secret not in caplog.text

    def test_retry_succeeds_after_server_comes_back(self, client, db, Session, down_server):
        rid = pending(db)
        assert approve_http(client, rid).json()["file_written"] is False

        failed = writer.retry_pending_writes(Session)
        assert failed == {"attempted": 1, "written": 0, "failed": 1}
        assert row(Session, rid).file_written is False

        down_server.start()
        ok = writer.retry_pending_writes(Session)
        assert ok == {"attempted": 1, "written": 1, "failed": 0}
        saved = row(Session, rid)
        assert saved.file_written is True and saved.file_written_at is not None
        assert len(down_server.data_lines()) == 1
        assert writer.retry_pending_writes(Session)["attempted"] == 0  # queue drained

    def test_worker_stops_after_first_failure_while_down(self, db, Session, down_server):
        for _ in range(3):
            approved_unwritten(db)
        assert writer.retry_pending_writes(Session) == {"attempted": 1, "written": 0, "failed": 1}

    def test_worker_drains_whole_backlog_when_up(self, db, Session, server):
        ids = [approved_unwritten(db, name=f"Cust {i}") for i in range(4)]
        assert writer.retry_pending_writes(Session)["written"] == 4
        assert len(server.data_lines()) == 4
        assert all(row(Session, i).file_written for i in ids)

    def test_background_worker_in_api_lifespan_writes_queued_rows(
        self, db, Session, server, monkeypatch
    ):
        rid = approved_unwritten(db)
        import app.api.main as main
        monkeypatch.setattr(main, "create_tables", lambda: None)  # never touch parking.db
        monkeypatch.setattr(main, "SessionLocal", Session)
        monkeypatch.setattr(settings, "mcp_retry_interval_seconds", 0.1)
        with TestClient(app):
            deadline = time.time() + 15
            while time.time() < deadline and not row(Session, rid).file_written:
                time.sleep(0.1)
        assert row(Session, rid).file_written is True
        assert len(server.data_lines()) == 1

    def test_background_worker_not_started_when_unconfigured(self, monkeypatch):
        import app.api.main as main
        monkeypatch.setattr(main, "create_tables", lambda: None)
        monkeypatch.setattr(settings, "mcp_api_key", "")
        started = []
        monkeypatch.setattr(main, "_retry_worker", lambda: started.append(1))
        with TestClient(app):
            pass
        assert started == []


# ── double approval / idempotency ────────────────────────────────────────────


class TestNoDuplicates:
    def test_double_approve_second_is_409_and_one_line(self, client, db, server):
        rid = pending(db)
        assert approve_http(client, rid).status_code == 200
        assert approve_http(client, rid).status_code == 409
        assert len(server.data_lines()) == 1

    def test_double_email_link_and_rest_still_one_line(self, client, db, server, monkeypatch):
        rid = pending(db)
        assert approve_http(client, rid).status_code == 200
        token = approve_token(db, rid, monkeypatch)
        assert client.get("/admin/approve", params={"token": token}).status_code == 409
        assert len(server.data_lines()) == 1

    def test_flag_reset_then_retry_hits_server_duplicate_not_second_line(self, client, db, Session, server):
        rid = pending(db)
        approve_http(client, rid)
        with Session() as s:                      # simulate "write ok but crash before marking"
            s.query(LotReservation).filter_by(id=rid).update({"file_written": False})
            s.commit()
        assert writer.retry_pending_writes(Session)["written"] == 1
        assert row(Session, rid).file_written is True
        assert len(server.data_lines()) == 1

    def test_export_approved_called_twice_writes_once(self, client, db, server):
        rid = pending(db)
        approve_http(client, rid)
        assert writer.export_approved(db, rid) is True
        assert writer.export_approved(db, rid) is True
        assert len(server.data_lines()) == 1


# ── LLM / any caller without admin approval is blocked ───────────────────────


class TestBlockedWithoutApproval:
    @pytest.fixture
    def spy(self, monkeypatch):
        m = AsyncMock(return_value="ok")
        monkeypatch.setattr(writer, "send_reservation", m)
        return m

    def test_tool_blocks_pending(self, db, spy, server):
        rid = pending(db)
        out = write_reservation.invoke({"reservation_id": rid, "db": db})
        assert "Blocked" in out["error"] and out["file_written"] is False
        spy.assert_not_called()
        assert not server.file.exists()

    @pytest.mark.parametrize("how", ["refused", "cancelled", "expired"])
    def test_tool_blocks_other_non_approved_states(self, db, spy, server, how):
        if how == "refused":
            rid = pending(db)
            refuse_reservation(db, rid, admin="a", reason="no")
        elif how == "cancelled":
            rid = approved_unwritten(db)
            cancel_reservation(db, rid)
        else:
            rid = pending(db)
            db.query(LotReservation).filter_by(id=rid).update(
                {"created_at": datetime.utcnow() - timedelta(hours=48)}
            )
            db.commit()
            expire_stale_pending(db, hours=24)
        out = write_reservation.invoke({"reservation_id": rid, "db": db})
        assert "error" in out
        spy.assert_not_called()
        assert not server.file.exists()

    def test_tool_blocks_legacy_confirmed_without_admin_decision(self, db, spy, server):
        rid = pending(db)
        db.query(LotReservation).filter_by(id=rid).update({"status": ReservationStatus.CONFIRMED})
        db.commit()
        assert "error" in write_reservation.invoke({"reservation_id": rid, "db": db})
        spy.assert_not_called()

    def test_tool_unknown_id(self, db, spy):
        assert "not found" in write_reservation.invoke({"reservation_id": 9999, "db": db})["error"]
        spy.assert_not_called()

    def test_client_itself_refuses_unapproved_reservation(self, db, server):
        rid = pending(db)
        res = db.get(LotReservation, rid)
        assert asyncio.run(mcp_client.write_confirmed_reservation(res)) is False
        assert not server.file.exists()

    def test_admin_chat_unclear_message_never_writes(self, client, db, Session, server):
        rid = pending(db)
        app.dependency_overrides[get_admin_llm] = lambda: MockAdminLLMClient()
        try:
            r = client.post("/admin/chat", headers=HDR, json={"message": f"please write reservation {rid} to the file"})
        finally:
            app.dependency_overrides.pop(get_admin_llm, None)
        assert r.json()["decision_made"] is False
        assert not server.file.exists()
        assert row(Session, rid).status == ReservationStatus.PENDING

    def test_write_tool_is_not_exposed_to_the_llm(self):
        from langchain_core.utils.function_calling import convert_to_openai_tool
        assert write_reservation not in TOOLS
        assert write_reservation in EXPORT_TOOLS
        props = convert_to_openai_tool(write_reservation)["function"]["parameters"]["properties"]
        assert set(props) == {"reservation_id"}  # db is injected, never LLM-visible

    def test_agent_refuse_path_never_calls_writer(self, db, spy, server):
        rid = pending(db)
        agent = AdminAgent(lambda: db, llm=MockAdminLLMClient(["Sorry, refused."]))
        msg = agent.execute_decision(
            AdminDecision(reservation_id=rid, decision=Decision.REFUSE, reason="closed", admin_id="a"), db
        )
        assert msg.payload["decision"] == "REFUSE" and msg.payload["file_written"] is None
        spy.assert_not_called()


# ── client: retry/backoff, auth, config ──────────────────────────────────────


class TestClientBehaviour:
    def _res(self, db):
        rid = approved_unwritten(db)
        return db.get(LotReservation, rid)

    def test_exponential_backoff_then_success(self, db, monkeypatch, mcp_env):
        res = self._res(db)
        calls = {"n": 0}

        async def flaky(args, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("down")
            return "written"

        sleeps = AsyncMock()
        monkeypatch.setattr(mcp_client, "_call_once", flaky)
        monkeypatch.setattr(mcp_client.asyncio, "sleep", sleeps)
        monkeypatch.setattr(settings, "mcp_client_backoff_seconds", 0.5)
        assert asyncio.run(mcp_client.write_confirmed_reservation(res, max_attempts=3)) is True
        assert calls["n"] == 3
        assert [c.args[0] for c in sleeps.call_args_list] == [0.5, 1.0]

    def test_gives_up_after_max_attempts(self, db, monkeypatch, mcp_env):
        res = self._res(db)
        boom = AsyncMock(side_effect=ConnectionError("down"))
        monkeypatch.setattr(mcp_client, "_call_once", boom)
        assert asyncio.run(mcp_client.write_confirmed_reservation(res, max_attempts=3)) is False
        assert boom.await_count == 3

    def test_permanent_rejection_is_not_retried(self, db, monkeypatch, mcp_env):
        res = self._res(db)
        boom = AsyncMock(side_effect=mcp_client.PermanentMcpError("bad"))
        monkeypatch.setattr(mcp_client, "_call_once", boom)
        assert asyncio.run(mcp_client.write_confirmed_reservation(res, max_attempts=3)) is False
        assert boom.await_count == 1

    def test_duplicate_status_counts_as_success(self, db, monkeypatch, mcp_env):
        res = self._res(db)
        monkeypatch.setattr(mcp_client, "_call_once", AsyncMock(return_value="duplicate"))
        assert asyncio.run(mcp_client.write_confirmed_reservation(res)) is True

    def test_wrong_token_returns_false_and_writes_nothing(self, db, server, monkeypatch):
        res = self._res(db)
        monkeypatch.setattr(settings, "mcp_api_key", "not-the-token")
        assert asyncio.run(mcp_client.write_confirmed_reservation(res, max_attempts=1)) is False
        assert not server.file.exists()

    def test_wrong_token_is_401_and_not_retried(self, db, server, monkeypatch, caplog):
        res = self._res(db)
        monkeypatch.setattr(settings, "mcp_api_key", "not-the-token")
        with caplog.at_level(logging.WARNING):
            assert asyncio.run(mcp_client.write_confirmed_reservation(res, max_attempts=3)) is False
        assert caplog.text.count("attempt_failed") == 1  # permanent: no further attempts
        assert "HTTPStatusError" in caplog.text

    def test_not_configured_returns_false_without_network(self, db, monkeypatch, mcp_env):
        res = self._res(db)
        monkeypatch.setattr(settings, "mcp_api_key", "")
        spy = AsyncMock()
        monkeypatch.setattr(mcp_client, "_call_once", spy)
        assert asyncio.run(mcp_client.write_confirmed_reservation(res)) is False
        spy.assert_not_called()

    def test_unsafe_customer_name_is_sanitised_so_the_approval_is_still_recorded(self, db, server):
        rid = approved_unwritten(db, name="Bad|Name")  # '|' would forge a column
        res = db.get(LotReservation, rid)
        assert asyncio.run(mcp_client.write_confirmed_reservation(res, max_attempts=1)) is True
        assert server.data_lines()[0].startswith("Bad/Name | ")

    def test_conflicting_content_for_same_id_is_permanent_and_not_retried(self, db, server, caplog):
        rid = approved_unwritten(db, name="Real Person")
        # someone/something already recorded a DIFFERENT booking under this id
        from app.mcp_server.server import ReservationRecord, ReservationStore
        ReservationStore(server.file).write(ReservationRecord(
            reservation_id=rid, customer_name="Other Person", car_plate="ZZ-999",
            start_time="2026-01-01T10:00:00", end_time="2026-01-01T12:00:00",
            approved_at="2026-01-01T09:00:00"))
        res = db.get(LotReservation, rid)
        with caplog.at_level(logging.WARNING):
            outcome = asyncio.run(mcp_client.send_reservation(res, max_attempts=3))
        assert outcome == mcp_client.PERMANENT
        assert "attempt_failed" not in caplog.text
        assert len(server.data_lines()) == 1 and "Real Person" not in server.file.read_text(encoding="utf-8")

    def test_server_validation_error_is_permanent_and_not_retried(self, db, server, monkeypatch, caplog):
        res = self._res(db)
        bad = {**mcp_client._payload(res), "car_plate": "NOT A PLATE"}  # server-side validation fails
        monkeypatch.setattr(mcp_client, "_payload", lambda _r: bad)
        with caplog.at_level(logging.WARNING):
            outcome = asyncio.run(mcp_client.send_reservation(res, max_attempts=3))
        assert outcome == mcp_client.PERMANENT
        assert "rejected" in caplog.text and "attempt_failed" not in caplog.text
        assert not server.file.exists()

    def test_unexpected_status_is_failure(self, db, monkeypatch, mcp_env):
        res = self._res(db)
        monkeypatch.setattr(mcp_client, "_call_once", AsyncMock(return_value="weird"))
        assert asyncio.run(mcp_client.write_confirmed_reservation(res)) is False

    def test_cancellation_propagates_instead_of_being_swallowed(self, db, monkeypatch, mcp_env):
        res = self._res(db)
        monkeypatch.setattr(mcp_client, "_call_once", AsyncMock(side_effect=asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(mcp_client.write_confirmed_reservation(res))

    def test_run_sync_works_inside_running_loop(self):
        async def inner():
            return mcp_client.run_sync(asyncio.sleep(0, result=7))
        assert asyncio.run(inner()) == 7


# ── DB migration ─────────────────────────────────────────────────────────────


class TestMigration:
    def test_adds_columns_to_pre_stage3_table_and_is_idempotent(self, tmp_path):
        eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
        try:
            with eng.begin() as c:
                c.execute(text("CREATE TABLE lot_reservations (id INTEGER PRIMARY KEY, customer_name TEXT)"))
                c.execute(text("INSERT INTO lot_reservations (id, customer_name) VALUES (1, 'x')"))
            assert ensure_stage3_columns(eng) == ["file_written", "file_written_at"]
            assert ensure_stage3_columns(eng) == []
            with eng.connect() as c:
                found = c.execute(text("SELECT file_written, file_written_at FROM lot_reservations")).one()
            assert tuple(found) == (0, None)
        finally:
            eng.dispose()

    def test_noop_when_table_missing(self, tmp_path):
        eng = create_engine(f"sqlite:///{tmp_path / 'e.db'}")
        try:
            assert ensure_stage3_columns(eng) == []
        finally:
            eng.dispose()

    def test_new_rows_default_to_not_written(self, db):
        rid = pending(db)
        res = db.get(LotReservation, rid)
        assert res.file_written is False and res.file_written_at is None
