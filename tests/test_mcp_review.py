"""
Regression tests for the Stage-3 security/reliability review.

One class per finding; each test fails on the pre-review code:

  auth        websocket scope, path variants, weak keys, memory growth
  injection   spreadsheet-formula names, client-side sanitising, crash-truncated line
  idempotency same id + different content, rotated/deleted output file
  races       real multi-process writers
  queue       one permanently rejected row must not block the rows behind it
  secrets     admin identity no longer embeds ADMIN_API_KEY characters; no values in logs
"""
import asyncio
import hashlib
import logging
import subprocess
import sys
import textwrap
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.agents.mcp_client as mcp_client
import app.agents.reservation_writer as writer
import app.mcp_server.server as server_mod
from app.api.main import app as fastapi_app
from app.core.config import settings
from app.db.models import Base, LotReservation, ParkingLot
from app.db.repository import approve_reservation, create_pending_reservation
from app.db.session import get_db
from app.mcp_server.server import (
    HEADER,
    McpSettings,
    ReservationRecord,
    ReservationStore,
    SecurityMiddleware,
    build_app,
)
from starlette.testclient import TestClient as StarletteClient

ROOT = Path(__file__).resolve().parents[1]
KEY = "review-key-" + "r" * 24


def rec(**o):
    base = dict(
        reservation_id=1, customer_name="Ann Lee", car_plate="AB-123",
        start_time="2026-11-04T10:00:00", end_time="2026-11-04T12:00:00",
        approved_at="2026-11-03T09:00:00",
    )
    base.update(o)
    return ReservationRecord(**base)


@pytest.fixture
def store(tmp_path):
    return ReservationStore(tmp_path / "out" / "c.txt")


# ── auth ─────────────────────────────────────────────────────────────────────


class TestAuthHardening:
    def test_websocket_is_closed_and_never_forwarded(self):
        inner = AsyncMock()
        mw = SecurityMiddleware(inner, KEY, 100)
        sent = []

        async def send(m):
            sent.append(m)

        asyncio.run(mw({"type": "websocket", "path": "/mcp", "headers": []}, None, send))
        inner.assert_not_awaited()
        assert sent == [{"type": "websocket.close", "code": 1008}]

    def test_unknown_scope_type_is_dropped(self):
        inner = AsyncMock()
        asyncio.run(SecurityMiddleware(inner, KEY, 100)({"type": "weird"}, None, AsyncMock()))
        inner.assert_not_awaited()

    @pytest.mark.parametrize("key", ["", "   ", "short", "x" * 15])
    def test_weak_or_blank_key_refuses_to_start(self, key):
        with pytest.raises(RuntimeError, match="MCP_API_KEY"):
            SecurityMiddleware(object(), key, 10)

    def test_sixteen_char_key_is_accepted_and_whitespace_stripped(self):
        mw = SecurityMiddleware(object(), "  " + "k" * 16 + "  ", 10)
        assert mw._authorized({"headers": [(b"authorization", b"Bearer " + b"k" * 16)]})

    @pytest.mark.parametrize(
        "path", ["/health/", "//mcp", "/MCP", "/mcp/", "/mcp/x", "/%6dcp", "/health/../mcp", "/admin", "/"]
    )
    def test_no_path_variant_bypasses_the_token(self, tmp_path, path):
        app_ = build_app(McpSettings(_env_file=None, mcp_api_key=KEY,
                                     reservations_file=str(tmp_path / "f.txt"),
                                     mcp_rate_limit_per_minute=1000))
        with StarletteClient(app_) as c:
            for method in ("get", "post", "put", "delete"):
                r = getattr(c, method)(path)
                assert r.status_code == 401, (method, path, r.status_code)

    def test_only_exact_health_is_public(self, tmp_path):
        app_ = build_app(McpSettings(_env_file=None, mcp_api_key=KEY,
                                     reservations_file=str(tmp_path / "f.txt")))
        with StarletteClient(app_) as c:
            assert c.get("/health").status_code == 200

    def test_rate_limiter_memory_is_bounded(self, monkeypatch):
        mw = SecurityMiddleware(object(), KEY, 5)
        t = [0.0]
        monkeypatch.setattr(server_mod.time, "monotonic", lambda: t[0])
        for i in range(10_050):
            mw._rate_limited(f"10.0.{i // 256}.{i % 256}")
        assert len(mw._hits) > 10_000
        t[0] += 120  # every window expired
        mw._rate_limited("fresh")
        assert len(mw._hits) < 10

    def test_oversized_request_body_is_rejected(self, tmp_path):
        app_ = build_app(McpSettings(_env_file=None, mcp_api_key=KEY,
                                     reservations_file=str(tmp_path / "f.txt")))
        with StarletteClient(app_) as c:
            r = c.post("/mcp", headers={
                "Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"},
                content=b'{"jsonrpc":"2.0","id":1,"method":"x","params":{"p":"' + b"A" * 5_000_000 + b'"}}')
        assert r.status_code == 413
        assert not (tmp_path / "f.txt").exists()

    def test_non_loopback_bind_logs_a_warning(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setenv("MCP_API_KEY", KEY)
        monkeypatch.setenv("MCP_HOST", "0.0.0.0")
        monkeypatch.setenv("RESERVATIONS_FILE", str(tmp_path / "o.txt"))
        monkeypatch.setattr("uvicorn.run", MagicMock())
        with caplog.at_level(logging.WARNING):
            server_mod.main()
        assert "non_loopback_bind" in caplog.text

    def test_loopback_bind_logs_no_warning(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setenv("MCP_API_KEY", KEY)
        monkeypatch.setenv("MCP_HOST", "127.0.0.1")
        monkeypatch.setenv("RESERVATIONS_FILE", str(tmp_path / "o.txt"))
        monkeypatch.setattr("uvicorn.run", MagicMock())
        with caplog.at_level(logging.WARNING):
            server_mod.main()
        assert "non_loopback_bind" not in caplog.text


# ── injection ────────────────────────────────────────────────────────────────


class TestInjection:
    @pytest.mark.parametrize("name", ["=HYPERLINK(1)", "+1+1", "-2+3", "@SUM(A1)", "  =x"])
    def test_formula_leading_names_rejected_by_server(self, name):
        with pytest.raises(ValidationError, match="must not start"):
            rec(customer_name=name)

    @pytest.mark.parametrize("name", ["Anne-Marie", "O'Brien", "Zoë Müller", "A. B. Smith", "X=Y"])
    def test_legitimate_names_still_accepted(self, name):
        assert rec(customer_name=name).customer_name == name

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Shalini Singh", "Shalini Singh"),
            ("Eve | Mallory", "Eve / Mallory"),
            ("=HYPERLINK(1)", "HYPERLINK(1)"),
            ("a\r\nb\x00c", "a bc"),
            ("  -x  ", "x"),
            ("|||", "///"),
            ("\x00\x01\x1b", "Unknown"),
            ("", "Unknown"),
            ("A" * 300, "A" * 100),
            ("Eve Mallory", "Eve Mallory"),
        ],
    )
    def test_client_sanitises_instead_of_dropping_the_booking(self, raw, expected):
        assert mcp_client.sanitize_name(raw) == expected

    @pytest.mark.parametrize("raw", ["a|b", "x\ny", "=1+1", "\x00", "q‮"])
    def test_sanitised_output_always_passes_server_validation(self, raw):
        assert rec(customer_name=mcp_client.sanitize_name(raw))

    def test_crash_truncated_line_is_repaired_not_glued(self, store):
        store.write(rec(reservation_id=1))
        with open(store.path, "a", encoding="utf-8", newline="") as fh:
            fh.write("PARTIAL LINE WITHOUT NEWLINE")
        assert store.write(rec(reservation_id=2, customer_name="Bob")) == "written"
        lines = store.path.read_text(encoding="utf-8").split("\n")
        assert "PARTIAL LINE WITHOUT NEWLINE" in lines  # kept on its own line
        assert any(ln.startswith("Bob | ") for ln in lines)  # not glued to the garbage

    def test_clean_file_gets_no_extra_blank_lines(self, store):
        store.write(rec(reservation_id=1))
        store.write(rec(reservation_id=2))
        assert "\n\n" not in store.path.read_text(encoding="utf-8")


# ── idempotency ──────────────────────────────────────────────────────────────


class TestIdempotencyIsContentAware:
    def test_same_id_same_content_is_duplicate(self, store):
        assert store.write(rec()) == "written"
        assert store.write(rec()) == "duplicate"

    def test_same_id_different_content_is_conflict_not_silent_success(self, store):
        store.write(rec(customer_name="First Person"))
        assert store.write(rec(customer_name="Other Person")) == "conflict"
        text = store.path.read_text(encoding="utf-8")
        assert "First Person" in text and "Other Person" not in text

    def test_conflict_is_visible_through_the_tool(self, tmp_path):
        mcp = server_mod.create_mcp(ReservationStore(tmp_path / "t.txt"), McpSettings(_env_file=None, mcp_api_key=KEY))
        fn = mcp._tool_manager.get_tool("write_reservation").fn
        args = ("A B", "AB-123", "2026-11-04T10:00:00", "2026-11-04T12:00:00", "2026-11-03T09:00:00")
        assert fn(1, *args)["status"] == "written"
        assert fn(1, "C D", *args[1:])["status"] == "conflict"

    def test_legacy_id_without_digest_is_treated_as_duplicate(self, store):
        store.write(rec())
        store.ids_path.write_text("1\n", encoding="utf-8")  # older format: id only
        assert store.write(rec(customer_name="Whoever")) == "duplicate"

    def test_deleted_output_file_resets_ids_so_nothing_is_silently_lost(self, store):
        store.write(rec(reservation_id=1))
        store.path.unlink()  # operator rotates / deletes the log
        assert store.write(rec(reservation_id=1)) == "written"
        assert store.path.read_text(encoding="utf-8").startswith(HEADER)
        assert len([ln for ln in store.path.read_text(encoding="utf-8").splitlines() if ln]) == 2

    def test_emptied_output_file_also_resets_ids(self, store):
        store.write(rec(reservation_id=1))
        store.path.write_text("", encoding="utf-8")
        assert store.write(rec(reservation_id=1)) == "written"
        assert store.path.read_text(encoding="utf-8").count(HEADER) == 1

    def test_existing_output_file_without_ids_sidecar_is_appended_to(self, store):
        store.path.parent.mkdir(parents=True, exist_ok=True)
        store.path.write_text(HEADER + "\nOld Person | OL-111 | 2026-01-01 10:00 - 11:00 | 2026-01-01 09:00:00\n", encoding="utf-8")
        assert store.write(rec(reservation_id=7)) == "written"
        lines = [ln for ln in store.path.read_text(encoding="utf-8").splitlines() if ln]
        assert len(lines) == 3 and lines[1].startswith("Old Person")
        assert store.write(rec(reservation_id=7)) == "duplicate"

    def test_ids_file_does_not_contain_plain_names(self, store):
        store.write(rec(customer_name="Zelda Quux"))
        assert "Zelda" not in store.ids_path.read_text(encoding="utf-8")


# ── real multi-process races ─────────────────────────────────────────────────

_WORKER = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    from app.mcp_server.server import ReservationRecord, ReservationStore
    store, proc = ReservationStore(sys.argv[1]), int(sys.argv[2])

    def rec(i):
        return ReservationRecord(reservation_id=i, customer_name=f"Person {{i}}", car_plate="AB-123",
                                 start_time="2026-11-04T10:00:00", end_time="2026-11-04T12:00:00",
                                 approved_at="2026-11-03T09:00:00")

    for i in range(1, 16):                                   # SAME ids in every process
        store.write(rec(i))
    for i in range(1000 + proc * 100, 1000 + proc * 100 + 10):   # unique per process
        store.write(rec(i))
    """
)


class TestMultiProcessRaces:
    def test_four_real_processes_no_lost_duplicate_or_interleaved_lines(self, tmp_path):
        target = tmp_path / "mp" / "c.txt"
        script = tmp_path / "worker.py"
        script.write_text(_WORKER.format(root=str(ROOT)), encoding="utf-8")
        procs = [
            subprocess.Popen([sys.executable, str(script), str(target), str(n)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(tmp_path))
            for n in range(4)
        ]
        for p in procs:
            out, err = p.communicate(timeout=120)
            assert p.returncode == 0, err.decode()[-600:]
        lines = target.read_text(encoding="utf-8").splitlines()
        assert lines[0] == HEADER and lines.count(HEADER) == 1
        body = lines[1:]
        assert len(body) == 15 + 4 * 10  # shared ids once each + all unique ids, no more, no less
        assert len(set(body)) == len(body)  # no duplicate lines
        for ln in body:
            parts = ln.split(" | ")
            assert len(parts) == 4 and parts[0].startswith("Person ") and parts[1] == "AB-123"
        ids = [int(ln.split()[0]) for ln in target.with_name("c.txt.ids").read_text().splitlines()]
        assert len(ids) == len(set(ids)) == 15 + 40


# ── retry queue: head-of-line blocking ───────────────────────────────────────


@pytest.fixture
def Session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'rv.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    S = sessionmaker(bind=engine, autoflush=False)
    with S() as s:
        s.add(ParkingLot(name="RvLot", address="1 Rv St", total_spaces=10, available_spaces=10,
                         hourly_price=1.0, daily_price=5.0, opens_at=dtime(0, 0), closes_at=dtime(23, 59)))
        s.commit()
    yield S
    engine.dispose()


@pytest.fixture
def db(Session):
    s = Session()
    yield s
    s.close()


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(settings, "mcp_export_enabled", True)
    monkeypatch.setattr(settings, "mcp_api_key", KEY)


_n = [0]


def approved(db, name="Rv User") -> int:
    _n[0] += 1
    start = datetime.utcnow() + timedelta(days=30 + _n[0])
    r = create_pending_reservation(db, db.query(ParkingLot).first().id, name, "RV-0001", start, start + timedelta(hours=2))
    approve_reservation(db, r.id, admin="rv-admin")
    return r.id


class TestQueueNeverBlocks:
    def test_permanently_rejected_row_does_not_block_later_rows(self, db, Session, configured, monkeypatch):
        bad, good = approved(db), approved(db)
        monkeypatch.setattr(writer, "send_reservation", AsyncMock(side_effect=["permanent", "ok"]))
        summary = writer.retry_pending_writes(Session)
        assert summary == {"attempted": 2, "written": 1, "failed": 1}
        db.expire_all()
        assert db.get(LotReservation, bad).file_written is False
        assert db.get(LotReservation, good).file_written is True

    def test_transient_failure_still_stops_the_sweep(self, db, Session, configured, monkeypatch):
        approved(db), approved(db)
        spy = AsyncMock(side_effect=["transient", "ok"])
        monkeypatch.setattr(writer, "send_reservation", spy)
        assert writer.retry_pending_writes(Session) == {"attempted": 1, "written": 0, "failed": 1}
        assert spy.await_count == 1

    def test_rejected_row_is_retried_on_the_next_sweep(self, db, Session, configured, monkeypatch):
        rid = approved(db)
        monkeypatch.setattr(writer, "send_reservation", AsyncMock(return_value="permanent"))
        writer.retry_pending_writes(Session)
        assert writer.pending_ids(db) == [rid]  # still queued, not silently dropped or marked written


class TestSendOutcomes:
    def test_blocked_for_pending(self, db, configured):
        _n[0] += 1
        start = datetime.utcnow() + timedelta(days=60 + _n[0])
        r = create_pending_reservation(db, db.query(ParkingLot).first().id, "P", "PD-0001", start, start + timedelta(hours=1))
        assert asyncio.run(mcp_client.send_reservation(db.get(LotReservation, r.id))) == mcp_client.BLOCKED

    def test_disabled_when_unconfigured(self, db, monkeypatch):
        monkeypatch.setattr(settings, "mcp_api_key", "")
        rid = approved(db)
        assert asyncio.run(mcp_client.send_reservation(db.get(LotReservation, rid))) == mcp_client.DISABLED

    def test_transient_after_exhausting_attempts(self, db, configured, monkeypatch):
        rid = approved(db)
        monkeypatch.setattr(settings, "mcp_client_backoff_seconds", 0.0)
        monkeypatch.setattr(mcp_client, "_call_once", AsyncMock(side_effect=ConnectionError("x")))
        assert asyncio.run(mcp_client.send_reservation(db.get(LotReservation, rid), max_attempts=2)) == mcp_client.TRANSIENT

    @pytest.mark.parametrize("status", ["conflict", "weird", ""])
    def test_non_success_status_is_permanent(self, db, configured, monkeypatch, status):
        rid = approved(db)
        monkeypatch.setattr(mcp_client, "_call_once", AsyncMock(return_value=status))
        assert asyncio.run(mcp_client.send_reservation(db.get(LotReservation, rid))) == mcp_client.PERMANENT

    def test_bool_wrapper_matches_outcome(self, db, configured, monkeypatch):
        rid = approved(db)
        monkeypatch.setattr(mcp_client, "_call_once", AsyncMock(return_value="written"))
        assert asyncio.run(mcp_client.write_confirmed_reservation(db.get(LotReservation, rid))) is True


class TestInsecureUrlWarning:
    @pytest.mark.parametrize("url", ["http://10.1.2.3:8765/mcp", "http://example.com/mcp"])
    def test_cleartext_http_to_remote_host_warns(self, monkeypatch, caplog, url):
        monkeypatch.setattr(settings, "mcp_server_url", url)
        with caplog.at_level(logging.WARNING):
            mcp_client.warn_if_insecure_url()
        assert "insecure_url" in caplog.text

    @pytest.mark.parametrize(
        "url", ["http://127.0.0.1:8765/mcp", "http://localhost:8765/mcp", "http://mcp-server:8765/mcp", "https://mcp.example.com/mcp"]
    )
    def test_loopback_internal_and_https_do_not_warn(self, monkeypatch, caplog, url):
        monkeypatch.setattr(settings, "mcp_server_url", url)
        with caplog.at_level(logging.WARNING):
            mcp_client.warn_if_insecure_url()
        assert "insecure_url" not in caplog.text


# ── secrets ──────────────────────────────────────────────────────────────────


class TestSecrets:
    ADMIN_KEY = "citypark-secret-admin-key-123"

    @pytest.fixture
    def client(self, Session, monkeypatch):
        monkeypatch.setattr(settings, "admin_api_key", self.ADMIN_KEY)

        def _db():
            s = Session()
            try:
                yield s
            finally:
                s.close()

        fastapi_app.dependency_overrides[get_db] = _db
        yield TestClient(fastapi_app, raise_server_exceptions=False)
        fastapi_app.dependency_overrides.clear()

    def test_decided_by_is_a_hash_not_a_slice_of_the_admin_key(self, client, db):
        rid = approved_pending(db)
        r = client.post(f"/admin/reservations/{rid}/approve", headers={"X-Admin-Key": self.ADMIN_KEY}, json={})
        assert r.status_code == 200
        who = r.json()["decided_by"]
        assert who == "api:" + hashlib.sha256(self.ADMIN_KEY.encode()).hexdigest()[:8]
        assert self.ADMIN_KEY[:8] not in who
        db.expire_all()
        assert self.ADMIN_KEY[:8] not in (db.get(LotReservation, rid).decided_by or "")

    def test_refuse_also_uses_the_hashed_identity(self, client, db):
        rid = approved_pending(db)
        r = client.post(f"/admin/reservations/{rid}/refuse", headers={"X-Admin-Key": self.ADMIN_KEY}, json={"reason": "closed"})
        assert r.json()["decided_by"].startswith("api:") and self.ADMIN_KEY[:8] not in r.json()["decided_by"]

    def test_bad_argument_values_are_not_written_to_logs(self, tmp_path, caplog):
        app_ = build_app(McpSettings(_env_file=None, mcp_api_key=KEY, reservations_file=str(tmp_path / "f.txt")))
        h = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
        with caplog.at_level(logging.DEBUG), StarletteClient(app_) as c:
            for args in (
                dict(reservation_id="SECRET-ID-xyz", customer_name="A B", car_plate="AB-123"),
                dict(reservation_id=1, customer_name=987654321, car_plate="AB-123"),
                dict(reservation_id=1, customer_name="Pat Q|Private", car_plate="PLATE-SECRET"),
            ):
                c.post("/mcp", headers=h, json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "write_reservation", "arguments": {
                        "start_time": "2026-11-04T10:00:00", "end_time": "2026-11-04T12:00:00",
                        "approved_at": "2026-11-03T09:00:00", **args}}})
        for secret in ("SECRET-ID-xyz", "987654321", "Private", "PLATE-SECRET"):
            assert secret not in caplog.text


def approved_pending(db) -> int:
    _n[0] += 1
    start = datetime.utcnow() + timedelta(days=90 + _n[0])
    return create_pending_reservation(
        db, db.query(ParkingLot).first().id, "Sec User", "SC-0001", start, start + timedelta(hours=2)
    ).id
