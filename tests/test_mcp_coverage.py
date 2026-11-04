"""
Edge-case tests for the Stage-3 modules that need no running MCP server:
reservation_writer, the server's main()/locking/rate-limit internals, the
API retry worker and the two scripts.  Resources are disposed in fixtures so
Windows can delete the temp SQLite files.
"""
import asyncio
import importlib.util
import runpy
import sys
import types
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.agents.reservation_writer as writer
import app.api.main as main_mod
import app.mcp_server.server as server_mod
from app.core.config import settings
from app.db.models import Base, LotReservation, ParkingLot, ReservationStatus
from app.db.repository import approve_reservation, create_pending_reservation
from app.mcp_server.server import McpSettings, ReservationRecord, ReservationStore, SecurityMiddleware

ROOT = Path(__file__).resolve().parents[1]
KEY = "coverage-key-" + "c" * 20  # >= 16 chars, as the server now requires


@pytest.fixture
def Session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'cov.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    S = sessionmaker(bind=engine, autoflush=False)
    with S() as s:
        s.add(ParkingLot(
            name="CovLot", address="1 Cov St", total_spaces=10, available_spaces=10,
            hourly_price=1.0, daily_price=5.0, opens_at=dtime(0, 0), closes_at=dtime(23, 59),
        ))
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
    monkeypatch.setattr(settings, "mcp_api_key", "cov-token-" + "k" * 16)


_n = [0]


def make(db, approved=True) -> int:
    _n[0] += 1
    start = datetime.utcnow() + timedelta(days=10 + _n[0])
    lot = db.query(ParkingLot).first()
    r = create_pending_reservation(db, lot.id, "Cov User", "CV-0001", start, start + timedelta(hours=2))
    if approved:
        approve_reservation(db, r.id, admin="cov-admin")
    return r.id


# ── reservation_writer ───────────────────────────────────────────────────────


class TestExportApproved:
    def test_unknown_reservation_is_false(self, db, configured):
        assert writer.export_approved(db, 424242) is False

    def test_pending_reservation_is_not_eligible(self, db, configured, monkeypatch):
        spy = AsyncMock(return_value="ok")
        monkeypatch.setattr(writer, "send_reservation", spy)
        assert writer.export_approved(db, make(db, approved=False)) is False
        spy.assert_not_called()

    def test_already_written_short_circuits_without_network(self, db, configured, monkeypatch):
        rid = make(db)
        db.query(LotReservation).filter_by(id=rid).update({"file_written": True})
        db.commit()
        spy = AsyncMock(return_value="ok")
        monkeypatch.setattr(writer, "send_reservation", spy)
        assert writer.export_approved(db, rid) is True
        spy.assert_not_called()

    def test_unconfigured_returns_false_and_leaves_row_queued(self, db, monkeypatch):
        monkeypatch.setattr(settings, "mcp_api_key", "")
        rid = make(db)
        assert writer.export_approved(db, rid) is False
        assert db.get(LotReservation, rid).file_written is False

    def test_success_marks_row(self, db, configured, monkeypatch):
        monkeypatch.setattr(writer, "send_reservation", AsyncMock(return_value="ok"))
        rid = make(db)
        assert writer.export_approved(db, rid) is True
        row = db.get(LotReservation, rid)
        assert row.file_written is True and row.file_written_at is not None

    def test_client_exception_never_escapes_and_session_stays_usable(self, db, configured, monkeypatch):
        monkeypatch.setattr(writer, "send_reservation", AsyncMock(side_effect=RuntimeError("boom")))
        rid = make(db)
        assert writer.export_approved(db, rid) is False
        assert db.get(LotReservation, rid).status == ReservationStatus.CONFIRMED

    def test_notes(self):
        assert "recorded" in writer.export_note(True)
        assert "pending" in writer.export_note(False)


class TestRetryPendingWrites:
    def test_unconfigured_counts_one_failure_and_stops(self, db, Session, monkeypatch):
        monkeypatch.setattr(settings, "mcp_api_key", "")
        make(db)
        make(db)
        assert writer.retry_pending_writes(Session) == {"attempted": 1, "written": 0, "failed": 1}

    def test_exception_in_client_counts_as_failure(self, db, Session, configured, monkeypatch):
        monkeypatch.setattr(writer, "send_reservation", AsyncMock(side_effect=ValueError("x")))
        make(db)
        assert writer.retry_pending_writes(Session) == {"attempted": 1, "written": 0, "failed": 1}

    def test_row_changed_since_listing_is_skipped(self, db, Session, configured, monkeypatch):
        rid = make(db)
        db.query(LotReservation).filter_by(id=rid).update({"file_written": True})
        db.commit()
        monkeypatch.setattr(writer, "pending_ids", lambda *_a, **_k: [rid, 999_999])
        spy = AsyncMock(return_value="ok")
        monkeypatch.setattr(writer, "send_reservation", spy)
        assert writer.retry_pending_writes(Session) == {"attempted": 0, "written": 0, "failed": 0}
        spy.assert_not_called()

    def test_success_marks_every_row(self, db, Session, configured, monkeypatch):
        monkeypatch.setattr(writer, "send_reservation", AsyncMock(return_value="ok"))
        ids = [make(db), make(db)]
        assert writer.retry_pending_writes(Session)["written"] == 2
        db.expire_all()
        assert all(db.get(LotReservation, i).file_written for i in ids)


# ── API background worker ────────────────────────────────────────────────────


class TestRetryWorkerLoop:
    def test_worker_survives_a_failing_sweep_and_keeps_looping(self, monkeypatch):
        calls = []

        def flaky(_factory):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("db hiccup")
            return {"attempted": 0, "written": 0, "failed": 0}

        monkeypatch.setattr(main_mod, "retry_pending_writes", flaky)
        monkeypatch.setattr(settings, "mcp_retry_interval_seconds", 0.01)

        async def run():
            task = asyncio.create_task(main_mod._retry_worker())
            for _ in range(200):
                if len(calls) >= 3:
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run())
        assert len(calls) >= 3  # first sweep raised, worker carried on

    def test_worker_is_cancellable(self, monkeypatch):
        monkeypatch.setattr(main_mod, "retry_pending_writes", lambda _f: {})
        monkeypatch.setattr(settings, "mcp_retry_interval_seconds", 60)

        async def run():
            task = asyncio.create_task(main_mod._retry_worker())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())


# ── server internals ─────────────────────────────────────────────────────────


def _rec(**o):
    base = dict(reservation_id=1, customer_name="A B", car_plate="AB-123",
                start_time="2026-11-04T10:00:00", end_time="2026-11-04T12:00:00",
                approved_at="2026-11-03T09:00:00")
    base.update(o)
    return ReservationRecord(**base)


class TestServerInternals:
    def test_main_runs_uvicorn_with_configured_host_port(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MCP_API_KEY", KEY)
        monkeypatch.setenv("MCP_HOST", "127.0.0.1")
        monkeypatch.setenv("MCP_PORT", "9123")
        monkeypatch.setenv("RESERVATIONS_FILE", str(tmp_path / "o.txt"))
        run = MagicMock()
        monkeypatch.setattr("uvicorn.run", run)
        server_mod.main()
        run.assert_called_once()
        assert run.call_args.kwargs["host"] == "127.0.0.1" and run.call_args.kwargs["port"] == 9123
        assert isinstance(run.call_args.args[0], SecurityMiddleware)

    def test_main_refuses_to_start_without_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MCP_API_KEY", "")
        monkeypatch.setenv("RESERVATIONS_FILE", str(tmp_path / "o.txt"))
        monkeypatch.setattr("uvicorn.run", MagicMock())
        with pytest.raises(RuntimeError, match="MCP_API_KEY"):
            server_mod.main()

    def test_write_oserror_becomes_generic_tool_error_without_paths(self, tmp_path, monkeypatch):
        store = ReservationStore(tmp_path / "x" / "f.txt")
        monkeypatch.setattr(store, "write", MagicMock(side_effect=PermissionError(str(tmp_path / "secret-path"))))
        mcp = server_mod.create_mcp(store, McpSettings(_env_file=None, mcp_api_key="k"))
        tool = mcp._tool_manager.get_tool("write_reservation")
        with pytest.raises(RuntimeError) as ei:
            tool.fn(1, "A B", "AB-123", "2026-11-04T10:00:00", "2026-11-04T12:00:00", "2026-11-03T09:00:00")
        assert "secret-path" not in str(ei.value) and "Could not write" in str(ei.value)

    @pytest.mark.skipif(sys.platform != "win32", reason="msvcrt locking is Windows-only")
    def test_windows_lock_retries_until_acquired(self, tmp_path, monkeypatch):
        import msvcrt
        real, attempts = msvcrt.locking, []

        def flaky(fd, mode, nbytes):
            if mode == msvcrt.LK_LOCK and not attempts:
                attempts.append(1)
                raise OSError("deadlock timeout")
            return real(fd, mode, nbytes)

        monkeypatch.setattr(msvcrt, "locking", flaky)
        with server_mod._os_lock(tmp_path / "t.lock"):
            pass
        assert attempts == [1]

    def test_posix_lock_branch_uses_flock(self, tmp_path, monkeypatch):
        calls = []
        fake = types.SimpleNamespace(
            LOCK_EX=2, LOCK_UN=8, flock=lambda fd, op: calls.append(op)
        )
        monkeypatch.setitem(sys.modules, "fcntl", fake)
        monkeypatch.setattr(server_mod.sys, "platform", "linux")
        with server_mod._os_lock(tmp_path / "p.lock"):
            assert calls == [2]
        assert calls == [2, 8]

    def test_rate_limit_window_expires(self, monkeypatch):
        mw = SecurityMiddleware(object(), KEY, rate_limit_per_minute=2)
        t = [1000.0]
        monkeypatch.setattr(server_mod.time, "monotonic", lambda: t[0])
        assert [mw._rate_limited("c") for _ in range(3)] == [False, False, True]
        t[0] += 61
        assert mw._rate_limited("c") is False  # old hits aged out

    def test_lifespan_scope_passes_through_untouched(self):
        inner = AsyncMock()
        mw = SecurityMiddleware(inner, KEY, 1)
        asyncio.run(mw({"type": "lifespan"}, None, None))
        inner.assert_awaited_once()

    def test_rate_limit_is_per_client(self):
        mw = SecurityMiddleware(object(), KEY, 1)
        assert mw._rate_limited("a") is False and mw._rate_limited("a") is True
        assert mw._rate_limited("b") is False

    def test_tail_on_missing_file_and_limit(self, tmp_path):
        store = ReservationStore(tmp_path / "t.txt")
        assert store.tail(3) == []
        for i in (1, 2, 3):
            store.write(_rec(reservation_id=i, customer_name=f"N{i}"))
        assert [ln.split(" | ")[0] for ln in store.tail(2)] == ["N2", "N3"]


# ── scripts ──────────────────────────────────────────────────────────────────


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRetryScript:
    @pytest.fixture
    def script(self, Session, monkeypatch):
        mod = _load_script("retry_pending_writes")
        monkeypatch.setattr(mod, "create_tables", lambda: None)   # never touch parking.db
        monkeypatch.setattr(mod, "SessionLocal", Session)
        return mod

    def test_exit_2_when_export_not_configured(self, script, monkeypatch, capsys):
        monkeypatch.setattr(settings, "mcp_api_key", "")
        assert script.main() == 2
        assert "not configured" in capsys.readouterr().out

    def test_exit_0_and_rows_marked_on_success(self, script, db, configured, monkeypatch, capsys):
        monkeypatch.setattr(writer, "send_reservation", AsyncMock(return_value="ok"))
        rid = make(db)
        assert script.main() == 0
        out = capsys.readouterr().out
        assert "1 approved reservation(s) waiting" in out and "written=1 failed=0" in out
        db.expire_all()
        assert db.get(LotReservation, rid).file_written is True

    def test_exit_1_when_server_unreachable(self, script, db, configured, monkeypatch, capsys):
        monkeypatch.setattr(writer, "send_reservation", AsyncMock(return_value="transient"))
        make(db)
        assert script.main() == 1
        assert "failed=1" in capsys.readouterr().out


class TestRunServerScript:
    def test_dunder_main_invokes_server_main(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setattr("app.mcp_server.server.main", fake)
        runpy.run_path(str(ROOT / "scripts" / "run_mcp_server.py"), run_name="__main__")
        fake.assert_called_once_with()

    def test_importing_does_not_start_the_server(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setattr("app.mcp_server.server.main", fake)
        mod = _load_script("run_mcp_server")
        fake.assert_not_called()
        assert callable(mod.main)
