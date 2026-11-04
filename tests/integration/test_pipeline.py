"""Full pipeline through the HTTP API: chat -> escalate -> admin decision -> MCP file.

TestClient, mocked LLM, tmp-file SQLite for the app DB and the LangGraph checkpointer
(the ``orch`` fixture in tests/conftest.py).  The MCP *transport* is mocked, but the
mock writes through the REAL server-side store (ReservationRecord + ReservationStore),
so the output-file format, idempotency and locking are the production code.
"""
from __future__ import annotations

import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.agents import mcp_client
from app.db.models import LotReservation, ReservationStatus
from app.mcp_server.server import ReservationRecord, ReservationStore


@pytest.fixture
def pipe(orch):
    """orch whose MCP client writes into a real ReservationStore file."""
    store = ReservationStore(orch.tmp_path / "out" / "confirmed.txt")
    orch.store = store
    orch.mcp_down = False

    async def fake_send(res, max_attempts=None):
        if not mcp_client.is_admin_approved(res):
            return mcp_client.BLOCKED
        if orch.mcp_down:
            return mcp_client.TRANSIENT
        status = store.write(ReservationRecord(**mcp_client._payload(res)))
        return mcp_client.OK if status in ("written", "duplicate") else mcp_client.PERMANENT

    orch.send.side_effect = fake_send
    orch.lines = lambda: store.tail(10_000)
    return orch


def _sid(tag):
    return f"{tag}-{uuid.uuid4().hex[:6]}"


def test_info_then_booking_then_approval_writes_exactly_one_correct_line(pipe):
    sid = _sid("flow")
    info = pipe.chat(sid, "How much does parking cost?")
    assert info["intent"] == "DYNAMIC_INFO" and info["reservation_id"] is None

    rid = pipe.book(sid, name="Shalini Singh", plate="TS09AB1234")
    assert pipe.lines() == []                                   # PENDING: nothing in the file

    body = pipe.approve(rid).json()
    assert body["status"] == "confirmed" and body["file_written"] is True
    lines = pipe.lines()
    assert len(lines) == 1
    name, plate, period, approved_at = [p.strip() for p in lines[0].split("|")]
    assert (name, plate) == ("Shalini Singh", "TS09AB1234")
    assert period.count(":") == 2 and " - " in period          # "YYYY-MM-DD HH:MM - HH:MM"
    assert len(approved_at) == 19                               # "YYYY-MM-DD HH:MM:SS"
    assert pipe.row(rid).file_written is True


def test_refusal_and_cancellation_never_reach_the_file(pipe):
    refused, cancelled, approved = _sid("r"), _sid("c"), _sid("a")
    r_id, c_id, a_id = pipe.book(refused), pipe.book(cancelled, "Cara Cole", "CC-1111"), pipe.book(
        approved, "Ann Lee", "AL-2222")
    assert pipe.refuse(r_id).status_code == 200
    assert "cancelled" in pipe.chat(cancelled, "cancel my request")["answer"].lower()
    assert pipe.approve(c_id).status_code == 409 and pipe.approve(a_id).status_code == 200

    lines = pipe.lines()
    assert len(lines) == 1 and lines[0].startswith("Ann Lee | AL-2222")
    assert pipe.row(r_id).status == ReservationStatus.REFUSED
    assert pipe.row(c_id).status == ReservationStatus.CANCELLED


def test_mcp_outage_keeps_the_approval_and_the_retry_sweep_fills_the_file(pipe):
    from app.agents.reservation_writer import retry_pending_writes

    sid = _sid("out")
    rid = pipe.book(sid, "Olga Ng", "OL-3333")
    pipe.mcp_down = True
    body = pipe.approve(rid).json()
    assert body["status"] == "confirmed" and body["file_written"] is False
    assert pipe.lines() == []

    pipe.mcp_down = False
    assert retry_pending_writes(pipe.Session)["written"] == 1
    assert len(pipe.lines()) == 1
    assert retry_pending_writes(pipe.Session)["attempted"] == 0   # nothing left; no duplicate line


def test_concurrent_users_and_admins_produce_no_lost_or_duplicate_lines(pipe):
    """8 customers book in parallel, then 8 admin calls (+ 8 duplicate clicks) race."""
    sids = [_sid(f"u{i}") for i in range(8)]
    with ThreadPoolExecutor(8) as pool:
        rids = list(pool.map(
            lambda a: pipe.book(a[1], f"User Number{a[0]}", f"LD-{a[0]:04d}"), enumerate(sids)))
    assert len(set(rids)) == 8                                   # no shared/duplicated reservation ids
    for sid, rid in zip(sids, rids):
        assert pipe.graph(sid)["reservation_id"] == rid          # no cross-talk between sessions

    approve_ids, refuse_ids = rids[:4], rids[4:]
    jobs = [("a", r) for r in approve_ids] * 2 + [("r", r) for r in refuse_ids]   # each approve fired twice
    with ThreadPoolExecutor(12) as pool:
        codes = list(pool.map(
            lambda j: (j, (pipe.approve if j[0] == "a" else pipe.refuse)(j[1]).status_code), jobs))

    for rid in approve_ids:                                      # exactly one winner per duplicated approve
        assert sorted(c for (k, r), c in codes if r == rid) == [200, 409]
    assert all(c == 200 for (k, r), c in codes if k == "r")

    lines = pipe.lines()
    assert len(lines) == 4 and len(set(lines)) == 4              # nothing lost, nothing duplicated
    db = pipe.Session()
    try:
        confirmed = db.query(LotReservation).filter_by(status=ReservationStatus.CONFIRMED).count()
        assert confirmed == 4
        assert all(r.file_written for r in db.query(LotReservation).filter_by(status=ReservationStatus.CONFIRMED))
    finally:
        db.close()
    assert Counter(l.split(" | ")[1] for l in lines) == Counter(f"LD-{i:04d}" for i in range(4))
