"""Regression tests for the Stage-4 review: races, crash recovery, auth gaps, log hygiene."""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta
from unittest.mock import patch

import pytest

from app.agents.channels import tokens
from app.agents.channels.tokens import TokenInvalid
from app.chat import reservation as resv
from app.db.models import LotReservation, ParkingLot, ReservationStatus
from app.db.repository import (
    ValidationError,
    approve_reservation,
    create_pending_reservation,
    refuse_reservation,
)
from app.orchestration import graph as graph_mod
from app.orchestration.service import reconcile_decided, run_turn


def _sid(tag):
    return f"{tag}-{uuid.uuid4().hex[:6]}"


def _lot(Session, spaces):
    s = Session()
    lot = ParkingLot(name=f"Lot{uuid.uuid4().hex[:4]}", address="x", total_spaces=spaces,
                     available_spaces=spaces, hourly_price=8.0, daily_price=40.0,
                     opens_at=time(0, 0), closes_at=time(23, 59))
    s.add(lot)
    s.commit()
    lot_id = lot.id
    s.close()
    return lot_id


def _pending(Session, lot_id, n, same_window=True):
    ids = []
    s = Session()
    base = datetime.now() + timedelta(days=3)
    for i in range(n):
        start = base if same_window else base + timedelta(hours=3 * i)
        r = create_pending_reservation(
            s, lot_id, f"Race User{i}", f"RC{uuid.uuid4().hex[:6].upper()}", start,
            start + timedelta(hours=2), session_id=f"race-{i}")
        ids.append(r.id)
    s.close()
    return ids


def _decide_in_threads(Session, jobs):
    """jobs: list of (fn, reservation_id); each runs in its own thread + session."""
    def run(job):
        fn, rid = job
        s = Session()
        try:
            fn(s, rid, admin="race")
            return "ok"
        except ValidationError as exc:
            return f"rejected: {str(exc)[:30]}"
        finally:
            s.close()

    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(run, jobs))


# ---------------------------------------------------------------------------
# Race conditions
# ---------------------------------------------------------------------------


def test_same_reservation_approved_twice_at_once_has_exactly_one_winner(orch):
    lot_id = _lot(orch.Session, 10)
    (rid,) = _pending(orch.Session, lot_id, 1)
    results = _decide_in_threads(orch.Session, [(approve_reservation, rid)] * 12)
    assert results.count("ok") == 1 and len(results) == 12
    s = orch.Session()
    assert s.get(ParkingLot, lot_id).available_spaces == 9          # decremented exactly once
    assert s.get(LotReservation, rid).status == ReservationStatus.CONFIRMED
    s.close()


def test_concurrent_approvals_never_exceed_lot_capacity(orch):
    lot_id = _lot(orch.Session, 3)
    ids = _pending(orch.Session, lot_id, 10)
    results = _decide_in_threads(orch.Session, [(approve_reservation, r) for r in ids])
    s = orch.Session()
    confirmed = s.query(LotReservation).filter_by(lot_id=lot_id, status=ReservationStatus.CONFIRMED).count()
    refused = s.query(LotReservation).filter_by(lot_id=lot_id, status=ReservationStatus.REFUSED).count()
    assert confirmed == 3 and results.count("ok") == 3
    assert refused == 7                                              # the rest were auto-refused
    assert s.get(ParkingLot, lot_id).available_spaces == 0
    s.close()


def test_approve_and_refuse_racing_for_one_reservation_yield_one_decision(orch):
    lot_id = _lot(orch.Session, 5)
    (rid,) = _pending(orch.Session, lot_id, 1)
    results = _decide_in_threads(orch.Session, [(approve_reservation, rid), (refuse_reservation, rid)] * 4)
    assert results.count("ok") == 1
    s = orch.Session()
    row = s.get(LotReservation, rid)
    spaces = s.get(ParkingLot, lot_id).available_spaces
    assert (row.status == ReservationStatus.CONFIRMED) == (spaces == 4)   # space taken iff approved
    s.close()


def test_chat_turns_and_an_admin_decision_for_the_same_session_do_not_corrupt_the_thread(orch):
    sid = _sid("mix")
    rid = orch.book(sid)
    with ThreadPoolExecutor(8) as pool:
        futs = [pool.submit(orch.chat, sid, "what's the status?") for _ in range(6)]
        futs.append(pool.submit(orch.approve, rid))
        for f in futs:
            f.result()
    g = orch.graph(sid)
    assert g["status"] == "approved" and g["next"] == [] and g["file_written"] is True
    assert orch.send.await_count == 1


# ---------------------------------------------------------------------------
# Crash windows
# ---------------------------------------------------------------------------


class _Crash(BaseException):
    """Simulates the process dying (not an Exception, so no node swallows it)."""


def _restart(orch, sid):
    graph_mod.reset_graph()
    resv._sessions.pop(sid, None)


def test_crash_after_decision_commit_before_graph_resume_is_healed_by_reconcile(orch):
    sid = _sid("heal")
    rid = orch.book(sid)
    s = orch.Session()
    approve_reservation(s, rid, admin="admin-before-crash")        # committed; resume never happened
    assert orch.graph(sid)["interrupted"] is True and orch.send.await_count == 0
    _restart(orch, sid)

    assert reconcile_decided(s) == 1                                # startup / sweep recovery
    s.close()
    g = orch.graph(sid)
    assert g["status"] == "approved" and g["next"] == [] and g["file_written"] is True
    assert orch.send.await_count == 1 and orch.row(rid).file_written is True


def test_crash_between_pending_row_and_checkpoint_recovers_from_the_database(orch):
    sid = _sid("crash")
    orch.book(sid, confirm=False)                                   # SM is in CONFIRM, nothing created
    real = resv.finalize_confirmed

    def die_after_creating_the_row(session_id, db):
        real(session_id, db)
        raise _Crash()

    db = orch.Session()
    with patch("app.orchestration.nodes.resv.finalize_confirmed", die_after_creating_the_row):
        with pytest.raises(_Crash):
            run_turn(sid, "confirm", db)
    db.close()
    s = orch.Session()
    rows = s.query(LotReservation).filter_by(session_id=sid).all()
    s.close()
    assert len(rows) == 1 and rows[0].status == ReservationStatus.PENDING   # row exists, checkpoint stale
    rid = rows[0].id

    _restart(orch, sid)
    r = orch.chat(sid, "what's the status?")                        # must NOT ask "type confirm" again
    assert "PENDING" in r["answer"] and "confirm" not in r["answer"].lower().replace("confirmed", "")
    g = orch.graph(sid)
    assert g["interrupted"] is True and g["reservation_id"] == rid
    assert orch.approve(rid).json()["file_written"] is True
    assert orch.graph(sid)["status"] == "approved" and len(orch.row(rid).customer_name) > 0


def test_losing_the_whole_checkpoint_database_does_not_lose_a_pending_approval(orch):
    sid = _sid("lost")
    rid = orch.book(sid)
    graph_mod.reset_graph()
    resv._sessions.pop(sid, None)
    for f in orch.tmp_path.glob("checkpoints.db*"):
        f.unlink()                                                  # disk wiped / volume replaced

    r = orch.chat(sid, "any news?")
    assert "CPK-" in r["answer"] and "awaiting" in r["answer"].lower()
    g = orch.graph(sid)
    assert g["interrupted"] is True and g["reservation_id"] == rid  # re-created from the DB row
    assert orch.approve(rid).json()["status"] == "confirmed"
    assert orch.graph(sid)["status"] == "approved"
    assert orch.send.await_count == 1


# ---------------------------------------------------------------------------
# Auth gaps
# ---------------------------------------------------------------------------


def test_placeholder_email_token_secret_is_refused_both_ways(orch, monkeypatch):
    monkeypatch.setattr(tokens.settings, "email_token_secret", "change-me-in-production")
    assert tokens.secret_is_configured() is False
    with pytest.raises(TokenInvalid):
        tokens.sign(1, 1, "APPROVE")                                # never mint a forgeable link
    # an attacker who knows the public placeholder cannot approve anything with it
    forged = tokens.URLSafeTimedSerializer("change-me-in-production", salt=tokens._SALT).dumps(
        {"rid": 1, "aid": 1, "dec": "APPROVE", "rsn": ""})
    r = orch.client.get("/admin/approve", params={"token": forged})
    assert r.status_code == 400
    monkeypatch.setattr(tokens.settings, "email_token_secret", "a-real-random-secret-value-123")
    assert tokens.secret_is_configured() is True
    assert tokens.verify(tokens.sign(7, 3, "REFUSE", "r"))["rid"] == 7


def test_stage1_reservation_listing_requires_the_admin_key(orch):
    assert orch.client.get("/reservations/").status_code == 401
    assert orch.client.get("/reservations/1").status_code == 401
    assert orch.client.get("/reservations/", headers={"X-Admin-Key": "wrong"}).status_code == 403
    assert orch.client.get("/reservations/", headers=orch.headers).status_code == 200
    assert orch.client.get("/reservations/99/status").status_code == 404   # customer status stays public


# ---------------------------------------------------------------------------
# Log hygiene
# ---------------------------------------------------------------------------


def test_unhandled_errors_log_the_class_not_the_message(orch, caplog):
    from fastapi.testclient import TestClient

    from app.api.main import app

    secret = "ABC-1234 Jane Doe"
    with patch("app.api.routes.graph.get_state", side_effect=ValueError(secret)):
        r = TestClient(app, raise_server_exceptions=False).get("/graph/x/state", headers=orch.headers)
    assert r.status_code == 500 and secret not in r.text
    assert "error=ValueError" in caplog.text
    assert secret not in caplog.text and "Jane" not in caplog.text
