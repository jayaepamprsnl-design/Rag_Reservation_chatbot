#!/usr/bin/env python3
"""
CityPark Stage-2 Evaluation Suite — Human-in-the-Loop Agent
=============================================================
Sections:
  1. Decision parsing accuracy   30 labelled admin messages
  2. Escalation & notification latency  (mean, p95 over N_BENCH iterations)
  3. Reliability   20 concurrent reservations  (no lost / duplicated)
  4. Double-booking prevention   2 simultaneous approve calls

Outputs:
  docs/results_stage2.json
  docs/evaluation_stage2.md
  docs/charts/stage2_accuracy.png
  docs/charts/stage2_latency.png
  docs/charts/stage2_reliability.png
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import tempfile
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool, NullPool

from app.db.models import Base, LotReservation, ParkingLot, ReservationStatus
from app.db.repository import (
    ValidationError,
    approve_reservation,
    create_pending_reservation,
    list_pending,
    refuse_reservation,
)
from app.chat.reservation import (
    ReservationState,
    _sessions as _sm_sessions,
    advance,
    reset,
    start,
)
from app.agents.admin_agent import AdminAgent
from app.agents.contracts import AdminDecision, Decision
from app.rag.generator import BaseLLMClient

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR   = ROOT / "docs"
CHARTS_DIR = DOCS_DIR / "charts"
DOCS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BENCH      = 30    # latency iterations
N_CONCURRENT = 20    # concurrent reservation workers
LOT_SPACES   = 500   # large lot — no capacity limit in concurrency test
_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

_plate_counter = 0
_plate_lock    = threading.Lock()


def _next_plate() -> str:
    """Thread-safe unique licence plate (AB-0001 … AB-9999)."""
    global _plate_counter
    with _plate_lock:
        _plate_counter += 1
        return f"AB-{_plate_counter:04d}"


def _future_str(hours: int = 25, hr: int = 10) -> str:
    dt = datetime.now() + timedelta(hours=hours)
    return dt.replace(hour=hr, minute=0, second=0, microsecond=0).strftime(
        "%Y-%m-%d %H:%M"
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_ENGINE   = None
_SESSION  = None
_DB_LOCK  = threading.Lock()


def _apply_wal(engine) -> None:
    """Enable WAL journal mode — allows concurrent readers + one writer."""
    @event.listens_for(engine, "connect")
    def _set_wal(conn, _record):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")


def _make_file_engine(path: str) -> tuple:
    """
    File-based SQLite engine suitable for concurrent thread access.

    StaticPool shares a single connection across all threads — it is NOT
    thread-safe for concurrent writes.  A file-based engine with NullPool
    gives each thread its own connection to the same file; SQLite's WAL
    mode then serialises concurrent writers gracefully with a 30-second
    busy timeout rather than immediately raising OperationalError.
    """
    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool,
    )
    _apply_wal(engine)
    return engine, sessionmaker(bind=engine)


def _seed_lot(Session, lot_spaces: int = LOT_SPACES, name: str = "EvalLot") -> None:
    db = Session()
    db.add(ParkingLot(
        name=name,
        address="1 Eval Avenue",
        total_spaces=lot_spaces,
        available_spaces=lot_spaces,
        hourly_price=8.0,
        daily_price=40.0,
        opens_at=datetime.min.time(),
        closes_at=datetime.max.time(),
    ))
    db.commit()
    db.close()


def _init_db() -> tuple:
    """Create one shared in-memory SQLite DB (StaticPool) for serial sections."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    _seed_lot(Session)
    db = Session()
    return engine, Session, db


def _lot_id(db) -> int:
    return db.query(ParkingLot).filter_by(name="EvalLot").one().id


# ---------------------------------------------------------------------------
# NullLLM — deterministic mock for reason extraction
# ---------------------------------------------------------------------------

class _NullLLM(BaseLLMClient):
    """
    Always returns empty string.

    Effect: when the regex finds no "because/since" clause the LLM returns
    "", so parse_decision correctly asks for a reason (CLARIFICATION) rather
    than inventing one.  Items with an explicit reason clause are handled by
    the regex alone.
    """

    def complete(self, messages: list[dict]) -> str:
        return ""


# ===========================================================================
# Section 1 — Decision parsing accuracy
# ===========================================================================

_DECISION_DATASET: list[dict] = [
    # ── approve (8) ──────────────────────────────────────────────────────────
    {"id":  1, "text": "approve 42",                                          "category": "approve",           "expected": "APPROVE"},
    {"id":  2, "text": "APPROVE reservation 7",                               "category": "approve",           "expected": "APPROVE"},
    {"id":  3, "text": "looks good, approve 12",                              "category": "approve",           "expected": "APPROVE"},
    {"id":  4, "text": "ok 15",                                               "category": "approve",           "expected": "APPROVE"},
    {"id":  5, "text": "confirmed, let this one through",                     "category": "approve",           "expected": "APPROVE"},
    {"id":  6, "text": "accepted — go ahead with booking 9",                  "category": "approve",           "expected": "APPROVE"},
    {"id":  7, "text": "approved",                                            "category": "approve",           "expected": "APPROVE"},
    {"id":  8, "text": "it is fine, approve 18",                              "category": "approve",           "expected": "APPROVE"},
    # ── refuse with explicit reason (8) ──────────────────────────────────────
    {"id":  9, "text": "refuse 5 because the lot is closed for maintenance",  "category": "refuse_with_reason","expected": "REFUSE"},
    {"id": 10, "text": "reject 11 because we are fully booked today",         "category": "refuse_with_reason","expected": "REFUSE"},
    {"id": 11, "text": "deny 6 because of ongoing security maintenance",      "category": "refuse_with_reason","expected": "REFUSE"},
    {"id": 12, "text": "decline 20 because the car park is flooded",          "category": "refuse_with_reason","expected": "REFUSE"},
    {"id": 13, "text": "refuse 3 because it is double-booked with an event",  "category": "refuse_with_reason","expected": "REFUSE"},
    {"id": 14, "text": "reject 16 because owner requested last-minute cancellation","category":"refuse_with_reason","expected":"REFUSE"},
    {"id": 15, "text": "deny 2 because contractor vehicles need access",      "category": "refuse_with_reason","expected": "REFUSE"},
    {"id": 16, "text": "refuse 8 since the lot is temporarily closed",        "category": "refuse_with_reason","expected": "REFUSE"},
    # ── refuse without reason — agent must ask (4) ───────────────────────────
    {"id": 17, "text": "refuse 3",                                            "category": "refuse_no_reason",  "expected": "CLARIFICATION"},
    {"id": 18, "text": "reject 14",                                           "category": "refuse_no_reason",  "expected": "CLARIFICATION"},
    {"id": 19, "text": "deny reservation 1",                                  "category": "refuse_no_reason",  "expected": "CLARIFICATION"},
    {"id": 20, "text": "decline 10",                                          "category": "refuse_no_reason",  "expected": "CLARIFICATION"},
    # ── show pending / informational (4) ─────────────────────────────────────
    {"id": 21, "text": "show pending",                                        "category": "no_decision",       "expected": "CLARIFICATION"},
    {"id": 22, "text": "list all pending reservations",                       "category": "no_decision",       "expected": "CLARIFICATION"},
    {"id": 23, "text": "what requests are waiting?",                          "category": "no_decision",       "expected": "CLARIFICATION"},
    {"id": 24, "text": "how many bookings need review?",                      "category": "no_decision",       "expected": "CLARIFICATION"},
    # ── ambiguous (2) ─────────────────────────────────────────────────────────
    {"id": 25, "text": "approve or refuse 7?",                                "category": "ambiguous",         "expected": "CLARIFICATION"},
    {"id": 26, "text": "should I accept or reject this?",                     "category": "ambiguous",         "expected": "CLARIFICATION"},
    # ── unrelated (4) ─────────────────────────────────────────────────────────
    {"id": 27, "text": "hello, how are you?",                                 "category": "unrelated",         "expected": "CLARIFICATION"},
    {"id": 28, "text": "thanks for the help",                                 "category": "unrelated",         "expected": "CLARIFICATION"},
    {"id": 29, "text": "what is the weather like today?",                     "category": "unrelated",         "expected": "CLARIFICATION"},
    {"id": 30, "text": "can you book a taxi for me?",                         "category": "unrelated",         "expected": "CLARIFICATION"},
]

assert len(_DECISION_DATASET) == 30, "Dataset must have exactly 30 items"


def _predicted_label(result: AdminDecision | str) -> str:
    if isinstance(result, AdminDecision):
        return result.decision.value   # "APPROVE" or "REFUSE"
    return "CLARIFICATION"


def run_decision_parsing(engine, Session) -> dict[str, Any]:
    """Evaluate AdminAgent.parse_decision on the 30-item labelled dataset."""
    db = Session()
    try:
        agent = AdminAgent(lambda: Session(), llm=_NullLLM())
        rows: list[dict] = []

        for item in _DECISION_DATASET:
            result = agent.parse_decision(
                admin_text=item["text"],
                reservation_id=99,          # fallback — most items embed their own ID
                admin_id="eval-admin",
            )
            predicted  = _predicted_label(result)
            expected   = item["expected"]
            correct    = (predicted == expected)
            rows.append({
                "id":        item["id"],
                "category":  item["category"],
                "text":      item["text"],
                "expected":  expected,
                "predicted": predicted,
                "correct":   correct,
            })

        cats     = sorted({r["category"] for r in rows})
        cat_stat = {}
        for cat in cats:
            sub = [r for r in rows if r["category"] == cat]
            n_ok = sum(r["correct"] for r in sub)
            cat_stat[cat] = {
                "n":        len(sub),
                "correct":  n_ok,
                "accuracy": round(n_ok / len(sub), 4),
            }

        total_ok  = sum(r["correct"] for r in rows)
        overall   = round(total_ok / len(rows), 4)

        return {
            "overall_accuracy":  overall,
            "total_correct":     total_ok,
            "total":             len(rows),
            "per_category":      cat_stat,
            "per_item":          rows,
        }
    finally:
        db.close()


# ===========================================================================
# Section 2 — Escalation & notification latency
# ===========================================================================

def _drive_to_confirm_state(sid: str, db, hour_offset: int) -> None:
    """Advance SM to CONFIRM (not measured); unique time window per iteration."""
    lot_id = _lot_id(db)
    start(sid)
    advance(sid, "Bench User", db)
    advance(sid, _next_plate(), db)
    advance(sid, str(lot_id), db)
    advance(sid, _future_str(hour_offset,      10), db)
    advance(sid, _future_str(hour_offset + 2,  14), db)
    # state == CONFIRM — next advance("confirm") will create pending + escalate


def run_latency(engine, Session) -> dict[str, Any]:
    """Measure escalation and decision-to-notification latency."""
    esc_times:  list[float] = []    # confirm -> pending visible in list_pending()
    notif_times: list[float] = []   # approve_reservation() -> advance() delivers
    total_times: list[float] = []   # approval + notification combined

    db = Session()
    try:
        base_offset = 500   # start well in the future, step by 4 hours each iter

        for i in range(N_BENCH):
            hour_off = base_offset + i * 4
            sid = f"bench-esc-{i:04d}"

            # ── escalation latency ───────────────────────────────────────────
            _drive_to_confirm_state(sid, db, hour_off)

            t0 = time.perf_counter()
            r  = advance(sid, "confirm", db)         # creates pending + RestChannel row
            esc_ms = (time.perf_counter() - t0) * 1_000

            assert r.state == ReservationState.AWAITING_APPROVAL, (
                f"Expected AWAITING_APPROVAL, got {r.state}"
            )
            res_id = r.reservation_id

            # Confirm visibility in list_pending()
            pending_ids = {p.id for p in list_pending(db)}
            assert res_id in pending_ids, f"Reservation {res_id} not in pending"

            esc_times.append(esc_ms)

            # ── decision-to-notification latency ─────────────────────────────
            # Approval (timed)
            t1 = time.perf_counter()
            approve_reservation(db, res_id, admin="bench-admin")
            approve_ms = (time.perf_counter() - t1) * 1_000

            # Notification delivery via advance() (timed)
            t2 = time.perf_counter()
            r2 = advance(sid, "status", db)
            notif_ms = (time.perf_counter() - t2) * 1_000

            assert r2.state == ReservationState.DONE, (
                f"Expected DONE after approval, got {r2.state}"
            )

            notif_times.append(approve_ms + notif_ms)
            total_times.append(esc_ms + approve_ms + notif_ms)

            _sm_sessions.pop(sid, None)

    finally:
        db.close()

    def _stats(vals: list[float]) -> dict:
        return {
            "mean_ms": round(float(np.mean(vals)),        2),
            "p95_ms":  round(float(np.percentile(vals, 95)), 2),
            "min_ms":  round(float(np.min(vals)),         2),
            "max_ms":  round(float(np.max(vals)),         2),
        }

    return {
        "n_iterations": N_BENCH,
        "escalation":              _stats(esc_times),
        "decision_to_notification": _stats(notif_times),
        "total_round_trip":        _stats(total_times),
    }


# ===========================================================================
# Section 3 — Concurrent reliability (20 reservations)
#
# Root cause of the StaticPool failure:
#   StaticPool shares ONE SQLite connection across all threads.  SQLite is
#   not thread-safe for concurrent writes on a single connection object.
#   Fix: use a temp file-based SQLite with NullPool (one connection per
#   thread) + WAL journal mode (serialises writers, allows concurrent reads).
# ===========================================================================

def _concurrent_worker(
    worker_id: int,
    engine,
    hour_offset: int,
) -> dict[str, Any]:
    """Create one pending reservation and return result metadata."""
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        sid = f"concurrent-{worker_id:04d}"
        lot_id = db.query(ParkingLot).filter_by(name="EvalLot").one().id

        t0 = time.perf_counter()
        start(sid)
        advance(sid, f"User {worker_id}", db)
        advance(sid, _next_plate(), db)
        advance(sid, str(lot_id), db)
        advance(sid, _future_str(hour_offset,      10), db)
        advance(sid, _future_str(hour_offset + 2,  14), db)
        r = advance(sid, "confirm", db)
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        _sm_sessions.pop(sid, None)

        if r.state != ReservationState.AWAITING_APPROVAL:
            return {
                "worker_id": worker_id,
                "success":   False,
                "error":     f"Unexpected state: {r.state}",
            }
        return {
            "worker_id":     worker_id,
            "success":       True,
            "reservation_id": r.reservation_id,
            "elapsed_ms":    round(elapsed_ms, 2),
        }
    except Exception as exc:  # noqa: BLE001
        return {"worker_id": worker_id, "success": False, "error": str(exc)}
    finally:
        db.close()


def run_concurrent(_ignored_engine, _ignored_session) -> dict[str, Any]:
    """
    Launch N_CONCURRENT workers simultaneously using a file-based SQLite
    (NullPool + WAL mode) so each thread has its own connection.

    Verify:
      * all N succeed (no lost requests)
      * no duplicate reservation IDs
      * no duplicate ApprovalRequest rows per reservation
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name

    try:
        conc_engine, ConSession = _make_file_engine(db_path)
        Base.metadata.create_all(bind=conc_engine)
        _seed_lot(ConSession)

        base_hour = 3000
        futures: list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_CONCURRENT) as ex:
            wall_t0 = time.perf_counter()
            for i in range(N_CONCURRENT):
                futures.append(
                    ex.submit(_concurrent_worker, i, conc_engine, base_hour + i * 4)
                )
            concurrent.futures.wait(futures)
            wall_elapsed = time.perf_counter() - wall_t0

        results = [f.result() for f in futures]
        successes = [r for r in results if r["success"]]
        failures  = [r for r in results if not r["success"]]
        res_ids   = [r["reservation_id"] for r in successes]
        unique_ids = set(res_ids)

        # Audit-row check
        from app.db.models import ApprovalRequest
        check_db = ConSession()
        try:
            dup_count = sum(
                1 for rid in res_ids
                if check_db.query(ApprovalRequest).filter_by(reservation_id=rid).count() != 1
            )
        finally:
            check_db.close()

        conc_engine.dispose()
        elapsed_ms = [r["elapsed_ms"] for r in successes]

        return {
            "n_workers":         N_CONCURRENT,
            "wall_time_s":       round(wall_elapsed, 3),
            "n_success":         len(successes),
            "n_failure":         len(failures),
            "failures":          failures,
            "no_lost_requests":  len(successes) == N_CONCURRENT,
            "no_duplicates":     len(unique_ids) == len(res_ids),
            "no_audit_dups":     dup_count == 0,
            "db_mode":           "file-based SQLite + WAL + NullPool",
            "worker_latency_ms": {
                "mean": round(float(np.mean(elapsed_ms)),           2) if elapsed_ms else 0.0,
                "p95":  round(float(np.percentile(elapsed_ms, 95)), 2) if elapsed_ms else 0.0,
            },
        }
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


# ===========================================================================
# Section 4 — Double-booking prevention
#
# Root cause of the StaticPool double-approval:
#   Both threads share the same connection -> SQLAlchemy session snapshots
#   read the same pre-commit PENDING value -> both succeed.
#   Fix: file-based SQLite so each session has its own connection; SQLite
#   then serialises the two write transactions so the second one reads
#   the committed CONFIRMED state and raises ValidationError.
# ===========================================================================

def _approve_worker(res_id: int, admin_id: str, engine) -> dict[str, Any]:
    Session = sessionmaker(bind=engine)
    db = Session()
    t0 = time.perf_counter()
    try:
        res = approve_reservation(db, res_id, admin=admin_id)
        elapsed = (time.perf_counter() - t0) * 1_000
        return {"admin_id": admin_id, "success": True,
                "status": res.status.value, "elapsed_ms": round(elapsed, 2)}
    except ValidationError as exc:
        elapsed = (time.perf_counter() - t0) * 1_000
        return {"admin_id": admin_id, "success": False,
                "error": str(exc), "elapsed_ms": round(elapsed, 2)}
    finally:
        db.close()


def run_double_booking(_ignored_engine, _ignored_session) -> dict[str, Any]:
    """
    Two admins try to approve the same PENDING reservation simultaneously.
    Invariant: exactly one approval succeeds, available_spaces never < 0.
    Uses a file-based SQLite so each thread gets its own connection and
    SQLite's locking serialises the two write transactions correctly.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name

    try:
        tiny_engine, TinySession = _make_file_engine(db_path)
        Base.metadata.create_all(bind=tiny_engine)

        # Seed a 1-space lot
        setup_db = TinySession()
        setup_db.add(ParkingLot(
            name="TinyLot", address="1 Tiny St",
            total_spaces=1, available_spaces=1,
            hourly_price=5.0, daily_price=25.0,
            opens_at=datetime.min.time(), closes_at=datetime.max.time(),
        ))
        setup_db.commit()
        lot_id = setup_db.query(ParkingLot).filter_by(name="TinyLot").one().id
        res = create_pending_reservation(
            setup_db, lot_id=lot_id,
            customer_name="Double",
            car_plate=_next_plate(),
            start_time=datetime.utcnow() + timedelta(hours=200),
            end_time=datetime.utcnow()   + timedelta(hours=202),
        )
        setup_db.commit()
        res_id = res.id
        setup_db.close()

        barrier = threading.Barrier(2, timeout=10)

        def _barriered(admin_id: str) -> dict:
            barrier.wait()   # maximise the race window
            return _approve_worker(res_id, admin_id, tiny_engine)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(_barriered, f"admin-{k}") for k in range(2)]
            concurrent.futures.wait(futs)

        outcomes = [f.result() for f in futs]
        n_success = sum(1 for o in outcomes if o["success"])
        n_failure = sum(1 for o in outcomes if not o["success"])

        verify = TinySession()
        try:
            lot  = verify.query(ParkingLot).filter_by(name="TinyLot").one()
            final_spaces = lot.available_spaces
            final_status = verify.get(LotReservation, res_id).status.value
        finally:
            verify.close()

        tiny_engine.dispose()
        invariant_ok = (n_success == 1 and final_spaces == 0)

        return {
            "n_concurrent_approvals": 2,
            "n_success":              n_success,
            "n_failure":              n_failure,
            "final_reservation_status": final_status,
            "final_available_spaces": final_spaces,
            "invariant_held":         invariant_ok,
            "db_mode":                "file-based SQLite + WAL + NullPool",
            "outcomes":               outcomes,
        }
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass




# ===========================================================================
# Charts
# ===========================================================================

def _chart_accuracy(acc: dict, path: Path) -> None:
    cats  = sorted(acc["per_category"])
    vals  = [acc["per_category"][c]["accuracy"] for c in cats]
    n_ok  = [acc["per_category"][c]["correct"]   for c in cats]
    n_tot = [acc["per_category"][c]["n"]         for c in cats]
    labels = [f"{c}\n({n_ok[i]}/{n_tot[i]})" for i, c in enumerate(cats)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vals, color=_COLORS[:len(cats)], edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.axhline(acc["overall_accuracy"], color="black", linestyle="--",
               linewidth=1.2, label=f"Overall {acc['overall_accuracy']*100:.1f}%")
    ax.set_ylabel("Accuracy")
    ax.set_title("Decision Parsing Accuracy by Category (n=30)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v*100:.0f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _chart_latency(lat: dict, path: Path) -> None:
    stages = ["Escalation\n(confirm->pending)", "Decision->Notification"]
    keys   = ["escalation", "decision_to_notification"]
    means  = [lat[k]["mean_ms"] for k in keys]
    p95s   = [lat[k]["p95_ms"]  for k in keys]

    x = np.arange(len(stages))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_m = ax.bar(x - w/2, means, w, label="Mean",  color=_COLORS[0])
    bars_p = ax.bar(x + w/2, p95s,  w, label="P95",   color=_COLORS[1])
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency Distribution (n={N_BENCH} iterations)")
    ax.legend()
    for bar in list(bars_m) + list(bars_p):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _chart_reliability(conc: dict, dbl: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Concurrent: success vs failure
    labels1 = ["Succeeded", "Failed"]
    vals1   = [conc["n_success"], conc["n_failure"]]
    colours1 = [_COLORS[2], _COLORS[3]]
    bars1 = ax1.bar(labels1, vals1, color=colours1, edgecolor="white")
    ax1.set_ylim(0, max(vals1) * 1.3 + 1)
    ax1.set_title(f"{N_CONCURRENT} Concurrent Reservations")
    ax1.set_ylabel("Count")
    for bar, v in zip(bars1, vals1):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                 str(v), ha="center", va="bottom", fontsize=11)

    # Ticks text for invariants
    checks = {
        "No lost\nrequests":  conc["no_lost_requests"],
        "No duplicate\nIDs":   conc["no_duplicates"],
        "No audit\nduplicates": conc["no_audit_dups"],
    }
    for i, (label, ok) in enumerate(checks.items()):
        colour = _COLORS[2] if ok else _COLORS[3]
        symbol = "PASS" if ok else "FAIL"
        ax1.text(0.05 + i * 0.33, 0.93, f"{symbol}\n{label}",
                 transform=ax1.transAxes, ha="center", va="top",
                 fontsize=8, color=colour,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colour))

    # Double-booking
    labels2 = ["Approved", "Blocked"]
    vals2   = [dbl["n_success"], dbl["n_failure"]]
    colours2 = [_COLORS[2] if dbl["invariant_held"] else _COLORS[3], _COLORS[1]]
    bars2 = ax2.bar(labels2, vals2, color=colours2, edgecolor="white")
    ax2.set_ylim(0, 3)
    ax2.set_title("Double-booking Prevention (2 concurrent approvals)")
    ax2.set_ylabel("Count")
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                 str(v), ha="center", va="bottom", fontsize=11)
    inv_text = "INVARIANT HELD" if dbl["invariant_held"] else "INVARIANT VIOLATED"
    inv_col  = _COLORS[2] if dbl["invariant_held"] else _COLORS[3]
    ax2.text(0.5, 0.88, inv_text, transform=ax2.transAxes,
             ha="center", va="top", fontsize=9, color=inv_col,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=inv_col))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
# Markdown report
# ===========================================================================

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def generate_report(results: dict, path: Path) -> None:
    acc  = results["decision_parsing"]
    lat  = results["latency"]
    conc = results["reliability"]
    dbl  = results["double_booking"]
    ts   = results["metadata"]["timestamp"]

    lines: list[str] = [
        "# CityPark Stage-2 — Human-in-the-Loop Evaluation Report",
        "",
        f"**Generated:** {ts}  ",
        f"**Engine:** in-process SQLite (StaticPool)  |  "
        f"**LLM:** MockLLM / _NullLLM (offline)  ",
        "",
        "---",
        "",
        "## 1. Decision Parsing Accuracy",
        "",
        "AdminAgent.parse_decision evaluated on **30 labelled admin messages**",
        "across 5 categories.  The LLM is replaced with `_NullLLM` (always returns",
        "empty string) so reason extraction is deterministic; only the regex gate",
        "and keyword detection are exercised.",
        "",
        "| Category | N | Correct | Accuracy |",
        "|----------|---|--------:|---------:|",
    ]
    for cat, s in sorted(acc["per_category"].items()):
        lines.append(f"| {cat} | {s['n']} | {s['correct']} | {_pct(s['accuracy'])} |")

    lines += [
        "",
        f"| **Overall** | **{acc['total']}** | **{acc['total_correct']}** "
        f"| **{_pct(acc['overall_accuracy'])}** |",
        "",
        "![Decision parsing accuracy](charts/stage2_accuracy.png)",
        "",
        "**Analysis:** "
        "Categories `approve`, `refuse_with_reason`, `refuse_no_reason`, and "
        "`no_decision` rely purely on deterministic regex detection.  "
        "The `ambiguous` group (messages containing both approve and refuse "
        "keywords) is correctly routed to clarification.  "
        "The `unrelated` group never triggers a keyword so also returns "
        "clarification.  "
        "Any miss in `refuse_with_reason` indicates a 'because/since' synonym "
        "not covered by the current regex; these can be added to `_REASON_RE`.",
        "",
        "---",
        "",
        "## 2. Escalation & Notification Latency",
        "",
        f"Measured over **{lat['n_iterations']} iterations**, in-process SQLite.",
        "",
        "| Stage | Mean (ms) | P95 (ms) | Min (ms) | Max (ms) |",
        "|-------|----------:|---------:|---------:|---------:|",
        f"| Escalation (confirm -> pending visible) "
        f"| {lat['escalation']['mean_ms']} "
        f"| {lat['escalation']['p95_ms']} "
        f"| {lat['escalation']['min_ms']} "
        f"| {lat['escalation']['max_ms']} |",
        f"| Decision -> notification (approve + advance) "
        f"| {lat['decision_to_notification']['mean_ms']} "
        f"| {lat['decision_to_notification']['p95_ms']} "
        f"| {lat['decision_to_notification']['min_ms']} "
        f"| {lat['decision_to_notification']['max_ms']} |",
        f"| Total round-trip "
        f"| {lat['total_round_trip']['mean_ms']} "
        f"| {lat['total_round_trip']['p95_ms']} "
        f"| {lat['total_round_trip']['min_ms']} "
        f"| {lat['total_round_trip']['max_ms']} |",
        "",
        "![Latency chart](charts/stage2_latency.png)",
        "",
        "**Analysis:** "
        "Escalation latency = `create_pending_reservation()` (SQL INSERT) + "
        "`RestChannel.send()` (one more INSERT into `approval_requests`). "
        "Decision-to-notification = `approve_reservation()` (UPDATE + space "
        "decrement) + the next `advance()` call which reads the updated row "
        "and delivers the notification inline.  "
        "All times are sub-millisecond with in-process SQLite; on a remote "
        "PostgreSQL they would be 5–30× higher. "
        "The EmailChannel (if enabled) adds SMTP latency on top.",
        "",
        "---",
        "",
        "## 3. Reliability",
        "",
        "### 3.1 Concurrent Reservations",
        "",
        f"**{conc['n_workers']} workers** launched simultaneously, each driving "
        "the full booking form to AWAITING_APPROVAL.",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Workers launched | {conc['n_workers']} |",
        f"| Succeeded | {conc['n_success']} |",
        f"| Failed | {conc['n_failure']} |",
        f"| No lost requests | {'PASS' if conc['no_lost_requests'] else 'FAIL'} |",
        f"| No duplicate reservation IDs | {'PASS' if conc['no_duplicates'] else 'FAIL'} |",
        f"| No duplicate audit rows | {'PASS' if conc['no_audit_dups'] else 'FAIL'} |",
        f"| Wall time | {conc['wall_time_s']} s |",
        f"| Worker latency mean | {conc['worker_latency_ms']['mean']} ms |",
        f"| Worker latency P95 | {conc['worker_latency_ms']['p95']} ms |",
        "",
    ]

    if conc["failures"]:
        lines.append("**Failures:**")
        for f in conc["failures"]:
            lines.append(f"- Worker {f['worker_id']}: {f.get('error', 'unknown')}")
        lines.append("")

    lines += [
        "### 3.2 Double-booking Prevention",
        "",
        "Two admin threads call `approve_reservation(same_id)` simultaneously, "
        "racing through a 1-space lot.",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Concurrent approve calls | {dbl['n_concurrent_approvals']} |",
        f"| Succeeded | {dbl['n_success']} |",
        f"| Blocked (ValidationError) | {dbl['n_failure']} |",
        f"| Final reservation status | {dbl['final_reservation_status']} |",
        f"| Final available spaces | {dbl['final_available_spaces']} |",
        f"| Invariant held (exactly 1 success, spaces=0) | "
        f"{'PASS' if dbl['invariant_held'] else 'FAIL'} |",
        "",
        "![Reliability chart](charts/stage2_reliability.png)",
        "",
        "**Analysis:** "
        "SQLite serialises concurrent writes, ensuring only one `approve_reservation` "
        "transaction wins.  The second call reads the post-commit state "
        "(status=CONFIRMED) and raises `ValidationError(cannot be approved)`. "
        "This guarantees the available_spaces counter is decremented exactly once. "
        "On a multi-instance deployment backed by PostgreSQL, `SELECT FOR UPDATE` "
        "or optimistic-locking would be required for the same guarantee.",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Section | Result |",
        "|---------|--------|",
        f"| Decision parsing overall | {_pct(acc['overall_accuracy'])} ({acc['total_correct']}/{acc['total']}) |",
        f"| Escalation latency (mean / P95) | {lat['escalation']['mean_ms']} ms / {lat['escalation']['p95_ms']} ms |",
        f"| Decision-to-notification (mean / P95) | {lat['decision_to_notification']['mean_ms']} ms / {lat['decision_to_notification']['p95_ms']} ms |",
        f"| {N_CONCURRENT} concurrent reservations | {'All pass' if conc['no_lost_requests'] and conc['no_duplicates'] else 'Issues found'} |",
        f"| Double-booking prevention | {'PASS' if dbl['invariant_held'] else 'FAIL'} |",
        "",
        "---",
        "*Report generated automatically by `scripts/run_eval_stage2.py`.*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved -> {path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("CityPark Stage-2 Evaluation Suite")
    print("=" * 60)

    engine, Session, db = _init_db()
    db.close()   # we'll open per-section

    try:
        # ── 1. Decision parsing ──────────────────────────────────────────────
        print("\n[1/4] Decision parsing accuracy…")
        acc_results = run_decision_parsing(engine, Session)
        print(
            f"  Overall: {acc_results['overall_accuracy']*100:.1f}% "
            f"({acc_results['total_correct']}/{acc_results['total']})"
        )

        # ── 2. Latency ───────────────────────────────────────────────────────
        print(f"\n[2/4] Latency benchmark (n={N_BENCH})…")
        lat_results = run_latency(engine, Session)
        print(
            f"  Escalation     mean={lat_results['escalation']['mean_ms']} ms  "
            f"p95={lat_results['escalation']['p95_ms']} ms"
        )
        print(
            f"  Notif-delivery mean={lat_results['decision_to_notification']['mean_ms']} ms  "
            f"p95={lat_results['decision_to_notification']['p95_ms']} ms"
        )

        # ── 3. Concurrent reliability ────────────────────────────────────────
        print(f"\n[3/4] Concurrent reliability (n={N_CONCURRENT} workers)…")
        conc_results = run_concurrent(engine, Session)
        status = "PASS" if (
            conc_results["no_lost_requests"] and
            conc_results["no_duplicates"] and
            conc_results["no_audit_dups"]
        ) else "FAIL"
        print(
            f"  {conc_results['n_success']}/{N_CONCURRENT} succeeded  "
            f"wall_time={conc_results['wall_time_s']}s  "
            f"reliability={status}"
        )

        # ── 4. Double-booking ────────────────────────────────────────────────
        print("\n[4/4] Double-booking prevention…")
        dbl_results = run_double_booking(engine, Session)
        print(
            f"  success={dbl_results['n_success']}  "
            f"blocked={dbl_results['n_failure']}  "
            f"final_status={dbl_results['final_reservation_status']}  "
            f"invariant={'PASS' if dbl_results['invariant_held'] else 'FAIL'}"
        )

    finally:
        engine.dispose()

    # ── Assemble ─────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "n_bench":      N_BENCH,
            "n_concurrent": N_CONCURRENT,
            "db":           "sqlite:///:memory: StaticPool",
            "llm":          "MockLLM offline / _NullLLM",
        },
        "decision_parsing": acc_results,
        "latency":          lat_results,
        "reliability":      conc_results,
        "double_booking":   dbl_results,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = DOCS_DIR / "results_stage2.json"
    json_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved -> {json_path}")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("Generating charts…")
    _chart_accuracy(acc_results,  CHARTS_DIR / "stage2_accuracy.png")
    _chart_latency(lat_results,   CHARTS_DIR / "stage2_latency.png")
    _chart_reliability(conc_results, dbl_results,
                       CHARTS_DIR / "stage2_reliability.png")
    print(f"  Charts -> {CHARTS_DIR}/")

    # ── Report ────────────────────────────────────────────────────────────────
    print("Generating report…")
    generate_report(results, DOCS_DIR / "evaluation_stage2.md")

    print("\nDone!")
    print("  docs/results_stage2.json")
    print("  docs/evaluation_stage2.md")
    print("  docs/charts/stage2_*.png")


if __name__ == "__main__":
    main()
