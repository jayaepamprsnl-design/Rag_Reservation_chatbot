#!/usr/bin/env python3
"""
Load test for the parking chatbot (asyncio + httpx; real processes, real measurements).

Starts its own throw-away stack - MCP server(s) + API (uvicorn, 1 worker) - on a fresh
SQLite DB / checkpoint / output file under a temp dir, drives it over HTTP, then verifies
the database and the output file.  Nothing in the repo's parking.db is touched.  The LLM is
the offline MockLLMClient (OPENAI_API_KEY is forced to a placeholder).

Scenarios
  1  Chatbot   : 50 concurrent users, each a dialogue of info questions + a full booking
  2a Admin     : 20 concurrent decisions (10 approve / 10 refuse) on scenario-1 bookings
  2b Admin     : capacity contention  - 12 concurrent approvals for a 5-space lot
  2c Admin     : duplicate-click race - each of 10 approvals fired twice at once
  3  MCP server: 100 concurrent writes (+ 100 concurrent re-sends = idempotency)
  3b MCP server: 100 concurrent writes against the DEFAULT rate limit (120 req/min)

Usage
  python load/load_test.py run --label before     # -> load/results_before.json
  python load/load_test.py report                 # -> docs/load_test_report.md + chart PNG
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import platform
import random
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOAD_DIR = ROOT / "load"
DOCS = ROOT / "docs"

API_PORT, MCP_PORT, MCP_BURST_PORT, MCP_DEFAULT_PORT = 8788, 8789, 8790, 8791

SEED = """
from datetime import time
from app.db.session import create_tables, SessionLocal
from app.db.models import ParkingLot
create_tables()
db = SessionLocal()
for name, total in (("LoadLot A", 1000), ("Tight Lot", 5)):
    db.add(ParkingLot(name=name, address="1 Load Test Way", total_spaces=total,
                      available_spaces=total, hourly_price=8.0, daily_price=40.0,
                      opens_at=time(0, 0), closes_at=time(23, 59)))
db.commit()
"""


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    label: str
    ms: float
    ok: bool
    status: int | None = None
    err: str | None = None


def pct(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    if not sorted_vals:
        return float("nan")
    return sorted_vals[max(0, math.ceil(q / 100 * len(sorted_vals)) - 1)]


def summarize(samples: list[Sample], wall_s: float) -> dict:
    ms = sorted(s.ms for s in samples)
    n = len(samples)
    errors = [s for s in samples if not s.ok]
    return {
        "n": n,
        "errors": len(errors),
        "error_rate_pct": round(100 * len(errors) / n, 2) if n else 0.0,
        "avg_ms": round(sum(ms) / n, 1) if n else 0.0,
        "p50_ms": round(pct(ms, 50), 1),
        "p95_ms": round(pct(ms, 95), 1),
        "p99_ms": round(pct(ms, 99), 1),
        "max_ms": round(ms[-1], 1) if ms else 0.0,
        "wall_s": round(wall_s, 2),
        "throughput_rps": round(n / wall_s, 2) if wall_s > 0 else 0.0,
        "statuses": dict(Counter(str(s.status) for s in samples)),
        "error_kinds": dict(Counter(s.err or str(s.status) for s in errors)),
        "latencies_ms": [round(x, 1) for x in ms],
    }


def by_label(samples: list[Sample]) -> dict[str, dict]:
    out = {}
    for lab in dict.fromkeys(s.label for s in samples):
        sub = [s for s in samples if s.label == lab]
        d = summarize(sub, 1.0)
        out[lab] = {k: d[k] for k in ("n", "errors", "avg_ms", "p95_ms", "p99_ms", "max_ms")}
    return out


# ---------------------------------------------------------------------------
# Stack management
# ---------------------------------------------------------------------------

procs: dict[str, subprocess.Popen] = {}
logs: list = []


def start(name: str, args: list[str], env: dict) -> None:
    log = open(TMP / f"{name}.log", "w")
    logs.append(log)
    procs[name] = subprocess.Popen(
        [sys.executable, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
    )


def stop(name: str) -> None:
    p = procs.pop(name, None)
    if p:
        p.terminate()
        with contextlib.suppress(Exception):
            p.wait(timeout=15)


def wait_http(url: str, secs: int = 180) -> None:
    end = time.time() + secs
    while time.time() < end:
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(url, timeout=2).status_code == 200:
                return
        time.sleep(0.5)
    raise SystemExit(f"timeout waiting for {url}")


def db_all(sql: str, params: tuple = ()) -> list[tuple]:
    with contextlib.closing(sqlite3.connect(DB_FILE)) as c:
        return c.execute(sql, params).fetchall()


def file_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines()[1:] if ln]


def expected_lines(where: str = "", params: tuple = ()) -> list[str]:
    """The lines the MCP server SHOULD hold for admin-approved rows (built by its own formatter)."""
    from app.mcp_server.server import ReservationRecord

    rows = db_all(
        "select customer_name, car_plate, start_time, end_time, decided_at, id from lot_reservations "
        "where status='confirmed' and decided_by is not null " + where, params)
    out = []
    for name, plate, st, en, dec, rid in rows:
        out.append(ReservationRecord(
            reservation_id=rid, customer_name=name, car_plate=plate,
            start_time=datetime.fromisoformat(st).isoformat(),
            end_time=datetime.fromisoformat(en).isoformat(),
            approved_at=datetime.fromisoformat(dec).isoformat()).to_line())
    return out


def compare_lines(expected: list[str], actual: list[str]) -> tuple[int, int, int]:
    """(lost, duplicated, unexpected) between the expected and the actual file content."""
    e, a = Counter(expected), Counter(actual)
    lost = sum((e - a).values())
    dup = sum(c - 1 for c in a.values() if c > 1)
    unexpected = sum((a - e).values()) - dup
    return lost, dup, max(unexpected, 0)


checks: list[dict] = []


def check(name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
# Scenario 1: chatbot users
# ---------------------------------------------------------------------------


async def timed(client: httpx.AsyncClient, samples: list[Sample], label: str, method: str,
                url: str, accept: tuple[int, ...] = (200,), **kw):
    t0 = time.perf_counter()
    try:
        r = await client.request(method, url, **kw)
    except Exception as exc:  # noqa: BLE001
        samples.append(Sample(label, (time.perf_counter() - t0) * 1000, False, None, type(exc).__name__))
        return None
    samples.append(Sample(label, (time.perf_counter() - t0) * 1000, r.status_code in accept, r.status_code))
    return r


async def chat_user(client, samples, i: int, lot: str, tag: str, think: bool, gate: asyncio.Event,
                    results: list[dict], day_offset: int = 2):
    """One customer: info questions (optional) then a full booking up to 'confirm'."""
    rnd = random.Random(1000 + i)
    sid = f"{tag}-{i:03d}-{uuid.uuid4().hex[:6]}"
    prefix = {"ld": "LD", "tight": "TG", "race": "RC", "warm": "WM"}[tag]
    name, plate = f"{prefix} User{i:03d}", f"{prefix}{i:04d}"
    # contention users all want the SAME window so that capacity (not time) decides
    hour = 10 if tag == "tight" else 6 + (i % 12)
    start_dt = (datetime.now() + timedelta(days=day_offset)).replace(
        hour=hour, minute=0, second=0, microsecond=0)
    end_dt = start_dt + timedelta(hours=2)
    steps = []
    if think:
        steps += [("info: price (SQL)", "How much does parking cost?"),
                  ("info: location (RAG)", "Where is the Airport lot located?")]
    steps += [("book: start", "I want to book a parking space"), ("book: name", name),
              ("book: plate", plate), ("book: lot", lot),
              ("book: arrival", start_dt.strftime("%Y-%m-%d %H:%M")),
              ("book: departure", end_dt.strftime("%Y-%m-%d %H:%M")), ("book: confirm", "confirm")]
    await gate.wait()
    t_user = time.perf_counter()
    rid, functional_ok, last = None, True, None
    for label, msg in steps:
        r = await timed(client, samples, label, "POST", "/chat", json={"message": msg, "session_id": sid})
        if r is None or r.status_code != 200:
            functional_ok = False
            break
        last = r.json()
        if think:
            await asyncio.sleep(rnd.uniform(0.05, 0.25))
    if functional_ok:
        rid = last.get("reservation_id")
        functional_ok = bool(rid) and "administrator" in last.get("answer", "")
    results.append({"i": i, "sid": sid, "name": name, "plate": plate, "rid": rid, "ok": functional_ok,
                    "dialogue_ms": (time.perf_counter() - t_user) * 1000})


async def run_users(client, n: int, lot: str, tag: str, think: bool, samples: list[Sample]) -> tuple[list[dict], float]:
    gate, results = asyncio.Event(), []
    tasks = [asyncio.create_task(chat_user(client, samples, i, lot, tag, think, gate, results)) for i in range(n)]
    await asyncio.sleep(0.2)
    t0 = time.perf_counter()
    gate.set()
    await asyncio.gather(*tasks)
    return results, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# MCP direct writes
# ---------------------------------------------------------------------------


def _leaf(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return ",".join(_leaf(e) for e in exc.exceptions)
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP{exc.response.status_code}"
    return type(exc).__name__


def _cheap_http_client(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
    """MCP SDK client factory without the per-instance TLS-context build.

    httpx.AsyncClient() loads the CA bundle every time (~470 ms of CPU on the test machine,
    ~1 ms with verify=False).  With one load-generator process that cost serializes and would
    be measured instead of the server.  The target is plain http://127.0.0.1, so TLS is unused.
    """
    return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth, follow_redirects=True, verify=False)


async def mcp_write(url: str, key: str, payload: dict, samples: list[Sample], label: str,
                    gate: asyncio.Event, statuses: list[str], default_client: bool = False):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    kw = {} if default_client else {"httpx_client_factory": _cheap_http_client}

    await gate.wait()
    t0 = time.perf_counter()
    try:
        async def go():
            async with streamablehttp_client(url, headers={"Authorization": f"Bearer {key}"}, timeout=30, **kw) as (r, w, _):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    return await s.call_tool("write_reservation", payload)
        res = await asyncio.wait_for(go(), timeout=60)
        status = "error" if res.isError else json.loads(res.content[0].text).get("status", "?")
        statuses.append(status)
        samples.append(Sample(label, (time.perf_counter() - t0) * 1000, status in ("written", "duplicate"), 200, None if status != "error" else "tool_error"))
    except BaseException as exc:  # noqa: BLE001 - incl. anyio ExceptionGroup
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        statuses.append("exception")
        samples.append(Sample(label, (time.perf_counter() - t0) * 1000, False, None, _leaf(exc)))


def mcp_payload(i: int) -> dict:
    s = datetime(2027, 3, 1, 8, 0) + timedelta(hours=i)
    return {"reservation_id": 500_000 + i, "customer_name": f"Mcp User{i:03d}", "car_plate": f"MC{i:04d}",
            "start_time": s.isoformat(), "end_time": (s + timedelta(hours=2)).isoformat(),
            "approved_at": datetime(2027, 2, 1, 12, 0, i % 60).isoformat()}


async def burst(url, key, payloads, samples, label, default_client: bool = False) -> tuple[float, list[str]]:
    gate, statuses = asyncio.Event(), []
    tasks = [asyncio.create_task(mcp_write(url, key, p, samples, label, gate, statuses, default_client))
             for p in payloads]
    await asyncio.sleep(0.2)
    t0 = time.perf_counter()
    gate.set()
    await asyncio.gather(*tasks)
    return time.perf_counter() - t0, statuses


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


async def run_all(args) -> dict:
    admin_key, mcp_key = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    base = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{DB_FILE.as_posix()}",
        "CHECKPOINT_DB": str(TMP / "checkpoints.db"),
        "OPENAI_API_KEY": "sk-placeholder",          # forces the offline MockLLMClient
        "ADMIN_API_KEY": admin_key,
        "APPROVAL_EXPIRY_ENABLED": "false",
        "MCP_API_KEY": mcp_key,
        "MCP_SERVER_URL": f"http://127.0.0.1:{MCP_PORT}/mcp",
        "MCP_CLIENT_TIMEOUT": "5",
        "MCP_RETRY_INTERVAL_SECONDS": "3",
        "ANONYMIZED_TELEMETRY": "false",
    }
    subprocess.run([sys.executable, "-c", SEED], cwd=ROOT, env=base, check=True, capture_output=True)

    def mcp_env(port, out, rate):
        return {**base, "MCP_HOST": "127.0.0.1", "MCP_PORT": str(port), "RESERVATIONS_FILE": str(out),
                "MCP_RATE_LIMIT_PER_MINUTE": str(rate)}

    out_api, out_burst, out_default = TMP / "api_out.txt", TMP / "burst_out.txt", TMP / "default_out.txt"
    start("mcp", ["scripts/run_mcp_server.py"], mcp_env(MCP_PORT, out_api, 1_000_000))
    start("mcp_burst", ["scripts/run_mcp_server.py"], mcp_env(MCP_BURST_PORT, out_burst, 1_000_000))
    start("mcp_default", ["scripts/run_mcp_server.py"], mcp_env(MCP_DEFAULT_PORT, out_default, 120))
    for p in (MCP_PORT, MCP_BURST_PORT, MCP_DEFAULT_PORT):
        wait_http(f"http://127.0.0.1:{p}/health")
    start("api", ["-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)], base)
    api = f"http://127.0.0.1:{API_PORT}"
    wait_http(f"{api}/health")

    H = {"X-Admin-Key": admin_key}
    limits = httpx.Limits(max_connections=300, max_keepalive_connections=300)
    result: dict = {"scenarios": {}}
    async with httpx.AsyncClient(base_url=api, timeout=180, limits=limits) as client:
        # ---- warm-up (reported separately as cold start; excluded from the measurements)
        cold = {}
        for label, msg in (("dynamic", "How much does parking cost?"), ("static_rag", "Where is the Airport lot located?")):
            t0 = time.perf_counter()
            r = await client.post("/chat", json={"message": msg, "session_id": "warmup"})
            cold[label] = {"ms": round((time.perf_counter() - t0) * 1000, 1), "status": r.status_code,
                           "answer_head": r.json().get("answer", "")[:70] if r.status_code == 200 else ""}
        await client.get("/admin/verify", headers=H)
        # one complete booking dialogue to prime the booking path (date parser, slot extractor, ...)
        warm_samples: list[Sample] = []
        await run_users(client, 1, "1", "warm", False, warm_samples)
        cold["booking_steps"] = {s_.label: round(s_.ms, 1) for s_ in warm_samples}
        result["cold_start"] = cold
        print("cold start:", {k: v["ms"] for k, v in cold.items() if k != "booking_steps"})
        print("cold booking steps (ms):", cold["booking_steps"])

        # ================= Scenario 1: 50 chat users =================
        print(f"\n[1] Chatbot: {args.users} concurrent users")
        s1: list[Sample] = []
        users, wall1 = await run_users(client, args.users, "1", "ld", True, s1)
        result["scenarios"]["chat"] = {**summarize(s1, wall1), "by_step": by_label(s1)}
        dlg = sorted(u["dialogue_ms"] for u in users)
        result["scenarios"]["chat"]["dialogue"] = {
            "users": len(users), "completed": sum(u["ok"] for u in users),
            "avg_s": round(sum(dlg) / len(dlg) / 1000, 2), "p95_s": round(pct(dlg, 95) / 1000, 2),
            "max_s": round(dlg[-1] / 1000, 2)}
        print("  ", {k: result["scenarios"]["chat"][k] for k in ("n", "avg_ms", "p95_ms", "p99_ms", "throughput_rps", "error_rate_pct")})
        done = [u for u in users if u["ok"]]
        check("S1 every user completed the dialogue", len(done) == args.users, f"{len(done)}/{args.users}")
        rows = db_all("select id, session_id, car_plate, status from lot_reservations where session_id like 'ld-%'")
        check("S1 no lost / duplicated bookings in the DB",
              len(rows) == args.users and len({r[0] for r in rows}) == args.users
              and len({r[1] for r in rows}) == args.users and all(r[3].lower() == "pending" for r in rows),
              f"{len(rows)} rows, {len({r[1] for r in rows})} distinct sessions, "
              f"statuses={dict(Counter(r[3] for r in rows))}")
        plate_by_sid = {r[1]: r[2] for r in rows}
        check("S1 no cross-talk between sessions",
              all(plate_by_sid.get(u["sid"]) == u["plate"] for u in done), "each session's plate matches its user")
        paused = 0
        for u in done:
            g = (await client.get(f"/graph/{u['sid']}/state", headers=H)).json()
            paused += bool(g.get("interrupted")) and g.get("reservation_id") == u["rid"]
        check("S1 every booking is paused at admin_approval in the graph", paused == len(done), f"{paused}/{len(done)}")

        # ================= Scenario 2a: 20 concurrent decisions =================
        print("\n[2a] Admin: 20 concurrent approvals/refusals")
        s2: list[Sample] = []
        decide = sorted(done, key=lambda u: u["i"])[:20]
        gate = asyncio.Event()

        async def decision(u, k):
            await gate.wait()
            if k % 2 == 0:
                return await timed(client, s2, "approve", "POST", f"/admin/reservations/{u['rid']}/approve", headers=H, json={})
            return await timed(client, s2, "refuse", "POST", f"/admin/reservations/{u['rid']}/refuse", headers=H,
                               json={"reason": "load test refusal"})

        tasks = [asyncio.create_task(decision(u, k)) for k, u in enumerate(decide)]
        await asyncio.sleep(0.2)
        t0 = time.perf_counter()
        gate.set()
        await asyncio.gather(*tasks)
        wall2 = time.perf_counter() - t0
        result["scenarios"]["admin"] = {**summarize(s2, wall2), "by_step": by_label(s2)}
        print("  ", {k: result["scenarios"]["admin"][k] for k in ("n", "avg_ms", "p95_ms", "p99_ms", "throughput_rps", "error_rate_pct")})
        ids_ok = [u["rid"] for k, u in enumerate(decide) if k % 2 == 0]
        ids_ref = [u["rid"] for k, u in enumerate(decide) if k % 2 == 1]
        st = {r: s for r, s in db_all("select id, status from lot_reservations where id in (%s)" % ",".join(map(str, ids_ok + ids_ref)))}
        check("S2a statuses match the decisions", all(st[i] == "confirmed" for i in ids_ok) and all(st[i] == "refused" for i in ids_ref),
              f"{sum(st[i] == 'confirmed' for i in ids_ok)}/10 confirmed, {sum(st[i] == 'refused' for i in ids_ref)}/10 refused")
        fw = db_all("select count(*) from lot_reservations where id in (%s) and file_written=1" % ",".join(map(str, ids_ok)))[0][0]
        check("S2a every approval was written to the file (file_written)", fw == len(ids_ok), f"{fw}/{len(ids_ok)}")
        avail = db_all("select available_spaces from parking_lots where id=1")[0][0]
        check("S2a lot 1 space accounting", avail == 1000 - len(ids_ok), f"available={avail}, expected {1000 - len(ids_ok)}")
        lost, dup, unexp = compare_lines(expected_lines(), file_lines(out_api))
        check("S2a output file: no lost / duplicated / unexpected lines", (lost, dup, unexp) == (0, 0, 0),
              f"lost={lost} duplicated={dup} unexpected={unexp}, file has {len(file_lines(out_api))} lines")
        settled = 0
        for i in ids_ok + ids_ref:
            sid = next(u["sid"] for u in decide if u["rid"] == i)
            g = (await client.get(f"/graph/{sid}/state", headers=H)).json()
            settled += (g.get("next") == [] and g.get("status") in ("approved", "refused"))
        check("S2a every graph thread ran to completion", settled == 20, f"{settled}/20")

        # ================= Scenario 2b: capacity contention =================
        print("\n[2b] Admin: 12 concurrent approvals for a 5-space lot")
        sink: list[Sample] = []
        tight, _ = await run_users(client, 12, "2", "tight", False, sink)
        tight_ids = [u["rid"] for u in tight if u["ok"]]
        s2b: list[Sample] = []
        gate = asyncio.Event()

        async def approve_one(rid, out):
            await gate.wait()
            out.append(await timed(client, s2b, "approve", "POST", f"/admin/reservations/{rid}/approve",
                                   accept=(200, 409, 422), headers=H, json={}))

        outs: list = []
        tasks = [asyncio.create_task(approve_one(r, outs)) for r in tight_ids]
        await asyncio.sleep(0.2)
        gate.set()
        await asyncio.gather(*tasks)
        conf = db_all("select count(*) from lot_reservations where lot_id=2 and status='confirmed'")[0][0]
        avail2 = db_all("select available_spaces from parking_lots where id=2")[0][0]
        codes = dict(Counter(str(o.status_code) for o in outs if o is not None))
        result["scenarios"]["contention"] = {"requests": len(tight_ids), "http": codes, "confirmed": conf,
                                             "capacity": 5, "available_after": avail2}
        check("S2b no overbooking (confirmed <= capacity)", conf <= 5, f"confirmed={conf}, capacity=5, http={codes}")
        check("S2b space counter consistent with confirmed rows", avail2 == 5 - conf and avail2 >= 0,
              f"available={avail2}, expected {5 - conf}")
        lost, dup, unexp = compare_lines(expected_lines("and lot_id=2"),
                                         [ln for ln in file_lines(out_api) if " | TG" in ln])
        check("S2b file lines match confirmed rows", (lost, dup, unexp) == (0, 0, 0), f"lost={lost} duplicated={dup} unexpected={unexp}")

        # ================= Scenario 2c: duplicate-click race =================
        print("\n[2c] Admin: each of 10 approvals fired twice at the same instant")
        sink = []
        racers, _ = await run_users(client, 10, "1", "race", False, sink)
        race_ids = [u["rid"] for u in racers if u["ok"]]
        before_avail = db_all("select available_spaces from parking_lots where id=1")[0][0]
        s2c: list[Sample] = []
        gate = asyncio.Event()
        outs = []

        async def approve_dup(rid):
            await gate.wait()
            r = await timed(client, s2c, "approve (duplicate click)", "POST", f"/admin/reservations/{rid}/approve",
                            accept=(200, 409), headers=H, json={})
            outs.append((rid, r.status_code if r is not None else None))

        tasks = [asyncio.create_task(approve_dup(r)) for r in race_ids for _ in range(2)]
        await asyncio.sleep(0.2)
        gate.set()
        await asyncio.gather(*tasks)
        per_id = {r: sorted(c for rr, c in outs if rr == r) for r in race_ids}
        double_ok = [r for r, c in per_id.items() if c == [200, 200]]
        after_avail = db_all("select available_spaces from parking_lots where id=1")[0][0]
        result["scenarios"]["duplicate_click"] = {"pairs": len(race_ids), "both_200": len(double_ok),
                                                  "decrement": before_avail - after_avail}
        check("S2c each duplicated approve has exactly one 200 + one 409", not double_ok,
              f"{len(double_ok)}/{len(race_ids)} pairs returned 200 twice")
        check("S2c lot 1 decremented exactly once per reservation", before_avail - after_avail == len(race_ids),
              f"decrement={before_avail - after_avail}, expected {len(race_ids)}")
        await asyncio.sleep(1.0)
        exp_all = expected_lines()
        lost, dup, unexp = compare_lines(exp_all, file_lines(out_api))
        check("S2 output file after all admin scenarios: no lost / duplicated lines",
              (lost, dup) == (0, 0), f"lost={lost} duplicated={dup} unexpected={unexp}; "
              f"file={len(file_lines(out_api))} lines vs {len(exp_all)} CONFIRMED rows")

    # ================= Scenario 3: MCP direct writes =================
    print(f"\n[3] MCP server: {args.writes} concurrent writes")
    payloads = [mcp_payload(i) for i in range(args.writes)]
    s3: list[Sample] = []
    wall3, st3 = await burst(f"http://127.0.0.1:{MCP_BURST_PORT}/mcp", mcp_key, payloads, s3, "write_reservation")
    result["scenarios"]["mcp"] = {**summarize(s3, wall3), "tool_status": dict(Counter(st3))}
    print("  ", {k: result["scenarios"]["mcp"][k] for k in ("n", "avg_ms", "p95_ms", "p99_ms", "throughput_rps", "error_rate_pct")})
    from app.mcp_server.server import ReservationRecord
    exp3 = [ReservationRecord(**p).to_line() for p in payloads]
    lost, dup, unexp = compare_lines(exp3, file_lines(out_burst))
    check("S3 100 concurrent writes: none lost, none duplicated", (lost, dup, unexp) == (0, 0, 0) and len(file_lines(out_burst)) == args.writes,
          f"lines={len(file_lines(out_burst))}/{args.writes}, lost={lost} duplicated={dup} unexpected={unexp}, tool={dict(Counter(st3))}")
    ids_n = len((TMP / "burst_out.txt.ids").read_text().splitlines()) if (TMP / "burst_out.txt.ids").exists() else 0
    check("S3 idempotency ledger (.ids) has one entry per write", ids_n == args.writes, f"{ids_n} entries")
    s3b: list[Sample] = []
    wall3b, st3b = await burst(f"http://127.0.0.1:{MCP_BURST_PORT}/mcp", mcp_key, payloads, s3b, "re-send")
    result["scenarios"]["mcp_resend"] = {**summarize(s3b, wall3b), "tool_status": dict(Counter(st3b))}
    check("S3 re-sending all 100 concurrently adds nothing (idempotent)",
          len(file_lines(out_burst)) == args.writes and Counter(st3b).get("duplicate", 0) == args.writes,
          f"lines={len(file_lines(out_burst))}, tool={dict(Counter(st3b))}")

    print("\n[3b] MCP server with the DEFAULT rate limit (120 req/min): same 100 writes")
    s3c: list[Sample] = []
    wall3c, st3c = await burst(f"http://127.0.0.1:{MCP_DEFAULT_PORT}/mcp", mcp_key, payloads, s3c, "write_reservation")
    result["scenarios"]["mcp_default_limit"] = {**summarize(s3c, wall3c), "tool_status": dict(Counter(st3c))}
    lines_c = file_lines(out_default)
    ok_c = sum(s.ok for s in s3c)
    check("S3b rate-limited run: file holds exactly the writes that were acknowledged, no duplicates",
          len(lines_c) == len(set(lines_c)) == ok_c, f"acknowledged={ok_c}, lines={len(lines_c)}, "
          f"rejected={len(s3c) - ok_c} ({dict(Counter(s.err for s in s3c if not s.ok))})")

    print("\n[3c] MCP client cost: 10 concurrent writes with the SDK's DEFAULT client (what app/agents/mcp_client.py uses)")
    s3d: list[Sample] = []
    extra = [mcp_payload(10_000 + i) for i in range(10)]
    wall3d, st3d = await burst(f"http://127.0.0.1:{MCP_BURST_PORT}/mcp", mcp_key, extra, s3d, "write (default client)",
                               default_client=True)
    result["scenarios"]["mcp_default_client"] = {**summarize(s3d, wall3d), "tool_status": dict(Counter(st3d))}
    print("  ", {k: result["scenarios"]["mcp_default_client"][k] for k in ("n", "avg_ms", "p95_ms", "throughput_rps", "error_rate_pct")})

    stop("api")
    return result


def main_run(args) -> None:
    global TMP, DB_FILE
    TMP = Path(tempfile.mkdtemp(prefix="load_test_"))
    DB_FILE = TMP / "parking.db"
    t_start = time.time()
    try:
        res = asyncio.run(run_all(args))
    finally:
        for n in list(procs):
            stop(n)
        for log in logs:
            log.close()
        for _ in range(10):
            shutil.rmtree(TMP, ignore_errors=True)
            if not TMP.exists():
                break
            time.sleep(0.5)
    import fastapi
    import uvicorn
    from importlib.metadata import version

    res.update({
        "label": args.label, "checks": checks, "finished_at": datetime.now().isoformat(timespec="seconds"),
        "total_s": round(time.time() - t_start, 1),
        "env": {"os": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count(),
                "fastapi": fastapi.__version__, "uvicorn": uvicorn.__version__, "langgraph": version("langgraph"),
                "mcp": version("mcp"), "api": "1 uvicorn worker, SQLite (file), SQLite checkpointer, MockLLM",
                "users": args.users, "writes": args.writes},
    })
    out = LOAD_DIR / f"results_{args.label}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nsaved {out}")
    failed = [c for c in checks if not c["pass"]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} verification checks passed")
    for c in failed:
        print(f"  FAILED: {c['name']} - {c['detail']}")


# ---------------------------------------------------------------------------
# Report + chart
# ---------------------------------------------------------------------------


def make_chart(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = res["scenarios"]
    groups = [("Chat turn\n(50 users)", sc["chat"]), ("Admin decision\n(20 concurrent)", sc["admin"]),
              ("MCP write\n(100 concurrent)", sc["mcp"])]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0][0]
    x = range(len(groups))
    for k, (key, col) in enumerate((("avg_ms", "#4c78a8"), ("p95_ms", "#f58518"), ("p99_ms", "#e45756"))):
        vals = [g[1][key] for g in groups]
        bars = ax.bar([i + (k - 1) * 0.26 for i in x], vals, 0.26, label=key.replace("_ms", ""), color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("latency (ms, log scale)"); ax.set_title("Latency: avg / p95 / p99"); ax.legend()

    ax = axes[0][1]
    names = [g[0].replace("\n", " ") for g in groups]
    tp = [g[1]["throughput_rps"] for g in groups]
    bars = ax.bar(names, tp, color=["#4c78a8", "#f58518", "#54a24b"])
    for b, v in zip(bars, tp):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("requests / second"); ax.set_title("Throughput (completed requests / wall time)")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1][0]
    steps = list(sc["chat"]["by_step"].items())
    ypos = range(len(steps))
    ax.barh([i + 0.2 for i in ypos], [s[1]["avg_ms"] for s in steps], 0.4, label="avg", color="#4c78a8")
    ax.barh([i - 0.2 for i in ypos], [s[1]["p95_ms"] for s in steps], 0.4, label="p95", color="#f58518")
    ax.set_yticks(list(ypos)); ax.set_yticklabels([s[0] for s in steps], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("ms"); ax.set_title("Chat latency by dialogue step (50 concurrent users)"); ax.legend()

    ax = axes[1][1]
    for (lab, g), col in zip(groups, ("#4c78a8", "#f58518", "#54a24b")):
        v = g["latencies_ms"]
        ax.plot(v, [(i + 1) / len(v) * 100 for i in range(len(v))], label=lab.replace("\n", " "), color=col)
    ax.set_xscale("log"); ax.set_xlabel("latency (ms, log scale)"); ax.set_ylabel("% of requests <= x")
    ax.set_title("Latency distribution (ECDF)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"CityPark load test ({res['label']}) - {res['env']['users']} users / 20 decisions / "
                 f"{res['env']['writes']} MCP writes", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def row(name: str, d: dict) -> str:
    return (f"| {name} | {d['n']} | {d['avg_ms']:.0f} | {d['p50_ms']:.0f} | {d['p95_ms']:.0f} | {d['p99_ms']:.0f} | "
            f"{d['max_ms']:.0f} | {d['throughput_rps']:.1f} | {d['error_rate_pct']:.1f}% |")


HEAD = ("| Scenario | Requests | avg ms | p50 ms | p95 ms | p99 ms | max ms | Throughput (req/s) | Error rate |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|")


def main_report(args) -> None:
    labels = [p.stem.removeprefix("results_") for p in sorted(LOAD_DIR.glob("results_*.json"))]
    if not labels:
        raise SystemExit("no results_*.json found; run `run` first")
    final = "after" if "after" in labels else labels[-1]
    data = {lab: json.loads((LOAD_DIR / f"results_{lab}.json").read_text(encoding="utf-8")) for lab in labels}
    res = data[final]
    sc, env = res["scenarios"], res["env"]
    chart = DOCS / "load_test_chart.png"
    make_chart(res, chart)

    md = ["# Load test report", "",
          f"Measured {res['finished_at']} - run label **{final}** (all numbers below are copied from "
          f"`load/results_{final}.json`, produced by `load/load_test.py`; nothing is estimated).", "",
          "## Setup", "",
          f"- Host: {env['os']}, {env['cpus']} logical CPUs, Python {env['python']}",
          f"- API: FastAPI {env['fastapi']} / uvicorn {env['uvicorn']}, **{env['api']}**; LangGraph {env['langgraph']}; MCP SDK {env['mcp']}",
          "- Client: `asyncio` + `httpx` in one process on the same machine (client and server compete for CPU), "
          "real HTTP over loopback, fresh temp DB / checkpoint file / output file per run",
          "- LLM: the offline `MockLLMClient` (`OPENAI_API_KEY` forced to a placeholder), so LLM latency is **not** included; "
          "static questions use the real Chroma store + embedding model",
          "- Why not Locust: it installs on this machine, so this is a fit decision, not a Windows failure. Scenario 3 needs the "
          "async MCP client and exact counts (100 simultaneous writes, 20 simultaneous decisions), and every scenario is "
          "followed by DB / file verification, which is simpler in one asyncio script",
          "- Percentiles are nearest-rank over every request in the scenario; throughput = completed requests / wall time "
          "from the moment all clients are released; error rate = non-expected HTTP status, exceptions and tool errors", "",
          "## Results", "", HEAD,
          row("1. Chatbot - all turns (50 concurrent users)", sc["chat"]),
          row("2a. Admin - approve/refuse (20 concurrent)", sc["admin"]),
          row("3. MCP server - 100 concurrent writes", sc["mcp"]),
          row("3. MCP server - 100 concurrent re-sends (idempotent)", sc["mcp_resend"]),
          row("3b. MCP server - 100 writes, default rate limit", sc["mcp_default_limit"]),
          row("3c. MCP client cost - 10 writes, SDK default client", sc["mcp_default_client"]), ""]
    d = sc["chat"]["dialogue"]
    md += [f"Cold start (first request after boot, excluded above): dynamic question "
           f"{res['cold_start']['dynamic']['ms']:.0f} ms, static RAG question {res['cold_start']['static_rag']['ms']:.0f} ms; "
           f"first booking dialogue, step by step (ms): {res['cold_start']['booking_steps']}.", "",
           f"Whole dialogue (9 turns incl. think time 50-250 ms/turn): {d['completed']}/{d['users']} users completed, "
           f"avg {d['avg_s']} s, p95 {d['p95_s']} s, max {d['max_s']} s.", "",
           "### Scenario 1 - latency by dialogue step", "", "| Step | Requests | avg ms | p95 ms | p99 ms | max ms | Errors |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in sc["chat"]["by_step"].items():
        md.append(f"| {k} | {v['n']} | {v['avg_ms']:.0f} | {v['p95_ms']:.0f} | {v['p99_ms']:.0f} | {v['max_ms']:.0f} | {v['errors']} |")
    md += ["", "### Scenario 2 - by decision", "", "| Decision | Requests | avg ms | p95 ms | p99 ms | max ms | Errors |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in sc["admin"]["by_step"].items():
        md.append(f"| {k} | {v['n']} | {v['avg_ms']:.0f} | {v['p95_ms']:.0f} | {v['p99_ms']:.0f} | {v['max_ms']:.0f} | {v['errors']} |")
    md += ["", f"HTTP statuses: chat {sc['chat']['statuses']}, admin {sc['admin']['statuses']}, MCP tool results "
           f"{sc['mcp']['tool_status']}, re-send {sc['mcp_resend']['tool_status']}.", "",
           "### Scenario 3b - default MCP rate limit", "",
           f"With `MCP_RATE_LIMIT_PER_MINUTE` left at its default (120 requests/min per client address), the same 100 concurrent "
           f"writes gave {sc['mcp_default_limit']['n'] - sc['mcp_default_limit']['errors']} acknowledged and "
           f"{sc['mcp_default_limit']['errors']} rejected (`{sc['mcp_default_limit']['error_kinds']}`). Each write is "
           f"several HTTP requests (initialize + call), so the limit is reached after a few dozen writes. The numbers in the main "
           f"table use a raised limit so that throughput of the writer itself is measured.", "",
           "![load test chart](load_test_chart.png)", "", "## Correctness checks", "",
           "| Result | Check | Detail |", "|---|---|---|"]
    for c in res["checks"]:
        md.append(f"| {'PASS' if c['pass'] else '**FAIL**'} | {c['name']} | {c['detail']} |")
    if len(labels) > 1:
        md += ["", "## Before / after the fix", "", "| Check | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for c in data[labels[0]]["checks"]:
            cells = []
            for lab in labels:
                m = next((x for x in data[lab]["checks"] if x["name"] == c["name"]), None)
                cells.append("-" if m is None else f"{'PASS' if m['pass'] else '**FAIL**'} - {m['detail']}")
            md.append(f"| {c['name']} | " + " | ".join(cells) + " |")
        md += ["", "| Admin decision latency | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
        for key in ("avg_ms", "p95_ms", "p99_ms", "throughput_rps"):
            md.append(f"| {key} | " + " | ".join(str(data[lab]['scenarios']['admin'][key]) for lab in labels) + " |")
    findings = LOAD_DIR / "findings.md"
    if findings.exists():
        md += ["", findings.read_text(encoding="utf-8").strip()]
    (DOCS / "load_test_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {DOCS / 'load_test_report.md'} and {chart}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", default="run")
    r.add_argument("--users", type=int, default=50)
    r.add_argument("--writes", type=int, default=100)
    sub.add_parser("report")
    a = ap.parse_args()
    main_run(a) if a.cmd == "run" else main_report(a)
