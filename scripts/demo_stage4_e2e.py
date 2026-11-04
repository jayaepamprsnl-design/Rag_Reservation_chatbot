#!/usr/bin/env python3
"""
Stage-4 end-to-end demo: the LangGraph orchestration across real processes.

  A  book via POST /chat -> graph PAUSES at admin_approval -> admin approves over REST
     -> record (MCP file line) -> notify
  B  book, then RESTART THE API while the approval is pending -> the paused graph is
     restored from the SQLite checkpoint -> admin approves -> record -> notify
  C  book, admin REFUSES -> nothing is written, customer gets the reason
  D  second approve of the same request -> HTTP 409, still exactly one line
  E  MCP server DOWN at approval -> approval stands, file_written=False, queued;
     server back -> the retry worker writes the line

Uses a throw-away DB copy, checkpoint file and output file (deleted afterwards unless
--keep).  Needs ADMIN_API_KEY in .env (never printed).  Ports 8798 (API) / 8799 (MCP).

Usage:  python scripts/demo_stage4_e2e.py [--keep]
"""
import contextlib
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402  (reads .env for ADMIN_API_KEY)

KEEP = "--keep" in sys.argv
API_PORT, MCP_PORT = 8798, 8799
API = f"http://127.0.0.1:{API_PORT}"
SEP = "=" * 74
tmp = Path(tempfile.mkdtemp(prefix="stage4_e2e_"))
out_file = tmp / "out" / "confirmed_reservations.txt"
db_copy = tmp / "parking.db"
cp_file = tmp / "checkpoints.db"
shutil.copy(ROOT / "parking.db", db_copy)

env = {
    **os.environ,
    "DATABASE_URL": f"sqlite:///{db_copy.as_posix()}",
    "CHECKPOINT_DB": str(cp_file),
    "MCP_API_KEY": secrets.token_urlsafe(24),
    "MCP_HOST": "127.0.0.1",
    "MCP_PORT": str(MCP_PORT),
    "RESERVATIONS_FILE": str(out_file),
    "MCP_SERVER_URL": f"http://127.0.0.1:{MCP_PORT}/mcp",
    "MCP_CLIENT_TIMEOUT": "2",
    "MCP_CLIENT_BACKOFF_SECONDS": "0.2",
    "MCP_RETRY_INTERVAL_SECONDS": "3",
    "ANONYMIZED_TELEMETRY": "false",
}
procs: dict[str, subprocess.Popen] = {}
logs: list = []
H = {"X-Admin-Key": settings.admin_api_key}


def start(name, args):
    log = open(tmp / f"{name}-{len(logs)}.log", "w")
    logs.append(log)
    procs[name] = subprocess.Popen(
        [sys.executable, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def stop(name):
    p = procs.pop(name, None)
    if p:
        p.terminate()
        p.wait(timeout=15)


def wait_http(url, secs=120):
    end = time.time() + secs
    while time.time() < end:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"timeout waiting for {url}")


def start_api():
    start("api", ["-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)])
    wait_http(f"{API}/health")


def db_row(rid):
    with contextlib.closing(sqlite3.connect(db_copy)) as c:
        return c.execute(
            "select status, decided_by is not null, file_written from lot_reservations where id=?", (rid,)
        ).fetchone()


def gstate(sid):
    g = httpx.get(f"{API}/graph/{sid}/state", headers=H, timeout=10).json()
    print(f"  graph> next={g['next']} interrupted={g['interrupted']} status={g['status']} "
          f"file_written={g['file_written']} errors={g['errors']}")
    return g


def show_file(title):
    print(f"  --- {title}\n" + "".join(
        f"      {ln}\n" for ln in (out_file.read_text(encoding='utf-8').splitlines() if out_file.exists()
                                  else ["(file does not exist)"])).rstrip())


def book(sid, name, plate, days_ahead):
    start_dt = (datetime.now() + timedelta(days=days_ahead)).replace(hour=10, minute=0, second=0, microsecond=0)
    end_dt = start_dt.replace(hour=14)
    for msg in ["I want to book a parking space", name, plate, "1",
                start_dt.strftime("%Y-%m-%d %H:%M"), end_dt.strftime("%Y-%m-%d %H:%M"), "confirm"]:
        r = httpx.post(f"{API}/chat", json={"message": msg, "session_id": sid}, timeout=60).json()
    print(f"  chat> {r['answer'][:100].replace(chr(10), ' ')}...")
    return r["reservation_id"]


def notifications(sid):
    n = httpx.get(f"{API}/sessions/{sid}/notifications", timeout=10).json()
    for x in n["notifications"]:
        print(f"  notify> {x['message']}")


def approve(rid):
    r = httpx.post(f"{API}/admin/reservations/{rid}/approve", headers=H, json={}, timeout=60)
    body = r.json()
    print(f"  POST approve -> HTTP {r.status_code}  {body.get('status', body)}  file_written={body.get('file_written')}")
    return r


try:
    print(SEP + f"\nStarting MCP server + API (logs/temp in {tmp})")
    start("mcp", ["scripts/run_mcp_server.py"])
    wait_http(f"http://127.0.0.1:{MCP_PORT}/health")
    start_api()

    print(SEP + "\nA. chat -> PAUSE at admin_approval -> approve -> record -> notify")
    info = httpx.post(f"{API}/chat", json={"message": "How much does parking cost?", "session_id": "e2e-info"},
                      timeout=60).json()
    print(f"  info question> intent={info['intent']} sources={info['sources']}  (no booking flow)")
    a = book("e2e-a", "Jane Smith", "ABC-1234", 1)
    gstate("e2e-a")
    print(f"  db row {db_row(a)}")
    show_file("file while PENDING (must be empty)")
    approve(a)
    gstate("e2e-a")
    notifications("e2e-a")
    show_file("file after approval")

    print(SEP + "\nB. API RESTART while the approval is pending -> resume from the SQLite checkpoint")
    b = book("e2e-b", "John Doe", "XYZ-9876", 2)
    gstate("e2e-b")
    print("  >>> stopping the API process (in-memory booking state is lost) ...")
    stop("api")
    print(f"  >>> checkpoint file on disk: {cp_file.name} ({cp_file.stat().st_size} bytes)")
    start_api()
    print("  >>> API restarted")
    g = gstate("e2e-b")
    assert g["interrupted"], "checkpoint did not restore the paused graph"
    r = httpx.post(f"{API}/chat", json={"message": "what's the status?", "session_id": "e2e-b"}, timeout=60).json()
    print(f"  chat after restart> {r['answer'][:90].replace(chr(10), ' ')}...")
    approve(b)
    gstate("e2e-b")
    notifications("e2e-b")

    print(SEP + "\nC. admin REFUSES -> nothing written, customer told why")
    c = book("e2e-c", "Eve Adams", "REF-0001", 3)
    r = httpx.post(f"{API}/admin/reservations/{c}/refuse", headers=H,
                   json={"reason": "lot closed for maintenance"}, timeout=30)
    print(f"  POST refuse -> HTTP {r.status_code} {r.json()['status']}")
    gstate("e2e-c")
    notifications("e2e-c")

    print(SEP + "\nD. double decision")
    r = approve(a)
    print(f"  (second approve of #{a}) detail: {r.json().get('detail')}")
    r = httpx.post(f"{API}/admin/reservations/{c}/approve", headers=H, json={}, timeout=30)
    print(f"  approve of a REFUSED request -> HTTP {r.status_code}")

    print(SEP + "\nE. MCP server DOWN at approval, then back")
    stop("mcp")
    d = book("e2e-d", "Dan Brown", "MCP-2222", 4)
    approve(d)
    gstate("e2e-d")
    print(f"  db row {db_row(d)}   (file_written must be 0: queued for retry)")
    notifications("e2e-d")
    start("mcp", ["scripts/run_mcp_server.py"])
    wait_http(f"http://127.0.0.1:{MCP_PORT}/health")
    t0 = time.time()
    while time.time() - t0 < 45 and not db_row(d)[2]:
        time.sleep(1)
    print(f"  after {time.time() - t0:.0f}s retry worker: db row {db_row(d)}")
    show_file("FINAL file (refused #C and the 409 never appear; exactly one line per approval)")
finally:
    for n in list(procs):
        stop(n)
    if KEEP:
        print(SEP + f"\n(temp files kept in {tmp})")
    else:
        for log in logs:
            log.close()
        for _ in range(10):
            shutil.rmtree(tmp, ignore_errors=True)
            if not tmp.exists():
                break
            time.sleep(0.5)
        print(SEP + ("\n(temp files deleted)" if not tmp.exists() else f"\n(WARNING: could not delete {tmp})"))
