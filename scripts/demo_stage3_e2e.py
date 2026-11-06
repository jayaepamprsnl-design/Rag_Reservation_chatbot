#!/usr/bin/env python3
"""
Stage-3 end-to-end demo (real processes, throw-away DB copy and output file).

  1. start the MCP server and the API as separate processes
  2. book a space through POST /chat  (-> PENDING)
  3. admin approves over REST         (-> CONFIRMED, line appended to the file)
  4. kill the MCP server, book + approve again (-> CONFIRMED, file_written=False)
  5. restart the MCP server; the API's retry worker writes the queued line
  6. print the final file

Usage:  python scripts/demo_stage3_e2e.py [--keep]
        --keep  leave the temp dir (DB copy, output file, logs) for inspection;
                by default it is deleted because the DB copy holds customer data.
Needs ADMIN_API_KEY in .env (never printed).  Uses ports 8798 (API) / 8799 (MCP).
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
API, MCP_PORT, API_PORT = "http://127.0.0.1:8798", 8799, 8798
SEP = "=" * 70
tmp = Path(tempfile.mkdtemp(prefix="stage3_e2e_"))
out_file = tmp / "out" / "confirmed_reservations.txt"
db_copy = tmp / "parking.db"
shutil.copy(ROOT / "parking.db", db_copy)

env = {
    **os.environ,
    "DATABASE_URL": f"sqlite:///{db_copy.as_posix()}",
    "MCP_API_KEY": secrets.token_urlsafe(24),       # generated per run, shared by both processes
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
logs: list = []  # log file handles; closed before cleanup (Windows cannot delete open files)
H = {"X-Admin-Key": settings.admin_api_key}


def start(name, args):
    log = open(tmp / f"{name}.log", "w")
    logs.append(log)
    procs[name] = subprocess.Popen(
        [sys.executable, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
    )


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


def db_row(rid):
    # closing(): "with sqlite3.connect()" only commits, it leaves the file handle (and the lock) open
    with contextlib.closing(sqlite3.connect(db_copy)) as c:
        return c.execute(
            "select status, decided_by is not null, file_written, file_written_at "
            "from lot_reservations where id=?", (rid,)
        ).fetchone()


def show_file(title):
    print(f"\n--- {title}\n" + (out_file.read_text(encoding="utf-8") if out_file.exists() else "(file does not exist)"))


def book(session, name, plate, days_ahead):
    start_dt = (datetime.now() + timedelta(days=days_ahead)).replace(hour=10, minute=0, second=0, microsecond=0)
    end_dt = start_dt.replace(hour=14)
    for msg in ["I want to book a parking space", name, plate, "1",
                start_dt.strftime("%Y-%m-%d %H:%M"), end_dt.strftime("%Y-%m-%d %H:%M"), "confirm"]:
        r = httpx.post(f"{API}/chat", json={"message": msg, "session_id": session}, timeout=60).json()
        last = r["answer"]
    print(f"  chat> {last[:110].replace(chr(10), ' ')}")
    pend = httpx.get(f"{API}/admin/reservations/pending", headers=H, timeout=10).json()
    return max(p["id"] for p in pend)


def approve(rid):
    r = httpx.post(f"{API}/admin/reservations/{rid}/approve", headers=H, json={}, timeout=60)
    print(f"  POST approve -> HTTP {r.status_code}  status={r.json()['status']}  "
          f"file_written={r.json()['file_written']}")
    print(f"  note: {r.json()['file_note']}")
    print(f"  db row (status, admin_decided, file_written, file_written_at): {db_row(rid)}")


try:
    print(SEP + "\nStarting MCP server and API (logs in %s)" % tmp)
    start("mcp", ["scripts/run_mcp_server.py"])
    wait_http(f"http://127.0.0.1:{MCP_PORT}/health")
    start("api", ["-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)])
    wait_http(f"{API}/health")
    show_file("file before anything is booked")

    print(SEP + "\nSCENARIO A - book via chat, admin approves, MCP server up")
    rid_a = book("e2e-a", "Jane Smith", "ABC-1234", 1)
    print(f"  reservation #{rid_a} is PENDING; db row: {db_row(rid_a)}")
    show_file("file while PENDING (must still be empty)")
    approve(rid_a)
    show_file("file after approval")

    print(SEP + "\nSCENARIO B - MCP server DOWN at approval time")
    stop("mcp")
    rid_b = book("e2e-b", "John Doe", "XYZ-9876", 2)
    approve(rid_b)
    show_file("file while server is down (unchanged)")

    print(SEP + "\nSCENARIO C - server comes back; background retry worker catches up")
    start("mcp", ["scripts/run_mcp_server.py"])
    wait_http(f"http://127.0.0.1:{MCP_PORT}/health")
    t0 = time.time()
    while time.time() - t0 < 45 and not db_row(rid_b)[2]:
        time.sleep(1)
    print(f"  after {time.time() - t0:.0f}s db row: {db_row(rid_b)}")
    show_file("FINAL file content")
finally:
    for n in list(procs):
        stop(n)
    if KEEP:
        print(SEP + f"\n(temp files kept in {tmp})")
    else:
        for log in logs:
            log.close()
        for _ in range(10):  # child processes may need a moment to release the DB on Windows
            shutil.rmtree(tmp, ignore_errors=True)
            if not tmp.exists():
                break
            time.sleep(0.5)
        print(SEP + ("\n(temp files deleted)" if not tmp.exists() else f"\n(WARNING: could not delete {tmp})"))
