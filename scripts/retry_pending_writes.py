#!/usr/bin/env python
"""
Write every admin-approved (CONFIRMED) reservation whose MCP file write is still
pending, then mark it file_written=True.  Safe to re-run: the MCP server is
idempotent per reservation_id.

Usage:
    python scripts/retry_pending_writes.py          # exit 1 if any write failed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.mcp_client import export_configured  # noqa: E402
from app.agents.reservation_writer import pending_ids, retry_pending_writes  # noqa: E402
from app.db.session import SessionLocal, create_tables  # noqa: E402


def main() -> int:
    create_tables()  # also adds the Stage-3 columns to an older parking.db
    if not export_configured():
        print("MCP export is not configured (set MCP_API_KEY in .env). Nothing done.")
        return 2
    db = SessionLocal()
    try:
        waiting = len(pending_ids(db))
    finally:
        db.close()
    print(f"{waiting} approved reservation(s) waiting for a file write.")
    summary = retry_pending_writes(SessionLocal)
    print(f"attempted={summary['attempted']} written={summary['written']} failed={summary['failed']}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
