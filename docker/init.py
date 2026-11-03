"""Docker container initialisation — runs once at startup.

Idempotent: seeds the database only when empty; ingests static documents
into ChromaDB only when the collection has no chunks.
"""
from __future__ import annotations

import subprocess
import sys

sys.path.insert(0, "/app")


def _seed_db() -> None:
    from app.db.models import ParkingLot
    from app.db.session import SessionLocal, create_tables

    create_tables()
    db = SessionLocal()
    count = db.query(ParkingLot).count()
    db.close()

    if count == 0:
        print("[init] Database empty — seeding parking lots…", flush=True)
        subprocess.run(["python", "scripts/seed_db.py"], check=True)
    else:
        print(f"[init] Database already seeded ({count} lots).", flush=True)


def _ingest_docs() -> None:
    from app.rag.chroma_store import get_collection

    try:
        col = get_collection()
        chunk_count = col.count()
    except Exception:
        chunk_count = 0

    if chunk_count == 0:
        print("[init] ChromaDB empty — ingesting static documents…", flush=True)
        subprocess.run(["python", "scripts/ingest.py"], check=True)
    else:
        print(f"[init] ChromaDB already has {chunk_count} chunks.", flush=True)


def _migrate_db() -> None:
    """Apply Stage-2 schema migrations (idempotent — safe on every startup).

    On a fresh volume create_tables() already builds the full schema, so the
    migration script just prints 'Column already exists' and exits cleanly.
    On an upgraded volume (Stage-1 SQLite) it adds the missing columns.
    """
    print("[init] Running Stage-2 schema migration…", flush=True)
    result = subprocess.run(
        ["python", "scripts/migrate_stage2.py"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"[init]   {line}", flush=True)
    if result.returncode != 0:
        print(f"[init] Migration stderr: {result.stderr}", flush=True)
        # Non-fatal — app can still start with existing schema
    else:
        print("[init] Migration complete.", flush=True)


if __name__ == "__main__":
    _seed_db()
    _migrate_db()
    _ingest_docs()
    print("[init] Initialisation complete.", flush=True)
