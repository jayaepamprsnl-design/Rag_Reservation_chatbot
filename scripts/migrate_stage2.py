#!/usr/bin/env python3
"""
Stage-2 schema migration for SQLite.

Safe to re-run: each ALTER TABLE is skipped if the column already exists,
and the new table is created with CREATE TABLE IF NOT EXISTS.

Usage:
    python scripts/migrate_stage2.py

For a fresh development environment, deleting parking.db and re-running
scripts/seed_db.py achieves the same result without needing this script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db.session import engine

# New columns to add to lot_reservations
_NEW_COLUMNS = [
    ("session_id",              "ALTER TABLE lot_reservations ADD COLUMN session_id VARCHAR(100)"),
    ("admin_decision_reason",   "ALTER TABLE lot_reservations ADD COLUMN admin_decision_reason TEXT"),
    ("decided_at",              "ALTER TABLE lot_reservations ADD COLUMN decided_at DATETIME"),
    ("decided_by",              "ALTER TABLE lot_reservations ADD COLUMN decided_by VARCHAR(100)"),
    ("notified_user",           "ALTER TABLE lot_reservations ADD COLUMN notified_user BOOLEAN NOT NULL DEFAULT 0"),
]

_CREATE_APPROVAL_TABLE = """
CREATE TABLE IF NOT EXISTS approval_requests (
    id              INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    reservation_id  INTEGER NOT NULL REFERENCES lot_reservations(id),
    status          VARCHAR(20) NOT NULL DEFAULT 'sent',
    created_at      DATETIME NOT NULL,
    sent_at         DATETIME,
    responded_at    DATETIME,
    channel         VARCHAR(50) NOT NULL DEFAULT 'api',
    request_text    TEXT
)
"""


def migrate() -> None:
    with engine.connect() as conn:
        # Inspect existing columns on lot_reservations
        rows = conn.execute(text("PRAGMA table_info(lot_reservations)")).fetchall()
        existing_cols = {row[1] for row in rows}

        for col_name, stmt in _NEW_COLUMNS:
            if col_name not in existing_cols:
                conn.execute(text(stmt))
                print(f"  + Added column: lot_reservations.{col_name}")
            else:
                print(f"  ~ Skipped (exists): lot_reservations.{col_name}")

        conn.execute(text(_CREATE_APPROVAL_TABLE))
        print("  + Ensured table: approval_requests")

        conn.commit()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
