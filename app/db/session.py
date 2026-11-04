from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import Base

# SQLite requires check_same_thread=False; other drivers don't support that arg.
_connect_args = {"check_same_thread": False} if "sqlite" in settings.database_url else {}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


_STAGE3_COLUMNS = [
    ("file_written", "ALTER TABLE lot_reservations ADD COLUMN file_written BOOLEAN NOT NULL DEFAULT 0"),
    ("file_written_at", "ALTER TABLE lot_reservations ADD COLUMN file_written_at DATETIME"),
]


def ensure_stage3_columns(bind=None) -> list[str]:
    """Add the Stage-3 columns to an existing lot_reservations table (idempotent).

    create_all() creates missing tables but never alters existing ones, so a
    pre-Stage-3 parking.db would otherwise fail on the first query.
    Returns the names of the columns that were added.
    """
    bind = bind or engine
    insp = inspect(bind)
    if "lot_reservations" not in insp.get_table_names():
        return []
    existing = {c["name"] for c in insp.get_columns("lot_reservations")}
    added = []
    with bind.begin() as conn:
        for name, stmt in _STAGE3_COLUMNS:
            if name not in existing:
                conn.execute(text(stmt))
                added.append(name)
    return added


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_stage3_columns(engine)


def get_db():
    """FastAPI dependency — yields a database session and ensures it is closed."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
