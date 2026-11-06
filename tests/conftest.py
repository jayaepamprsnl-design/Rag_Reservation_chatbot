import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Base


@pytest.fixture(autouse=True)
def _no_mcp_export(monkeypatch):
    """Existing tests must never reach a real MCP server; tests/test_mcp_integration.py opts in."""
    monkeypatch.setattr(settings, "mcp_export_enabled", False)


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()
