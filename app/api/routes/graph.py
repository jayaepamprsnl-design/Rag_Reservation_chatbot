"""Debug view of the orchestration graph (admin key required)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.routes.admin import require_admin
from app.db.session import get_db
from app.orchestration.service import get_state

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/{session_id}/state", summary="Checkpointed graph state for a session")
def graph_state(
    session_id: str,
    db: Session = Depends(get_db),
    _: str = Depends(require_admin),
) -> dict:
    """Where the conversation is (next node, paused at admin approval?, status, ids).

    Customer names are never returned and the plate is masked.
    """
    return get_state(session_id, db)
