"""Main chat route — runs one turn on the LangGraph orchestration (app/orchestration)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.orchestration.service import run_turn

router = APIRouter(tags=["chat"])

class ChatRequest(BaseModel):
    message: str
    # WARNING: callers MUST supply a unique session_id per conversation.
    # The default "default" value means all clients that omit it share one
    # in-memory session, corrupting each other's reservation state.
    # A UUID generated client-side is the recommended value.
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = []
    intent: str = "UNKNOWN"
    guardrail_flags: list[str] = []
    reservation_id: int | None = None  # set when a LotReservation is active


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """One customer turn.  The pipeline (guardrail in -> router -> RAG/SQL/booking ->
    guardrail out, then escalation / admin approval / MCP record / notification) is
    the LangGraph in app/orchestration; this route only runs a turn on it."""
    result = run_turn(request.session_id, request.message, db)
    return ChatResponse(
        answer=result.answer,
        session_id=request.session_id,
        sources=result.sources,
        intent=result.intent,
        guardrail_flags=result.guardrail_flags,
        reservation_id=result.reservation_id,
    )
