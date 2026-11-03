"""Main chat route — wires guardrail(in) → router → RAG/SQL/reservation → guardrail(out)."""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.chat.conversation import get_or_create_session
from app.chat.reservation import (
    ReservationState,
    _COLLECTING_STATES,
    advance as reservation_advance,
    get_state as reservation_get_state,
    prefill_name as reservation_prefill_name,
    start as reservation_start,
)
from app.chat.router import RouterIntent, handle as router_handle
from app.core.logging_config import logger
from app.db.session import get_db
from app.guardrails.input_guard import check_input
from app.guardrails.output_guard import check_output

router = APIRouter(tags=["chat"])

# PII types the reservation SM receives unmasked (name + plate must be stored raw).
_RESERVATION_SKIP = ["PERSON", "LICENSE_PLATE"]

# _COLLECTING_STATES is now imported from reservation.py

# Pattern: "book/reserve for <Name>" — used to pre-populate the name field so
# the user does not have to repeat themselves.
_NAME_IN_REQUEST_RE = re.compile(
    r"\b(?:for|reserve\s+for|book(?:ing)?\s+for|slot\s+for|parking\s+for)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
    re.I,
)


def _extract_name_from_request(text: str) -> str:
    """Return a name extracted from the reservation request, or empty string."""
    m = _NAME_IN_REQUEST_RE.search(text)
    if not m:
        return ""
    candidate = m.group(1).strip()
    return candidate if len(candidate) >= 2 else ""


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
    sid = request.session_id
    ctx = get_or_create_session(sid)

    res_state      = reservation_get_state(sid)
    in_reservation = res_state not in (ReservationState.IDLE, ReservationState.DONE)
    collecting_field = res_state in _COLLECTING_STATES

    # ── Input guardrail ──────────────────────────────────────────────────────
    guard_in = check_input(
        request.message,
        skip_entities=_RESERVATION_SKIP if collecting_field else None,
    )

    if guard_in.blocked:
        logger.warning("session=%s guardrail=input_blocked", sid)
        return ChatResponse(
            answer=guard_in.block_reason,
            session_id=sid,
            intent="BLOCKED",
            guardrail_flags=["injection_blocked"],
        )

    safe_text = guard_in.text   # for LLM / logs / history
    raw_text  = request.message # for reservation SM field collection

    ctx.add_message("user", safe_text)

    answer: str
    sources: list[str] = []
    intent: str = "UNKNOWN"
    guardrail_flags: list[str] = []
    _active_res_id: int | None = None   # tracks LotReservation.id across branches

    # ── Route ────────────────────────────────────────────────────────────────
    if in_reservation:
        # SM receives raw text so name/plate are stored unmasked.
        reply   = reservation_advance(sid, raw_text, db)
        answer  = reply.message
        intent  = RouterIntent.RESERVATION.value
        if reply.reservation_id:
            sources = ["database"]
            _active_res_id = reply.reservation_id

    else:
        route_resp = router_handle(
            safe_text,
            chat_history=ctx.last_n_messages(6),
            db=db,
        )
        answer  = route_resp.answer
        sources = route_resp.sources
        intent  = route_resp.intent.value

        if route_resp.intent == RouterIntent.RESERVATION:
            # advance() handles IDLE→start internally and applies multi-slot
            # extraction, so any name / date / time given in the first message
            # is captured without asking again.
            reply   = reservation_advance(sid, raw_text, db)
            answer  = reply.message
            intent  = RouterIntent.RESERVATION.value
            if reply.reservation_id:
                sources = ["database"]
                _active_res_id = reply.reservation_id

            # The SM is now active — allow the output guard to pass through
            # PERSON/PLATE so the customer sees their name in the response.
            in_reservation = True

    # ── Output guardrail ─────────────────────────────────────────────────────
    # In reservation context the booking summary contains the customer's own
    # name/plate — those are intentional echoes, not data leaks.
    out_skip  = ["PERSON", "LICENSE_PLATE"] if in_reservation else None
    guard_out = check_output(answer, skip_entities=out_skip)

    if guard_out.blocked:
        logger.warning(
            "session=%s guardrail=output_blocked reason=%s", sid, guard_out.block_reason
        )
        guardrail_flags.append(f"output_blocked:{guard_out.block_reason}")
        answer = guard_out.block_reason
    else:
        answer = guard_out.text
        if guard_out.entities_redacted:
            guardrail_flags.append(
                f"pii_redacted:{','.join(guard_out.entities_redacted)}"
            )

    if guard_in.entities_found:
        guardrail_flags.append(f"pii_masked:{','.join(guard_in.entities_found)}")

    logger.info(
        "session=%s intent=%s sources=%s flags=%s", sid, intent, sources, guardrail_flags
    )
    ctx.add_message("assistant", answer)

    return ChatResponse(
        answer=answer,
        session_id=sid,
        sources=sources,
        intent=intent,
        guardrail_flags=guardrail_flags,
        reservation_id=_active_res_id,
    )
