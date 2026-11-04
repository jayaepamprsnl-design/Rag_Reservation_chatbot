"""Graph nodes.  Each one wraps existing code; none re-implements it.

chat_node            guardrail(in) -> router -> RAG / SQL / slot filling -> guardrail(out)
escalate_node        reservation.finalize_confirmed: PENDING row + send to the admin channels
admin_approval_node  human-in-the-loop: pauses with interrupt() until an admin decides
record_node          MCP write of an APPROVED reservation (never runs otherwise)
notify_user_node     builds the customer message for a decided reservation
handle_error_node    turns an internal failure into a safe reply

Safety invariants
-----------------
* Nothing in this module calls an LLM to decide anything.  The ONLY way out of
  admin_approval_node is a resume value from service.sync_decision(), which is
  built from the database row an administrator's approve/refuse call committed.
* After resuming, the node trusts the DATABASE, not the resume value: it routes
  to "approved" only if reservation_writer/mcp_client.is_admin_approved(row), and
  a resume value that disagrees with the row is an error, not a decision.
* record_node re-checks is_admin_approved itself, and export_approved checks it a
  third time, so a PENDING / REFUSED / CANCELLED / EXPIRED row can never be sent.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt
from sqlalchemy.orm import Session

from app.agents.mcp_client import is_admin_approved
from app.agents.reservation_writer import export_approved
from app.chat import reservation as resv
from app.chat.reservation import ReservationState, _COLLECTING_STATES
from app.chat.router import RouterIntent, handle as router_handle
from app.core.logging_config import logger
from app.db.models import LotReservation, ReservationStatus
from app.db.repository import get_reservations_by_session
from app.guardrails.input_guard import check_input
from app.guardrails.output_guard import check_output
from app.orchestration import state as S
from app.orchestration.messages import build_user_message

# PII types the reservation SM receives unmasked (name + plate must be stored raw).
_RESERVATION_SKIP = ["PERSON", "LICENSE_PLATE"]

_ERROR_REPLY = (
    "Sorry, something went wrong on our side. Please try again in a moment."
)
_VALID_DECISIONS = {"APPROVE", "REFUSE", "EXPIRE"}


@dataclass
class TurnContext:
    """Per-run objects that must not be checkpointed (passed via config['configurable']['turn'])."""

    db: Session
    raw_text: str = ""     # unmasked user message: the booking SM needs the real name/plate


def _turn(config: RunnableConfig) -> TurnContext:
    return config["configurable"]["turn"]


def _safe(node: str, error_status: str | None = None) -> Callable:
    """Turn an unexpected exception into state (-> handle_error) instead of a 500.

    GraphBubbleUp (interrupt / resume control flow) is re-raised untouched.
    The log carries the exception class only, never a message that could hold PII.
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: S.GraphState, config: RunnableConfig) -> dict:
            try:
                return fn(state, config)
            except GraphBubbleUp:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("orchestration|node_failed node=%s error=%s", node, type(exc).__name__)
                update: dict[str, Any] = {
                    "errors": [*(state.get("errors") or []), f"{node}: {type(exc).__name__}"]
                }
                if error_status:
                    update["status"] = error_status
                return update

        return wrapper

    return deco


# ---------------------------------------------------------------------------
# chat_node
# ---------------------------------------------------------------------------


def _rehydrate(state: S.GraphState, db: Session) -> None:
    """After a restart the in-memory booking SM is empty: restore it from the checkpoint.

    The database wins over the snapshot: a crash between "PENDING row created" and "checkpoint
    saved" leaves a stale CONFIRM snapshot next to a real PENDING row.  In that case the snapshot
    is skipped and reservation.advance() recovers the session from the row.
    """
    snap = state.get("reservation")
    if not snap:
        return
    sid = state["session_id"]
    if snap.get("state") == ReservationState.CONFIRM.value and any(
        r.status == ReservationStatus.PENDING for r in get_reservations_by_session(db, sid)
    ):
        return
    resv.import_session(sid, snap)


def _status_after_chat(state: S.GraphState, sid: str, db: Session) -> str:
    rs = resv.get_state(sid)
    if rs == ReservationState.AWAITING_APPROVAL:
        return S.AWAITING_ADMIN
    rid = state.get("reservation_id")
    if state.get("status") == S.AWAITING_ADMIN and rid:
        # The thread is paused at admin_approval.  If the customer did not cancel,
        # the row was decided elsewhere (delivered lazily by the SM): keep the
        # status so admin_approval reconciles it (record / notify) instead of
        # leaving a dangling pause.
        row = db.get(LotReservation, rid)
        if row is not None and row.status != ReservationStatus.CANCELLED:
            return S.AWAITING_ADMIN
        return S.CANCELLED
    if rs in _COLLECTING_STATES or rs == ReservationState.CONFIRM:
        return S.COLLECTING
    prev = state.get("status")
    if prev in (S.APPROVED, S.REFUSED, S.EXPIRED, S.CANCELLED):
        return prev   # keep the outcome of the last booking until a new one starts
    return S.IDLE


@_safe("chat_node")
def chat_node(state: S.GraphState, config: RunnableConfig) -> dict:
    turn = _turn(config)
    db, sid, raw_text = turn.db, state["session_id"], turn.raw_text
    _rehydrate(state, db)
    resv.recover_pending_session(sid, db)

    res_state = resv.get_state(sid)
    in_reservation = res_state not in (ReservationState.IDLE, ReservationState.DONE)
    collecting_field = res_state in _COLLECTING_STATES

    # ── input guardrail ──────────────────────────────────────────────────────
    guard_in = check_input(
        raw_text, skip_entities=_RESERVATION_SKIP if collecting_field else None
    )
    if guard_in.blocked:
        logger.warning("session=%s guardrail=input_blocked", sid)
        return {
            "answer": guard_in.block_reason,
            "intent": "BLOCKED",
            "sources": [],
            "guardrail_flags": ["injection_blocked"],
            "status": state.get("status", S.IDLE),
        }
    safe_text = guard_in.text   # for LLM / logs / history
    user_msg = {"role": "user", "content": safe_text}

    flags: list[str] = []
    sources: list[str] = []
    active_res_id: int | None = None

    # ── route ────────────────────────────────────────────────────────────────
    if in_reservation:
        reply = resv.advance(sid, raw_text, db, defer_confirm=True)
        answer, intent = reply.message, RouterIntent.RESERVATION.value
    else:
        history = [*(state.get("messages") or []), user_msg][-6:]
        route_resp = router_handle(safe_text, chat_history=history, db=db)
        answer, sources = route_resp.answer, route_resp.sources
        intent = route_resp.intent.value
        reply = None
        if route_resp.intent == RouterIntent.RESERVATION:
            # advance() starts the booking and applies multi-slot extraction, so
            # anything given in the first message is captured without asking again.
            reply = resv.advance(sid, raw_text, db, defer_confirm=True)
            answer, intent = reply.message, RouterIntent.RESERVATION.value
            in_reservation = True   # let the output guard pass the customer's own name/plate

    if reply is not None and reply.reservation_id:
        sources, active_res_id = ["database"], reply.reservation_id

    if guard_in.entities_found:
        flags.append(f"pii_masked:{','.join(guard_in.entities_found)}")

    snapshot = resv.export_session(sid) or {}
    update: dict[str, Any] = {
        "intent": intent,
        "sources": sources,
        "guardrail_flags": flags,
        "active_reservation_id": active_res_id,
        "reservation": snapshot,
        "status": _status_after_chat(state, sid, db),
    }
    if snapshot.get("state") == ReservationState.AWAITING_APPROVAL.value and snapshot.get("reservation_id"):
        # e.g. recovered from the database after a crash: the graph must wait on the same row
        update["reservation_id"] = snapshot["reservation_id"]

    if reply is not None and reply.confirmed:
        # The user said "confirm": escalate_node creates the row and produces the reply.
        update.update(confirmed=True, answer="", messages=[user_msg])
        return update

    # ── output guardrail ─────────────────────────────────────────────────────
    # A booking summary echoes the customer's own name/plate on purpose.
    guard_out = check_output(answer, skip_entities=_RESERVATION_SKIP if in_reservation else None)
    if guard_out.blocked:
        logger.warning("session=%s guardrail=output_blocked reason=%s", sid, guard_out.block_reason)
        flags.insert(0, f"output_blocked:{guard_out.block_reason}")
        answer = guard_out.block_reason
    else:
        answer = guard_out.text
        if guard_out.entities_redacted:
            flags.insert(0, f"pii_redacted:{','.join(guard_out.entities_redacted)}")

    logger.info("session=%s intent=%s sources=%s flags=%s", sid, intent, sources, flags)
    update["answer"] = answer
    update["messages"] = [user_msg, {"role": "assistant", "content": answer}]
    return update


# ---------------------------------------------------------------------------
# escalate_node
# ---------------------------------------------------------------------------


@_safe("escalate_node")
def escalate_node(state: S.GraphState, config: RunnableConfig) -> dict:
    turn = _turn(config)
    sid = state["session_id"]
    reply = resv.finalize_confirmed(sid, turn.db)

    guard_out = check_output(reply.message, skip_entities=_RESERVATION_SKIP)
    flags = list(state.get("guardrail_flags") or [])
    if guard_out.blocked:
        flags.insert(0, f"output_blocked:{guard_out.block_reason}")
        answer = guard_out.block_reason
    else:
        answer = guard_out.text
        if guard_out.entities_redacted:
            flags.insert(0, f"pii_redacted:{','.join(guard_out.entities_redacted)}")

    update: dict[str, Any] = {
        "answer": answer,
        "guardrail_flags": flags,
        "confirmed": False,
        "file_written": False,   # a new request: nothing recorded yet
        "reservation": resv.export_session(sid) or {},
        "messages": [{"role": "assistant", "content": answer}],
    }
    if reply.state == ReservationState.AWAITING_APPROVAL and reply.reservation_id:
        update.update(
            reservation_id=reply.reservation_id,
            active_reservation_id=reply.reservation_id,
            sources=["database"],
            status=S.AWAITING_ADMIN,
        )
    else:   # e.g. the slot was taken meanwhile: the SM went back to ASK_LOT
        update["status"] = S.COLLECTING
    return update


# ---------------------------------------------------------------------------
# admin_approval_node  (human in the loop)
# ---------------------------------------------------------------------------


def _outcome_from_row(res: LotReservation) -> str:
    """The graph status implied by the database row (the source of truth)."""
    if is_admin_approved(res):
        return S.APPROVED
    return {
        ReservationStatus.REFUSED: S.REFUSED,
        ReservationStatus.EXPIRED: S.EXPIRED,
        ReservationStatus.CANCELLED: S.CANCELLED,
    }.get(res.status, "")


_DECISION_TO_STATUS = {"APPROVE": S.APPROVED, "REFUSE": S.REFUSED, "EXPIRE": S.EXPIRED}


@_safe("admin_approval_node", error_status=S.ERROR)
def admin_approval_node(state: S.GraphState, config: RunnableConfig) -> dict:
    db = _turn(config).db
    rid = state.get("reservation_id")
    res = db.get(LotReservation, rid) if rid else None
    if res is None:
        return {"status": S.ERROR, "errors": [*(state.get("errors") or []), "admin_approval: no reservation"]}
    db.refresh(res)

    if res.status == ReservationStatus.PENDING:
        # Pauses the graph here.  The run only continues when service.sync_decision()
        # resumes this thread with the committed decision; any other input to the
        # thread re-enters this node and simply pauses again.
        payload = interrupt({"reservation_id": rid, "session_id": state["session_id"]})
        db.refresh(res)
        decision = payload.get("decision") if isinstance(payload, dict) else None
        if decision not in _VALID_DECISIONS or _DECISION_TO_STATUS[decision] != _outcome_from_row(res):
            # A resume that the database does not back up is not a decision.
            logger.error(
                "orchestration|resume_rejected reservation_id=%s row_status=%s",
                rid, getattr(res.status, "value", res.status),
            )
            return {
                "status": S.ERROR,
                "errors": [*(state.get("errors") or []), "admin_approval: resume not backed by a committed decision"],
            }

    outcome = _outcome_from_row(res)
    if not outcome:
        return {"status": S.ERROR, "errors": [*(state.get("errors") or []), "admin_approval: undecided row"]}
    logger.info("orchestration|decision_applied reservation_id=%s outcome=%s", rid, outcome)
    return {"status": outcome}


# ---------------------------------------------------------------------------
# record_node
# ---------------------------------------------------------------------------


@_safe("record_node")
def record_node(state: S.GraphState, config: RunnableConfig) -> dict:
    db = _turn(config).db
    rid = state.get("reservation_id")
    res = db.get(LotReservation, rid) if rid else None
    if res is None or state.get("status") != S.APPROVED or not is_admin_approved(res):
        # Unreachable through the graph edges; kept as a hard stop.
        logger.error("orchestration|record_blocked reservation_id=%s", rid)
        return {"file_written": False, "errors": [*(state.get("errors") or []), "record: not approved"]}

    written = export_approved(db, rid)   # idempotent; never raises; gates on is_admin_approved
    update: dict[str, Any] = {"file_written": written}
    if not written:
        # The row stays CONFIRMED with file_written=False: that IS the retry marker the
        # background worker (reservation_writer.retry_pending_writes) sweeps.  Carry on.
        logger.warning("orchestration|record_deferred reservation_id=%s (queued for retry)", rid)
        update["errors"] = [*(state.get("errors") or []), "record: MCP write deferred, queued for retry"]
    return update


# ---------------------------------------------------------------------------
# notify_user_node / handle_error_node
# ---------------------------------------------------------------------------


@_safe("notify_user_node")
def notify_user_node(state: S.GraphState, config: RunnableConfig) -> dict:
    """Prepare the customer message.

    It is returned in state['notification'] and left for delivery by the existing
    channels (GET /sessions/{id}/notifications, or the SM on the customer's next
    message), which also own the notified_user flag.  state['answer'] is untouched.
    """
    db = _turn(config).db
    res = db.get(LotReservation, state["reservation_id"])
    if res is None:
        return {"notification": ""}
    text = build_user_message(res)
    if state.get("status") == S.EXPIRED:
        text = (
            f"Your parking request #{res.id} expired because no administrator replied in time. "
            "Please try booking again."
        )
    return {"notification": text}


def handle_error_node(state: S.GraphState, config: RunnableConfig) -> dict:
    logger.error("orchestration|handled_errors session=%s errors=%s", state.get("session_id"), state.get("errors"))
    update: dict[str, Any] = {"guardrail_flags": [*(state.get("guardrail_flags") or []), "error"]}
    if not state.get("answer"):
        update["answer"] = _ERROR_REPLY
    return update
