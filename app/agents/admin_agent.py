"""
Agent 2 — Admin Agent (LangChain tool-calling pattern).

Three-phase pipeline
--------------------
Phase 1  generate_notification(request)          → str
Phase 2  parse_decision(text, res_id, admin_id)  → AdminDecision | str
Phase 3  execute_decision(decision, db)           → AgentMessage(USER_NOTIFICATION)

Safety invariant
----------------
record_admin_decision() is NEVER called unless the administrator's raw text
contains an explicit decision keyword (approve / refuse / reject / deny …).
The LLM parses and formats text; it CANNOT initiate a decision autonomously.
The deterministic keyword gate (Phase 2, Step 2) runs BEFORE any LLM call.

MockLLM mode
------------
When OPENAI_API_KEY is a placeholder the existing get_llm_client() factory
returns MockLLMClient.  For unit tests inject MockAdminLLMClient directly so
each test controls the scripted LLM responses.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Annotated, Any, Optional

from langchain_core.tools import InjectedToolArg, tool
from sqlalchemy.orm import Session

from app.agents.contracts import (
    AdminDecision,
    AgentMessage,
    Decision,
    MessageType,
    ReservationRequest,
)
from app.agents.mcp_client import is_admin_approved
from app.agents.reservation_writer import export_approved, export_note
from app.core.logging_config import logger
from app.db.models import LotReservation
from app.db.repository import (
    ValidationError,
    approve_reservation,
    check_conflicts,
    get_lot,
    get_reservation,
    list_pending,
    refuse_reservation,
)
from app.guardrails.input_guard import check_input
from app.rag.generator import BaseLLMClient, _call_with_retry, get_llm_client


# ---------------------------------------------------------------------------
# Inline PII helper — avoids importing private symbol from reservation.py
# ---------------------------------------------------------------------------


def _mask_plate(plate: str) -> str:
    if len(plate) <= 4:
        return plate[0] + "***"
    return plate[:3] + "****" + plate[-2:]


# ---------------------------------------------------------------------------
# Deterministic parsing helpers (no LLM involved)
# ---------------------------------------------------------------------------

_APPROVE_RE = re.compile(
    r"\b(approve[ds]?|confirm(?:s|ed)?|accept(?:s|ed)?|ok(?:ay)?)\b", re.I
)
_REFUSE_RE = re.compile(
    r"\b(refuse[ds]?|reject(?:ed|s)?|den(?:y|ied)|decline[ds]?)\b", re.I
)
_ID_RE = re.compile(r"\b(\d+)\b")
_REASON_RE = re.compile(
    r"\b(?:because|reason|since|as|due\s+to|for)\s+(.+)", re.I | re.DOTALL
)


def _detect_decision_keyword(text: str) -> str | None:
    """
    Return 'APPROVE', 'REFUSE', or None.

    None means either both keywords appear (ambiguous) or neither does (no
    decision).  This is the safety gate — the LLM is never consulted for
    the decision itself.
    """
    has_approve = bool(_APPROVE_RE.search(text))
    has_refuse  = bool(_REFUSE_RE.search(text))
    if has_refuse and not has_approve:
        return "REFUSE"
    if has_approve and not has_refuse:
        return "APPROVE"
    return None  # ambiguous or absent


def _extract_id(text: str) -> int | None:
    m = _ID_RE.search(text)
    return int(m.group(1)) if m else None


def _extract_reason_regex(text: str) -> str:
    m = _REASON_RE.search(text)
    return m.group(1).strip() if m else ""


def _extract_reason_llm(text: str, llm: BaseLLMClient) -> str:
    """
    Ask the LLM to extract the refusal reason from a complex sentence.
    Used only when regex finds no explicit 'because/since/…' clause.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a text-extraction assistant for a parking management system. "
                "Extract ONLY the stated reason or justification for the admin's refusal. "
                "Reply with the reason text only, or an empty string if none is present. "
                "Do not add anything else."
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        result = _call_with_retry(llm, messages)
        return result.strip()
    except Exception:  # noqa: BLE001
        return ""


def _extract_reason(text: str, llm: BaseLLMClient) -> str:
    reason = _extract_reason_regex(text)
    return reason if reason else _extract_reason_llm(text, llm)


# ---------------------------------------------------------------------------
# Mock LLM for offline / test mode
# ---------------------------------------------------------------------------


class MockAdminLLMClient(BaseLLMClient):
    """
    Scripted mock that returns pre-supplied responses in order.

    Once the script is exhausted the sentinel string is returned so tests
    can assert that no unexpected extra LLM calls occurred.
    """

    SENTINEL = "[MOCK: no more scripted responses]"

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses: list[str] = list(responses or [])
        self._call_count = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = self.SENTINEL
        self._call_count += 1
        logger.debug("MockAdminLLM call #%d → %r", self._call_count, resp[:60])
        return resp

    @property
    def call_count(self) -> int:
        return self._call_count


# ---------------------------------------------------------------------------
# LangChain tools
#
# Each tool takes 'db' as an InjectedToolArg: the parameter is excluded
# from the LLM-visible OpenAI function schema (verified via
# convert_to_openai_tool) but is provided at invocation time by the agent.
# The LLM therefore never sees 'db' and cannot call any tool autonomously.
# ---------------------------------------------------------------------------


@tool
def get_pending_reservations(
    db: Annotated[Session, InjectedToolArg],
) -> list[dict]:
    """Return all parking reservations currently awaiting admin approval."""
    rows = list_pending(db)
    return [
        {
            "reservation_id": r.id,
            "customer_name":  r.customer_name,
            "car_plate":      _mask_plate(r.car_plate),
            "lot_id":         r.lot_id,
            "start_time":     r.start_time.isoformat(),
            "end_time":       r.end_time.isoformat(),
            "session_id":     r.session_id,
            "created_at":     r.created_at.isoformat(),
        }
        for r in rows
    ]


@tool
def get_reservation_details(
    reservation_id: int,
    db: Annotated[Session, InjectedToolArg],
) -> dict:
    """Return full details for a specific reservation including lot information."""
    try:
        r   = get_reservation(db, reservation_id)
        lot = get_lot(db, r.lot_id)
        return {
            "reservation_id":  r.id,
            "customer_name":   r.customer_name,
            "car_plate":       _mask_plate(r.car_plate),
            "lot_id":          r.lot_id,
            "lot_name":        lot.name,
            "lot_address":     lot.address,
            "total_spaces":    lot.total_spaces,
            "available_spaces": lot.available_spaces,
            "hourly_price":    lot.hourly_price,
            "daily_price":     lot.daily_price,
            "start_time":      r.start_time.isoformat(),
            "end_time":        r.end_time.isoformat(),
            "status":          r.status.value,
            "session_id":      r.session_id,
            "created_at":      r.created_at.isoformat(),
        }
    except ValidationError as exc:
        return {"error": str(exc)}


@tool
def check_lot_availability(
    lot_id: int,
    start_time: str,
    end_time: str,
    db: Annotated[Session, InjectedToolArg],
) -> dict:
    """
    Check current space availability and overlapping confirmed reservations
    for a given lot and time window.  start_time and end_time are ISO strings.
    """
    try:
        lot       = get_lot(db, lot_id)
        start_dt  = datetime.fromisoformat(start_time)
        end_dt    = datetime.fromisoformat(end_time)
        conflicts = check_conflicts(db, lot_id, start_dt, end_dt)
        return {
            "lot_id":              lot.id,
            "lot_name":            lot.name,
            "total_spaces":        lot.total_spaces,
            "available_spaces":    lot.available_spaces,
            "conflicts_in_window": len(conflicts),
            "conflict_ids":        [c.id for c in conflicts],
            "can_accommodate":     (
                lot.available_spaces > 0 and len(conflicts) < lot.total_spaces
            ),
        }
    except (ValidationError, ValueError) as exc:
        return {"error": str(exc)}


@tool
def send_admin_notification(
    reservation_id: int,
    notification_text: str,
    db: Annotated[Session, InjectedToolArg],
) -> dict:
    """
    Persist the notification text as an ApprovalRequest audit row and return
    a delivery receipt.  In production, extend this to dispatch email/webhook.
    """
    from app.db.models import ApprovalRequest

    audit = ApprovalRequest(
        reservation_id=reservation_id,
        status="sent",
        sent_at=datetime.utcnow(),
        channel="api",
        request_text=notification_text,
    )
    db.add(audit)
    db.commit()
    logger.info(
        "admin_agent|notification_sent reservation_id=%d", reservation_id
    )
    return {
        "reservation_id": reservation_id,
        "status":         "sent",
        "audit_id":       audit.id,
        "preview":        notification_text[:120],
    }


@tool
def write_reservation(
    reservation_id: int,
    db: Annotated[Session, InjectedToolArg],
) -> dict:
    """
    Append an admin-APPROVED reservation to the confirmed-reservations file
    through the MCP server.

    Deliberately NOT in TOOLS (the LLM-facing list): it is invoked only by the
    code path of a successful admin approval (record_admin_decision, the REST
    approve route, the email-link route and the retry worker).  As defence in
    depth it re-verifies against the database that an admin approved the
    reservation, so even a direct call for a PENDING / REFUSED / CANCELLED /
    EXPIRED reservation is blocked and nothing is sent.
    """
    res = db.get(LotReservation, reservation_id)
    if res is None:
        return {"error": f"Reservation {reservation_id} not found."}
    if not is_admin_approved(res):
        logger.warning(
            "admin_agent|write_blocked reservation_id=%d status=%s",
            reservation_id,
            getattr(res.status, "value", res.status),
        )
        return {
            "error": (
                f"Blocked: reservation {reservation_id} has not been approved by an "
                "administrator; nothing was written."
            ),
            "file_written": False,
        }
    written = export_approved(db, reservation_id)
    return {
        "reservation_id": reservation_id,
        "file_written":   written,
        "note":           export_note(written),
    }


@tool
def record_admin_decision(
    reservation_id: int,
    decision: str,
    db: Annotated[Session, InjectedToolArg],
    reason: str = "",
    admin_id: str = "admin-agent",
) -> dict:
    """
    Apply an APPROVE or REFUSE decision to a PENDING reservation.

    Only PENDING reservations can be decided — the repository enforces this
    and raises ValidationError on any other status (double-click guard).
    """
    dec = decision.strip().upper()
    try:
        if dec == "APPROVE":
            res = approve_reservation(db, reservation_id, admin=admin_id, note=reason)
            # Approval is committed. Only now (and only because it succeeded) try the
            # file write; failure leaves the reservation CONFIRMED with file_written=False.
            export = write_reservation.invoke({"reservation_id": res.id, "db": db})
            return {
                "status":         "approved",
                "reservation_id": res.id,
                "decided_by":     admin_id,
                "decided_at":     res.decided_at.isoformat() if res.decided_at else None,
                "file_written":   bool(export.get("file_written")),
                "file_note":      export.get("note") or export.get("error"),
            }
        if dec == "REFUSE":
            res = refuse_reservation(db, reservation_id, admin=admin_id, reason=reason)
            return {
                "status":         "refused",
                "reservation_id": res.id,
                "reason":         reason,
                "decided_by":     admin_id,
                "decided_at":     res.decided_at.isoformat() if res.decided_at else None,
            }
        return {"error": f"Unknown decision '{decision}'. Must be APPROVE or REFUSE."}
    except ValidationError as exc:
        return {"error": str(exc)}


# Convenience list — useful for building LLM tool schemas in future
TOOLS = [
    get_pending_reservations,
    get_reservation_details,
    check_lot_availability,
    send_admin_notification,
    record_admin_decision,
]

# Tools that code (never the LLM) invokes after a verified admin action.
# Kept out of TOOLS so they can never be bound to a model.
EXPORT_TOOLS = [write_reservation]


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_NOTIFICATION_SYSTEM = """\
You are the CityPark admin assistant preparing an approval request.

Rules:
1. Use ONLY the facts in the TOOL RESULTS section provided by the user.
2. Never invent numbers, prices, names, or times.
3. Include a conflict warning only if conflicts_in_window > 0.
4. Keep the text under 200 words.
5. End with: "Please reply: approve <id>  or  refuse <id> because <reason>"
"""

_USER_NOTIFICATION_SYSTEM = """\
You are the CityPark booking assistant writing to a customer about their
parking reservation.

Rules:
1. Be polite and concise (under 80 words).
2. APPROVE: confirm the booking details and wish them a good visit.
3. REFUSE:  apologise, state the reason given, invite them to try again.
4. Use only the decision, reason, and reservation ID provided — no invention.
"""


# ---------------------------------------------------------------------------
# AdminAgent
# ---------------------------------------------------------------------------


class AdminAgent:
    """
    Orchestrates the three-phase admin workflow.

    Inject *session_factory* (a zero-arg callable that returns a
    ``sqlalchemy.orm.Session``) and optionally a *llm* override for tests.
    """

    def __init__(
        self,
        session_factory: Any,
        llm: Optional[BaseLLMClient] = None,
    ) -> None:
        self._session_factory = session_factory
        self._llm = llm or get_llm_client()

    # ------------------------------------------------------------------
    # Phase 1 — generate admin notification
    # ------------------------------------------------------------------

    def generate_notification(self, request: ReservationRequest) -> str:
        """
        Fetch live facts from the DB via tools, then call the LLM to
        compose a human-readable admin notification.

        All numbers and facts in the output are grounded in tool results;
        the LLM only handles prose formatting.
        """
        db = self._session_factory()
        try:
            details = get_reservation_details.invoke(
                {"reservation_id": request.reservation_id, "db": db}
            )
            if "error" in details:
                logger.error(
                    "admin_agent|generate_notification error %s", details["error"]
                )
                return f"Error fetching reservation: {details['error']}"

            avail = check_lot_availability.invoke(
                {
                    "lot_id":     details["lot_id"],
                    "start_time": request.start_time.isoformat(),
                    "end_time":   request.end_time.isoformat(),
                    "db":         db,
                }
            )
        finally:
            db.close()

        context = json.dumps(
            {"reservation": details, "availability": avail},
            indent=2,
            default=str,
        )
        messages = [
            {"role": "system", "content": _NOTIFICATION_SYSTEM},
            {
                "role": "user",
                "content": f"TOOL RESULTS:\n{context}\n\nWrite the admin notification now.",
            },
        ]
        text = _call_with_retry(self._llm, messages)
        logger.info(
            "admin_agent|notification_generated reservation_id=%d chars=%d",
            request.reservation_id,
            len(text),
        )
        return text

    # ------------------------------------------------------------------
    # Phase 2 — parse admin decision
    # ------------------------------------------------------------------

    def parse_decision(
        self,
        admin_text: str,
        reservation_id: int,
        admin_id: str,
    ) -> AdminDecision | str:
        """
        Convert the administrator's natural-language input into a validated
        AdminDecision, or return a clarification prompt string.

        Safety gate
        -----------
        Step 2 is deterministic: if the raw message contains no explicit
        decision keyword the method returns a clarification and the LLM is
        never called.  This prevents autonomous LLM approvals.
        """
        # Step 1 — PII filter (admin message may quote customer data)
        guard = check_input(admin_text, skip_entities=["PERSON", "LICENSE_PLATE"])
        if guard.blocked:
            logger.warning(
                "admin_agent|parse_decision blocked admin_id=%s", admin_id
            )
            return guard.block_reason

        clean = guard.text

        # Step 2 — SAFETY GATE: deterministic keyword detection, no LLM
        keyword = _detect_decision_keyword(clean)
        if keyword is None:
            return (
                "I couldn't detect an explicit decision in your message.\n"
                "Please reply with:\n"
                "  • approve <reservation_id>\n"
                "  • refuse  <reservation_id> because <reason>"
            )

        # Step 3 — extract reservation ID (text wins; fallback to argument)
        parsed_id = _extract_id(clean) or reservation_id

        # Step 4 — reason extraction for REFUSE (regex → LLM fallback)
        reason: str | None = None
        if keyword == "REFUSE":
            reason = _extract_reason(clean, self._llm) or None
            if not (reason or "").strip():
                return (
                    f"Please provide a reason for refusing reservation "
                    f"#{parsed_id}.\n"
                    "Example: refuse 12 because the lot is closed for maintenance"
                )

        try:
            return AdminDecision(
                reservation_id=parsed_id,
                decision=Decision(keyword),
                reason=reason,
                admin_id=admin_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("admin_agent|parse_decision_build_error %s", exc)
            return f"Could not build a valid decision: {exc}"

    # ------------------------------------------------------------------
    # Phase 3 — execute decision
    # ------------------------------------------------------------------

    def execute_decision(
        self,
        decision: AdminDecision,
        db: Session,
    ) -> AgentMessage:
        """
        Apply the admin's decision via the record_admin_decision tool and
        return a USER_NOTIFICATION AgentMessage for Agent 1 to relay.
        """
        result = record_admin_decision.invoke(
            {
                "reservation_id": decision.reservation_id,
                "decision":       decision.decision.value,
                "reason":         decision.reason or "",
                "admin_id":       decision.admin_id,
                "db":             db,
            }
        )

        if "error" in result:
            logger.error(
                "admin_agent|execute_error reservation_id=%d %s",
                decision.reservation_id,
                result["error"],
            )
            return AgentMessage(
                type=MessageType.ERROR,
                payload={
                    "error":          result["error"],
                    "reservation_id": decision.reservation_id,
                },
                correlation_id=decision.reservation_id,
            )

        # Generate user-facing notification text
        messages = [
            {"role": "system", "content": _USER_NOTIFICATION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Decision: {decision.decision.value}\n"
                    f"Reason: {decision.reason or 'N/A'}\n"
                    f"Reservation ID: #{decision.reservation_id}"
                ),
            },
        ]
        notification_text = _call_with_retry(self._llm, messages)

        logger.info(
            "admin_agent|decision_executed reservation_id=%d decision=%s",
            decision.reservation_id,
            decision.decision.value,
        )
        return AgentMessage(
            type=MessageType.USER_NOTIFICATION,
            payload={
                "reservation_id":    decision.reservation_id,
                "decision":          decision.decision.value,
                "reason":            decision.reason,
                "notification_text": notification_text,
                # Admin-side only: Agent 1 must not relay this to the customer.
                "file_written":      result.get("file_written"),
                "file_note":         result.get("file_note"),
            },
            correlation_id=decision.reservation_id,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_admin_agent(session_factory: Any) -> AdminAgent:
    """Return a default AdminAgent using the module-level LLM singleton."""
    return AdminAgent(session_factory)
