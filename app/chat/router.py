"""Message router: classify each user message and dispatch to the right handler.

Intent taxonomy
---------------
STATIC_INFO   – general knowledge in the vector DB (location, rules, EV charging, …)
DYNAMIC_INFO  – live SQL data (availability, prices, working hours)
RESERVATION   – start or continue the booking flow
SMALL_TALK    – greetings / thanks / farewells
OUT_OF_SCOPE  – unrelated to parking

Public surface
--------------
classify(text, chat_history)   -> ClassifyResult
handle(text, chat_history, db) -> RouteResponse
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging_config import logger
from app.db.repository import get_availability, get_prices, get_working_hours
from app.rag.generator import call_llm

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RouterIntent(str, Enum):
    STATIC_INFO = "STATIC_INFO"
    DYNAMIC_INFO = "DYNAMIC_INFO"
    RESERVATION = "RESERVATION"
    SMALL_TALK = "SMALL_TALK"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass
class ClassifyResult:
    intent: RouterIntent
    lot_id: int | None = None           # 1=Central, 2=Eastside, 3=Airport; None=all
    sub_intent: str | None = None       # "availability" | "prices" | "hours"
    entities: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteResponse:
    answer: str
    intent: RouterIntent
    sub_intent: str | None
    lot_id: int | None
    sources: list[str]
    entities: dict[str, Any]


# ---------------------------------------------------------------------------
# Lot-name resolution
# ---------------------------------------------------------------------------

_LOT_RE: dict[int, re.Pattern[str]] = {
    1: re.compile(r"\b(central|citypark\s+central|downtown)\b", re.I),
    2: re.compile(r"\b(eastside|citypark\s+eastside|east\s+business|east\s+district|east)\b", re.I),
    3: re.compile(r"\b(airport|citypark\s+airport|connector|transit\s+zone)\b", re.I),
}

_LOT_ID_TO_NAME: dict[int, str] = {
    1: "CityPark Central",
    2: "CityPark Eastside",
    3: "CityPark Airport",
}


def _resolve_lot(text: str) -> int | None:
    """Return the first matching lot ID, or None if no lot is mentioned."""
    for lot_id, pattern in _LOT_RE.items():
        if pattern.search(text):
            return lot_id
    return None


def _resolve_lot_from_name(name: str | None) -> int | None:
    """Resolve a lot name string (from LLM JSON) to a lot ID."""
    if not name:
        return None
    return _resolve_lot(name)


# ---------------------------------------------------------------------------
# Keyword / regex fallback classifier
# ---------------------------------------------------------------------------

_SMALL_TALK_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|good\s+(morning|afternoon|evening|day)|howdy|"
    r"thanks?(\s+[\w\s]{0,30})?|thank\s+you(\s+[\w\s]{0,20})?|"
    r"bye+|goodbye|see\s+you|cheers|great|awesome|perfect|cool)"
    r"\s*[!?.]*\s*$",
    re.I,
)

_KEYWORD_MAP: list[tuple[re.Pattern[str], RouterIntent, str | None]] = [
    # STATIC_INFO – cancellation (must come before RESERVATION to avoid misclassifying "cancel my reservation")
    (re.compile(r"\b(cancel|cancell|cancellation|refund)\b", re.I),
     RouterIntent.STATIC_INFO, None),
    # RESERVATION
    (re.compile(r"\b(book|reserve|reservation|make\s+a\s+booking|want\s+(a\s+spot|to\s+park)|"
                r"need\s+(a\s+space|parking)|i\s+(want|need)\s+to\s+park)\b", re.I),
     RouterIntent.RESERVATION, None),
    # DYNAMIC_INFO – availability  (use spaces? so both "space" and "spaces" match)
    (re.compile(r"\b(available|availability|free\s+spaces?|free\s+spots?|any\s+spaces?|"
                r"empty\s+spaces?|how\s+many\s+spaces?|spots\s+left|spaces\s+left)\b", re.I),
     RouterIntent.DYNAMIC_INFO, "availability"),
    # DYNAMIC_INFO – prices
    (re.compile(r"\b(price|cost|rate|fee|charge|how\s+much|pricing|tariff|affordable)\b", re.I),
     RouterIntent.DYNAMIC_INFO, "prices"),
    # DYNAMIC_INFO – hours
    (re.compile(r"\b(hours|hour|open|close|closing|opening|schedule|when\s+do\s+you|"
                r"what\s+time\s+do|what\s+time\s+does)\b", re.I),
     RouterIntent.DYNAMIC_INFO, "hours"),
    # STATIC_INFO – explicit topics
    (re.compile(r"\b(where|location|direction|how\s+to\s+get|address|gps|transit|subway|"
                r"bus|metro|rule|regulation|policy|ev\s+charg|electric|wheelchair|accessible|"
                r"accessibility|cancel|payment|method|pay|security|safe|cctv|facility|"
                r"facilities|about|general|info|what\s+is|tell\s+me|explain|"
                r"park(ing)?\s+rule|overnight|permit)\b", re.I),
     RouterIntent.STATIC_INFO, None),
]

_OUT_OF_SCOPE_RE = re.compile(
    r"\b(weather|restaurant|food|hotel|flight|train|taxi|uber|lyft|"
    r"movie|sport|news|stock|crypto|bitcoin|recipe|health|doctor|"
    r"hospital|school|university)\b",
    re.I,
)


def classify_keyword(text: str) -> ClassifyResult:
    """Keyword/regex classifier — always offline-safe, used as fallback."""
    if _SMALL_TALK_RE.match(text):
        return ClassifyResult(intent=RouterIntent.SMALL_TALK)

    if _OUT_OF_SCOPE_RE.search(text):
        return ClassifyResult(intent=RouterIntent.OUT_OF_SCOPE)

    lot_id = _resolve_lot(text)

    for pattern, intent, sub in _KEYWORD_MAP:
        if pattern.search(text):
            return ClassifyResult(
                intent=intent,
                lot_id=lot_id,
                sub_intent=sub,
                entities={"lot_name": _LOT_ID_TO_NAME.get(lot_id) if lot_id else None},
            )

    # Default: treat as general parking info query.
    return ClassifyResult(
        intent=RouterIntent.STATIC_INFO,
        lot_id=lot_id,
        entities={"lot_name": _LOT_ID_TO_NAME.get(lot_id) if lot_id else None},
    )


# ---------------------------------------------------------------------------
# LLM-based classifier
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """\
You are a classifier for a parking chatbot. Respond with ONLY a single-line JSON object — no prose, \
no markdown fences.

Intent values:
- "STATIC_INFO"  — questions about parking rules, location/directions, facilities, booking process,
                   EV charging, accessibility, cancellation policy, payment methods, general info
- "DYNAMIC_INFO" — questions about current availability, prices/rates, opening/closing hours
- "RESERVATION"  — user wants to book, reserve, or make a booking
- "SMALL_TALK"   — greetings, thanks, farewells ("hi", "thanks", "bye")
- "OUT_OF_SCOPE" — completely unrelated to parking

Entity fields (always include all three, use null when absent):
- "lot_name": one of "CityPark Central" | "CityPark Eastside" | "CityPark Airport" | null
- "sub_intent": for DYNAMIC_INFO — "availability" | "prices" | "hours"; else null
- "date": any date mentioned in ISO format YYYY-MM-DD, else null

Examples:
{"intent":"DYNAMIC_INFO","entities":{"lot_name":"CityPark Airport","sub_intent":"prices","date":null}}
{"intent":"STATIC_INFO","entities":{"lot_name":null,"sub_intent":null,"date":null}}
{"intent":"RESERVATION","entities":{"lot_name":null,"sub_intent":null,"date":"2026-11-15"}}
{"intent":"SMALL_TALK","entities":{"lot_name":null,"sub_intent":null,"date":null}}
{"intent":"OUT_OF_SCOPE","entities":{"lot_name":null,"sub_intent":null,"date":null}}
"""

_VALID_INTENTS = {e.value for e in RouterIntent}
_VALID_SUBS = {"availability", "prices", "hours", None}


def _parse_classify_response(raw: str) -> ClassifyResult | None:
    """Parse the LLM's JSON response into a ClassifyResult; return None on error."""
    try:
        # Strip any accidental markdown fences.
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None

    intent_str = obj.get("intent", "")
    if intent_str not in _VALID_INTENTS:
        return None

    entities: dict[str, Any] = obj.get("entities", {})
    sub = entities.get("sub_intent")
    if sub not in _VALID_SUBS:
        sub = None

    lot_id = _resolve_lot_from_name(entities.get("lot_name"))
    return ClassifyResult(
        intent=RouterIntent(intent_str),
        lot_id=lot_id,
        sub_intent=sub,
        entities=entities,
    )


def _classify_llm(text: str, chat_history: list[dict[str, str]]) -> ClassifyResult | None:
    """Call the LLM classifier; return None if the response is invalid."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
        *chat_history[-4:],                         # at most 4 prior turns for context
        {"role": "user", "content": text},
    ]
    try:
        raw = call_llm(messages)
        result = _parse_classify_response(raw)
        if result is None:
            logger.warning("LLM classifier returned non-JSON: %.80s", raw)
        return result
    except Exception as exc:
        logger.warning("LLM classifier failed (%s) — falling back to keyword.", exc)
        return None


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------


def classify(
    text: str,
    chat_history: list[dict[str, str]] | None = None,
) -> ClassifyResult:
    """Classify *text* — tries LLM first, keyword fallback on any failure."""
    result = _classify_llm(text, chat_history or [])
    if result is None:
        result = classify_keyword(text)
        logger.debug("Using keyword classifier → %s", result.intent)
    else:
        logger.debug("LLM classifier → %s", result.intent)
    return result


# ---------------------------------------------------------------------------
# DYNAMIC_INFO: SQL → facts → grounded LLM answer
# ---------------------------------------------------------------------------

_DYNAMIC_SYSTEM = """\
You are CityPark's parking assistant. Answer the customer's question using ONLY \
the factual data provided below (queried live from the database). \
Rules:
1. Never invent, round, or modify any number — use the exact values given.
2. Keep the answer under 80 words.
3. Be friendly and direct.
4. Do not add a "Sources:" line for database answers.
"""


def _fetch_dynamic_facts(
    sub_intent: str | None,
    lot_id: int | None,
    db: Session,
) -> str:
    """Query SQL and return a compact fact string for every relevant lot."""
    lot_ids = [lot_id] if lot_id else list(_LOT_ID_TO_NAME.keys())
    lines: list[str] = []

    for lid in lot_ids:
        try:
            if sub_intent == "availability":
                data = get_availability(db, lid)
                lines.append(
                    f"{data['name']}: {data['available_spaces']} of "
                    f"{data['total_spaces']} spaces available."
                )
            elif sub_intent == "prices":
                data = get_prices(db, lid)
                lines.append(
                    f"{data['name']}: ${data['hourly_price']:.2f}/hr, "
                    f"${data['daily_price']:.2f}/day ({data['currency']})."
                )
            elif sub_intent == "hours":
                data = get_working_hours(db, lid)
                lines.append(f"{data['name']}: open {data['opens_at']} – {data['closes_at']}.")
            else:
                # Unknown sub_intent — return all three data types.
                av = get_availability(db, lid)
                pr = get_prices(db, lid)
                wh = get_working_hours(db, lid)
                lines.append(
                    f"{av['name']}: {av['available_spaces']}/{av['total_spaces']} spaces, "
                    f"${pr['hourly_price']:.2f}/hr, open {wh['opens_at']}–{wh['closes_at']}."
                )
        except Exception as exc:
            logger.warning("Could not fetch data for lot %d: %s", lid, exc)

    return "\n".join(lines) if lines else "No data available."


def _answer_from_facts(facts: str, query: str, history: list[dict[str, str]]) -> str:
    """Feed SQL facts to the LLM for a strictly grounded natural-language answer."""
    user_msg = f"Live parking data:\n{facts}\n\nCustomer question: {query}"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _DYNAMIC_SYSTEM},
        *history[-4:],
        {"role": "user", "content": user_msg},
    ]
    return call_llm(messages)


# ---------------------------------------------------------------------------
# STATIC_INFO handler (thin wrapper over generator.answer_question)
# ---------------------------------------------------------------------------


def _answer_static(query: str, history: list[dict[str, str]]) -> tuple[str, list[str]]:
    from app.rag.generator import answer_question

    result = answer_question(query, history)
    return result["answer"], result["sources"]


# ---------------------------------------------------------------------------
# Public handle function
# ---------------------------------------------------------------------------

_SMALL_TALK_REPLY = (
    "Hello! I'm the CityPark assistant. I can help you with parking availability, "
    "prices, directions, or making a reservation. What can I do for you today?"
)
_OUT_OF_SCOPE_REPLY = (
    "I'm only able to help with parking-related questions. "
    "For other queries, please try a general search engine."
)


def handle(
    text: str,
    chat_history: list[dict[str, str]] | None = None,
    db: Session | None = None,
) -> RouteResponse:
    """Classify *text* and return a fully resolved RouteResponse.

    Args:
        text:         The user's message.
        chat_history: Prior conversation turns.
        db:           SQLAlchemy session (opened internally if not provided).
    """
    history = chat_history or []
    result = classify(text, history)

    answer: str
    sources: list[str] = []

    if result.intent == RouterIntent.SMALL_TALK:
        answer = _SMALL_TALK_REPLY

    elif result.intent == RouterIntent.OUT_OF_SCOPE:
        answer = _OUT_OF_SCOPE_REPLY

    elif result.intent == RouterIntent.RESERVATION:
        answer = (
            "I'd love to help you make a reservation! "
            "Please provide your full name, vehicle plate number, "
            "preferred facility, and your desired arrival and departure times."
        )

    elif result.intent == RouterIntent.DYNAMIC_INFO:
        _db_provided = db is not None
        if not _db_provided:
            from app.db.session import SessionLocal
            db = SessionLocal()
        try:
            facts = _fetch_dynamic_facts(result.sub_intent, result.lot_id, db)
            answer = _answer_from_facts(facts, text, history)
            sources = ["database"]
        finally:
            if not _db_provided:
                db.close()  # type: ignore[union-attr]

    else:  # STATIC_INFO (default)
        answer, sources = _answer_static(text, history)

    return RouteResponse(
        answer=answer,
        intent=result.intent,
        sub_intent=result.sub_intent,
        lot_id=result.lot_id,
        sources=sources,
        entities=result.entities,
    )
