"""Tests for app/chat/router.py — classifier and handler."""
from __future__ import annotations

from datetime import time
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.router import (
    ClassifyResult,
    RouteResponse,
    RouterIntent,
    classify,
    classify_keyword,
    handle,
)
from app.db.models import Base, ParkingLot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_with_lots():
    """In-memory SQLite session seeded with 3 parking lots."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    db.add_all([
        ParkingLot(
            name="CityPark Central",
            address="123 Central Ave",
            total_spaces=350,
            available_spaces=287,
            hourly_price=8.00,
            daily_price=45.00,
            opens_at=time(6, 0),
            closes_at=time(22, 0),
        ),
        ParkingLot(
            name="CityPark Eastside",
            address="45 Eastside Blvd",
            total_spaces=220,
            available_spaces=194,
            hourly_price=5.00,
            daily_price=28.00,
            opens_at=time(6, 0),
            closes_at=time(22, 0),
        ),
        ParkingLot(
            name="CityPark Airport",
            address="1 Airport Connector Rd",
            total_spaces=480,
            available_spaces=431,
            hourly_price=6.00,
            daily_price=35.00,
            opens_at=time(0, 0),
            closes_at=time(23, 59),
        ),
    ])
    db.commit()
    yield db
    db.close()


# ---------------------------------------------------------------------------
# classify_keyword — 20 parametrized cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, exp_intent, exp_sub, exp_lot", [
    # ── STATIC_INFO ────────────────────────────────────────────────────────
    ("Where is the parking lot?",                  RouterIntent.STATIC_INFO,  None,           None),
    ("What are the parking rules?",                RouterIntent.STATIC_INFO,  None,           None),
    ("Do you have EV charging?",                   RouterIntent.STATIC_INFO,  None,           None),
    ("Is the lot wheelchair accessible?",          RouterIntent.STATIC_INFO,  None,           None),
    ("How do I cancel my reservation?",            RouterIntent.STATIC_INFO,  None,           None),
    ("What payment methods do you accept?",        RouterIntent.STATIC_INFO,  None,           None),
    ("How do I get to CityPark by subway?",        RouterIntent.STATIC_INFO,  None,           None),
    ("Tell me about the facilities",               RouterIntent.STATIC_INFO,  None,           None),
    # ── DYNAMIC_INFO ───────────────────────────────────────────────────────
    ("How many spaces are available at CityPark Eastside?",
                                                   RouterIntent.DYNAMIC_INFO, "availability", 2),
    ("What is the hourly rate at the airport?",    RouterIntent.DYNAMIC_INFO, "prices",       3),
    ("What time does the downtown lot close?",     RouterIntent.DYNAMIC_INFO, "hours",        1),
    ("How much does it cost to park?",             RouterIntent.DYNAMIC_INFO, "prices",       None),
    ("Are there any free spaces?",                 RouterIntent.DYNAMIC_INFO, "availability", None),
    ("What are your opening hours?",               RouterIntent.DYNAMIC_INFO, "hours",        None),
    ("What's the daily rate at CityPark Central?", RouterIntent.DYNAMIC_INFO, "prices",       1),
    # ── RESERVATION ────────────────────────────────────────────────────────
    ("I want to book a parking space",             RouterIntent.RESERVATION,  None,           None),
    ("Can I make a reservation for tomorrow?",     RouterIntent.RESERVATION,  None,           None),
    # ── SMALL_TALK ─────────────────────────────────────────────────────────
    ("Hello!",                                     RouterIntent.SMALL_TALK,   None,           None),
    ("Thanks for the help",                        RouterIntent.SMALL_TALK,   None,           None),
    # ── OUT_OF_SCOPE ───────────────────────────────────────────────────────
    ("What's the weather today?",                  RouterIntent.OUT_OF_SCOPE, None,           None),
    ("Can you recommend a restaurant nearby?",     RouterIntent.OUT_OF_SCOPE, None,           None),
])
def test_classify_keyword(text, exp_intent, exp_sub, exp_lot):
    result = classify_keyword(text)
    assert result.intent == exp_intent, f"Expected {exp_intent!r}, got {result.intent!r} for: {text!r}"
    if exp_sub is not None:
        assert result.sub_intent == exp_sub
    if exp_lot is not None:
        assert result.lot_id == exp_lot


# ---------------------------------------------------------------------------
# classify() — LLM path mocked, verify JSON parse + fallback
# ---------------------------------------------------------------------------


def test_classify_uses_llm_when_available():
    """When LLM returns valid JSON, classify() uses it."""
    mock_json = '{"intent":"DYNAMIC_INFO","entities":{"lot_name":"CityPark Airport","sub_intent":"prices","date":null}}'
    with patch("app.chat.router.call_llm", return_value=mock_json):
        result = classify("What does it cost at the airport?")
    assert result.intent == RouterIntent.DYNAMIC_INFO
    assert result.sub_intent == "prices"
    assert result.lot_id == 3


def test_classify_falls_back_when_llm_returns_garbage():
    """Invalid LLM JSON triggers keyword fallback."""
    with patch("app.chat.router.call_llm", return_value="sorry I cannot help"):
        result = classify("How much does parking cost?")
    assert result.intent == RouterIntent.DYNAMIC_INFO
    assert result.sub_intent == "prices"


def test_classify_falls_back_when_llm_raises():
    """LLM exception triggers keyword fallback."""
    with patch("app.chat.router.call_llm", side_effect=RuntimeError("timeout")):
        result = classify("Are there any free spaces?")
    assert result.intent == RouterIntent.DYNAMIC_INFO
    assert result.sub_intent == "availability"


def test_classify_llm_small_talk():
    mock_json = '{"intent":"SMALL_TALK","entities":{"lot_name":null,"sub_intent":null,"date":null}}'
    with patch("app.chat.router.call_llm", return_value=mock_json):
        result = classify("hey there!")
    assert result.intent == RouterIntent.SMALL_TALK


def test_classify_llm_out_of_scope():
    mock_json = '{"intent":"OUT_OF_SCOPE","entities":{"lot_name":null,"sub_intent":null,"date":null}}'
    with patch("app.chat.router.call_llm", return_value=mock_json):
        result = classify("What's the best pizza place nearby?")
    assert result.intent == RouterIntent.OUT_OF_SCOPE


def test_classify_llm_reservation_with_date():
    mock_json = '{"intent":"RESERVATION","entities":{"lot_name":"CityPark Central","sub_intent":null,"date":"2026-12-01"}}'
    with patch("app.chat.router.call_llm", return_value=mock_json):
        result = classify("I want to park at Central on December 1st")
    assert result.intent == RouterIntent.RESERVATION
    assert result.lot_id == 1
    assert result.entities.get("date") == "2026-12-01"


def test_classify_handles_markdown_fenced_json():
    """LLM sometimes wraps JSON in code fences — parser must strip them."""
    fenced = '```json\n{"intent":"STATIC_INFO","entities":{"lot_name":null,"sub_intent":null,"date":null}}\n```'
    with patch("app.chat.router.call_llm", return_value=fenced):
        result = classify("Tell me about the parking rules")
    assert result.intent == RouterIntent.STATIC_INFO


# ---------------------------------------------------------------------------
# handle() — DYNAMIC_INFO with real DB fixture
# ---------------------------------------------------------------------------


def test_handle_dynamic_availability(db_with_lots):
    with patch("app.chat.router.call_llm") as mock_llm:
        # First call: LLM classifier
        mock_llm.return_value = '{"intent":"DYNAMIC_INFO","entities":{"lot_name":"CityPark Eastside","sub_intent":"availability","date":null}}'
        # Second call: facts-to-answer LLM
        mock_llm.side_effect = [
            '{"intent":"DYNAMIC_INFO","entities":{"lot_name":"CityPark Eastside","sub_intent":"availability","date":null}}',
            "CityPark Eastside has 194 out of 220 spaces available right now.",
        ]
        resp = handle("How many spaces are free at Eastside?", db=db_with_lots)

    assert resp.intent == RouterIntent.DYNAMIC_INFO
    assert resp.sub_intent == "availability"
    assert resp.lot_id == 2
    assert "194" in resp.answer or "Eastside" in resp.answer


def test_handle_dynamic_prices(db_with_lots):
    with patch("app.chat.router.call_llm") as mock_llm:
        mock_llm.side_effect = [
            '{"intent":"DYNAMIC_INFO","entities":{"lot_name":"CityPark Airport","sub_intent":"prices","date":null}}',
            "CityPark Airport costs $6.00 per hour or $35.00 per day.",
        ]
        resp = handle("How much is parking at the airport?", db=db_with_lots)

    assert resp.intent == RouterIntent.DYNAMIC_INFO
    assert resp.sub_intent == "prices"
    assert resp.lot_id == 3
    assert "6.00" in resp.answer or "35" in resp.answer or "Airport" in resp.answer


def test_handle_dynamic_hours(db_with_lots):
    with patch("app.chat.router.call_llm") as mock_llm:
        mock_llm.side_effect = [
            '{"intent":"DYNAMIC_INFO","entities":{"lot_name":"CityPark Central","sub_intent":"hours","date":null}}',
            "CityPark Central is open from 06:00 to 22:00.",
        ]
        resp = handle("When does the downtown lot close?", db=db_with_lots)

    assert resp.intent == RouterIntent.DYNAMIC_INFO
    assert resp.sub_intent == "hours"
    assert "06:00" in resp.answer or "22:00" in resp.answer or "Central" in resp.answer


def test_handle_dynamic_all_lots(db_with_lots):
    """When no lot is specified, facts cover all three lots."""
    with patch("app.chat.router.call_llm") as mock_llm:
        mock_llm.side_effect = [
            '{"intent":"DYNAMIC_INFO","entities":{"lot_name":null,"sub_intent":"prices","date":null}}',
            "Central: $8/hr, Eastside: $5/hr, Airport: $6/hr.",
        ]
        resp = handle("What are the parking rates?", db=db_with_lots)

    assert resp.intent == RouterIntent.DYNAMIC_INFO
    assert resp.lot_id is None  # no specific lot
    assert "sources" in resp.__dataclass_fields__


# ---------------------------------------------------------------------------
# handle() — non-DB intents
# ---------------------------------------------------------------------------


def test_handle_small_talk():
    with patch("app.chat.router.call_llm",
               return_value='{"intent":"SMALL_TALK","entities":{"lot_name":null,"sub_intent":null,"date":null}}'):
        resp = handle("Hi there!")
    assert resp.intent == RouterIntent.SMALL_TALK
    assert "hello" in resp.answer.lower() or "help" in resp.answer.lower()


def test_handle_out_of_scope():
    with patch("app.chat.router.call_llm",
               return_value='{"intent":"OUT_OF_SCOPE","entities":{"lot_name":null,"sub_intent":null,"date":null}}'):
        resp = handle("What stocks should I buy?")
    assert resp.intent == RouterIntent.OUT_OF_SCOPE
    assert "parking" in resp.answer.lower()


def test_handle_reservation_intent():
    with patch("app.chat.router.call_llm",
               return_value='{"intent":"RESERVATION","entities":{"lot_name":null,"sub_intent":null,"date":null}}'):
        resp = handle("I'd like to book a space")
    assert resp.intent == RouterIntent.RESERVATION
    assert "reservation" in resp.answer.lower() or "book" in resp.answer.lower()


# ---------------------------------------------------------------------------
# RouteResponse structure
# ---------------------------------------------------------------------------


def test_route_response_has_all_fields():
    with patch("app.chat.router.call_llm",
               return_value='{"intent":"SMALL_TALK","entities":{"lot_name":null,"sub_intent":null,"date":null}}'):
        resp = handle("Hello")
    assert isinstance(resp, RouteResponse)
    assert resp.intent is not None
    assert isinstance(resp.sources, list)
    assert isinstance(resp.entities, dict)


def test_classify_result_defaults():
    r = ClassifyResult(intent=RouterIntent.STATIC_INFO)
    assert r.lot_id is None
    assert r.sub_intent is None
    assert r.entities == {}
