import pytest

from app.chat.intent_router import Intent, detect_intent


@pytest.mark.parametrize("text,expected", [
    ("I want to book a parking spot",        Intent.MAKE_RESERVATION),
    ("I need to make a reservation",         Intent.MAKE_RESERVATION),
    ("Are there any available spaces?",      Intent.CHECK_AVAILABILITY),
    ("Is the parking lot free right now?",   Intent.CHECK_AVAILABILITY),
    ("How much does it cost to park?",       Intent.CHECK_PRICES),
    ("What are the pricing rates?",          Intent.CHECK_PRICES),
    ("What are your opening hours?",         Intent.CHECK_HOURS),
    ("When do you close?",                   Intent.CHECK_HOURS),
    ("Tell me about your parking facility",  Intent.GENERAL_INFO),
    ("Where are you located?",               Intent.GENERAL_INFO),
])
def test_detect_intent(text: str, expected: Intent):
    assert detect_intent(text) == expected
