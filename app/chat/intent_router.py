from enum import Enum
from typing import Dict, List


class Intent(str, Enum):
    MAKE_RESERVATION = "make_reservation"
    CHECK_AVAILABILITY = "check_availability"
    CHECK_PRICES = "check_prices"
    CHECK_HOURS = "check_hours"
    GENERAL_INFO = "general_info"


_KEYWORDS: Dict[Intent, List[str]] = {
    Intent.MAKE_RESERVATION: ["book", "reserve", "reservation", "want a spot", "need parking", "make a booking"],
    Intent.CHECK_AVAILABILITY: ["available", "availability", "free", "open spot", "empty space", "any spaces"],
    Intent.CHECK_PRICES: ["price", "cost", "fee", "rate", "charge", "how much", "pricing"],
    Intent.CHECK_HOURS: ["hours", "open", "close", "schedule", "when do you", "opening time", "closing time"],
}


def detect_intent(text: str) -> Intent:
    lower = text.lower()
    for intent, keywords in _KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent
    return Intent.GENERAL_INFO
