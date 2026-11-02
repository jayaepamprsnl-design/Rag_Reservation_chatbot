"""Output guardrail: block system-prompt reveals and data leaks; redact residual PII.

Applied to every LLM response before it is returned to the user.

Blocking conditions
-------------------
1. **System-prompt reveal** — the model attempts to repeat its own instructions.
2. **Customer data leak** — the response contains more than one distinct booking
   reference (CPK-XXXXXX), suggesting it may be mixing up customer records.

For anything that is not blocked, any residual PII in the output is redacted
with entity-type labels so the user never accidentally receives raw PII.

Public API
----------
check_output(text) -> OutputGuardResult
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging_config import logger
from app.guardrails.pii_detector import mask

# ---------------------------------------------------------------------------
# System-prompt reveal patterns
# ---------------------------------------------------------------------------

_REVEAL_RAW: list[str] = [
    r"my\s+(system\s+prompt|initial\s+prompt)\s+(is|was|are|states?|says?)",
    r"i\s+(was|am|have\s+been)\s+(instructed|told|programmed|asked|directed)\s+to",
    r"as\s+stated\s+in\s+my\s+(instructions?|prompt|system\s+context)",
    r"the\s+instructions?\s+(given|provided)\s+to\s+me\s+(are|is|were|say)",
    r"in\s+my\s+system\s+prompt",
    r"my\s+instructions?\s+say",
]

_REVEAL_RE: list[re.Pattern[str]] = [re.compile(p, re.I) for p in _REVEAL_RAW]

# A single booking reference in a user's own session is fine; more than one
# unique reference in a single response suggests cross-customer leakage.
_BOOKING_REF_RE = re.compile(r"\bCPK-\d{6}\b")

_BLOCKED_RESPONSE = (
    "I'm sorry, I can't provide that information. "
    "Please contact our support team at info@citypark.example.com."
)

# ---------------------------------------------------------------------------
# Public type and function
# ---------------------------------------------------------------------------


@dataclass
class OutputGuardResult:
    """Result returned by :func:`check_output`."""

    text: str                                   # safe text to return to the user
    blocked: bool = False                       # True → original response suppressed
    block_reason: str = ""                      # "system-prompt-reveal" | "customer-data-leak"
    entities_redacted: list[str] = field(default_factory=list)


def check_output(
    text: str,
    skip_entities: list[str] | None = None,
) -> OutputGuardResult:
    """Validate and clean the LLM response before it reaches the user.

    Steps:
    1. Check for system-prompt reveal phrases.
    2. Check for customer data leakage (multiple booking references).
    3. Redact any residual PII in non-blocked responses.

    Args:
        text:           Raw LLM output.
        skip_entities:  PII entity types to leave unredacted.  Pass
                        ``["PERSON"]`` for reservation confirmation messages
                        where showing the customer's own name is intentional.

    Returns:
        :class:`OutputGuardResult`.  If ``blocked=True``, use ``block_reason``
        (a safe, user-facing string) instead of the original text.
    """
    if not text:
        return OutputGuardResult(text=text)

    # Step 1: System-prompt reveal.
    for pattern in _REVEAL_RE:
        if pattern.search(text):
            logger.warning(
                "guardrail|output_blocked: system_prompt_reveal pattern=%.70s",
                pattern.pattern,
            )
            return OutputGuardResult(
                text=_BLOCKED_RESPONSE,
                blocked=True,
                block_reason="system-prompt-reveal",
            )

    # Step 2: Customer data leakage (> 1 distinct booking reference).
    booking_refs = set(_BOOKING_REF_RE.findall(text))
    if len(booking_refs) > 1:
        logger.warning(
            "guardrail|output_blocked: customer_data_leak refs=%d", len(booking_refs)
        )
        return OutputGuardResult(
            text=_BLOCKED_RESPONSE,
            blocked=True,
            block_reason="customer-data-leak",
        )

    # Step 3: Redact residual PII.
    redacted, entities = mask(text, skip_entities=skip_entities)
    for et in entities:
        logger.warning("guardrail|output_redacted: %s", et)

    return OutputGuardResult(text=redacted, entities_redacted=entities)
