"""Input guardrail: prompt-injection detection + PII masking.

Applied before any user message reaches the LLM or the application log.

Exception for reservation flows
---------------------------------
During field collection (name, plate number), pass ``skip_entities`` to leave
those specific entity types unmasked so the state machine can store the raw
value.  The *masked* text returned here is still safe for logs and LLM context.

Public API
----------
check_input(text, skip_entities=None) -> InputGuardResult
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging_config import logger
from app.guardrails.pii_detector import mask

# ---------------------------------------------------------------------------
# Prompt-injection detection patterns
# ---------------------------------------------------------------------------

_RAW_PATTERNS: list[str] = [
    r"ignore\s+(previous|all|prior|my)\s+(instructions?|prompts?|rules?|context)",
    r"forget\s+(previous|all|prior|your)\s+(instructions?|prompts?|context|rules?)",
    r"disregard\s+(previous|all|prior|your)\s+(instructions?|prompts?|context)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions?|initial\s+prompt)",
    r"show\s+(me\s+)?(your|the)\s+system\s+prompt",
    r"print\s+(your\s+)?(system\s+prompt|instructions?)",
    r"what\s+(is|are|were)\s+your\s+(instructions?|system\s+prompt|initial\s+prompt)",
    r"(repeat|output|tell\s+me|give\s+me)\s+(your|the)\s+(system\s+prompt|instructions?)",
    r"\bjailbreak\b",
    r"\bdan\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"\bmaintenance\s+mode\b",
    r"bypass\s+(?:all\s+)?(?:the\s+)?(filter|guardrail|safety|restriction|rule)",
    r"override\s+(safety|guardrail|filter|restriction)",
    r"pretend\s+(you\s+(are|were)|to\s+be)\s+(?!a\s+parking)",
    r"act\s+as\s+if\s+you\s+(have\s+no|don.t\s+have)\s+(restrictions?|rules?|filter)",
    r"you\s+are\s+now\s+(a|an)\s+(?!parking\s+assistant)",
    r"roleplay\s+as\s+(?!a\s+parking\s+assistant)",
    r"\bdo\s+anything\s+now\b",
    r"as\s+an?\s+(ai|llm|language\s+model)\s+(without|with\s+no)\s+restrictions?",
    r"in\s+this\s+hypothetical\s+(scenario|situation).{0,40}you\s+can",
    r"simulate\s+(being\s+a?\s*)?(?:unrestricted|uncensored|jailbroken)",
    r"(output|print|display)\s+(?:the\s+)?(?:full\s+)?(?:your\s+)?system\s+(prompt|message|instructions?)",
]

_INJECTION_RE: list[re.Pattern[str]] = [
    re.compile(p, re.I) for p in _RAW_PATTERNS
]

_SAFE_REFUSAL = (
    "I'm here to help with parking questions only. "
    "I can't process that kind of request."
)

# ---------------------------------------------------------------------------
# Public types and function
# ---------------------------------------------------------------------------


@dataclass
class InputGuardResult:
    """Result returned by :func:`check_input`."""

    text: str                               # PII-masked; safe for logs and LLM
    blocked: bool = False                   # True → do not proceed; return block_reason
    block_reason: str = ""
    entities_found: list[str] = field(default_factory=list)


def check_input(
    text: str,
    skip_entities: list[str] | None = None,
) -> InputGuardResult:
    """Validate and clean *text* before it reaches the LLM or log system.

    Steps:
    1. Detect prompt injection in the raw text (before masking, in case the
       attack is embedded in what looks like PII).
    2. Mask PII using entity-specific labels (``<EMAIL>``, ``<NAME>``, …).

    Args:
        text:           Raw user message.
        skip_entities:  PII entity types to leave unmasked.  Use
                        ``["PERSON", "LICENSE_PLATE"]`` during reservation
                        field collection so the state machine can read the
                        raw name/plate, while still masking everything else.

    Returns:
        :class:`InputGuardResult` — ``blocked=True`` if injection is detected.
    """
    if not text:
        return InputGuardResult(text=text)

    # Step 1: Prompt-injection check on the *raw* text.
    for pattern in _INJECTION_RE:
        if pattern.search(text):
            logger.warning(
                "guardrail|injection_blocked: pattern=%.70s", pattern.pattern
            )
            return InputGuardResult(
                text=text,
                blocked=True,
                block_reason=_SAFE_REFUSAL,
            )

    # Step 2: PII masking.
    masked, entities = mask(text, skip_entities=skip_entities)
    return InputGuardResult(text=masked, entities_found=entities)
