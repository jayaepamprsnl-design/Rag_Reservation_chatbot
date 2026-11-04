"""Backward-compatible PII filter — delegates to pii_detector.

Kept so existing call-sites (chat route, tests) continue to work unchanged.
New code should use pii_detector.mask() / input_guard.check_input() directly.
"""
from app.guardrails.pii_detector import detect as _detect, mask as _mask


def mask_pii(text: str, language: str = "en") -> str:
    """Replace detected PII with entity-type labels (``<EMAIL>``, ``<NAME>``, …)."""
    masked, _ = _mask(text, language=language)
    return masked


def has_pii(text: str, language: str = "en") -> bool:
    """Return True if *text* contains any recognised PII."""
    return bool(_detect(text, language=language))
