"""PII detection engine: Presidio with spaCy en_core_web_lg + custom recognizers.

Custom entities
---------------
API_KEY       — OpenAI sk-* keys, GitHub tokens, Bearer JWT tokens
PASSWORD      — 'password is/: <value>' style phrases
LICENSE_PLATE — vehicle plate numbers (e.g. ABC-1234, XY9900)

Public API
----------
detect(text)           -> list[RecognizerResult]
mask(text, skip=None)  -> (masked_text, sorted_entity_types_triggered)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.core.logging_config import logger

# ---------------------------------------------------------------------------
# Custom recognizers
# ---------------------------------------------------------------------------


class ApiKeyRecognizer(PatternRecognizer):
    """Detect API keys / bearer tokens / JWTs."""

    PATTERNS = [
        Pattern("openai_key",    r"\bsk-[A-Za-z0-9]{20,}\b",                                   0.95),
        Pattern("bearer_token",  r"\bBearer\s+[A-Za-z0-9\-._~+/]{10,}={0,2}\b",               0.95),
        Pattern("github_token",  r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",                            0.92),
        Pattern("generic_jwt",
                r"\beyJ[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b",             0.88),
    ]
    CONTEXT = ["token", "key", "api", "secret", "auth", "authorization", "bearer"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="API_KEY",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )


class PasswordRecognizer(PatternRecognizer):
    """Detect inline password disclosures such as 'password is hunter2'."""

    PATTERNS = [
        Pattern(
            "password_inline",
            r"(?i)\b(?:password|passwd|pwd|passphrase|secret)\s*(?:is|was|=|:)\s*\S{4,}",
            0.87,
        ),
    ]
    CONTEXT = ["login", "credentials", "authenticate", "auth"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="PASSWORD",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )


class LicensePlateRecognizer(PatternRecognizer):
    """Detect vehicle licence plates in formats common to CityPark customers."""

    PATTERNS = [
        # With hyphen separator — higher confidence: ABC-1234, XY-99
        Pattern("plate_hyphen",    r"\b[A-Z]{2,3}-[0-9]{2,4}[A-Z]{0,2}\b", 0.70),
        # Without separator — lower confidence: ABC1234, XY9900
        Pattern("plate_nohyphen",  r"\b[A-Z]{2,3}[0-9]{3,4}[A-Z]{0,2}\b",  0.55),
    ]
    CONTEXT = ["plate", "license", "registration", "vehicle", "car", "my plate"]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="LICENSE_PLATE",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
        )


# ---------------------------------------------------------------------------
# Entity catalog and replacement labels
# ---------------------------------------------------------------------------

ENTITIES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "API_KEY",
    "PASSWORD",
    "LICENSE_PLATE",
]

_LABELS: dict[str, str] = {
    "PERSON":         "<NAME>",
    "EMAIL_ADDRESS":  "<EMAIL>",
    "PHONE_NUMBER":   "<PHONE>",
    "CREDIT_CARD":    "<CARD>",
    "IBAN_CODE":      "<IBAN>",
    "IP_ADDRESS":     "<IP>",
    "URL":            "<URL>",
    "API_KEY":        "<API_KEY>",
    "PASSWORD":       "<PASSWORD>",
    "LICENSE_PLATE":  "<PLATE>",
}
_DEFAULT_LABEL = "<REDACTED>"

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_analyzer() -> AnalyzerEngine:
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        }
    )
    engine = AnalyzerEngine(nlp_engine=provider.create_engine())
    engine.registry.add_recognizer(ApiKeyRecognizer())
    engine.registry.add_recognizer(PasswordRecognizer())
    engine.registry.add_recognizer(LicensePlateRecognizer())
    return engine


@lru_cache(maxsize=1)
def _build_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect(
    text: str,
    entities: Optional[list[str]] = None,
    language: str = "en",
) -> list[RecognizerResult]:
    """Return Presidio results for all detected PII in *text*."""
    if not text:
        return []
    return _build_analyzer().analyze(
        text=text,
        language=language,
        entities=entities or ENTITIES,
    )


def mask(
    text: str,
    skip_entities: Optional[list[str]] = None,
    language: str = "en",
) -> tuple[str, list[str]]:
    """Replace PII in *text* with entity-type labels (``<EMAIL>``, ``<NAME>``, …).

    Args:
        text:           Input string.
        skip_entities:  Entity types to leave unmasked (e.g. during reservation
                        field collection where the SM needs the raw value).
        language:       Language code passed to Presidio.

    Returns:
        ``(masked_text, sorted_list_of_triggered_entity_types)``
        Only entity *types* are logged — raw values are never written to logs.
    """
    if not text:
        return text, []

    active = [e for e in ENTITIES if e not in (skip_entities or [])]
    if not active:
        return text, []

    results = detect(text, entities=active, language=language)
    if not results:
        return text, []

    entity_types = sorted({r.entity_type for r in results})
    for et in entity_types:
        logger.warning("guardrail|pii_detected: %s", et)

    operators: dict[str, OperatorConfig] = {
        et: OperatorConfig("replace", {"new_value": _LABELS.get(et, _DEFAULT_LABEL)})
        for et in entity_types
    }
    operators["DEFAULT"] = OperatorConfig("replace", {"new_value": _DEFAULT_LABEL})

    anonymized = _build_anonymizer().anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    return anonymized.text, entity_types
