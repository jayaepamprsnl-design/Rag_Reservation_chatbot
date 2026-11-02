"""RAG answer generation: OpenAI-compatible LLM with offline MockLLM fallback.

Public API
----------
answer_question(query, chat_history) -> {answer, sources, retrieved_chunks}
generate(context_docs, query)        -> str   (backward-compat shim for chat.py)
"""
from __future__ import annotations

import abc
import random
import re
import time
from typing import Any

import openai
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from app.core.config import settings
from app.core.logging_config import logger
from app.rag.retriever import retrieve

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are CityPark's parking assistant. Answer the customer's question using ONLY \
the context passages provided below.

Rules:
1. Give a concise, direct answer (under 150 words unless the question requires more detail).
2. End every answer with a "Sources:" line listing the document names you drew from, \
   e.g. "Sources: general_info, location_and_directions".
3. If the context does not contain enough information to answer, reply with exactly:
   "I don't have that information. Please contact us at \
info@citypark.example.com or call +1 (555) 200-7000."
4. Never invent, infer, or add any detail that is not explicitly stated in the context.
"""


def _build_messages(
    chunks: list[dict[str, Any]],
    query: str,
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source", "")
        sec = c.get("section", "")
        tag = f"[{i}] source={src!r}" + (f", section={sec!r}" if sec else "")
        lines.append(f"{tag}\n{c['text']}")
    context_block = "\n\n".join(lines)

    user_msg = (
        f"Context from parking documentation:\n\n{context_block}"
        f"\n\n---\nCustomer question: {query}"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_msg})
    return messages


# ---------------------------------------------------------------------------
# LLM client hierarchy
# ---------------------------------------------------------------------------

class BaseLLMClient(abc.ABC):
    """Swap implementations by replacing the singleton returned by get_llm_client()."""

    @abc.abstractmethod
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAIClient(BaseLLMClient):
    """Thin wrapper around the openai SDK.  Retries are handled at the call site."""

    def __init__(self) -> None:
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            timeout=settings.llm_timeout,
            max_retries=0,          # retries managed by _call_with_retry
        )
        self._model = settings.openai_model
        self._temperature = settings.llm_temperature

    def complete(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""


class MockLLMClient(BaseLLMClient):
    """Offline fallback — synthesises a deterministic response from the context block.

    Used automatically when no real API key is configured, and in tests.
    """

    # cosine score below which we declare the context insufficient
    _MIN_SCORE = 0.35

    # Lines that carry no standalone semantic value in a paragraph context.
    _SKIP_PREFIXES = ("#", "|", "---", "-|-", "+-")
    _CHUNK_TAG_RE = re.compile(r"^\[\d+\]\s+source=", re.MULTILINE)

    def complete(self, messages: list[dict[str, str]]) -> str:
        user_content = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        # ── Dynamic-facts format ("Live parking data:") ───────────────────
        # Used by router._answer_from_facts — return the facts verbatim.
        if "Live parking data:" in user_content:
            data_marker = "Live parking data:\n"
            q_marker_dyn = "\n\nCustomer question:"
            start = user_content.find(data_marker)
            end   = user_content.find(q_marker_dyn)
            if start != -1:
                facts = user_content[
                    start + len(data_marker): end if end != -1 else start + 600
                ].strip()
                if facts:
                    return facts
            return self._no_info()

        # ── RAG context format ("Context from parking documentation:") ────
        header = "Context from parking documentation:\n\n"
        divider = "\n\n---\n"
        h_pos = user_content.find(header)
        d_pos = user_content.find(divider)
        if h_pos == -1:
            return self._no_info()

        raw_context = user_content[h_pos + len(header): d_pos if d_pos != -1 else h_pos + 800]

        # Split into individual chunk blocks on "[N] source=" markers.
        blocks = self._CHUNK_TAG_RE.split(raw_context)
        tag_lines = self._CHUNK_TAG_RE.findall(raw_context)

        # Reconstruct (tag_line, body) pairs so we have the source label per block.
        pairs: list[tuple[str, str]] = []
        for i, block in enumerate(blocks):
            if i == 0:
                # text before the first tag — usually empty
                if block.strip():
                    pairs.append(("", block))
            else:
                tag = tag_lines[i - 1] if i - 1 < len(tag_lines) else ""
                pairs.append((tag + block.split("\n")[0], block))

        # Extract (src, body_lines) for each block.
        candidates: list[tuple[str, list[str]]] = []
        for tag_with_rest, block in pairs:
            src = ""
            if "source=" in tag_with_rest:
                s = tag_with_rest.split("source=", 1)[1]
                src = s.split(",")[0].strip("'\" ")
            candidates.append((src, block.splitlines()[1:]))

        # Re-rank by query-term overlap so the most relevant block is tried first.
        # Extract the query from the last user message (after "Customer question:").
        q_marker = "Customer question:"
        q_pos = user_content.rfind(q_marker)
        raw_query = user_content[q_pos + len(q_marker):].strip() if q_pos != -1 else ""
        query_terms = set(re.findall(r"\w{3,}", raw_query.lower()))

        def _overlap(body_lines: list[str]) -> int:
            text = " ".join(body_lines).lower()
            return sum(1 for t in query_terms if t in text)

        candidates.sort(key=lambda pair: _overlap(pair[1]), reverse=True)

        # Walk re-ranked blocks; pick the first substantive sentence.
        for src, body_lines in candidates:
            for line in body_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if any(stripped.startswith(p) for p in self._SKIP_PREFIXES):
                    continue
                if self._CHUNK_TAG_RE.match(stripped):
                    continue
                if len(stripped) < 40:
                    continue
                suffix = f"\n\nSources: {src}" if src else ""
                return f"{stripped[:300]}{suffix}"

        return self._no_info()

    @staticmethod
    def _no_info() -> str:
        return (
            "I don't have that information. Please contact us at "
            "info@citypark.example.com or call +1 (555) 200-7000."
        )


# ---------------------------------------------------------------------------
# Exponential-backoff retry
# ---------------------------------------------------------------------------

_RETRYABLE = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)


def _call_with_retry(
    client: BaseLLMClient,
    messages: list[dict[str, str]],
) -> str:
    max_retries = settings.llm_max_retries
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.complete(messages)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = 2 ** attempt + random.uniform(0.0, 0.5)
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_llm: BaseLLMClient | None = None

_PLACEHOLDER_PREFIXES = ("sk-placeholder", "your-", "")


def _is_placeholder(key: str) -> bool:
    return not key or any(key.startswith(p) for p in _PLACEHOLDER_PREFIXES)


def get_llm_client() -> BaseLLMClient:
    global _llm
    if _llm is None:
        if _is_placeholder(settings.openai_api_key):
            logger.info("No real API key — MockLLMClient active (offline mode).")
            _llm = MockLLMClient()
        else:
            logger.info("OpenAIClient: model=%s base=%s", settings.openai_model, settings.openai_api_base)
            _llm = OpenAIClient()
    return _llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Full RAG pipeline: retrieve → prompt → generate.

    Args:
        query:        The user's question.
        chat_history: Previous turns as [{"role": "user"/"assistant", "content": "..."}].

    Returns:
        answer           – generated text string
        sources          – sorted list of unique source file stems cited
        retrieved_chunks – raw list of chunk dicts from the retriever
    """
    chunks = retrieve(query, k=settings.retrieval_top_k)
    messages = _build_messages(chunks, query, chat_history or [])
    answer = _call_with_retry(get_llm_client(), messages)
    sources = sorted({c["source"] for c in chunks if c.get("source")})
    logger.debug("answer_question: query=%.60s sources=%s", query, sources)
    return {"answer": answer, "sources": sources, "retrieved_chunks": chunks}


def call_llm(messages: list[dict[str, str]]) -> str:
    """Call the configured LLM with retry using the module-level singleton."""
    return _call_with_retry(get_llm_client(), messages)


def generate(context_docs: list[dict[str, Any]], query: str) -> str:
    """Backward-compatible shim — callers that already hold chunks skip re-retrieval."""
    messages = _build_messages(context_docs, query, [])
    return _call_with_retry(get_llm_client(), messages)
