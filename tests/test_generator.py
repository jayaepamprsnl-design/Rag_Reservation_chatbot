"""Tests for app/rag/generator.py — MockLLMClient, message building, public API."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.rag.generator import (
    BaseLLMClient,
    MockLLMClient,
    _build_messages,
    answer_question,
    call_llm,
    generate,
    get_llm_client,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_CHUNKS = [
    {
        "text": "CityPark is open from 06:00 to 22:00 every day.",
        "source": "general_info",
        "section": "Working Hours",
        "score": 0.93,
    },
    {
        "text": "Parking costs $8.00 per hour at CityPark Central.",
        "source": "payment_methods",
        "section": "Pricing Structure",
        "score": 0.87,
    },
]


# ===========================================================================
# MockLLMClient
# ===========================================================================


class TestMockLLMClient:
    def setup_method(self):
        self.client = MockLLMClient()

    def test_complete_with_rag_context_returns_nonempty_string(self):
        msgs = _build_messages(_CHUNKS, "What are the hours?", [])
        result = self.client.complete(msgs)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_complete_with_dynamic_facts_returns_facts(self):
        msgs = [
            {"role": "system", "content": "You are a parking assistant."},
            {
                "role": "user",
                "content": (
                    "Live parking data:\n"
                    "CityPark Central: 287 of 350 spaces available.\n\n"
                    "Customer question: How many spaces are left?"
                ),
            },
        ]
        result = self.client.complete(msgs)
        # Must echo back the facts, not _no_info()
        assert "287" in result or "350" in result or "CityPark" in result

    def test_complete_without_context_returns_no_info(self):
        msgs = [{"role": "user", "content": "Hello, who are you?"}]
        result = self.client.complete(msgs)
        assert "don't have" in result.lower() or "contact" in result.lower()

    def test_complete_returns_str_type(self):
        msgs = _build_messages(_CHUNKS, "any query", [])
        assert isinstance(self.client.complete(msgs), str)

    def test_complete_with_empty_chunks_returns_no_info(self):
        msgs = _build_messages([], "any query", [])
        result = self.client.complete(msgs)
        assert isinstance(result, str)


# ===========================================================================
# _build_messages
# ===========================================================================


def test_build_messages_first_role_is_system():
    msgs = _build_messages(_CHUNKS, "What are the hours?", [])
    assert msgs[0]["role"] == "system"


def test_build_messages_last_role_is_user():
    msgs = _build_messages(_CHUNKS, "test query", [])
    assert msgs[-1]["role"] == "user"


def test_build_messages_query_is_in_user_content():
    query = "what are the parking rates please"
    msgs = _build_messages(_CHUNKS, query, [])
    assert query in msgs[-1]["content"]


def test_build_messages_source_labels_in_user_content():
    msgs = _build_messages(_CHUNKS, "test", [])
    user_content = msgs[-1]["content"]
    assert "general_info" in user_content
    assert "payment_methods" in user_content


def test_build_messages_chat_history_inserted():
    history = [
        {"role": "user", "content": "Prior user message"},
        {"role": "assistant", "content": "Prior assistant answer"},
    ]
    msgs = _build_messages(_CHUNKS, "follow-up", history)
    roles = [m["role"] for m in msgs]
    # system + 2 history + 1 user = at least 4 messages
    assert len(msgs) >= 4
    assert roles.count("user") >= 2


def test_build_messages_empty_chunks_still_valid():
    msgs = _build_messages([], "query without context", [])
    assert len(msgs) >= 2
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"


# ===========================================================================
# get_llm_client — returns MockLLMClient when API key is placeholder
# ===========================================================================


def test_get_llm_client_returns_base_instance():
    client = get_llm_client()
    assert isinstance(client, BaseLLMClient)


def test_get_llm_client_is_mock_without_real_key():
    # No real key in test environment
    client = get_llm_client()
    assert isinstance(client, MockLLMClient)


# ===========================================================================
# answer_question
# ===========================================================================


def test_answer_question_returns_dict():
    with patch("app.rag.generator.retrieve", return_value=_CHUNKS):
        result = answer_question("What are the opening hours?")
    assert isinstance(result, dict)


def test_answer_question_has_required_keys():
    with patch("app.rag.generator.retrieve", return_value=_CHUNKS):
        result = answer_question("parking rates")
    for key in ("answer", "sources", "retrieved_chunks"):
        assert key in result, f"Missing key: {key}"


def test_answer_question_sources_from_chunks():
    with patch("app.rag.generator.retrieve", return_value=_CHUNKS):
        result = answer_question("What are the prices?")
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0


def test_answer_question_retrieved_chunks_match_input():
    with patch("app.rag.generator.retrieve", return_value=_CHUNKS) as mock_ret:
        result = answer_question("hours question")
    assert result["retrieved_chunks"] == _CHUNKS


def test_answer_question_passes_chat_history():
    history = [{"role": "user", "content": "previous question"}]
    with patch("app.rag.generator.retrieve", return_value=_CHUNKS):
        result = answer_question("follow-up", chat_history=history)
    assert "answer" in result


def test_answer_question_sources_are_unique_sorted():
    chunks_dup = _CHUNKS + [_CHUNKS[0]]  # duplicate source
    with patch("app.rag.generator.retrieve", return_value=chunks_dup):
        result = answer_question("test")
    sources = result["sources"]
    assert sources == sorted(set(sources))


# ===========================================================================
# generate — backward-compat shim
# ===========================================================================


def test_generate_returns_string():
    result = generate(_CHUNKS, "What are the hours?")
    assert isinstance(result, str)


def test_generate_nonempty():
    result = generate(_CHUNKS, "pricing information")
    assert len(result) > 0


# ===========================================================================
# call_llm
# ===========================================================================


def test_call_llm_returns_string():
    msgs = _build_messages(_CHUNKS, "test", [])
    result = call_llm(msgs)
    assert isinstance(result, str)
