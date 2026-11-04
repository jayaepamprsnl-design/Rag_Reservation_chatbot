"""Tests for app/rag/retriever.py — with mocked embed + chroma."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.rag.retriever import retrieve

# ---------------------------------------------------------------------------
# Mock data — simulates what similarity_search returns (chroma_store format)
# ---------------------------------------------------------------------------

_RAW_RESULTS = [
    {
        "text": "Zone A is the premium indoor parking area.",
        "metadata": {"source": "parking_info", "section": "Zones", "chunk_id": "aaa"},
        "score": 0.91,
    },
    {
        "text": "Zone B is the outdoor parking area at $3/hr.",
        "metadata": {"source": "parking_info", "section": "Pricing", "chunk_id": "bbb"},
        "score": 0.78,
    },
    {
        "text": "EV charging is available in Zone A.",
        "metadata": {"source": "ev_charging", "section": "Overview", "chunk_id": "ccc"},
        "score": 0.65,
    },
]


@pytest.fixture(autouse=True)
def mock_embed_and_search():
    """Patch all I/O so tests run without ChromaDB or SentenceTransformer."""
    with patch("app.rag.retriever.embed_query", return_value=[0.1] * 384), \
         patch("app.rag.retriever.similarity_search", return_value=_RAW_RESULTS):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrieve_returns_list():
    results = retrieve("where is the parking?")
    assert isinstance(results, list)


def test_retrieve_default_k_uses_settings():
    results = retrieve("parking info")
    # default retrieval_top_k=5, but mock returns 3 results
    assert len(results) == len(_RAW_RESULTS)


def test_retrieve_with_k_respected():
    # Patch to return exactly k results
    with patch("app.rag.retriever.similarity_search",
               return_value=_RAW_RESULTS[:1]):
        results = retrieve("anything", k=1)
    assert len(results) == 1


def test_retrieve_result_has_required_keys():
    results = retrieve("test query", k=3)
    for r in results:
        assert "text" in r
        assert "source" in r
        assert "section" in r
        assert "score" in r


def test_retrieve_score_is_rounded_float():
    results = retrieve("ev charging", k=3)
    for r in results:
        assert isinstance(r["score"], float)
        assert 0.0 <= r["score"] <= 1.0


def test_retrieve_source_comes_from_metadata():
    results = retrieve("zone pricing", k=2)
    sources = {r["source"] for r in results}
    assert "parking_info" in sources


def test_retrieve_section_comes_from_metadata():
    results = retrieve("zone pricing", k=2)
    sections = {r["section"] for r in results}
    assert "Zones" in sections or "Pricing" in sections


def test_retrieve_empty_results_returns_empty_list():
    with patch("app.rag.retriever.similarity_search", return_value=[]):
        results = retrieve("completely irrelevant query xyz")
    assert results == []


def test_retrieve_missing_metadata_keys_return_empty_strings():
    raw = [{"text": "Some text.", "metadata": {}, "score": 0.5}]
    with patch("app.rag.retriever.similarity_search", return_value=raw):
        results = retrieve("query")
    assert results[0]["source"] == ""
    assert results[0]["section"] == ""
