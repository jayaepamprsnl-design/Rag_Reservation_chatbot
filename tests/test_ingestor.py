"""Tests for app/rag/ingestor.py — splitter, heading helpers, ingest helpers."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.rag.ingestor import (
    RecursiveCharacterTextSplitter,
    _extract_headings,
    _heading_for_chunk,
    ingest_directory,
    ingest_file,
)


# ===========================================================================
# RecursiveCharacterTextSplitter
# ===========================================================================


def test_splitter_short_text_is_single_chunk():
    s = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    assert s.split_text("Short text.") == ["Short text."]


def test_splitter_respects_chunk_size_hard_text():
    s = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    long = "word " * 100
    chunks = s.split_text(long)
    # Every chunk must be at most chunk_size + some tolerance (merge boundary)
    for c in chunks:
        assert len(c) <= 60, f"Chunk too long: {len(c)}"


def test_splitter_splits_on_double_newline():
    # Each paragraph is 45 chars; chunk_size=50 forces a split between them.
    s = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
    text = "First paragraph with some extra words here.\n\nSecond paragraph with different words here."
    chunks = s.split_text(text)
    assert len(chunks) >= 2


def test_splitter_empty_string_returns_empty():
    s = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    assert s.split_text("") == []


def test_splitter_whitespace_only_returns_empty():
    s = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    result = s.split_text("   \n\n   ")
    assert result == []


def test_splitter_overlap_produces_multiple_chunks():
    s = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=15)
    text = "A" * 120
    chunks = s.split_text(text)
    assert len(chunks) > 2


def test_splitter_default_params():
    s = RecursiveCharacterTextSplitter()
    assert s.chunk_size == 500
    assert s.chunk_overlap == 80


# ===========================================================================
# _extract_headings
# ===========================================================================


def test_extract_headings_empty_text():
    assert _extract_headings("No headings.") == []


def test_extract_headings_h1():
    text = "# Main Title\n\nContent."
    headings = _extract_headings(text)
    assert len(headings) == 1
    assert headings[0][1] == "Main Title"


def test_extract_headings_h2_h3():
    text = "## Section\n\nParagraph.\n\n### Subsection\n\nMore."
    headings = _extract_headings(text)
    titles = {h[1] for h in headings}
    assert "Section" in titles
    assert "Subsection" in titles


def test_extract_headings_offsets_increase():
    text = "# First\n\nContent.\n\n# Second\n\nMore."
    headings = _extract_headings(text)
    assert headings[0][0] < headings[1][0]


def test_extract_headings_strips_whitespace():
    text = "#  Padded Title  \n\nContent."
    headings = _extract_headings(text)
    assert headings[0][1] == "Padded Title"


# ===========================================================================
# _heading_for_chunk
# ===========================================================================


def test_heading_for_chunk_returns_nearest_heading():
    text = "# Intro\n\nIntro text.\n\n## Details\n\nDetail content here."
    headings = _extract_headings(text)
    section = _heading_for_chunk("Detail content here.", text, headings)
    assert section == "Details"


def test_heading_for_chunk_before_any_heading_is_empty():
    text = "Preamble text before any heading.\n\n# Title\n\nContent."
    headings = _extract_headings(text)
    section = _heading_for_chunk("Preamble text before any heading.", text, headings)
    assert section == ""


def test_heading_for_chunk_not_found_returns_empty():
    text = "# Title\n\nContent."
    headings = _extract_headings(text)
    section = _heading_for_chunk("XYZZY not in text", text, headings)
    assert section == ""


# ===========================================================================
# ingest_file (mocked upsert + embed)
# ===========================================================================


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    md = tmp_path / "test_doc.md"
    md.write_text(
        "# Test Heading\n\nThis is test content for the parking chatbot evaluation system.\n\n"
        "## Details\n\nMore details about the parking facility and its services here.",
        encoding="utf-8",
    )
    return md


def test_ingest_file_returns_positive_count(markdown_file):
    with patch("app.rag.ingestor.upsert_documents") as mock_up, \
         patch("app.rag.ingestor.embed_texts", return_value=[[0.1] * 384]):
        count = ingest_file(markdown_file)
    assert count >= 1


def test_ingest_file_calls_upsert_once(markdown_file):
    with patch("app.rag.ingestor.upsert_documents") as mock_up, \
         patch("app.rag.ingestor.embed_texts", return_value=[[0.1] * 384]):
        ingest_file(markdown_file)
    mock_up.assert_called_once()


def test_ingest_file_metadata_has_source(markdown_file):
    captured: list[dict] = []

    def _capture(ids, documents, embeddings, metadatas):
        captured.extend(metadatas)

    with patch("app.rag.ingestor.upsert_documents", side_effect=_capture), \
         patch("app.rag.ingestor.embed_texts", return_value=[[0.1] * 384]):
        ingest_file(markdown_file)

    assert all("source" in m for m in captured)
    assert all(m["source"] == "test_doc" for m in captured)


def test_ingest_file_metadata_has_chunk_id(markdown_file):
    captured: list[dict] = []

    def _capture(ids, documents, embeddings, metadatas):
        captured.extend(metadatas)

    with patch("app.rag.ingestor.upsert_documents", side_effect=_capture), \
         patch("app.rag.ingestor.embed_texts", return_value=[[0.1] * 384]):
        ingest_file(markdown_file)

    assert all("chunk_id" in m for m in captured)
    # chunk_ids are hex strings
    assert all(len(m["chunk_id"]) == 32 for m in captured)


def test_ingest_file_metadata_has_section(markdown_file):
    captured: list[dict] = []

    def _capture(ids, documents, embeddings, metadatas):
        captured.extend(metadatas)

    with patch("app.rag.ingestor.upsert_documents", side_effect=_capture), \
         patch("app.rag.ingestor.embed_texts", return_value=[[0.1] * 384]):
        ingest_file(markdown_file)

    assert all("section" in m for m in captured)


# ===========================================================================
# ingest_directory (mocked)
# ===========================================================================


def test_ingest_directory_aggregates_all_files(tmp_path: Path):
    for i in range(3):
        (tmp_path / f"doc{i}.md").write_text(
            f"# Document {i}\n\nContent for document {i} in the parking facility.",
            encoding="utf-8",
        )

    total: list[int] = [0]

    def _capture(ids, documents, embeddings, metadatas):
        total[0] += len(ids)

    with patch("app.rag.ingestor.upsert_documents", side_effect=_capture), \
         patch("app.rag.ingestor.embed_texts", return_value=[[0.1] * 384]):
        count = ingest_directory(tmp_path)

    assert count == total[0]
    assert count >= 3


def test_ingest_directory_empty_dir(tmp_path: Path):
    with patch("app.rag.ingestor.upsert_documents"), \
         patch("app.rag.ingestor.embed_texts", return_value=[]):
        count = ingest_directory(tmp_path)
    assert count == 0


def test_ingest_directory_only_md_files(tmp_path: Path):
    (tmp_path / "doc.md").write_text("# Title\n\nContent here for testing.")
    (tmp_path / "skip.txt").write_text("This should be skipped.")

    processed: list[str] = []

    def _capture(ids, documents, embeddings, metadatas):
        processed.extend(m["source"] for m in metadatas)

    with patch("app.rag.ingestor.upsert_documents", side_effect=_capture), \
         patch("app.rag.ingestor.embed_texts", return_value=[[0.1] * 384]):
        ingest_directory(tmp_path)

    assert all(s == "doc" for s in processed)
    assert "skip" not in processed
