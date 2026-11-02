"""Load, chunk, and index static Markdown documents into ChromaDB."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from app.core.logging_config import logger
from app.rag.chroma_store import upsert_documents
from app.rag.embedder import embed_texts


# ---------------------------------------------------------------------------
# RecursiveCharacterTextSplitter (langchain-compatible, no extra dependency)
# ---------------------------------------------------------------------------

class RecursiveCharacterTextSplitter:
    """Split text on a cascade of separators until chunks fit within chunk_size."""

    _SEPARATORS = ["\n\n", "\n", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        return self._split(text, self._SEPARATORS)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        # Pick the first separator present in the text.
        sep = ""
        tail: list[str] = []
        for i, s in enumerate(separators):
            if s == "" or s in text:
                sep = s
                tail = separators[i + 1 :]
                break

        pieces = text.split(sep) if sep else list(text)

        good: list[str] = []
        for piece in pieces:
            if not piece.strip():
                continue
            if len(piece) > self.chunk_size and tail:
                good.extend(self._split(piece, tail))
            else:
                good.append(piece)

        return self._merge(good, sep)

    def _merge(self, pieces: list[str], sep: str) -> list[str]:
        """Greedily merge pieces into chunks of ≤ chunk_size with overlap."""
        chunks: list[str] = []
        window: list[str] = []
        window_len = 0

        for piece in pieces:
            p_len = len(piece)
            join = len(sep) if window else 0

            if window_len + join + p_len > self.chunk_size and window:
                chunk = sep.join(window).strip()
                if chunk:
                    chunks.append(chunk)
                # shrink window to fit overlap budget
                while window and window_len > self.chunk_overlap:
                    dropped = window.pop(0)
                    window_len -= len(dropped) + len(sep)
                    window_len = max(window_len, 0)

            window.append(piece)
            window_len += p_len + (len(sep) if len(window) > 1 else 0)

        if window:
            chunk = sep.join(window).strip()
            if chunk:
                chunks.append(chunk)

        return chunks


# ---------------------------------------------------------------------------
# Heading extraction helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def _extract_headings(text: str) -> list[tuple[int, str]]:
    """Return (char_offset, heading_text) pairs sorted by offset."""
    return [(m.start(), m.group(1).strip()) for m in _HEADING_RE.finditer(text)]


def _heading_for_chunk(chunk: str, text: str, headings: list[tuple[int, str]]) -> str:
    """Find the last heading whose offset is ≤ the chunk's position in text."""
    needle = chunk.lstrip()[:60]
    pos = text.find(needle)
    if pos == -1:
        return ""
    result = ""
    for offset, title in headings:
        if offset <= pos:
            result = title
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)


def ingest_file(path: Path, extra_meta: dict[str, Any] | None = None) -> int:
    text = path.read_text(encoding="utf-8")
    chunks = _splitter.split_text(text)
    headings = _extract_headings(text)

    ids, docs, embeds, metas = [], [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{path.name}:{i}".encode()).hexdigest()
        section = _heading_for_chunk(chunk, text, headings)
        ids.append(chunk_id)
        docs.append(chunk)
        metas.append(
            {
                **(extra_meta or {}),
                "source": path.stem,
                "section": section,
                "chunk_id": chunk_id,
            }
        )

    embeds = embed_texts(docs)
    upsert_documents(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
    logger.info("Ingested %d chunks from %s", len(chunks), path.name)
    return len(chunks)


def ingest_directory(directory: Path, glob: str = "**/*.md") -> int:
    total = 0
    for path in sorted(directory.glob(glob)):
        total += ingest_file(path)
    logger.info("Total chunks ingested: %d", total)
    return total
