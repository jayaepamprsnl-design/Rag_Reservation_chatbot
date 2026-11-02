"""Retrieve relevant chunks from ChromaDB for a natural-language query."""
from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.rag.chroma_store import similarity_search
from app.rag.embedder import embed_query


def retrieve(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """Return top-k chunks ranked by similarity.

    Each result dict has:
        text    – the chunk text
        source  – originating file stem (e.g. "location_and_directions")
        section – nearest markdown heading above the chunk
        score   – cosine similarity in [0, 1]
    """
    embedding = embed_query(query)
    raw = similarity_search(embedding, top_k=k or settings.retrieval_top_k)
    return [
        {
            "text": r["text"],
            "source": r["metadata"].get("source", ""),
            "section": r["metadata"].get("section", ""),
            "score": round(r["score"], 4),
        }
        for r in raw
    ]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "where is the parking located?"
    results = retrieve(query, k=4)
    print(f"\nQuery: {query!r}\n{'-' * 60}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']}  source={r['source']}  section={r['section']!r}")
        print(r["text"][:300].replace("\n", " "))
