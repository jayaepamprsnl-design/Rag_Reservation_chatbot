"""Embedding wrapper — swap the backend by subclassing or replacing the singleton."""
from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import settings


class Embedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Module-level singleton — replace this to swap the backend
# ---------------------------------------------------------------------------

_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(settings.embedding_model)
    return _embedder


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedder().embed_texts(texts)


def embed_query(text: str) -> List[float]:
    return get_embedder().embed_query(text)
