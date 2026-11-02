from typing import Any, Dict, List, Optional

import chromadb

from app.core.config import settings
from app.core.logging_config import logger

_client: Optional[chromadb.PersistentClient] = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection() -> chromadb.Collection:
    return get_client().get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_documents(
    ids: List[str],
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: Optional[List[Dict[str, Any]]] = None,
) -> None:
    get_collection().upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas or [{} for _ in ids],
    )
    logger.info("Upserted %d documents into ChromaDB", len(ids))


def similarity_search(
    query_embedding: List[float],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    k = top_k or settings.retrieval_top_k
    results = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "metadata": meta, "score": 1.0 - dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
