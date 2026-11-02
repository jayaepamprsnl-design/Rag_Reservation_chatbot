#!/usr/bin/env python3
"""Ingest static Markdown documents into the ChromaDB 'parking_static' collection.

Usage:
    python scripts/ingest.py            # idempotent upsert
    python scripts/ingest.py --reset    # drop and rebuild the collection
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.core.logging_config import logger
from app.rag.chroma_store import get_client
from app.rag.ingestor import ingest_directory


def _reset_collection() -> None:
    client = get_client()
    try:
        client.delete_collection(settings.chroma_collection)
        logger.info("Deleted existing collection '%s'.", settings.chroma_collection)
    except Exception:
        logger.info("Collection '%s' did not exist — nothing to delete.", settings.chroma_collection)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest static docs into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection before re-ingesting (full rebuild).",
    )
    args = parser.parse_args()

    static_dir = Path(__file__).resolve().parents[1] / "data" / "static"
    if not static_dir.exists():
        logger.error("Static data directory not found: %s", static_dir)
        sys.exit(1)

    if args.reset:
        _reset_collection()

    logger.info(
        "Ingesting documents from %s into collection '%s'.",
        static_dir,
        settings.chroma_collection,
    )
    total = ingest_directory(static_dir, glob="**/*.md")
    logger.info("Ingestion complete — %d chunks stored.", total)
    print(f"Done. {total} chunks in collection '{settings.chroma_collection}'.")


if __name__ == "__main__":
    main()
