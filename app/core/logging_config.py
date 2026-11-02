import logging
import sys

from app.core.config import settings

_NOISY_LOGGERS = ("chromadb", "httpx", "httpcore", "urllib3", "sentence_transformers")


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("parking_rag")


logger: logging.Logger = setup_logging()
