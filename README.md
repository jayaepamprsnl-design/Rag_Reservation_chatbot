# Parking RAG Chatbot

A RAG-powered parking reservation assistant built with **FastAPI**, **ChromaDB**, **SQLite**, and **Streamlit**.

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model (required for PII detection)
python -m spacy download en_core_web_lg

# 4. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and (optionally) OPENAI_API_BASE

# 5. Seed the database
python scripts/seed_db.py

# 6. Ingest static documents into ChromaDB
python scripts/ingest.py

# 7. Start the API server
uvicorn app.api.main:app --reload --port 8000

# 8. Start the Streamlit UI (separate terminal)
streamlit run app/ui/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## Docker

```bash
cp .env.example .env   # fill in your API key
docker-compose up --build
```

API → http://localhost:8000 | UI → http://localhost:8501

---

## Run Tests

```bash
pytest tests/ -v
```

> `test_pii_filter.py` requires the spaCy model (`python -m spacy download en_core_web_lg`).

---

## Project Structure

```
app/
├── api/          FastAPI app, /chat and /reservations routes
├── core/         Pydantic-settings config, logging setup
├── rag/          Ingestor, chunker, embedder, ChromaDB store, retriever, generator
├── db/           SQLAlchemy models (ParkingSpace, WorkingHours, Price, Reservation)
├── chat/         Conversation context, intent router, reservation state machine
├── guardrails/   PII detection & masking via Presidio + spaCy
└── ui/           Streamlit chat interface
data/
├── static/       Markdown documents ingested into ChromaDB
└── eval/         Golden Q&A dataset for retrieval evaluation
scripts/
├── ingest.py     Ingest static docs into ChromaDB
├── seed_db.py    Seed SQLite with parking spaces, hours, prices
└── run_eval.py   Run evaluation suite (Recall@K, Precision@K, MRR)
tests/            pytest test suite (≥2 tests per module)
docs/             Additional documentation
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | LLM endpoint — OpenAI or EPAM AI DIAL |
| `OPENAI_API_KEY` | *(required)* | API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `CHROMA_COLLECTION` | `parking_docs` | ChromaDB collection name |
| `DATABASE_URL` | `sqlite:///./parking.db` | SQLAlchemy database URL |
| `RETRIEVAL_TOP_K` | `5` | Documents returned per query |
| `LOG_LEVEL` | `INFO` | Python log level |

---

## Tech Stack

- **Python 3.11** · **FastAPI** · **Streamlit**
- **ChromaDB** (vector store) · **sentence-transformers** `all-MiniLM-L6-v2`
- **SQLite + SQLAlchemy 2** (dynamic data)
- **Presidio + spaCy** (PII detection & masking)
- **OpenAI SDK** (configurable endpoint)
- **pytest** (test suite)
