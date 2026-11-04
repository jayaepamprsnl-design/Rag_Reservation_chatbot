# CityPark RAG Chatbot — Stage 2

A RAG-powered parking reservation assistant with a **human-in-the-loop approval workflow**
built with FastAPI, LangChain, ChromaDB, SQLite, and Streamlit.

---

## Architecture overview

```
User (browser)
  |-- POST /chat ─────────────────────> FastAPI  (app/api/routes/chat.py)
                                           |
                                    Agent 1: RAG Chatbot
                                    (app/chat/reservation.py)
                                           |  user confirms booking
                                    create_pending_reservation()
                                           |
                                    ChannelRegistry
                                      RestChannel (always)
                                      EmailChannel (if ADMIN_EMAIL_ENABLED)
                                           |
                                    Admin sees pending in:
                                      GET /admin/reservations/pending
                                      Admin Panel page in Streamlit UI (:8501/admin)
                                           |
                               Admin approves or refuses
                               POST /admin/reservations/{id}/approve
                               POST /admin/reservations/{id}/refuse
                               POST /admin/chat  (natural language)
                               GET  /admin/approve?token=...  (email link)
                               GET  /admin/refuse?token=...   (email link)
                                           |
                                    Agent 2: AdminAgent (LangChain)
                                    (app/agents/admin_agent.py)
                                           |
                                    User's next message delivers notification
                                    GET /sessions/{id}/notifications (poll)
```

---

## Quick start (local)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 3. Configure environment
cp .env.example .env
# Edit .env — set ADMIN_API_KEY and (optionally) OPENAI_API_KEY

# 4. Seed DB, apply Stage-2 migration, ingest documents
python scripts/seed_db.py
python scripts/migrate_stage2.py
CHROMA_COLLECTION=parking_static python scripts/ingest.py

# 5. Start the API
uvicorn app.api.main:app --reload --port 8000

# 6. Start the unified UI (Chat + Admin Panel via sidebar)
streamlit run app/ui/streamlit_app.py --server.port 8501

```

| Service | URL |
|---------|-----|
| API + docs | http://localhost:8000/docs |
| Chat UI | http://localhost:8501 |
| Admin Panel | http://localhost:8501/admin (sidebar link; sign in with `ADMIN_API_KEY` from .env) |

---

## Docker (all services)

```bash
cp .env.example .env   # fill in ADMIN_API_KEY at minimum
docker compose up --build
```

Optional Mailpit SMTP capture for local email testing:

```bash
ADMIN_EMAIL_ENABLED=true SMTP_HOST=mailpit docker compose --profile mailpit up
# Mailpit web UI: http://localhost:8025
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `your-api-key-here` | LLM API key — leave as-is for offline/mock mode |
| `OPENAI_API_BASE` | OpenAI | Compatible with EPAM AI DIAL and other endpoints |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model identifier |
| `DATABASE_URL` | SQLite local | SQLAlchemy connection string |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION` | `parking_static` | Collection name |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks retrieved per query |
| **`ADMIN_API_KEY`** | **(required)** | Key for all `/admin/*` endpoints — **change before deploying** |
| `ADMIN_API_BASE_URL` | `http://localhost:8000` | Base URL embedded in email action links |
| `ADMIN_EMAIL_ENABLED` | `false` | Enable SMTP email notifications to admin |
| `SMTP_HOST` | `localhost` | SMTP server hostname (`mailpit` for local dev) |
| `SMTP_PORT` | `1025` | SMTP port |
| `SMTP_USER` | *(empty)* | SMTP username — **sensitive** |
| `SMTP_PASSWORD` | *(empty)* | SMTP password — **sensitive** |
| `ADMIN_EMAIL` | `admin@citypark.example.com` | Admin notification recipient |
| `EMAIL_FROM` | `no-reply@citypark.example.com` | Email sender address |
| `EMAIL_TOKEN_SECRET` | *(change-me)* | HMAC secret for signed email links — **change before deploying** |
| `EMAIL_TOKEN_MAX_AGE_SECONDS` | `86400` | Email link expiry (24 h) |

---

## API endpoints

### Chat (public)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a message; drives the reservation state machine |
| `GET` | `/health` | Liveness probe |
| `GET` | `/lots` | Live parking-lot availability |
| `POST` | `/reset-session` | Clear in-memory session state |

### Reservations (public)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/reservations/{id}/status` | Approval-workflow status of a LotReservation |
| `GET` | `/sessions/{session_id}/notifications` | Poll and mark delivered pending decisions |

### Admin (requires `X-Admin-Key` header)

| Method | Path | 
Description |
|--------|------|-------------|
| `GET` | `/admin/reservations/pending` | List all PENDING reservations |
| `GET` | `/admin/reservations/history` | List decided reservations |
| `GET` | `/admin/reservations/{id}` | Single reservation detail |
| `POST` | `/admin/reservations/{id}/approve` | Approve with optional note |
| `POST` | `/admin/reservations/{id}/refuse` | Refuse with required reason |
| `POST` | `/admin/chat` | Natural-language commands to the admin agent |

### Email-link (token-authenticated, no API key)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/approve?token=...` | One-click approve from email |
| `GET` | `/admin/refuse?token=...` | One-click refuse from email |

---

## Project structure

```
app/
  agents/
    admin_agent.py      Agent 2 — LangChain tool-calling admin agent
    bus.py              Message bus (InProcessBus / interface for queue swap)
    contracts.py        Pydantic contracts between agents
    channels/
      base.py           Channel ABC + ChannelResult
      rest_channel.py   Primary: persists to DB (always enabled)
      email_channel.py  Optional: SMTP with signed single-use links
      registry.py       Fan-out to all enabled channels
      tokens.py         sign / verify / consume (itsdangerous)
  api/
    main.py             FastAPI app entry point
    routes/
      admin.py          Admin API (protected by X-Admin-Key)
      chat.py           Main chat route
      reservations.py   Reservation status + notifications
  chat/
    reservation.py      State machine: IDLE -> ... -> AWAITING_APPROVAL -> DONE
    router.py           Intent routing (RAG / dynamic / reservation)
    conversation.py     Conversation context
    guardrails/         PII masking + injection detection
  db/
    models.py           SQLAlchemy models (LotReservation, ApprovalRequest, ...)
    repository.py       Repository layer (approval workflow functions)
    session.py          Database session management
  rag/
    generator.py        LLM client (OpenAI / MockLLM)
    retriever.py        ChromaDB retrieval
    ingestor.py         Static document ingestion
  ui/
    streamlit_app.py    User chat UI with notification polling
    admin_app.py        Admin dashboard (login, pending, history, agent chat)
  core/
    config.py           pydantic-settings (reads .env)
docker/
  entrypoint.sh         Container startup script
  init.py               DB seed + Stage-2 migration + ChromaDB ingest
infra/
  main.tf               Terraform Docker provider (api, ui, admin-ui)
  variables.tf          All configurable variables (secrets marked sensitive)
  outputs.tf            Service URLs
scripts/
  seed_db.py            Seed parking lots
  migrate_stage2.py     Idempotent Stage-2 schema migration
  ingest.py             Ingest static docs into ChromaDB
  run_eval.py           Stage-1 RAG evaluation suite
  run_eval_stage2.py    Stage-2 approval-workflow evaluation suite
```

---

## Tech stack

| Component | Library |
|-----------|---------|
| API framework | FastAPI 0.111 |
| Async server | uvicorn |
| LLM | openai (OpenAI / DIAL / MockLLM) |
| Agent orchestration | LangChain Core 0.3 (`@tool`, InjectedToolArg) |
| Vector DB | ChromaDB 0.5 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| ORM | SQLAlchemy 2 + SQLite |
| Settings | pydantic-settings |
| PII / guardrails | Presidio 2.2 + spaCy en_core_web_lg |
| Email tokens | itsdangerous 2.2 |
| User UI | Streamlit 1.36 |
| Tests | pytest 8 + httpx + coverage |
| IaC | Terraform (kreuzwerker/docker provider) |
| Local SMTP | Mailpit (optional Docker profile) |

---

## Running the test suite

```bash
pytest tests/ -v --cov=app --cov-fail-under=90
```

Current coverage: **93%** (534 tests, 0 failures).

---

## Security notes

- `ADMIN_API_KEY` is validated with `hmac.compare_digest` (constant-time — timing-attack resistant).
- Email action tokens are HMAC-signed (itsdangerous), time-limited (24 h default), and single-use (ApprovalRequest row tracks consumption).
- Plates are masked in all admin-facing outputs (e.g. `ABC****34`).
- PII in user messages is masked before reaching logs or the LLM.
- The LLM cannot approve or refuse autonomously — a deterministic keyword gate in `parse_decision` runs before any LLM call.
- `SMTP_PASSWORD` and `EMAIL_TOKEN_SECRET` are never included in log messages.
