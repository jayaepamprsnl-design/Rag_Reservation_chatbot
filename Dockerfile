# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Compiles wheels for packages that need build-essential and downloads the
# spaCy NER model into the virtual environment.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create an isolated virtual environment so we can copy it cleanly.
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Download the spaCy NER model required by Presidio (stored inside the venv).
RUN python -m spacy download en_core_web_lg


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Lean image — no build tools, just the pre-built venv and the application.
FROM python:3.11-slim AS runtime

# curl is the only runtime system dep (used by the Docker health check).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from the builder stage.
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Persistent-data directories (can be overridden by volume mounts at runtime).
RUN mkdir -p /data/chroma_db

# Copy application source.
COPY . .

# Make the entrypoint executable.
RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
