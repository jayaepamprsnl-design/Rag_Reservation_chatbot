#!/bin/sh
# Container entrypoint — seeds the DB and ingests static docs on first start,
# then hands off to the command passed by docker-compose (CMD / command:).
set -e

echo "[entrypoint] Running initialisation…"
python /app/docker/init.py

echo "[entrypoint] Starting service: $*"
exec "$@"
