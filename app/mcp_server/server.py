"""
MCP server that records APPROVED parking reservations in a text file.

Run as a separate process (see scripts/run_mcp_server.py); it speaks MCP over
streamable-http on MCP_HOST:MCP_PORT (default 127.0.0.1:8765, path /mcp).

Tools
-----
write_reservation(...)             append exactly one line, idempotent per id
list_confirmed_reservations(limit) read-only, newest last

File line format
----------------
    Name | Car Number | Reservation Period | Approval Time
    Shalini Singh | TS09AB1234 | 2026-11-04 14:00 - 17:00 | 2026-11-03 11:30:00

Security
--------
* Bearer token (MCP_API_KEY) on every route except /health, constant-time compare.
* The output path comes from RESERVATIONS_FILE only; clients never supply paths.
* All inputs are validated with Pydantic; "|", CR/LF and control characters
  are REJECTED so a client cannot forge columns or extra lines.
* Per-client rate limit, applied before authentication (slows token guessing).
* Logs contain reservation_id only: never names or plates.

Idempotency
-----------
The line format has no id column, so processed ids live in a sidecar file
(<output>.ids) as "<id> <sha256 of the line>".  The line is written first and
the id second, both under one lock and fsynced.  A crash between the two writes
could allow one duplicate line on retry; it can never lose a reservation.

* same id + same content  -> "duplicate" (nothing written)
* same id + DIFFERENT content -> "conflict" (nothing written, never reported as
  success: e.g. a recreated database reusing id 1 must not be silently dropped)
* the .ids file belongs to the output file: if the output file is missing or
  empty (deleted / rotated) the ids are discarded together with it
* a crash mid-line leaves no trailing newline; the next append repairs that
  first so records can never be glued together
"""
# NOTE: no `from __future__ import annotations` here — FastMCP 1.9 inspects the raw
# tool-function annotations and breaks on string annotations.
import hashlib
import hmac
import os
import re
import sys
import threading
import time
import unicodedata
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.logging_config import logger

HEADER = "Name | Car Number | Reservation Period | Approval Time"
_PLATE_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,18}[A-Z0-9]$")
_MAX_ID = 2**31 - 1


# ---------------------------------------------------------------------------
# Settings (env / .env — no secrets in code)
# ---------------------------------------------------------------------------


class McpSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765
    mcp_api_key: str = ""
    reservations_file: str = "data/output/confirmed_reservations.txt"
    mcp_rate_limit_per_minute: int = 120


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


_MIN_KEY_LEN = 16
_FORMULA_LEADERS = "=+-@"


def _reject_unsafe(value: str, field: str) -> str:
    """Reject the column separator and anything that is not a printable char."""
    if "|" in value:
        raise ValueError(f"{field} must not contain '|'")
    for ch in value:
        cat = unicodedata.category(ch)
        # C* = control/format/unassigned (CR, LF, TAB, NUL, BOM...); Zl/Zp = unicode line breaks
        if cat[0] == "C" or cat in ("Zl", "Zp"):
            raise ValueError(f"{field} contains a control or line-break character")
    return value


def _parse_dt(value: str, field: str) -> datetime:
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError(f"{field} must be an ISO-8601 datetime string")
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(f"{field} must be an ISO-8601 datetime, e.g. 2026-11-04T14:00:00")


class ReservationRecord(BaseModel):
    reservation_id: int = Field(gt=0, le=_MAX_ID)
    customer_name: str = Field(min_length=1, max_length=100)
    car_plate: str = Field(min_length=3, max_length=20)
    start_time: str
    end_time: str
    approved_at: str

    @field_validator("customer_name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("customer_name must not be blank")
        if v[0] in _FORMULA_LEADERS:
            # Spreadsheet formula injection if the file is ever opened in Excel/Sheets.
            raise ValueError(f"customer_name must not start with one of {_FORMULA_LEADERS}")
        return _reject_unsafe(v, "customer_name")

    @field_validator("car_plate")
    @classmethod
    def _plate(cls, v: str) -> str:
        v = v.strip().upper()
        if not _PLATE_RE.match(v):
            raise ValueError("car_plate must be 3-20 chars: letters, digits, hyphens")
        return v

    @model_validator(mode="after")
    def _times(self) -> "ReservationRecord":
        start = _parse_dt(self.start_time, "start_time")
        end = _parse_dt(self.end_time, "end_time")
        _parse_dt(self.approved_at, "approved_at")
        if (start.tzinfo is None) != (end.tzinfo is None):
            raise ValueError("start_time and end_time must both be naive or both timezone-aware")
        if end <= start:
            raise ValueError("end_time must be after start_time")
        return self

    def to_line(self) -> str:
        start = datetime.fromisoformat(self.start_time.strip())
        end = datetime.fromisoformat(self.end_time.strip())
        approved = datetime.fromisoformat(self.approved_at.strip())
        if start.date() == end.date():
            period = f"{start:%Y-%m-%d %H:%M} - {end:%H:%M}"
        else:
            period = f"{start:%Y-%m-%d %H:%M} - {end:%Y-%m-%d %H:%M}"
        return f"{self.customer_name} | {self.car_plate} | {period} | {approved:%Y-%m-%d %H:%M:%S}"


# ---------------------------------------------------------------------------
# File store: thread lock + OS file lock (cross-process), atomic append
# ---------------------------------------------------------------------------


@contextmanager
def _os_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive inter-process lock on a dedicated lock file (msvcrt / fcntl)."""
    with open(lock_path, "a+b") as fh:
        if sys.platform == "win32":
            import msvcrt

            while True:  # LK_LOCK gives up after ~10s with OSError; keep waiting
                fh.seek(0)
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class ReservationStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).resolve()
        self.ids_path = self.path.with_name(self.path.name + ".ids")
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._tlock = threading.Lock()

    def _read_ids(self) -> dict[int, str | None]:
        """id -> sha256 of the written line (None for ids recorded without a digest)."""
        if not self.ids_path.exists():
            return {}
        out: dict[int, str | None] = {}
        for ln in self.ids_path.read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if parts and parts[0].isdigit():
                out[int(parts[0])] = parts[1] if len(parts) > 1 else None
        return out

    @staticmethod
    def _append(path: Path, data: str) -> None:
        # newline="" keeps "\n" literal on Windows (no CRLF translation).
        with open(path, "a", encoding="utf-8", newline="") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

    def _ends_with_newline(self) -> bool:
        with open(self.path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) == b"\n"

    def write(self, rec: ReservationRecord) -> str:
        """Return 'written', 'duplicate' or 'conflict'."""
        line = rec.to_line()
        digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._tlock, _os_lock(self.lock_path):
            has_content = self.path.exists() and self.path.stat().st_size > 0
            if has_content:
                ids = self._read_ids()
            else:
                ids = {}
                if self.ids_path.exists():  # ids describe lines in the file; file is gone
                    self.ids_path.write_text("", encoding="utf-8")
            if rec.reservation_id in ids:
                known = ids[rec.reservation_id]
                return "duplicate" if known in (None, digest) else "conflict"
            if not has_content:
                prefix = HEADER + "\n"
            elif not self._ends_with_newline():
                prefix = "\n"  # repair a line truncated by a crash before appending
            else:
                prefix = ""
            self._append(self.path, prefix + line + "\n")
            self._append(self.ids_path, f"{rec.reservation_id} {digest}\n")
        return "written"

    def tail(self, limit: int) -> list[str]:
        if not self.path.exists():
            return []
        with self._tlock, _os_lock(self.lock_path):
            lines = self.path.read_text(encoding="utf-8").splitlines()
        return [ln for ln in lines[1:] if ln][-limit:]


# ---------------------------------------------------------------------------
# ASGI security middleware: rate limit -> bearer auth
# ---------------------------------------------------------------------------


class SecurityMiddleware:
    """Pure ASGI (no buffering, so streaming responses keep working)."""

    PUBLIC_PATHS = {"/health"}

    def __init__(self, app: Any, api_key: str, rate_limit_per_minute: int) -> None:
        api_key = (api_key or "").strip()
        if not api_key:
            raise RuntimeError("MCP_API_KEY is not set; refusing to start unauthenticated.")
        if len(api_key) < _MIN_KEY_LEN:
            raise RuntimeError(
                f"MCP_API_KEY is too short (minimum {_MIN_KEY_LEN} characters); "
                'generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        self.app = app
        self._key = api_key.encode("utf-8")
        self._limit = rate_limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._rl_lock = threading.Lock()

    def _rate_limited(self, client: str) -> bool:
        now = time.monotonic()
        with self._rl_lock:
            if len(self._hits) > 10_000:  # bound memory under many distinct client addresses
                for k in [k for k, d in self._hits.items() if not d or now - d[-1] > 60]:
                    del self._hits[k]
            q = self._hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self._limit:
                return True
            q.append(now)
            return False

    def _authorized(self, scope: dict) -> bool:
        token = b""
        scheme = b""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                scheme, _, rest = value.partition(b" ")
                token = rest.strip()
                break
        # Always run the constant-time compare so timing doesn't reveal header presence.
        match = hmac.compare_digest(token, self._key)
        return match and scheme.lower() == b"bearer"

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            # websocket (or anything new) is not part of this service: never forward it
            # to the app unauthenticated.
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return
        if scope.get("path") in self.PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        client = (scope.get("client") or ("unknown", 0))[0]
        if self._rate_limited(client):
            logger.warning("mcp_server|rate_limited")
            resp = JSONResponse({"error": "rate limit exceeded"}, 429, {"Retry-After": "60"})
        elif not self._authorized(scope):
            logger.warning("mcp_server|unauthorized")
            resp = JSONResponse(
                {"error": "unauthorized"}, 401, {"WWW-Authenticate": "Bearer"}
            )
        else:
            if scope.get("path") == "/mcp":
                # The SDK mounts the endpoint at /mcp/ and would answer /mcp with a 307;
                # many clients drop the Authorization header when following redirects.
                scope = {**scope, "path": "/mcp/", "raw_path": b"/mcp/"}
            await self.app(scope, receive, send)
            return
        await resp(scope, receive, send)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_mcp(store: ReservationStore, settings: McpSettings | None = None) -> FastMCP:
    s = settings or McpSettings()
    mcp = FastMCP(
        "citypark-reservations",
        host=s.mcp_host,
        port=s.mcp_port,
        stateless_http=True,
        json_response=True,
    )

    @mcp.tool()
    def write_reservation(
        reservation_id: int,
        customer_name: str,
        car_plate: str,
        start_time: str,
        end_time: str,
        approved_at: str,
    ) -> dict:
        """Append one approved reservation to the confirmed-reservations file.

        Idempotent: a reservation_id already recorded with the same content returns
        status "duplicate"; the same id with different content returns "conflict"
        (nothing is written either way).
        start_time, end_time and approved_at are ISO-8601 datetimes.
        """
        try:
            rec = ReservationRecord(
                reservation_id=reservation_id,
                customer_name=customer_name,
                car_plate=car_plate,
                start_time=start_time,
                end_time=end_time,
                approved_at=approved_at,
            )
        except ValidationError as exc:
            # Messages name the field and rule only; the offending value is never echoed.
            msgs = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'input'}: "
                f"{e['msg'].removeprefix('Value error, ')}"
                for e in exc.errors(include_input=False, include_url=False)
            )
            logger.warning("mcp_server|invalid_input reservation_id=%s", reservation_id)
            raise ValueError(f"Invalid reservation: {msgs}") from None
        try:
            status = store.write(rec)
        except OSError as exc:
            logger.error(
                "mcp_server|write_failed reservation_id=%d error=%s",
                rec.reservation_id, type(exc).__name__,
            )
            raise RuntimeError("Could not write to the reservations file.") from None
        if status == "conflict":
            logger.error("mcp_server|conflict reservation_id=%d", rec.reservation_id)
        else:
            logger.info("mcp_server|%s reservation_id=%d", status, rec.reservation_id)
        return {"status": status, "reservation_id": rec.reservation_id}

    @mcp.tool()
    def list_confirmed_reservations(limit: int = 20) -> dict:
        """Read-only: return the most recent confirmed reservation lines (limit 1-100)."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        lines = store.tail(limit)
        return {"count": len(lines), "reservations": lines}

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return mcp


def build_app(
    settings: McpSettings | None = None, store: ReservationStore | None = None
) -> Any:
    """Return the ASGI app (MCP at /mcp, health at /health) wrapped in security."""
    s = settings or McpSettings()
    store = store or ReservationStore(s.reservations_file)
    mcp = create_mcp(store, s)
    return SecurityMiddleware(
        mcp.streamable_http_app(), s.mcp_api_key, s.mcp_rate_limit_per_minute
    )


def main() -> None:
    import uvicorn

    s = McpSettings()
    app = build_app(s)
    if s.mcp_host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "mcp_server|non_loopback_bind host=%s: plain HTTP + bearer token; expose it "
            "only on a private/internal network or behind a TLS proxy", s.mcp_host,
        )
    logger.info("mcp_server|starting host=%s port=%d", s.mcp_host, s.mcp_port)
    uvicorn.run(app, host=s.mcp_host, port=s.mcp_port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover - covered via scripts/run_mcp_server.py
    main()
