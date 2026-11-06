"""
Async MCP client for the confirmed-reservations server (app/mcp_server).

write_confirmed_reservation(reservation) -> bool
    True  : the server recorded the reservation ("written") or already had it
            ("duplicate") — either way the line is in the file.
    False : not written (blocked, not configured, server down, rejected).

send_reservation(reservation) -> str
    Same call, but tells the caller WHY it failed so a retry sweep can tell a
    transient outage (stop and try later) from a permanent rejection (skip this
    row, keep going) and one bad row can never block the queue behind it.

Never raises for expected failures and never logs names or plates — only the
reservation id and exception class names.

Approval gate
-------------
The lowest layer refuses to send anything that is not an admin-approved
reservation (see is_admin_approved), so no caller — including a buggy or
prompt-injected one — can push a PENDING/REFUSED/CANCELLED/EXPIRED row.
"""
import asyncio
import json
import re
import unicodedata
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, Optional, TypeVar

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.core.config import settings
from app.core.logging_config import logger
from app.db.models import ReservationStatus

T = TypeVar("T")


class PermanentMcpError(Exception):
    """The server understood the request and rejected it; retrying cannot help."""


OK = "ok"                    # line is in the file (written or duplicate)
BLOCKED = "blocked"          # not an admin-approved reservation; nothing sent
DISABLED = "disabled"        # export not configured
PERMANENT = "permanent"      # server rejected it / content conflict; retry is pointless
TRANSIENT = "transient"      # server down, timeout, 5xx; retry later

_LOOPBACK_OR_INTERNAL = {"127.0.0.1", "localhost", "::1", "mcp-server"}


def warn_if_insecure_url() -> None:
    """The bearer token travels in clear text over http://; flag non-local targets."""
    parsed = urlparse(settings.mcp_server_url)
    if parsed.scheme != "https" and parsed.hostname not in _LOOPBACK_OR_INTERNAL:
        logger.warning(
            "mcp_client|insecure_url scheme=%s host=%s: bearer token is sent unencrypted; "
            "use https or a private network", parsed.scheme, parsed.hostname,
        )


def sanitize_name(name: str) -> str:
    """Make a stored customer name safe for the line format WITHOUT dropping the booking.

    The server (rightly) rejects '|', control characters and formula leaders. An
    approved reservation must still be recorded, so the trusted caller normalises:
    '|' -> '/', control/format characters removed, whitespace collapsed, leading
    =+-@ stripped, max 100 chars.  Falls back to "Unknown" if nothing is left.
    """
    cleaned = "".join(
        " " if unicodedata.category(c) in ("Zs", "Zl", "Zp") or c in "\t\r\n" else c
        for c in (name or "").replace("|", "/")
        if unicodedata.category(c)[0] != "C" or c in "\t\r\n"
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lstrip("=+-@ ").strip()
    return cleaned[:100] or "Unknown"


def export_configured() -> bool:
    return bool(settings.mcp_export_enabled and settings.mcp_api_key)


def is_admin_approved(res: Any) -> bool:
    """True only for CONFIRMED rows that an admin decision (decided_by/at) produced.

    approve_reservation() is the only code that sets decided_by on a CONFIRMED
    row; legacy Stage-1 direct-confirm rows have decided_by NULL and are excluded.
    """
    return (
        res.status == ReservationStatus.CONFIRMED
        and bool(res.decided_by)
        and res.decided_at is not None
    )


def _payload(res: Any) -> dict:
    return {
        "reservation_id": res.id,
        "customer_name": sanitize_name(res.customer_name),
        "car_plate": res.car_plate,
        "start_time": res.start_time.isoformat(),
        "end_time": res.end_time.isoformat(),
        "approved_at": res.decided_at.isoformat(),
    }


def _leaf_names(exc: BaseException) -> str:
    """Class names of the underlying errors (unwraps anyio/ExceptionGroups)."""
    if isinstance(exc, BaseExceptionGroup):
        return ",".join(_leaf_names(e) for e in exc.exceptions)
    return type(exc).__name__


def _is_permanent(exc: BaseException) -> bool:
    """4xx from the server (bad token, bad request) will not fix itself; 408/429 might."""
    if isinstance(exc, BaseExceptionGroup):
        return all(_is_permanent(e) for e in exc.exceptions)
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return 400 <= code < 500 and code not in (408, 429)
    return False


async def _call_once(args: dict, timeout: float) -> str:
    headers = {"Authorization": f"Bearer {settings.mcp_api_key}"}

    async def go() -> str:
        async with streamablehttp_client(
            settings.mcp_server_url, headers=headers, timeout=timeout
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("write_reservation", args)
        text = result.content[0].text if result.content else ""
        if result.isError:
            raise PermanentMcpError("server rejected the reservation")
        return json.loads(text).get("status", "")

    return await asyncio.wait_for(go(), timeout=timeout + 2)


async def send_reservation(reservation: Any, max_attempts: Optional[int] = None) -> str:
    """Send one approved reservation; return OK / BLOCKED / DISABLED / PERMANENT / TRANSIENT."""
    rid = getattr(reservation, "id", None)
    if not is_admin_approved(reservation):
        logger.warning("mcp_client|blocked_not_approved reservation_id=%s", rid)
        return BLOCKED
    if not export_configured():
        logger.info("mcp_client|export_disabled reservation_id=%s", rid)
        return DISABLED

    attempts = max(1, max_attempts or settings.mcp_client_max_attempts)
    args = _payload(reservation)
    for attempt in range(1, attempts + 1):
        try:
            status = await _call_once(args, settings.mcp_client_timeout)
        except PermanentMcpError:
            logger.error("mcp_client|rejected reservation_id=%s", rid)
            return PERMANENT
        except BaseException as exc:  # noqa: BLE001 - includes anyio ExceptionGroup
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            logger.warning(
                "mcp_client|attempt_failed reservation_id=%s attempt=%d/%d error=%s",
                rid, attempt, attempts, _leaf_names(exc),
            )
            if _is_permanent(exc):
                return PERMANENT
            if attempt < attempts:
                await asyncio.sleep(settings.mcp_client_backoff_seconds * 2 ** (attempt - 1))
            continue
        if status in ("written", "duplicate"):
            logger.info("mcp_client|%s reservation_id=%s", status, rid)
            return OK
        logger.error("mcp_client|unexpected_status reservation_id=%s status=%s", rid, status)
        return PERMANENT  # incl. "conflict": same id, different content — needs a human
    logger.error("mcp_client|gave_up reservation_id=%s attempts=%d", rid, attempts)
    return TRANSIENT


async def write_confirmed_reservation(
    reservation: Any, max_attempts: Optional[int] = None
) -> bool:
    """Send one approved reservation to the MCP server, retrying with backoff."""
    return await send_reservation(reservation, max_attempts) == OK


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code (FastAPI sync routes run in worker threads)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:  # called inside a running loop
        return pool.submit(asyncio.run, coro).result()
