"""
Shared helpers for the CityPark Streamlit UI.

Imported by both chat_page.py and admin_page.py.
Never calls any st.* function at module level — those are reserved for render().
"""
from __future__ import annotations

import functools
import os
import uuid
from typing import Any

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL: str = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT: float = 60.0
_ADMIN_KEY_HEADER: str = "X-Admin-Key"

# ---------------------------------------------------------------------------
# Backward-compat fragment shim (Streamlit >= 1.33 required for auto-refresh)
# ---------------------------------------------------------------------------

def auto_refresh(run_every: str = "5s"):
    """Return the best available fragment decorator for this Streamlit version."""
    for attr in ("fragment", "experimental_fragment"):
        fn = getattr(st, attr, None)
        if fn is not None:
            try:
                return fn(run_every=run_every)
            except Exception:
                pass
    # Fallback: plain decorator — works but no auto-refresh.
    def _noop(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return _wrapper
    return _noop

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def init_chat_state() -> None:
    """Ensure all chat-page session keys exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("lots_data", None)
    st.session_state.setdefault("awaiting_approval", False)
    st.session_state.setdefault("pending_reservation_id", None)


def init_admin_state() -> None:
    """Ensure all admin-page session keys exist."""
    # Pre-fill from env so docker-compose / local dev just works.
    if "admin_key" not in st.session_state:
        st.session_state.admin_key = os.getenv("ADMIN_API_KEY", "")
    st.session_state.setdefault("admin_chat_history", [])

# ---------------------------------------------------------------------------
# HTTP helpers (talk only to the REST API, never to the DB directly)
# ---------------------------------------------------------------------------

def admin_headers() -> dict[str, str]:
    """Return X-Admin-Key header using the stored key."""
    return {_ADMIN_KEY_HEADER: st.session_state.get("admin_key", "")}


def api_get(
    path: str,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    **params: Any,
) -> tuple[int, Any]:
    """GET ``{API_URL}{path}`` and return (status_code, json_body).

    Returns ``(0, {"detail": "..."})`` for connection errors,
    ``(-1, {"detail": "..."})`` for timeouts, ``(-2, …)`` for other errors.
    """
    try:
        r = httpx.get(
            f"{API_URL}{path}",
            headers=headers or {},
            params=params,
            timeout=timeout,
        )
        return r.status_code, r.json()
    except httpx.ConnectError:
        return 0, {"detail": f"Cannot reach API at {API_URL} — is the backend running?"}
    except httpx.TimeoutException:
        return -1, {"detail": "Request timed out."}
    except Exception as exc:  # noqa: BLE001
        return -2, {"detail": str(exc)}


def api_post(
    path: str,
    headers: dict[str, str] | None = None,
    json: dict | None = None,
    timeout: float = 20.0,
) -> tuple[int, Any]:
    """POST ``{API_URL}{path}`` and return (status_code, json_body)."""
    try:
        r = httpx.post(
            f"{API_URL}{path}",
            headers=headers or {},
            json=json or {},
            timeout=timeout,
        )
        return r.status_code, r.json()
    except httpx.ConnectError:
        return 0, {"detail": f"Cannot reach API at {API_URL} — is the backend running?"}
    except httpx.TimeoutException:
        return -1, {"detail": "Request timed out."}
    except Exception as exc:  # noqa: BLE001
        return -2, {"detail": str(exc)}

# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------

def show_api_error(code: int, body: Any) -> None:
    """Display a user-friendly (no stack trace) error for an API response."""
    detail = body.get("detail", "Unknown error") if isinstance(body, dict) else str(body)
    if code == 0:
        st.error(f"API unreachable: {detail}")
    elif code == -1:
        st.error(f"Timeout: {detail}")
    elif code == -2:
        st.error(f"Unexpected error: {detail}")
    elif code == 401:
        st.error("Authentication required — please sign in.")
    elif code == 403:
        st.error("Invalid admin key.")
    elif code == 404:
        st.warning("Not found.")
    elif code == 409:
        st.warning("Already decided.")
    elif code == 422:
        st.error(f"Validation error: {detail}")
    elif code == 503:
        st.error("Admin key not configured on server — set ADMIN_API_KEY in .env.")
    else:
        st.error(f"Error {code}: {detail}")

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def status_badge(status: str) -> str:
    """Return a coloured emoji badge for a reservation status string."""
    return {
        "pending":   "🟡 Pending",
        "confirmed": "🟢 Confirmed",
        "refused":   "🔴 Refused",
        "expired":   "⬜ Expired",
        "cancelled": "⬛ Cancelled",
    }.get(status.lower(), status.upper())
