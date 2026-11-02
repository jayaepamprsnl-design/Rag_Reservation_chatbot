"""CityPark Parking Assistant — Streamlit UI.

Features
--------
- Full chat history with bubble-style messages
- Sources expander under each assistant answer
- Visible notice when the input guardrail masked sensitive data
- Sidebar with live lot availability (auto-fetched, manual refresh)
- "New Chat" button that resets both the UI and the server-side session
"""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

# Lazy-import the router constant so the UI doesn't fail if the backend package
# is not importable in the Streamlit process (e.g. missing deps).
try:
    from app.chat.router import RouterIntent as _RouterIntent
    _RESERVATION_INTENT = _RouterIntent.RESERVATION.value
except Exception:
    _RESERVATION_INTENT = "RESERVATION"

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60.0

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CityPark Assistant",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []      # [{role, content, sources, guardrail_flags}]

if "lots_data" not in st.session_state:
    st.session_state.lots_data = None   # None = needs fetch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_lots() -> list[dict]:
    try:
        resp = httpx.get(f"{API_URL}/lots", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        return []
    except Exception:
        return []


def _post_chat(message: str, session_id: str) -> dict:
    resp = httpx.post(
        f"{API_URL}/chat",
        json={"message": message, "session_id": session_id},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _reset_server_session(session_id: str) -> None:
    try:
        httpx.post(
            f"{API_URL}/reset-session",
            json={"session_id": session_id},
            timeout=5.0,
        )
    except Exception:
        pass


def _availability_color(pct: float) -> str:
    if pct > 0.3:
        return "🟢"
    if pct > 0.1:
        return "🟡"
    return "🔴"


# ---------------------------------------------------------------------------
# Sidebar — live availability
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🅿️ CityPark")
    st.caption("Live parking availability")

    col_refresh, col_new = st.columns(2)

    if col_refresh.button("↻ Refresh", use_container_width=True):
        st.session_state.lots_data = None

    if col_new.button("🗑️ New Chat", use_container_width=True, type="secondary"):
        _reset_server_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.lots_data = None
        st.rerun()

    st.divider()

    if st.session_state.lots_data is None:
        with st.spinner("Loading lots…"):
            st.session_state.lots_data = _fetch_lots()

    lots = st.session_state.lots_data
    if not lots:
        st.warning("⚠️ Could not reach the backend API.")
    else:
        for lot in lots:
            avail = lot["available_spaces"]
            total = lot["total_spaces"]
            pct   = avail / max(total, 1)
            icon  = _availability_color(pct)

            st.markdown(f"**{icon} {lot['name']}**")
            st.progress(pct, text=f"{avail} / {total} spaces free")
            c1, c2 = st.columns(2)
            c1.metric("Hourly", f"${lot['hourly_price']:.2f}")
            c2.metric("Daily",  f"${lot['daily_price']:.2f}")
            st.caption(
                f"🕐 {lot['opens_at']} – {lot['closes_at']}   "
                f"📍 {lot.get('address', '')}"
            )
            st.divider()


# ---------------------------------------------------------------------------
# Main area — chat
# ---------------------------------------------------------------------------

st.title("Parking Reservation Assistant")
st.caption(
    "Ask about availability, prices, rules, EV charging, or directions — "
    "or say **\"I want to book\"** to start a reservation."
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            sources = msg.get("sources", [])
            flags   = msg.get("guardrail_flags", [])
            intent  = msg.get("intent", "")

            if sources:
                with st.expander(f"📎 Sources ({len(sources)})"):
                    for src in sources:
                        st.markdown(f"- `{src}`")

            pii_flags = [f for f in flags if f.startswith("pii_masked")]
            if pii_flags:
                types = pii_flags[0].split(":", 1)[1] if ":" in pii_flags[0] else ""
                st.info(
                    f"🔒 Sensitive data in your message was automatically masked"
                    + (f" ({types})" if types else "") + "."
                )

            if intent and intent not in ("UNKNOWN", "BLOCKED", ""):
                st.caption(f"Intent: `{intent}`")

# Chat input
if prompt := st.chat_input("Ask a question or type your answer…"):
    # --- Show user bubble immediately ---
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": [], "guardrail_flags": [], "intent": ""}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Call backend ---
    answer        = ""
    sources:  list[str] = []
    intent:   str = ""
    flags:    list[str] = []

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                data     = _post_chat(prompt, st.session_state.session_id)
                answer   = data.get("answer", "Sorry, something went wrong.")
                sources  = data.get("sources", [])
                intent   = data.get("intent", "")
                flags    = data.get("guardrail_flags", [])
            except httpx.ConnectError:
                answer = (
                    "⚠️ **Cannot reach the backend.** "
                    "Is the API running?  \n"
                    "```\nCHROMA_COLLECTION=parking_static uvicorn app.api.main:app "
                    "--host 0.0.0.0 --port 8000\n```"
                )
            except Exception as exc:
                answer = f"⚠️ Unexpected error: {exc}"

        st.markdown(answer)

        if sources:
            with st.expander(f"📎 Sources ({len(sources)})"):
                for src in sources:
                    st.markdown(f"- `{src}`")

        pii_flags = [f for f in flags if f.startswith("pii_masked")]
        if pii_flags:
            types = pii_flags[0].split(":", 1)[1] if ":" in pii_flags[0] else ""
            st.info(
                "🔒 Sensitive data in your message was automatically masked"
                + (f" ({types})" if types else "") + "."
            )

        if intent and intent not in ("UNKNOWN", "BLOCKED", ""):
            st.caption(f"Intent: `{intent}`")

    # Store assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "guardrail_flags": flags,
            "intent": intent,
        }
    )

    # After a confirmed reservation, refresh the availability sidebar.
    if intent == _RESERVATION_INTENT and "database" in sources:
        st.session_state.lots_data = None
