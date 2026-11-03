"""
Chat page — parking assistant with booking flow.

Exposes a single ``render()`` function called by the st.navigation entrypoint.
"""
from __future__ import annotations

import httpx
import streamlit as st

from app.ui.common import (
    API_URL,
    REQUEST_TIMEOUT,
    api_get,
    auto_refresh,
    init_chat_state,
    status_badge,
)

# Lazy import so the UI doesn't crash when the backend package is unavailable.
try:
    from app.chat.router import RouterIntent as _RouterIntent
    _RESERVATION_INTENT = _RouterIntent.RESERVATION.value
except Exception:
    _RESERVATION_INTENT = "RESERVATION"

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _availability_color(pct: float) -> str:
    if pct > 0.3:
        return "🟢"
    if pct > 0.1:
        return "🟡"
    return "🔴"


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
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Notification poller (auto-refresh fragment)
# ---------------------------------------------------------------------------

@auto_refresh("5s")
def _notification_poller() -> None:
    """Poll /sessions/{id}/notifications when a reservation is awaiting approval."""
    if not st.session_state.get("awaiting_approval"):
        return

    sid = st.session_state.session_id
    code, body = api_get(f"/sessions/{sid}/notifications", timeout=4.0)
    if code != 200 or body.get("count", 0) == 0:
        return

    for notif in body["notifications"]:
        msg_text = notif.get("message", "")
        decision = notif.get("decision", "")
        prefix = (
            "**Your parking request has been APPROVED!**\n\n"
            if decision == "confirmed"
            else "**Update on your parking request:**\n\n"
        )
        st.session_state.messages.append({
            "role":           "assistant",
            "content":        prefix + msg_text,
            "sources":        ["database"],
            "guardrail_flags": [],
            "intent":         _RESERVATION_INTENT,
        })

    st.session_state.awaiting_approval = False
    st.session_state.pending_reservation_id = None
    st.session_state.lots_data = None
    st.rerun()


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Chat page (sidebar + main area)."""
    init_chat_state()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🅿️ CityPark")
        st.caption("Live parking availability")

        col_refresh, col_new = st.columns(2)

        if col_refresh.button("↻ Refresh", use_container_width=True, key="chat_refresh"):
            st.session_state.lots_data = None

        if col_new.button("🗑️ New Chat", use_container_width=True,
                          type="secondary", key="chat_new"):
            _reset_server_session(st.session_state.session_id)
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.lots_data = None
            st.session_state.awaiting_approval = False
            st.session_state.pending_reservation_id = None
            st.rerun()

        st.divider()

        # Reservation status badge
        if (st.session_state.awaiting_approval
                and st.session_state.pending_reservation_id):
            rid = st.session_state.pending_reservation_id
            code, s = api_get(f"/reservations/{rid}/status", timeout=4.0)
            if code == 200:
                status_str = s.get("status", "unknown").upper()
                badge = {
                    "PENDING":   "🟡 **Awaiting admin approval**",
                    "CONFIRMED": "🟢 **Booking confirmed!**",
                    "REFUSED":   "🔴 **Booking refused**",
                    "EXPIRED":   "⬜ **Request expired**",
                }.get(status_str, f"ℹ️ {status_str}")
                st.markdown(f"**Your reservation CPK-{rid:06d}**")
                st.markdown(badge)
                reason = s.get("admin_decision_reason")
                if reason:
                    st.caption(f"Reason: {reason}")
                st.divider()

        # Lot availability
        if st.session_state.lots_data is None:
            with st.spinner("Loading lots…"):
                code, data = api_get("/lots", timeout=5.0)
                st.session_state.lots_data = data if code == 200 else []

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

    # ── Main area — conversation ──────────────────────────────────────────────
    st.title("Parking Reservation Assistant")
    st.caption(
        "Ask about availability, prices, rules, EV charging, or directions — "
        "or say **\"I want to book\"** to start a reservation."
    )

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
                        "🔒 Sensitive data in your message was automatically masked"
                        + (f" ({types})" if types else "") + "."
                    )

                if intent and intent not in ("UNKNOWN", "BLOCKED", ""):
                    st.caption(f"Intent: `{intent}`")

    # Chat input
    if prompt := st.chat_input("Ask a question or type your answer…",
                                key="chat_input"):
        st.session_state.messages.append({
            "role": "user", "content": prompt,
            "sources": [], "guardrail_flags": [], "intent": "",
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = ""
        sources: list[str] = []
        intent  = ""
        flags:  list[str] = []
        data:   dict = {}

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    data    = _post_chat(prompt, st.session_state.session_id)
                    answer  = data.get("answer", "Sorry, something went wrong.")
                    sources = data.get("sources", [])
                    intent  = data.get("intent", "")
                    flags   = data.get("guardrail_flags", [])
                except httpx.ConnectError:
                    answer = (
                        "⚠️ **Cannot reach the backend.** "
                        "Is the API running?  \n"
                        "```\nCHROMA_COLLECTION=parking_static "
                        "uvicorn app.api.main:app --host 0.0.0.0 --port 8000\n```"
                    )
                except Exception as exc:  # noqa: BLE001
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

        st.session_state.messages.append({
            "role":           "assistant",
            "content":        answer,
            "sources":        sources,
            "guardrail_flags": flags,
            "intent":         intent,
        })

        # Update awaiting-approval tracking
        res_id = data.get("reservation_id")
        if res_id and intent == _RESERVATION_INTENT:
            if ("awaiting" in answer.lower() or "queued" in answer.lower()
                    or "administrator" in answer.lower()):
                st.session_state.awaiting_approval = True
                st.session_state.pending_reservation_id = res_id
            if ("approved" in answer.lower() or "refused" in answer.lower()
                    or "not approved" in answer.lower()):
                st.session_state.awaiting_approval = False
                st.session_state.pending_reservation_id = None
                st.session_state.lots_data = None

        if intent == _RESERVATION_INTENT and "database" in sources:
            st.session_state.lots_data = None

    # Notification poller (auto-refresh when waiting for admin)
    _notification_poller()
