"""
CityPark Admin Dashboard — Streamlit UI.

Features
--------
* Login gate:  enter the ADMIN_API_KEY from .env to sign in
* Pending tab: live table of pending reservations with one-click Approve and
               Refuse-with-reason; auto-refreshes every 5 seconds via
               st.fragment(run_every=...)
* History tab: all decided reservations (confirmed / refused / expired)
* Agent chat:  free-form natural-language commands forwarded to /admin/chat
               (e.g. "approve 12", "refuse 7 because maintenance")

Run:
    ADMIN_API_KEY=<key> streamlit run app/ui/admin_app.py --server.port 8502
"""
from __future__ import annotations

import functools
import os

import httpx
import streamlit as st


# ---------------------------------------------------------------------------
# Backward-compatibility shim for st.fragment  (requires Streamlit >= 1.33)
# ---------------------------------------------------------------------------

def _auto_refresh(run_every: str = "5s"):
    """Return the best available auto-refresh decorator for this Streamlit version."""
    for attr in ("fragment", "experimental_fragment"):
        fn = getattr(st, attr, None)
        if fn is not None:
            try:
                return fn(run_every=run_every)
            except Exception:
                pass
    def _noop(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return _wrapper
    return _noop

API_URL = os.getenv("API_URL", "http://localhost:8000")
_KEY_HEADER = "X-Admin-Key"
_REFRESH = "5s"

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CityPark Admin",
    page_icon=":key:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "admin_key" not in st.session_state:
    # Pre-fill from env so docker-compose / local dev just works
    st.session_state.admin_key = os.getenv("ADMIN_API_KEY", "")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _hdrs() -> dict:
    return {_KEY_HEADER: st.session_state.admin_key}


def _get(path: str, **params) -> tuple[int, object]:
    try:
        r = httpx.get(f"{API_URL}{path}", headers=_hdrs(),
                      params=params, timeout=10.0)
        return r.status_code, r.json()
    except httpx.ConnectError:
        return 0, {"detail": "Cannot reach API"}
    except Exception as exc:
        return 0, {"detail": str(exc)}


def _post(path: str, json: dict | None = None) -> tuple[int, dict]:
    try:
        r = httpx.post(f"{API_URL}{path}", headers=_hdrs(),
                       json=json or {}, timeout=20.0)
        return r.status_code, r.json()
    except httpx.ConnectError:
        return 0, {"detail": "Cannot reach API"}
    except Exception as exc:
        return 0, {"detail": str(exc)}


def _status_badge(status: str) -> str:
    return {
        "pending":   "🟡 Pending",
        "confirmed": "🟢 Confirmed",
        "refused":   "🔴 Refused",
        "expired":   "⬜ Expired",
        "cancelled": "⬛ Cancelled",
    }.get(status.lower(), status.upper())


# ---------------------------------------------------------------------------
# Login gate
# ---------------------------------------------------------------------------

def _show_login() -> None:
    st.title("CityPark Admin — Sign In")
    st.caption("Enter the admin API key configured in your .env file.")

    with st.form("login_form"):
        key_input = st.text_input(
            "Admin API Key",
            type="password",
            placeholder="ADMIN_API_KEY value from .env",
        )
        if st.form_submit_button("Sign In", type="primary", use_container_width=True):
            if not key_input.strip():
                st.error("Please enter the API key.")
                return
            # Verify by calling a protected endpoint
            try:
                r = httpx.get(
                    f"{API_URL}/admin/reservations/pending",
                    headers={_KEY_HEADER: key_input.strip()},
                    timeout=5.0,
                )
                if r.status_code == 200:
                    st.session_state.admin_key = key_input.strip()
                    st.rerun()
                elif r.status_code in (401, 403):
                    st.error("Invalid admin key.")
                elif r.status_code == 503:
                    st.error("ADMIN_API_KEY is not set on the server.")
                else:
                    st.error(f"API returned {r.status_code}.")
            except httpx.ConnectError:
                st.error(f"Cannot reach API at {API_URL}. Is the server running?")


if not st.session_state.admin_key:
    _show_login()
    st.stop()


# ---------------------------------------------------------------------------
# Dashboard header
# ---------------------------------------------------------------------------

hcol, btn_col = st.columns([5, 1])
hcol.title(":key: CityPark Admin Dashboard")
if btn_col.button("Sign Out", use_container_width=True):
    st.session_state.admin_key = ""
    st.session_state.chat_history = []
    st.rerun()

st.caption(f"Connected to **{API_URL}**  |  auto-refreshes every {_REFRESH}")
st.divider()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_pending, tab_history, tab_chat = st.tabs(
    ["🔔 Pending Reservations", "📋 History", "💬 Ask Agent"]
)


# ── PENDING ─────────────────────────────────────────────────────────────────

with tab_pending:

    @_auto_refresh(_REFRESH)
    def _pending_panel() -> None:
        code, data = _get("/admin/reservations/pending")
        if code == 0:
            st.error(f"Cannot reach API: {data.get('detail', '')}")
            return
        if code != 200:
            st.error(f"API error {code}: {data.get('detail', '')}")
            return

        rows: list[dict] = data  # type: ignore[assignment]

        if not rows:
            st.info(
                "No pending reservations right now.  "
                f"This panel auto-refreshes every {_REFRESH}."
            )
            return

        st.markdown(
            f"**{len(rows)} pending reservation(s)** — "
            f"auto-refreshing every {_REFRESH}"
        )
        st.divider()

        for r in rows:
            rid = r["id"]
            header = (
                f"**CPK-{rid:06d}**  |  {r['customer_name']}  |  "
                f"{r['lot_name']}  |  {r['start_time'][:16]}"
            )
            with st.expander(header, expanded=True):
                info_col, act_col = st.columns([3, 2])

                with info_col:
                    st.markdown(f"**Customer:**  {r['customer_name']}")
                    st.markdown(f"**Plate:**     `{r['car_plate_masked']}`")
                    st.markdown(f"**Facility:**  {r['lot_name']}")
                    st.markdown(f"**Arrives:**   `{r['start_time'][:16]}`")
                    st.markdown(f"**Departs:**   `{r['end_time'][:16]}`")
                    if r.get("session_id"):
                        st.caption(f"Session: `{r['session_id']}`")
                    st.caption(f"Submitted: {r['created_at'][:19]}")

                with act_col:
                    # ── Approve ──────────────────────────────────────────────
                    if st.button(
                        "Approve",
                        key=f"ap_{rid}",
                        type="primary",
                        use_container_width=True,
                    ):
                        c, body = _post(
                            f"/admin/reservations/{rid}/approve",
                            json={"note": ""},
                        )
                        if c == 200:
                            st.success(f"CPK-{rid:06d} approved!")
                        elif c == 409:
                            st.warning("Already decided.")
                        else:
                            st.error(body.get("detail", f"Error {c}"))
                        st.rerun()

                    # ── Refuse ───────────────────────────────────────────────
                    with st.form(key=f"rf_{rid}"):
                        reason = st.text_input(
                            "Refusal reason (required)",
                            key=f"rsn_{rid}",
                            placeholder="e.g. Lot closed for maintenance",
                        )
                        if st.form_submit_button(
                            "Refuse", use_container_width=True
                        ):
                            if not reason.strip():
                                st.warning("A reason is required.")
                            else:
                                c, body = _post(
                                    f"/admin/reservations/{rid}/refuse",
                                    json={"reason": reason.strip()},
                                )
                                if c == 200:
                                    st.error(f"CPK-{rid:06d} refused.")
                                elif c == 409:
                                    st.warning("Already decided.")
                                else:
                                    st.error(body.get("detail", f"Error {c}"))
                                st.rerun()

    _pending_panel()


# ── HISTORY ──────────────────────────────────────────────────────────────────

with tab_history:

    @_auto_refresh(_REFRESH)
    def _history_panel() -> None:
        code, data = _get("/admin/reservations/history", limit=50)
        if code == 0:
            st.error(f"Cannot reach API: {data.get('detail', '')}")
            return
        if code != 200:
            st.error(f"API error {code}")
            return

        rows: list[dict] = data  # type: ignore[assignment]

        if not rows:
            st.info("No reservation history yet.")
            return

        st.markdown(f"**{len(rows)} decided reservation(s)**")

        table = []
        for r in rows:
            table.append({
                "Ref":      f"CPK-{r['id']:06d}",
                "Customer": r["customer_name"],
                "Plate":    r["car_plate_masked"],
                "Facility": r["lot_name"],
                "Arrives":  r["start_time"][:16],
                "Departs":  r["end_time"][:16],
                "Status":   _status_badge(r["status"]),
                "Reason":   r.get("admin_decision_reason") or "—",
                "Decided":  (r.get("decided_at") or "—")[:19],
            })

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Reason": st.column_config.TextColumn("Reason", width="large"),
            },
        )

    _history_panel()


# ── AGENT CHAT ───────────────────────────────────────────────────────────────

with tab_chat:
    st.markdown(
        "Chat with the admin agent in natural language.  \n"
        "Examples: **approve 5**, **refuse 3 because maintenance scheduled**,  \n"
        "**show pending**, **what reservations are waiting?**"
    )
    st.divider()

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Type a command or question…", key="admin_chat"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing…"):
                c, body = _post("/admin/chat", json={"message": prompt})

            if c == 200:
                reply_text = body.get("response", "Done.")
                if body.get("decision_made"):
                    res_id = body.get("reservation_id")
                    if res_id:
                        reply_text = (
                            f"Decision recorded for **CPK-{res_id:06d}**.\n\n"
                            + reply_text
                        )
            elif c == 422:
                reply_text = (
                    f"Could not apply decision: "
                    f"{body.get('detail', 'Unknown error')}"
                )
            elif c == 0:
                reply_text = f"Cannot reach API: {body.get('detail', '')}"
            else:
                reply_text = f"Error {c}: {body.get('detail', 'Unknown')}"

            st.markdown(reply_text)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply_text}
            )
