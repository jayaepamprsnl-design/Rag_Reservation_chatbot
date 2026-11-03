"""
Admin Panel page — reservation management.

Exposes a single ``render()`` function called by the st.navigation entrypoint.
All API calls go through common.py helpers (never touches the DB directly).
"""
from __future__ import annotations

import streamlit as st

from app.ui.common import (
    API_URL,
    _ADMIN_KEY_HEADER,
    admin_headers,
    api_get,
    api_post,
    auto_refresh,
    init_admin_state,
    show_api_error,
    status_badge,
)

_REFRESH = "5s"

# ---------------------------------------------------------------------------
# Login form
# ---------------------------------------------------------------------------

def _show_login() -> None:
    st.title("Admin Panel — Sign In")
    st.caption("Enter the admin API key configured in your .env file.")

    with st.form("admin_login_form"):
        key_input = st.text_input(
            "Admin API Key",
            type="password",
            placeholder="ADMIN_API_KEY value from .env",
        )
        submitted = st.form_submit_button(
            "Sign In", type="primary", use_container_width=True
        )

    if submitted:
        if not key_input.strip():
            st.error("Please enter the API key.")
            return
        import httpx as _httpx
        try:
            r = _httpx.get(
                f"{API_URL}/admin/verify",
                headers={_ADMIN_KEY_HEADER: key_input.strip()},
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
        except _httpx.ConnectError:
            st.error(f"Cannot reach API at {API_URL}. Is the server running?")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Pending panel
# ---------------------------------------------------------------------------

@auto_refresh(_REFRESH)
def _pending_panel() -> None:
    code, data = api_get("/admin/reservations/pending", headers=admin_headers())
    if code != 200:
        show_api_error(code, data)
        return

    rows: list[dict] = data  # type: ignore[assignment]

    if not rows:
        st.info(f"No pending reservations right now. Auto-refreshes every {_REFRESH}.")
        return

    st.markdown(
        f"**{len(rows)} pending reservation(s)** — auto-refreshing every {_REFRESH}"
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
                # Approve button
                if st.button(
                    "Approve",
                    key=f"adm_ap_{rid}",
                    type="primary",
                    use_container_width=True,
                ):
                    c, body = api_post(
                        f"/admin/reservations/{rid}/approve",
                        headers=admin_headers(),
                        json={"note": ""},
                    )
                    if c == 200:
                        st.success(f"CPK-{rid:06d} approved!")
                    else:
                        show_api_error(c, body)
                    st.rerun()

                # Refuse form
                with st.form(key=f"adm_rf_{rid}"):
                    reason = st.text_input(
                        "Refusal reason (required)",
                        key=f"adm_rsn_{rid}",
                        placeholder="e.g. Lot closed for maintenance",
                    )
                    if st.form_submit_button("Refuse", use_container_width=True):
                        if not reason.strip():
                            st.warning("A reason is required.")
                        else:
                            c, body = api_post(
                                f"/admin/reservations/{rid}/refuse",
                                headers=admin_headers(),
                                json={"reason": reason.strip()},
                            )
                            if c == 200:
                                st.error(f"CPK-{rid:06d} refused.")
                            else:
                                show_api_error(c, body)
                            st.rerun()


# ---------------------------------------------------------------------------
# History panel  (uses st.table to avoid pyarrow/NumPy version conflicts)
# ---------------------------------------------------------------------------

@auto_refresh(_REFRESH)
def _history_panel() -> None:
    code, data = api_get(
        "/admin/reservations/history",
        headers=admin_headers(),
        limit=50,
    )
    if code != 200:
        show_api_error(code, data)
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
            "Status":   status_badge(r["status"]),
            "Reason":   r.get("admin_decision_reason") or "—",
            "Decided":  (r.get("decided_at") or "—")[:19],
        })

    # Use st.table instead of st.dataframe to avoid pyarrow/NumPy version
    # conflicts (pyarrow >= 15 may require NumPy >= 2.0 at runtime).
    st.table(table)


# ---------------------------------------------------------------------------
# Agent chat panel
# ---------------------------------------------------------------------------

def _agent_chat_panel() -> None:
    st.markdown(
        "Chat with the admin agent in natural language.  \n"
        "Examples: **approve 5**, **refuse 3 because maintenance scheduled**"
    )
    st.divider()

    for msg in st.session_state.admin_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Type a command or question…",
                                key="adm_chat_input"):
        st.session_state.admin_chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing…"):
                c, body = api_post(
                    "/admin/chat",
                    headers=admin_headers(),
                    json={"message": prompt},
                )

            if c == 200:
                reply = body.get("response", "Done.")
                if body.get("decision_made"):
                    res_id = body.get("reservation_id")
                    if res_id:
                        reply = f"Decision recorded for **CPK-{res_id:06d}**.\n\n" + reply
            elif c == 0:
                reply = f"Cannot reach API: {body.get('detail', '')}"
            elif c == 422:
                reply = f"Could not apply decision: {body.get('detail', 'Unknown error')}"
            else:
                reply = f"Error {c}: {body.get('detail', 'Unknown')}"

            st.markdown(reply)
            st.session_state.admin_chat_history.append(
                {"role": "assistant", "content": reply}
            )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Admin Panel page."""
    init_admin_state()

    # Login gate
    if not st.session_state.admin_key:
        _show_login()
        return

    # Header + sign-out
    hcol, btn_col = st.columns([5, 1])
    hcol.title(":key: Admin Panel")
    if btn_col.button("Sign Out", use_container_width=True, key="adm_signout"):
        st.session_state.admin_key = ""
        st.session_state.admin_chat_history = []
        st.rerun()

    st.caption(f"Connected to **{API_URL}**  |  pending list auto-refreshes every {_REFRESH}")
    st.divider()

    # Tabs
    tab_pending, tab_history, tab_chat = st.tabs(
        ["🔔 Pending", "📋 History", "💬 Ask Agent"]
    )

    with tab_pending:
        _pending_panel()

    with tab_history:
        _history_panel()

    with tab_chat:
        _agent_chat_panel()
