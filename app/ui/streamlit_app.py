"""
CityPark — single Streamlit entrypoint.

Two pages accessible from the sidebar:
  Chat        (default) — parking assistant with multi-slot booking flow
  Admin Panel           — reservation management (requires API key)

Run locally:
    streamlit run app/ui/streamlit_app.py
"""
import streamlit as st

# st.set_page_config must be the very first Streamlit call.
st.set_page_config(
    page_title="CityPark",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise all session-state keys before any page renders.
# (Pages may be switched mid-session; absent keys would cause KeyErrors.)
from app.ui.common import init_admin_state, init_chat_state  # noqa: E402

init_chat_state()
init_admin_state()

# Import page render functions (they never call st.* at module level).
from app.ui.pages.admin_page import render as _admin_render  # noqa: E402
from app.ui.pages.chat_page import render as _chat_render    # noqa: E402

chat_page  = st.Page(_chat_render,  title="Chat",         icon="💬", default=True)
admin_page = st.Page(_admin_render, title="Admin Panel",  icon="🔑")

pg = st.navigation([chat_page, admin_page])
pg.run()
