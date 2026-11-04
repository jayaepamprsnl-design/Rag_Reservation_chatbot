"""The compiled LangGraph and its SQLite checkpointer.

    START -> chat --(user said "confirm")--> escalate --> admin_approval
               |                                              |
               |--(request pending)-------------------------->|  interrupt(): paused until
               |--(error)--> handle_error                     |  an admin decides
               '--(otherwise)--> END                          v
                                       approved --> record --> notify_user --> END
                                       refused / expired ----> notify_user
                                       cancelled -------------> END

State is checkpointed to SQLite with thread_id = session_id, so a paused approval
survives a process restart.  Any run on a paused thread re-enters admin_approval
(see chat_node / _status_after_chat), which pauses again, so a customer message
while waiting can never drop the pending approval.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.core.logging_config import logger
from app.orchestration import state as S
from app.orchestration.nodes import (
    admin_approval_node,
    chat_node,
    escalate_node,
    handle_error_node,
    notify_user_node,
    record_node,
)

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def after_chat(state: S.GraphState) -> str:
    if state.get("errors"):
        return "handle_error"
    if state.get("confirmed"):
        return "escalate"
    if state.get("status") == S.AWAITING_ADMIN:
        return "admin_approval"      # (re-)enter the wait; never drop a pending approval
    return END


def after_escalate(state: S.GraphState) -> str:
    if state.get("errors"):
        return "handle_error"
    return "admin_approval" if state.get("status") == S.AWAITING_ADMIN else END


def after_approval(state: S.GraphState) -> str:
    status = state.get("status")
    if status == S.APPROVED:
        return "record"
    if status in (S.REFUSED, S.EXPIRED):
        return "notify_user"
    if status == S.ERROR:
        return "handle_error"
    return END                       # cancelled by the customer


def after_handle_error(state: S.GraphState) -> str:
    # A failure on a user turn must not abandon a request that is still pending.
    return "admin_approval" if state.get("status") == S.AWAITING_ADMIN else END


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_graph(checkpointer):
    """Compile the graph with *checkpointer* (a SqliteSaver in production)."""
    g = StateGraph(S.GraphState)
    g.add_node("chat", chat_node)
    g.add_node("escalate", escalate_node)
    g.add_node("admin_approval", admin_approval_node)
    g.add_node("record", record_node)
    g.add_node("notify_user", notify_user_node)
    g.add_node("handle_error", handle_error_node)

    g.add_edge(START, "chat")
    g.add_conditional_edges("chat", after_chat, ["escalate", "admin_approval", "handle_error", END])
    g.add_conditional_edges("escalate", after_escalate, ["admin_approval", "handle_error", END])
    g.add_conditional_edges(
        "admin_approval", after_approval, ["record", "notify_user", "handle_error", END]
    )
    g.add_edge("record", "notify_user")
    g.add_edge("notify_user", END)
    g.add_conditional_edges("handle_error", after_handle_error, ["admin_approval", END])
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Process-wide instance (lazy; closed explicitly by tests / shutdown)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_graph = None
_conn: sqlite3.Connection | None = None


def get_graph():
    """The compiled graph on the SQLite checkpointer named by settings.checkpoint_db."""
    global _graph, _conn
    with _lock:
        if _graph is None:
            path = settings.checkpoint_db
            if path != ":memory:":
                Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(path, check_same_thread=False)
            _graph = build_graph(SqliteSaver(_conn))
            logger.info("orchestration|graph_ready checkpoint_db=%s", path)
        return _graph


def reset_graph() -> None:
    """Close the checkpointer connection and drop the instance (restart simulation / cleanup)."""
    global _graph, _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _graph, _conn = None, None


def delete_thread(session_id: str) -> None:
    """Forget a conversation's checkpoints (used by /reset-session)."""
    get_graph().checkpointer.delete_thread(session_id)
