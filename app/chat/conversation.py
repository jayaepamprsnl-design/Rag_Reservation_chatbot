from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ConversationState(str, Enum):
    GENERAL = "general"
    COLLECTING_RESERVATION = "collecting_reservation"
    CONFIRMING_RESERVATION = "confirming_reservation"
    DONE = "done"


@dataclass
class Message:
    role: str   # "user" | "assistant"
    content: str


@dataclass
class ConversationContext:
    session_id: str
    state: ConversationState = ConversationState.GENERAL
    messages: List[Message] = field(default_factory=list)
    reservation_data: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def last_n_messages(self, n: int = 10) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages[-n:]]


# In-memory session store — replace with Redis for multi-instance deployments.
_sessions: Dict[str, ConversationContext] = {}


def get_or_create_session(session_id: str) -> ConversationContext:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationContext(session_id=session_id)
    return _sessions[session_id]


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
