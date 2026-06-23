from typing import Optional

_sessions: dict = {}

def create_session(session_id: str, vector_store) -> None:
    """Store a new session's vector store."""
    _sessions[session_id] = vector_store


def get_session(session_id: str) -> Optional[object]:
    return _sessions.get(session_id)

def session_exists(session_id: str) -> bool:
    return session_id in _sessions

def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)