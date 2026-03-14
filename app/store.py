"""
SilentGuard AI — Session Store
In-memory session storage. Replace with Redis for production.

Usage:
    from app.store import session_store
    session_store.save(session_id, data)
    session_store.get(session_id)
"""

from typing import Dict, Optional, List, Any
from collections import deque
import time


class SessionStore:
    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 1800):
        self._store: Dict[str, dict] = {}
        self._order: deque = deque()           # for LRU eviction
        self._max   = max_sessions
        self._ttl   = ttl_seconds

    def save(self, session_id: str, data: dict) -> None:
        self._evict_expired()
        if len(self._store) >= self._max:
            oldest = self._order.popleft()
            self._store.pop(oldest, None)

        self._store[session_id] = {
            **data,
            "_saved_at": time.time()
        }
        self._order.append(session_id)

    def get(self, session_id: str) -> Optional[dict]:
        record = self._store.get(session_id)
        if record is None:
            return None
        if time.time() - record["_saved_at"] > self._ttl:
            self._store.pop(session_id, None)
            return None
        return record

    def all(self) -> List[dict]:
        self._evict_expired()
        return list(self._store.values())

    def count(self) -> int:
        return len(self._store)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            sid for sid, rec in self._store.items()
            if now - rec.get("_saved_at", 0) > self._ttl
        ]
        for sid in expired:
            self._store.pop(sid, None)


# Singleton — import this everywhere
session_store = SessionStore()