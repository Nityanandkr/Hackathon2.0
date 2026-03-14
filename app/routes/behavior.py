"""
SilentGuard AI — Behavior Collection Route

POST /api/collect-behavior
Receives raw behavioral signals from the frontend JS SDK.
Stores them in session store for later ML evaluation.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import BehaviorPayload, CollectResponse
from app.store import session_store
import time

router = APIRouter()


@router.post("/collect-behavior", response_model=CollectResponse)
async def collect_behavior(payload: BehaviorPayload):
    """
    Receives behavioral signals from the browser every ~3 seconds.

    The frontend JS SDK sends:
    - mouseMovements  : list of {x, y, time, speed}
    - keystrokes      : list of {key, time, interval}
    - scrollEvents    : list of {position, time}
    - browserFingerprint : device/browser metadata
    - requestTiming   : session-level timing data

    This endpoint ONLY stores signals.
    Call /verify-human to get the ML decision.
    """

    session_id = payload.session_id

    if not session_id or len(session_id) < 4:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # Get existing session data (if this is an update)
    existing = session_store.get(session_id) or {}

    # Merge new signals with any previously collected signals
    prev_mouse    = existing.get("mouse_movements", [])
    prev_keys     = existing.get("keystrokes", [])
    prev_scroll   = existing.get("scroll_events", [])

    merged_mouse  = prev_mouse  + [m.model_dump() for m in payload.mouse_movements]
    merged_keys   = prev_keys   + [k.model_dump() for k in payload.keystrokes]
    merged_scroll = prev_scroll + [s.model_dump() for s in payload.scroll_events]

    # Cap signal lists to prevent memory abuse
    MAX_SIGNALS = 500
    merged_mouse  = merged_mouse[-MAX_SIGNALS:]
    merged_keys   = merged_keys[-MAX_SIGNALS:]
    merged_scroll = merged_scroll[-MAX_SIGNALS:]

    record = {
        "session_id":          session_id,
        "mouse_movements":     merged_mouse,
        "keystrokes":          merged_keys,
        "scroll_events":       merged_scroll,
        "browser_fingerprint": payload.browser_fingerprint.model_dump() if payload.browser_fingerprint else {},
        "request_timing":      payload.request_timing or {},
        "first_seen":          existing.get("first_seen", time.time()),
        "last_updated":        time.time(),
        "collection_count":    existing.get("collection_count", 0) + 1,
    }

    session_store.save(session_id, record)

    total_signals = len(merged_mouse) + len(merged_keys) + len(merged_scroll)

    return CollectResponse(
        session_id    = session_id,
        received      = True,
        signals_count = total_signals,
        message       = f"Collected {total_signals} signals across {record['collection_count']} batch(es)"
    )