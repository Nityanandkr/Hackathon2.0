"""
SilentGuard AI — Dashboard Routes

GET /api/dashboard/stats    → aggregated bot vs human stats
GET /api/dashboard/sessions → recent session history
"""

from fastapi import APIRouter, Depends
from typing import List
import time

from app.models.schemas import DashboardStats, SessionRecord, VerificationStatus
from app.store import session_store
from app.auth.jwt import validate_token

router = APIRouter()


@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(token: dict = Depends(validate_token)):
    """
    Returns aggregated statistics across all sessions.
    Used by the admin dashboard to show bot traffic percentages.
    Requires a valid JWT.
    """
    sessions = session_store.all()

    total       = len(sessions)
    human_ct    = 0
    bot_ct      = 0
    suspicious  = 0
    score_sum   = 0.0
    scored      = 0

    for s in sessions:
        verdict = s.get("last_verdict")
        score   = s.get("last_score")

        if verdict == "verified":
            human_ct += 1
        elif verdict == "blocked":
            bot_ct += 1
        elif verdict == "suspicious":
            suspicious += 1

        if score is not None:
            score_sum += score
            scored += 1

    human_rate   = round(human_ct   / total * 100, 1) if total else 0.0
    bot_rate     = round(bot_ct     / total * 100, 1) if total else 0.0
    avg_score    = round(score_sum  / scored,       4) if scored else 0.0

    return DashboardStats(
        total_sessions      = total,
        human_sessions      = human_ct,
        bot_sessions        = bot_ct,
        suspicious_sessions = suspicious,
        human_rate          = human_rate,
        bot_rate            = bot_rate,
        avg_human_score     = avg_score,
    )


@router.get("/dashboard/sessions", response_model=List[SessionRecord])
def session_history(limit: int = 50, token: dict = Depends(validate_token)):
    """
    Returns the most recent sessions for dashboard table view.
    Sorted newest-first. Requires a valid JWT.
    """
    sessions = session_store.all()

    # Sort by last updated descending
    sessions.sort(key=lambda s: s.get("last_updated", 0), reverse=True)

    records = []
    for s in sessions[:limit]:
        verdict = s.get("last_verdict", "pending")
        action  = s.get("last_action",  "pending")
        score   = s.get("last_score",   0.0)

        try:
            status_enum = VerificationStatus(verdict)
        except ValueError:
            status_enum = VerificationStatus.SUSPICIOUS

        total_signals = (
            len(s.get("mouse_movements", [])) +
            len(s.get("keystrokes", [])) +
            len(s.get("scroll_events", []))
        )

        records.append(SessionRecord(
            session_id   = s["session_id"],
            human_score  = score,
            status       = status_enum,
            action       = action,
            timestamp    = s.get("last_updated", time.time()),
            signal_count = total_signals,
        ))

    return records