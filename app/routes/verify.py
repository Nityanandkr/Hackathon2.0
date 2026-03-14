"""
SilentGuard AI — Verification Route

POST /api/verify-human
The core endpoint. Takes a session_id, pulls all collected behavioral
signals, extracts features, runs the ML classifier, and returns a decision.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import time

from app.models.schemas import (
    VerifyResponse, VerificationStatus, DecisionAction,
    BehaviorPayload, MouseEvent, KeystrokeEvent, ScrollEvent, BrowserFingerprint
)
from app.features.extractor import extract_features
from app.ml.classifier import score_features, make_decision
from app.store import session_store

router = APIRouter()


class VerifyRequest(BaseModel):
    session_id: str
    include_features: bool = False      # Set True for debugging / dashboard


@router.post("/verify-human", response_model=VerifyResponse)
async def verify_human(request: VerifyRequest):
    """
    ML Human Verification endpoint.

    Flow:
    1. Look up session data from store
    2. Reconstruct BehaviorPayload from stored raw signals
    3. Extract feature vector (14 numeric features)
    4. Run ML classifier → Human Confidence Score (0–1)
    5. Decision engine → allow / otp_required / block
    6. Return verdict + optional feature breakdown

    Score thresholds:
    • >= 0.80 → VERIFIED  → allow
    • >= 0.50 → SUSPICIOUS → otp_required
    •  < 0.50 → BLOCKED   → block
    """

    session_id = request.session_id
    record     = session_store.get(session_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Collect behavior first via POST /api/collect-behavior"
        )

    # ── Reconstruct payload from stored raw data ──────────────────────
    try:
        fp_data = record.get("browser_fingerprint") or {}
        payload = BehaviorPayload(
            session_id=session_id,
            mouse_movements=[
                MouseEvent(**m) for m in record.get("mouse_movements", [])
            ],
            keystrokes=[
                KeystrokeEvent(**k) for k in record.get("keystrokes", [])
            ],
            scroll_events=[
                ScrollEvent(**s) for s in record.get("scroll_events", [])
            ],
            browser_fingerprint=BrowserFingerprint(**fp_data) if fp_data else None,
            request_timing=record.get("request_timing"),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse session data: {str(e)}")

    # ── Feature extraction ────────────────────────────────────────────
    features = extract_features(payload)

    # Debug: log extracted features to see actual values
    print(f"[DEBUG] Session {session_id} features:")
    for k, v in features.model_dump().items():
        print(f"  {k:30s} = {v}")

    # ── ML scoring ────────────────────────────────────────────────────
    prediction = score_features(features)

    # ── Decision ──────────────────────────────────────────────────────
    status_str, action_str, reason = make_decision(prediction.human_score)

    # ── Persist verdict to session (for dashboard) ────────────────────
    record["last_verdict"]    = status_str
    record["last_score"]      = prediction.human_score
    record["last_action"]     = action_str
    record["verified_at"]     = time.time()
    session_store.save(session_id, record)

    return VerifyResponse(
        session_id   = session_id,
        human_score  = prediction.human_score,
        status       = VerificationStatus(status_str),
        action       = DecisionAction(action_str),
        reason       = reason,
        features     = features if request.include_features else None,
        timestamp    = time.time(),
    )


@router.post("/verify-realtime", response_model=VerifyResponse)
async def verify_realtime(payload: BehaviorPayload):
    """
    High-speed, purely in-memory validation endpoint for real-time frontend updates.

    Flow:
    1. Accepts raw BehaviorPayload directly from the client.
    2. Extracts feature vector.
    3. Runs ML classifier.
    4. Returns score immediately without hitting the DB/SessionStore.
    """
    
    # ── Feature extraction ────────────────────────────────────────────
    features = extract_features(payload)

    # ── ML scoring ────────────────────────────────────────────────────
    prediction = score_features(features)

    # ── Decision ──────────────────────────────────────────────────────
    status_str, action_str, reason = make_decision(prediction.human_score)

    return VerifyResponse(
        session_id   = payload.session_id,
        human_score  = prediction.human_score,
        status       = VerificationStatus(status_str),
        action       = DecisionAction(action_str),
        reason       = reason,
        features     = None,
        timestamp    = time.time(),
    )