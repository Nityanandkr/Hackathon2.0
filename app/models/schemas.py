"""
SilentGuard AI — Data Schemas
All request and response models defined here.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────
class VerificationStatus(str, Enum):
    VERIFIED   = "verified"
    SUSPICIOUS = "suspicious"
    BLOCKED    = "blocked"


class DecisionAction(str, Enum):
    ALLOW   = "allow"
    OTP     = "otp_required"
    BLOCK   = "block"


# ── Incoming behavioral signal types ──────────────────────────────────
class MouseEvent(BaseModel):
    x:     float
    y:     float
    time:  int              # epoch ms
    speed: Optional[float] = 0.0


class KeystrokeEvent(BaseModel):
    key:      str
    time:     int           # epoch ms
    interval: Optional[int] = None   # ms since last key


class ScrollEvent(BaseModel):
    position: float         # scrollY in px
    time:     int           # epoch ms


class BrowserFingerprint(BaseModel):
    screen_width:    Optional[int]   = None
    screen_height:   Optional[int]   = None
    timezone:        Optional[str]   = None
    language:        Optional[str]   = None
    platform:        Optional[str]   = None
    color_depth:     Optional[int]   = None
    touch_support:   Optional[bool]  = None
    webgl_vendor:    Optional[str]   = None
    user_agent_hash: Optional[str]   = None   # hashed — never store raw UA
    hardware_concurrency: Optional[int]   = None
    device_memory:   Optional[float] = None
    is_mobile:       Optional[bool]  = None

class JSChallengeResult(BaseModel):
    execution_time_ms: float
    success: bool
    score: float

# ── Main payload sent from frontend every ~3 seconds ──────────────────
class BehaviorPayload(BaseModel):
    session_id:         str
    mouse_movements:    List[MouseEvent]      = Field(default_factory=list)
    keystrokes:         List[KeystrokeEvent]  = Field(default_factory=list)
    scroll_events:      List[ScrollEvent]     = Field(default_factory=list)
    browser_fingerprint: Optional[BrowserFingerprint] = None
    js_challenge:       Optional[JSChallengeResult] = None
    request_timing:     Optional[Dict[str, Any]] = None
    timestamp:          Optional[int] = None


class FeatureVector(BaseModel):
    mouse_speed_avg:          float = 0.0
    mouse_speed_variance:     float = 0.0
    mouse_direction_changes:  int   = 0
    mouse_curve_variance:     float = 0.0
    mouse_idle_ratio:         float = 0.0
    typing_interval_avg:      float = 0.0
    typing_interval_variance: float = 0.0
    typing_burst_count:       int   = 0
    backspace_frequency:      float = 0.0
    scroll_event_count:       int   = 0
    scroll_speed_avg:         float = 0.0
    click_frequency:          float = 0.0
    session_duration_ms:      int   = 0
    browser_entropy:          float = 0.0
    device_is_mobile:         float = 0.0
    device_memory_gb:         float = 0.0
    hardware_concurrency:     float = 0.0
    js_challenge_time_ms:     float = 0.0
    js_challenge_success:     float = 0.0
    js_challenge_score:       float = 0.0


# ── ML model output ───────────────────────────────────────────────────
class PredictionResult(BaseModel):
    human_score:       float               # 0.0 – 1.0
    confidence:        float               # model confidence
    top_signals:       List[str]           # most important features
    anomalies:         List[str]           # detected suspicious patterns


# ── API Responses ──────────────────────────────────────────────────────
class CollectResponse(BaseModel):
    session_id:      str
    received:        bool
    signals_count:   int
    message:         str


class VerifyResponse(BaseModel):
    session_id:      str
    human_score:     float
    status:          VerificationStatus
    action:          DecisionAction
    reason:          str
    features:        Optional[FeatureVector] = None
    timestamp:       float


class DashboardStats(BaseModel):
    total_sessions:     int
    human_sessions:     int
    bot_sessions:       int
    suspicious_sessions: int
    human_rate:         float   # percentage
    bot_rate:           float
    avg_human_score:    float


class SessionRecord(BaseModel):
    session_id:   str
    human_score:  float
    status:       VerificationStatus
    action:       DecisionAction
    timestamp:    float
    signal_count: int


class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None