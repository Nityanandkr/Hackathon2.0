"""
SilentGuard AI — ML Classification Engine (Random Forest only)

Uses a pre-trained Random Forest pipeline (StandardScaler + RandomForestClassifier)
loaded from models/silentguard_rf.pkl.

If the model file is missing, run:  python scripts/train_rf_model.py
"""

import os
from typing import Tuple

from app.models.schemas import FeatureVector, PredictionResult

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "silentguard_rf.pkl")
_model = None

FEATURE_ORDER = [
    "mouse_speed_avg", "mouse_speed_variance", "mouse_direction_changes",
    "mouse_curve_variance", "mouse_idle_ratio", "typing_interval_avg",
    "typing_interval_variance", "typing_burst_count", "backspace_frequency",
    "scroll_event_count", "scroll_speed_avg", "click_frequency",
    "session_duration_ms", "browser_entropy", "device_is_mobile",
    "device_memory_gb", "hardware_concurrency", "js_challenge_time_ms",
    "js_challenge_success", "js_challenge_score"
]


def _load_model():
    """Load the Random Forest model from disk. Raises if not found."""
    global _model
    if _model is not None:
        return _model

    import joblib

    path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(path):
        raise RuntimeError(
            f"[SilentGuard] Model file not found at '{path}'. "
            f"Please train the model first by running: python scripts/train_rf_model.py"
        )

    _model = joblib.load(path)
    print(f"[SilentGuard] Random Forest model loaded: {path}")
    return _model


def score_features(features: FeatureVector) -> PredictionResult:
    """
    Score behavioral features using the Random Forest classifier.

    Returns a PredictionResult with:
      - human_score  (0–1 probability of being human)
      - confidence   (distance from the 0.5 decision boundary, scaled to 0–1)
      - top_signals  (top-3 features by RF importance)
      - anomalies    (rule-based red-flags for explainability)
    """
    import pandas as pd

    model = _load_model()
    data = {col: [getattr(features, col)] for col in FEATURE_ORDER}
    arr = pd.DataFrame(data)

    proba = model.predict_proba(arr)[0]
    human_score = float(proba[1])
    confidence = abs(human_score - 0.5) * 2

    # Extract feature importances from the Random Forest step in the pipeline
    rf = model.named_steps["clf"]
    importances = rf.feature_importances_
    top3 = sorted(zip(FEATURE_ORDER, importances), key=lambda x: -x[1])[:3]
    top_signals = [f"{name} (importance: {imp:.3f})" for name, imp in top3]

    # Anomaly flags — explainability layer on top of the ML score
    anomalies = []
    if features.mouse_speed_variance < 100 and features.mouse_speed_avg > 0:
        anomalies.append("Constant mouse speed — linear bot movement")
    if features.typing_interval_avg < 10 and features.typing_interval_avg > 0:
        anomalies.append("Near-zero typing interval — programmatic input")
    if features.click_frequency > 5:
        anomalies.append(f"Click rate {features.click_frequency:.1f}/s exceeds human limit")
    if features.browser_entropy < 0.2:
        anomalies.append("Minimal browser fingerprint — headless browser suspected")

    return PredictionResult(
        human_score=round(human_score, 4),
        confidence=round(confidence, 4),
        top_signals=top_signals,
        anomalies=anomalies[:3],
    )


def make_decision(score: float) -> Tuple[str, str, str]:
    """
    Decision engine based on the RF human_score.

    Thresholds:
      >= 0.65 → VERIFIED  → allow
      >= 0.35 → SUSPICIOUS → otp_required
       < 0.35 → BLOCKED   → block
    """
    if score >= 0.65:
        return ("verified", "allow", "Behavioral patterns match human. Access granted.")
    elif score >= 0.35:
        return ("suspicious", "otp_required", "Insufficient confidence. Secondary check required.")
    else:
        return ("blocked", "block", "Automated access detected. Request blocked.")