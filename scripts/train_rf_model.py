"""
SilentGuard AI — Random Forest Model Training Script

Generates a large synthetic behavioral dataset (100,000 samples) with diverse
human and bot profiles, trains a Random Forest classifier, evaluates it with
cross-validation, and saves the pipeline to models/silentguard_rf.pkl.

Usage:
    python scripts/train_rf_model.py
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib

FEATURE_ORDER = [
    "mouse_speed_avg", "mouse_speed_variance", "mouse_direction_changes",
    "mouse_curve_variance", "mouse_idle_ratio", "typing_interval_avg",
    "typing_interval_variance", "typing_burst_count", "backspace_frequency",
    "scroll_event_count", "scroll_speed_avg", "click_frequency",
    "session_duration_ms", "browser_entropy",
    "device_is_mobile", "device_memory_gb", "hardware_concurrency",
    "js_challenge_time_ms", "js_challenge_success", "js_challenge_score"
]


# ══════════════════════════════════════════════════════════════════════════
#  HUMAN PROFILES — diverse, realistic user behaviors
# ══════════════════════════════════════════════════════════════════════════

def _human_casual_browser():
    """Casual user: slow mouse, minimal typing, lots of scrolling."""
    return [
        random.uniform(50, 400),                    # mouse_speed_avg
        random.uniform(3000, 200000),               # mouse_speed_variance
        random.randint(3, 40),                      # mouse_direction_changes
        random.uniform(200, 40000),                 # mouse_curve_variance
        random.uniform(0.15, 0.7),                  # mouse_idle_ratio (lots of pauses)
        random.uniform(150, 600),                   # typing_interval_avg (slow typer)
        random.uniform(1000, 40000),                # typing_interval_variance
        random.randint(0, 2),                       # typing_burst_count
        random.uniform(0.02, 0.12),                 # backspace_frequency
        random.randint(5, 60),                      # scroll_event_count (scrolls a lot)
        random.uniform(50, 500),                    # scroll_speed_avg
        random.uniform(0.01, 0.3),                  # click_frequency
        random.randint(8000, 120000),               # session_duration_ms (long sessions)
        random.uniform(0.4, 1.0),                   # browser_entropy
        0.0,                                        # device_is_mobile
        random.choice([4.0, 8.0, 16.0]),            # device_memory_gb
        random.choice([4.0, 8.0, 16.0]),            # hardware_concurrency
        random.uniform(30.0, 100.0),                # js_challenge_time_ms
        1.0,                                        # js_challenge_success
        random.uniform(800, 1200),                  # js_challenge_score
    ]


def _human_fast_typer():
    """Power user: fast typing, moderate mouse use, high variance."""
    return [
        random.uniform(200, 1200),                  # mouse_speed_avg
        random.uniform(10000, 500000),              # mouse_speed_variance
        random.randint(10, 80),                     # mouse_direction_changes
        random.uniform(1000, 100000),               # mouse_curve_variance
        random.uniform(0.03, 0.4),                  # mouse_idle_ratio
        random.uniform(40, 150),                    # typing_interval_avg (fast typer)
        random.uniform(500, 15000),                 # typing_interval_variance
        random.randint(1, 6),                       # typing_burst_count
        random.uniform(0.03, 0.20),                 # backspace_frequency
        random.randint(0, 20),                      # scroll_event_count
        random.uniform(0, 400),                     # scroll_speed_avg
        random.uniform(0.05, 0.6),                  # click_frequency
        random.randint(5000, 60000),                # session_duration_ms
        random.uniform(0.5, 1.0),                   # browser_entropy
        0.0,
        random.choice([8.0, 16.0, 32.0]),
        random.choice([8.0, 12.0, 16.0]),
        random.uniform(15.0, 50.0),
        1.0,
        random.uniform(1000, 1500),
    ]


def _human_mobile_user():
    """Mobile/touch user: no mouse, touch-based scrolling, smaller screen entropy."""
    return [
        random.uniform(100, 800),                   # mouse_speed_avg (touch drag)
        random.uniform(2000, 150000),               # mouse_speed_variance
        random.randint(2, 30),                      # mouse_direction_changes
        random.uniform(100, 30000),                 # mouse_curve_variance
        random.uniform(0.1, 0.6),                   # mouse_idle_ratio
        random.uniform(100, 500),                   # typing_interval_avg (on-screen keyboard)
        random.uniform(2000, 50000),                # typing_interval_variance (high variance)
        random.randint(0, 3),                       # typing_burst_count
        random.uniform(0.05, 0.25),                 # backspace_frequency (more typos on mobile)
        random.randint(10, 80),                     # scroll_event_count (lots of scrolling)
        random.uniform(100, 800),                   # scroll_speed_avg (swipe scrolling)
        random.uniform(0.02, 0.4),                  # click_frequency (taps)
        random.randint(5000, 90000),                # session_duration_ms
        random.uniform(0.3, 0.9),                   # browser_entropy
        1.0,
        random.choice([3.0, 4.0, 6.0, 8.0]),
        random.choice([4.0, 6.0, 8.0]),
        random.uniform(80.0, 200.0),
        1.0,
        random.uniform(400, 800),
    ]


def _human_elderly_user():
    """Elderly/slow user: very slow mouse, long pauses, slow typing."""
    return [
        random.uniform(30, 200),                    # mouse_speed_avg (very slow)
        random.uniform(1000, 80000),                # mouse_speed_variance
        random.randint(2, 20),                      # mouse_direction_changes
        random.uniform(50, 15000),                  # mouse_curve_variance
        random.uniform(0.3, 0.8),                   # mouse_idle_ratio (lots of hesitation)
        random.uniform(300, 1200),                  # typing_interval_avg (very slow typing)
        random.uniform(5000, 80000),                # typing_interval_variance
        random.randint(0, 1),                       # typing_burst_count
        random.uniform(0.05, 0.30),                 # backspace_frequency (more corrections)
        random.randint(1, 15),                      # scroll_event_count
        random.uniform(20, 200),                    # scroll_speed_avg
        random.uniform(0.01, 0.15),                 # click_frequency
        random.randint(15000, 180000),              # session_duration_ms (very long sessions)
        random.uniform(0.35, 1.0),                  # browser_entropy
        0.0,
        random.choice([4.0, 8.0]),
        random.choice([4.0, 8.0]),
        random.uniform(50.0, 150.0),
        1.0,
        random.uniform(600, 1000),
    ]


def _human_gamer():
    """Gamer/power user: very fast mouse with high variance, quick reflexes."""
    return [
        random.uniform(500, 2500),                  # mouse_speed_avg (fast)
        random.uniform(50000, 800000),              # mouse_speed_variance (very inconsistent)
        random.randint(20, 120),                    # mouse_direction_changes (lots of movement)
        random.uniform(5000, 150000),               # mouse_curve_variance
        random.uniform(0.01, 0.2),                  # mouse_idle_ratio (rarely idle)
        random.uniform(30, 120),                    # typing_interval_avg (fast)
        random.uniform(200, 8000),                  # typing_interval_variance
        random.randint(2, 8),                       # typing_burst_count
        random.uniform(0.01, 0.10),                 # backspace_frequency
        random.randint(0, 10),                      # scroll_event_count
        random.uniform(0, 300),                     # scroll_speed_avg
        random.uniform(0.1, 1.2),                   # click_frequency
        random.randint(3000, 45000),                # session_duration_ms
        random.uniform(0.5, 1.0),                   # browser_entropy
        0.0,
        random.choice([16.0, 32.0, 64.0]),
        random.choice([12.0, 16.0, 24.0]),
        random.uniform(10.0, 30.0),
        1.0,
        random.uniform(1500, 2500),
    ]


def _human_form_filler():
    """User filling out a form: structured typing, tab navigation, moderate mouse."""
    return [
        random.uniform(100, 600),                   # mouse_speed_avg
        random.uniform(5000, 200000),               # mouse_speed_variance
        random.randint(5, 35),                      # mouse_direction_changes
        random.uniform(300, 50000),                 # mouse_curve_variance
        random.uniform(0.1, 0.5),                   # mouse_idle_ratio
        random.uniform(80, 350),                    # typing_interval_avg
        random.uniform(1000, 25000),                # typing_interval_variance
        random.randint(1, 5),                       # typing_burst_count
        random.uniform(0.02, 0.18),                 # backspace_frequency
        random.randint(0, 8),                       # scroll_event_count
        random.uniform(0, 200),                     # scroll_speed_avg
        random.uniform(0.05, 0.5),                  # click_frequency
        random.randint(10000, 90000),               # session_duration_ms
        random.uniform(0.4, 1.0),                   # browser_entropy
        0.0,
        random.choice([8.0, 16.0]),
        random.choice([8.0, 16.0]),
        random.uniform(20.0, 80.0),
        1.0,
        random.uniform(800, 1300),
    ]


def generate_human_features():
    """Randomly pick one of several realistic human behavioral profiles."""
    profile = random.choice([
        _human_casual_browser,
        _human_fast_typer,
        _human_mobile_user,
        _human_elderly_user,
        _human_gamer,
        _human_form_filler,
    ])
    return profile()


# ══════════════════════════════════════════════════════════════════════════
#  BOT PROFILES — diverse automated attack patterns
# ══════════════════════════════════════════════════════════════════════════

def _bot_instant():
    """Instant bot: blazing fast, no variance, zero entropy."""
    return [
        random.uniform(3000, 15000),                # mouse_speed_avg (too fast)
        random.uniform(0, 500),                     # mouse_speed_variance (constant)
        random.randint(0, 1),                       # mouse_direction_changes (straight line)
        random.uniform(0, 50),                      # mouse_curve_variance (no curves)
        0.0,                                        # mouse_idle_ratio (never idles)
        random.uniform(0, 5),                       # typing_interval_avg (instant typing)
        random.uniform(0, 10),                      # typing_interval_variance (constant)
        random.randint(10, 30),                     # typing_burst_count (all bursts)
        0.0,                                        # backspace_frequency (never corrects)
        random.randint(0, 1),                       # scroll_event_count
        random.uniform(5000, 20000),                # scroll_speed_avg (instant scroll)
        random.uniform(3.0, 20.0),                  # click_frequency (superhuman)
        random.randint(50, 500),                    # session_duration_ms (too fast)
        random.uniform(0.0, 0.15),                  # browser_entropy (headless)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def _bot_linear():
    """Linear bot: steady constant speed, no direction changes."""
    speed = random.uniform(100, 400)
    return [
        speed,                                      # mouse_speed_avg
        random.uniform(0, 200),                     # mouse_speed_variance (too steady)
        random.randint(0, 1),                       # mouse_direction_changes (straight)
        random.uniform(0, 30),                      # mouse_curve_variance (no curves)
        random.uniform(0.0, 0.05),                  # mouse_idle_ratio
        random.uniform(95, 105),                    # typing_interval_avg (machine precise)
        random.uniform(0, 20),                      # typing_interval_variance (zero variance)
        0,                                          # typing_burst_count
        0.0,                                        # backspace_frequency
        0,                                          # scroll_event_count
        0.0,                                        # scroll_speed_avg
        random.uniform(0.3, 0.6),                   # click_frequency
        random.randint(500, 3000),                  # session_duration_ms
        random.uniform(0.0, 0.2),                   # browser_entropy
        0.0,
        0.0,
        0.0,
        random.uniform(0.0, 15.0),
        0.0,
        0.0,
    ]


def _bot_replay():
    """Replay bot: recorded human-like but too perfect/consistent."""
    return [
        random.uniform(200, 500),                   # mouse_speed_avg (looks normal)
        random.uniform(100, 1000),                  # mouse_speed_variance (too low)
        random.randint(1, 3),                       # mouse_direction_changes (few)
        random.uniform(10, 200),                    # mouse_curve_variance (low)
        random.uniform(0.0, 0.05),                  # mouse_idle_ratio (no natural pauses)
        random.uniform(100, 150),                   # typing_interval_avg
        random.uniform(0, 100),                     # typing_interval_variance (too steady)
        0,                                          # typing_burst_count
        0.0,                                        # backspace_frequency (never corrects)
        random.randint(2, 8),                       # scroll_event_count
        random.uniform(50, 200),                    # scroll_speed_avg
        random.uniform(0.5, 1.5),                   # click_frequency
        random.randint(1000, 5000),                 # session_duration_ms
        random.uniform(0.1, 0.3),                   # browser_entropy (suspicious)
        0.0,
        8.0,
        4.0,
        random.uniform(5.0, 20.0),
        random.choice([0.0, 1.0]),
        random.uniform(0.0, 300.0),
    ]


def _bot_headless_browser():
    """Headless browser bot: Puppeteer/Playwright with scripted actions."""
    return [
        random.uniform(400, 2000),                  # mouse_speed_avg (scripted movement)
        random.uniform(50, 2000),                   # mouse_speed_variance (low variance)
        random.randint(2, 8),                       # mouse_direction_changes (some, scripted)
        random.uniform(10, 500),                    # mouse_curve_variance (bezier curves)
        random.uniform(0.0, 0.1),                   # mouse_idle_ratio (minimal waits)
        random.uniform(20, 80),                     # typing_interval_avg (fast but not instant)
        random.uniform(5, 200),                     # typing_interval_variance (low)
        random.randint(0, 2),                       # typing_burst_count
        0.0,                                        # backspace_frequency
        random.randint(0, 5),                       # scroll_event_count
        random.uniform(100, 1000),                  # scroll_speed_avg
        random.uniform(0.2, 1.5),                   # click_frequency
        random.randint(800, 8000),                  # session_duration_ms
        random.uniform(0.0, 0.20),                  # browser_entropy (headless fingerprint)
        0.0,
        0.0,
        0.0,
        random.uniform(5.0, 40.0),
        random.choice([0.0, 1.0]),
        random.uniform(0.0, 500.0),
    ]


def _bot_credential_stuffer():
    """Credential stuffing bot: rapid form filling, zero scrolling."""
    return [
        random.uniform(1000, 8000),                 # mouse_speed_avg (fast tab/click)
        random.uniform(0, 300),                     # mouse_speed_variance (constant)
        random.randint(0, 2),                       # mouse_direction_changes
        random.uniform(0, 100),                     # mouse_curve_variance
        0.0,                                        # mouse_idle_ratio
        random.uniform(1, 15),                      # typing_interval_avg (paste-like speed)
        random.uniform(0, 30),                      # typing_interval_variance
        random.randint(5, 20),                      # typing_burst_count (all bursts)
        0.0,                                        # backspace_frequency
        0,                                          # scroll_event_count
        0.0,                                        # scroll_speed_avg
        random.uniform(2.0, 15.0),                  # click_frequency (rapid clicks)
        random.randint(200, 2000),                  # session_duration_ms (very fast)
        random.uniform(0.0, 0.15),                  # browser_entropy
        0.0,
        2.0,
        1.0,
        random.uniform(0.0, 10.0),
        0.0,
        0.0,
    ]


def _bot_slow_scraper():
    """Slow scraper bot: tries to mimic human speed but lacks randomness."""
    return [
        random.uniform(150, 500),                   # mouse_speed_avg (human-like speed)
        random.uniform(50, 800),                    # mouse_speed_variance (but too consistent)
        random.randint(1, 5),                       # mouse_direction_changes (few)
        random.uniform(5, 300),                     # mouse_curve_variance (minimal curves)
        random.uniform(0.0, 0.08),                  # mouse_idle_ratio (no natural pauses)
        random.uniform(90, 130),                    # typing_interval_avg (steady)
        random.uniform(0, 80),                      # typing_interval_variance (too uniform)
        0,                                          # typing_burst_count
        0.0,                                        # backspace_frequency
        random.randint(3, 15),                      # scroll_event_count (scripted)
        random.uniform(200, 600),                   # scroll_speed_avg (constant speed)
        random.uniform(0.3, 0.8),                   # click_frequency
        random.randint(3000, 12000),                # session_duration_ms (realistic)
        random.uniform(0.05, 0.25),                 # browser_entropy (low)
        0.0,
        4.0,
        2.0,
        random.uniform(10.0, 100.0),
        1.0,
        random.uniform(100.0, 600.0),
    ]


def _bot_click_farmer():
    """Click farm bot: moderate speed but zero typing, excessive clicks."""
    return [
        random.uniform(200, 1000),                  # mouse_speed_avg
        random.uniform(100, 3000),                  # mouse_speed_variance
        random.randint(2, 10),                      # mouse_direction_changes
        random.uniform(20, 1000),                   # mouse_curve_variance
        random.uniform(0.0, 0.1),                   # mouse_idle_ratio
        0.0,                                        # typing_interval_avg (no typing)
        0.0,                                        # typing_interval_variance
        0,                                          # typing_burst_count
        0.0,                                        # backspace_frequency
        random.randint(0, 3),                       # scroll_event_count
        random.uniform(0, 100),                     # scroll_speed_avg
        random.uniform(2.0, 10.0),                  # click_frequency (excessive)
        random.randint(1000, 8000),                 # session_duration_ms
        random.uniform(0.0, 0.20),                  # browser_entropy
        random.choice([0.0, 1.0]),
        random.choice([2.0, 4.0]),
        random.choice([2.0, 4.0]),
        random.uniform(50.0, 300.0),
        1.0,
        random.uniform(200.0, 800.0),
    ]


def _bot_api_abuser():
    """API-level bot: no mouse/scroll at all, instant form submission."""
    return [
        0.0,                                        # mouse_speed_avg (no mouse)
        0.0,                                        # mouse_speed_variance
        0,                                          # mouse_direction_changes
        0.0,                                        # mouse_curve_variance
        0.0,                                        # mouse_idle_ratio
        random.uniform(0, 3),                       # typing_interval_avg (injected)
        random.uniform(0, 5),                       # typing_interval_variance
        random.randint(0, 1),                       # typing_burst_count
        0.0,                                        # backspace_frequency
        0,                                          # scroll_event_count
        0.0,                                        # scroll_speed_avg
        0.0,                                        # click_frequency
        random.randint(10, 300),                    # session_duration_ms (instant)
        random.uniform(0.0, 0.10),                  # browser_entropy
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def generate_bot_features():
    """Randomly pick one of several bot behavioral profiles."""
    profile = random.choice([
        _bot_instant,
        _bot_linear,
        _bot_replay,
        _bot_headless_browser,
        _bot_credential_stuffer,
        _bot_slow_scraper,
        _bot_click_farmer,
        _bot_api_abuser,
    ])
    return profile()


# ══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════

def main():
    random.seed(42)
    np.random.seed(42)

    NUM_SAMPLES = 100_000          # 50k human + 50k bot
    print(f"Generating synthetic behavioral dataset ({NUM_SAMPLES:,} samples)...")

    X, y = [], []
    for _ in range(NUM_SAMPLES // 2):
        X.append(generate_human_features())
        y.append(1)     # Human

        X.append(generate_bot_features())
        y.append(0)     # Bot

    df = pd.DataFrame(X, columns=FEATURE_ORDER)
    y = np.array(y)

    print(f"Dataset ready: {len(df):,} samples ({sum(y):,} human, {len(y)-sum(y):,} bot)")
    print(f"\nFeature ranges (human samples):")
    human_df = df[y == 1]
    for col in FEATURE_ORDER:
        print(f"  {col:30s} min={human_df[col].min():12.2f}  max={human_df[col].max():12.2f}")

    # ── Build pipeline ────────────────────────────────────────────────
    print(f"\nTraining Random Forest Classifier (n_estimators=300, {NUM_SAMPLES:,} samples)...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=3,
            min_samples_split=5,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        ))
    ])

    # ── Cross-validation ──────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, df, y, cv=cv, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {scores.mean() * 100:.2f}% (+/- {scores.std() * 100:.2f}%)")

    # ── Train on full dataset ─────────────────────────────────────────
    pipeline.fit(df, y)

    train_acc = pipeline.score(df, y)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")

    # ── Classification report ─────────────────────────────────────────
    y_pred = pipeline.predict(df)
    print(f"\nClassification Report (on training data):")
    print(classification_report(y, y_pred, target_names=["Bot", "Human"]))

    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")

    # ── Feature importances ───────────────────────────────────────────
    rf = pipeline.named_steps['clf']
    importances = sorted(zip(FEATURE_ORDER, rf.feature_importances_), key=lambda x: -x[1])
    print(f"\nFeature Importances:")
    for name, imp in importances:
        bar = "#" * int(imp * 80)
        print(f"  {name:30s} {imp:.4f}  {bar}")

    # ── Sample predictions ────────────────────────────────────────────
    print(f"\nSample predictions:")
    for i in range(5):
        h = pd.DataFrame([generate_human_features()], columns=FEATURE_ORDER)
        b = pd.DataFrame([generate_bot_features()], columns=FEATURE_ORDER)
        print(f"  Human sample {i+1}: {pipeline.predict_proba(h)[0][1]:.4f}")
        print(f"  Bot   sample {i+1}: {pipeline.predict_proba(b)[0][1]:.4f}")

    # ── Save model ────────────────────────────────────────────────────
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "silentguard_rf.pkl")
    joblib.dump(pipeline, model_path)

    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nModel saved to: {model_path} ({file_size:.1f} MB)")
    print(f"Total training samples: {NUM_SAMPLES:,}")
    print("Done!")


if __name__ == "__main__":
    main()
