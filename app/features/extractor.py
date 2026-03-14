"""
SilentGuard AI — Feature Extraction Engine
Converts raw browser behavioral signals into a numeric feature vector
that the ML model can understand.
"""

import math
import hashlib
from typing import List, Optional

from app.models.schemas import (
    BehaviorPayload, FeatureVector,
    MouseEvent, KeystrokeEvent, ScrollEvent, BrowserFingerprint
)


# ── Mouse features ─────────────────────────────────────────────────────
def _downsample_mouse(events: List[MouseEvent], interval_ms: int = 50) -> List[MouseEvent]:
    """
    Browser sends mouse events at ~60fps (every ~16ms).
    Downsample to one event per interval_ms for meaningful feature extraction.
    Without this, direction changes and curve variance are artificially low
    because consecutive frames show nearly identical positions.
    """
    if len(events) < 2:
        return events
    sampled = [events[0]]
    for e in events[1:]:
        if e.time - sampled[-1].time >= interval_ms:
            sampled.append(e)
    # Always include the last event
    if sampled[-1].time != events[-1].time:
        sampled.append(events[-1])
    return sampled


def extract_mouse_features(events: List[MouseEvent]) -> dict:
    if len(events) < 2:
        return {
            "mouse_speed_avg": 0.0,
            "mouse_speed_variance": 0.0,
            "mouse_direction_changes": 0,
            "mouse_curve_variance": 0.0,
            "mouse_idle_ratio": 0.0,
        }

    # Downsample to ~50ms intervals for meaningful feature extraction
    events = _downsample_mouse(events, interval_ms=50)

    if len(events) < 2:
        return {
            "mouse_speed_avg": 0.0,
            "mouse_speed_variance": 0.0,
            "mouse_direction_changes": 0,
            "mouse_curve_variance": 0.0,
            "mouse_idle_ratio": 0.0,
        }

    speeds = []
    directions = []
    idle_count = 0

    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]

        dx = curr.x - prev.x
        dy = curr.y - prev.y
        dt = max(curr.time - prev.time, 1)          # avoid /0

        dist   = math.sqrt(dx * dx + dy * dy)
        speed  = dist / dt * 1000                   # px/second
        speeds.append(speed)

        direction = math.atan2(dy, dx)
        directions.append(direction)

        if speed < 5:
            idle_count += 1

    # Average speed
    speed_avg = sum(speeds) / len(speeds) if speeds else 0.0

    # Speed variance (humans are inconsistent)
    speed_var = (
        sum((s - speed_avg) ** 2 for s in speeds) / len(speeds)
        if len(speeds) > 1 else 0.0
    )

    # Direction changes
    dir_changes = 0
    for i in range(1, len(directions)):
        delta = abs(directions[i] - directions[i - 1])
        # Wrap to [0, π]
        delta = min(delta, 2 * math.pi - delta)
        if delta > 0.3:          # ~17 degrees — lowered for downsampled data
            dir_changes += 1

    # Curve variance — deviation from straight line
    curves = []
    for i in range(2, len(events)):
        a = events[i - 2]
        b = events[i - 1]
        c = events[i]
        # Cross product magnitude = area of triangle
        cross = abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))
        curves.append(cross)
    curve_var = sum(curves) / len(curves) if curves else 0.0

    idle_ratio = idle_count / len(events)

    return {
        "mouse_speed_avg":         round(speed_avg, 3),
        "mouse_speed_variance":    round(speed_var, 3),
        "mouse_direction_changes": dir_changes,
        "mouse_curve_variance":    round(min(curve_var, 999999), 3),
        "mouse_idle_ratio":        round(idle_ratio, 3),
    }


# ── Keystroke features ──────────────────────────────────────────────────
def extract_typing_features(events: List[KeystrokeEvent]) -> dict:
    if len(events) < 2:
        return {
            "typing_interval_avg":      0.0,
            "typing_interval_variance": 0.0,
            "typing_burst_count":       0,
            "backspace_frequency":      0.0,
        }

    intervals = [e.interval for e in events if e.interval is not None and e.interval > 0]

    if not intervals:
        return {
            "typing_interval_avg":      0.0,
            "typing_interval_variance": 0.0,
            "typing_burst_count":       0,
            "backspace_frequency":      0.0,
        }

    avg_interval = sum(intervals) / len(intervals)
    variance     = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)

    # Bursts = rapid sequences (interval < 80ms)
    burst_count = sum(1 for iv in intervals if iv < 80)

    backspace_count = sum(1 for e in events if e.key in ("Backspace", "Delete"))
    backspace_freq  = backspace_count / len(events) if events else 0.0

    return {
        "typing_interval_avg":      round(avg_interval, 2),
        "typing_interval_variance": round(variance, 2),
        "typing_burst_count":       burst_count,
        "backspace_frequency":      round(backspace_freq, 3),
    }


# ── Scroll features ─────────────────────────────────────────────────────
def extract_scroll_features(events: List[ScrollEvent]) -> dict:
    if len(events) < 2:
        return {
            "scroll_event_count": len(events),
            "scroll_speed_avg":   0.0,
        }

    speeds = []
    for i in range(1, len(events)):
        dp = abs(events[i].position - events[i - 1].position)
        dt = max(events[i].time - events[i - 1].time, 1)
        speeds.append(dp / dt * 1000)

    return {
        "scroll_event_count": len(events),
        "scroll_speed_avg":   round(sum(speeds) / len(speeds), 3),
    }


# ── Browser entropy ─────────────────────────────────────────────────────
def compute_browser_entropy(fp: Optional[BrowserFingerprint]) -> float:
    """
    Higher entropy = more unique = more likely real browser.
    Headless browsers often expose minimal/default fingerprint values.
    """
    if fp is None:
        return 0.0

    score = 0.0

    if fp.screen_width and fp.screen_height:
        # Common headless defaults: 800x600, 1024x768
        common = [(800, 600), (1024, 768), (1280, 720)]
        if (fp.screen_width, fp.screen_height) not in common:
            score += 0.2

    if fp.timezone:
        score += 0.15

    if fp.webgl_vendor:
        # Real browsers expose GPU vendor; headless often blank or "Google Inc."
        if fp.webgl_vendor not in ("", "Google Inc.", "Brian Paul"):
            score += 0.25

    if fp.color_depth and fp.color_depth >= 24:
        score += 0.1

    if fp.touch_support is not None:
        score += 0.1

    if fp.language:
        score += 0.1

    if fp.platform and fp.platform not in ("", "Linux x86_64"):
        score += 0.1

    return round(min(score, 1.0), 3)


# ── Session timing ─────────────────────────────────────────────────────
def compute_session_duration(payload: BehaviorPayload) -> int:
    all_times = []
    for m in payload.mouse_movements:
        all_times.append(m.time)
    for k in payload.keystrokes:
        all_times.append(k.time)
    for s in payload.scroll_events:
        all_times.append(s.time)

    if len(all_times) < 2:
        return 0
    return max(all_times) - min(all_times)


# ── Main extractor ──────────────────────────────────────────────────────
def extract_features(payload: BehaviorPayload) -> FeatureVector:
    mouse_f   = extract_mouse_features(payload.mouse_movements)
    typing_f  = extract_typing_features(payload.keystrokes)
    scroll_f  = extract_scroll_features(payload.scroll_events)
    entropy   = compute_browser_entropy(payload.browser_fingerprint)
    duration  = compute_session_duration(payload)

    # Click frequency from request_timing if provided
    click_freq = 0.0
    if payload.request_timing and "click_count" in payload.request_timing:
        click_count = payload.request_timing["click_count"]
        if duration > 0:
            click_freq = click_count / (duration / 1000)  # clicks per second

    # Extract new Extended Fingerprint features
    device_is_mobile = 0.0
    device_memory_gb = 0.0
    hardware_concurrency = 0.0
    
    if payload.browser_fingerprint:
        fp = payload.browser_fingerprint
        device_is_mobile = 1.0 if fp.is_mobile else 0.0
        device_memory_gb = float(fp.device_memory) if fp.device_memory is not None else 0.0
        hardware_concurrency = float(fp.hardware_concurrency) if fp.hardware_concurrency is not None else 0.0

    # Extract JS Challenge features
    js_time = 0.0
    js_success = 0.0
    js_score = 0.0
    if payload.js_challenge:
        js = payload.js_challenge
        js_time = float(js.execution_time_ms)
        js_success = 1.0 if js.success else 0.0
        js_score = float(js.score)

    return FeatureVector(
        mouse_speed_avg          = mouse_f["mouse_speed_avg"],
        mouse_speed_variance     = mouse_f["mouse_speed_variance"],
        mouse_direction_changes  = mouse_f["mouse_direction_changes"],
        mouse_curve_variance     = mouse_f["mouse_curve_variance"],
        mouse_idle_ratio         = mouse_f["mouse_idle_ratio"],
        typing_interval_avg      = typing_f["typing_interval_avg"],
        typing_interval_variance = typing_f["typing_interval_variance"],
        typing_burst_count       = typing_f["typing_burst_count"],
        backspace_frequency      = typing_f["backspace_frequency"],
        scroll_event_count       = scroll_f["scroll_event_count"],
        scroll_speed_avg         = scroll_f["scroll_speed_avg"],
        click_frequency          = round(click_freq, 4),
        session_duration_ms      = duration,
        browser_entropy          = entropy,
        device_is_mobile         = device_is_mobile,
        device_memory_gb         = device_memory_gb,
        hardware_concurrency     = hardware_concurrency,
        js_challenge_time_ms     = js_time,
        js_challenge_success     = js_success,
        js_challenge_score       = js_score,
    )