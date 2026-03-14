# 🛡 SilentGuard AI

**ML-Based Passive Human Verification System — A Modern CAPTCHA Alternative**

> *"Security without interruption."*

SilentGuard AI replaces traditional CAPTCHA with an intelligent passive verification layer that analyzes behavioral signals to distinguish humans from bots — without puzzles, image challenges, or user friction.

---

## 🧠 How It Works

```
User interacts with page
        ↓
Frontend JS SDK silently collects:
  • Mouse movement patterns
  • Typing rhythm & keystroke dynamics
  • Scroll behavior
  • Click timing
  • Browser/device fingerprint
        ↓
Data sent to Backend API
        ↓
Feature Extraction Engine
  → 14 numeric behavioral features
        ↓
ML Classification (Random Forest)
  → Human Confidence Score (0.0 – 1.0)
        ↓
Decision Engine:
  • ≥ 0.80 → ✅ Access granted (no CAPTCHA)
  • ≥ 0.50 → ⚠️ Secondary check (OTP / Turnstile)
  • < 0.50 → 🚫 Blocked (bot detected)
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────┐
│                  Frontend (Browser)              │
│  ┌───────────────────────────────────────────┐  │
│  │  Behavior Collector SDK (JavaScript)      │  │
│  │  • Mouse tracker   • Keystroke dynamics   │  │
│  │  • Scroll analyzer  • Click timing        │  │
│  │  • Browser fingerprint                    │  │
│  └──────────────────┬────────────────────────┘  │
└─────────────────────┼───────────────────────────┘
                      │ POST /api/collect-behavior
                      ▼
┌─────────────────────────────────────────────────┐
│                  Backend (FastAPI)                │
│  ┌──────────────┐  ┌─────────────────────────┐  │
│  │ Session Store │  │  Feature Extractor       │  │
│  │ (In-memory)   │  │  14 behavioral features  │  │
│  └──────────────┘  └────────────┬────────────┘  │
│                                  ▼               │
│  ┌───────────────────────────────────────────┐  │
│  │  ML Classifier (Random Forest Pipeline)    │  │
│  │  → Human Confidence Score (0–1)            │  │
│  │  → Decision: allow / otp / block           │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  Dashboard API (stats + session history)   │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## 📊 Behavioral Features Extracted

| # | Feature | Description |
|---|---------|-------------|
| 1 | `mouse_speed_avg` | Average mouse speed (px/s) |
| 2 | `mouse_speed_variance` | Inconsistency in mouse speed |
| 3 | `mouse_direction_changes` | Number of direction changes |
| 4 | `mouse_curve_variance` | Deviation from straight-line paths |
| 5 | `mouse_idle_ratio` | Time spent idle vs moving |
| 6 | `typing_interval_avg` | Average time between keystrokes |
| 7 | `typing_interval_variance` | Variability in typing rhythm |
| 8 | `typing_burst_count` | Rapid typing sequences |
| 9 | `backspace_frequency` | Rate of corrections (human trait) |
| 10 | `scroll_event_count` | Number of scroll interactions |
| 11 | `scroll_speed_avg` | Average scroll velocity |
| 12 | `click_frequency` | Clicks per second |
| 13 | `session_duration_ms` | Total interaction time |
| 14 | `browser_entropy` | Device/browser uniqueness score |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### 1. Install Dependencies
```bash
pip install -r Requirements.txt
```

### 2. Train the ML Model
```bash
python scripts/train_rf_model.py
```

### 3. Start the Server
```bash
python run.py
```
Server runs on `http://localhost:8000`

### 4. Open the Demo
Open `silentguard_behavior_collector.html` in your browser.

### 5. View the Dashboard
Open `dashboard.html` in your browser to see admin analytics.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + endpoint list |
| `GET` | `/health` | Server status |
| `POST` | `/api/collect-behavior` | Submit behavioral signals |
| `POST` | `/api/verify-human` | Get ML classification result |
| `GET` | `/api/dashboard/stats` | Aggregated statistics |
| `GET` | `/api/dashboard/sessions` | Recent session history |

### Collect Behavior
```json
POST /api/collect-behavior
{
  "session_id": "sg_abc123",
  "mouse_movements": [{"x": 100, "y": 200, "time": 1710000000, "speed": 300}],
  "keystrokes": [{"key": "h", "time": 1710000001, "interval": 150}],
  "scroll_events": [{"position": 0, "time": 1710000002}],
  "browser_fingerprint": {
    "screen_width": 1920,
    "screen_height": 1080,
    "timezone": "Asia/Kolkata",
    "language": "en-US",
    "platform": "Win32"
  }
}
```

### Verify Human
```json
POST /api/verify-human
{
  "session_id": "sg_abc123",
  "include_features": true
}

// Response
{
  "human_score": 0.92,
  "status": "verified",
  "action": "allow",
  "reason": "Behavioral patterns match human. Access granted."
}
```

---

## 🔒 Privacy-First Design

- ✅ **No personal data** collected — only behavioral patterns
- ✅ **No keystroke content** stored — only timing intervals
- ✅ **Device fingerprints hashed** — never stored raw
- ✅ **Session-scoped analysis** — data auto-expires (30 min TTL)
- ✅ **No cookies or tracking** — stateless verification
- ✅ **GDPR-friendly** — no PII processing

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, Vanilla JS, CSS3 |
| Backend | Python, FastAPI, Uvicorn |
| ML Model | Scikit-learn (Random Forest Pipeline) |
| Data | Pandas, NumPy |
| Store | In-memory (Redis-ready) |
| Deployment | Docker |

---

## 📁 Project Structure

```
Hackathon2.0/
├── app/
│   ├── main.py              # FastAPI application entry
│   ├── store.py             # In-memory session store
│   ├── features/
│   │   └── extractor.py     # Behavioral feature extraction
│   ├── ml/
│   │   └── classifier.py    # ML classification engine
│   ├── models/
│   │   └── schemas.py       # Pydantic data schemas
│   └── routes/
│       ├── behavior.py      # /api/collect-behavior
│       ├── verify.py        # /api/verify-human
│       └── dashboard.py     # /api/dashboard/*
├── models/
│   └── silentguard_rf.pkl   # Trained ML model
├── scripts/
│   └── train_rf_model.py    # Model training script
├── silentguard_behavior_collector.html  # Live demo page
├── dashboard.html           # Admin dashboard
├── run.py                   # Server launcher
├── Requirements.txt         # Python dependencies
├── Dockerfile               # Container deployment
└── README.md                # This file
```

---

## 🐳 Docker Deployment

```bash
docker build -t silentguard-ai .
docker run -p 8000:8000 silentguard-ai
```

---

## 🎯 Expected Impact

| Area | Improvement |
|------|-------------|
| **User Experience** | Zero friction — no puzzles or challenges |
| **API Protection** | Real-time bot traffic blocking |
| **Fraud Prevention** | Behavioral anomaly detection |
| **Enterprise Trust** | Seamless security integration |
| **Scalability** | Stateless, lightweight, pluggable |

---

## 👥 Team

SilentGuard AI — Built for Hackathon 2.0

---

*SilentGuard AI — Security without interruption.* 🛡
