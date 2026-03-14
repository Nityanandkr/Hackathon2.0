"""
SilentGuard AI — Backend API
ML-based passive human verification system
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import time
import os

from app.routes import behavior, verify, dashboard, auth
from app.models.schemas import ErrorResponse

app = FastAPI(
    title="SilentGuard AI",
    description="ML-based passive human verification — no CAPTCHA needed",
    version="1.0.0"
)

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production: set your frontend domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────
app.include_router(behavior.router,   prefix="/api", tags=["Behavior"])
app.include_router(verify.router,     prefix="/api", tags=["Verify"])
app.include_router(dashboard.router,  prefix="/api", tags=["Dashboard"])
app.include_router(auth.router,       prefix="/api/auth", tags=["Auth"])

# ── Static HTML pages ────────────────────────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

@app.get("/demo", tags=["Pages"], response_class=HTMLResponse)
def serve_demo():
    """Serve the live demo page."""
    path = os.path.join(PROJECT_ROOT, "silentguard_behavior_collector.html")
    return FileResponse(path, media_type="text/html")

@app.get("/dashboard", tags=["Pages"], response_class=HTMLResponse)
def serve_dashboard():
    """Serve the admin dashboard page."""
    path = os.path.join(PROJECT_ROOT, "dashboard.html")
    return FileResponse(path, media_type="text/html")


# ── Request timing middleware ─────────────────────────────────────────
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(round(time.time() - start, 4))
    return response


# ── Root health check ─────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "SilentGuard AI",
        "status": "running",
        "version": "1.0.0",
        "pages": {
            "demo":      "GET /demo",
            "dashboard": "GET /dashboard",
        },
        "endpoints": {
            "collect_behavior": "POST /api/collect-behavior",
            "verify_human":     "POST /api/verify-human",
            "dashboard_stats":  "GET  /api/dashboard/stats",
            "session_history":  "GET  /api/dashboard/sessions"
        }
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "timestamp": time.time()}