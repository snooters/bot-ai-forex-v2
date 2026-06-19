"""
FastAPI server for AI Forex Bot.

Provides:
  GET  /health          — basic liveness probe
  GET  /status          — bot state snapshot (heartbeat)
  GET  /metrics         — Prometheus-style key metrics
  GET  /api/state       — full bot state for web dashboard
  GET  /api/candles/{symbol}  — last 100 M5 candles
  WS   /ws             — real-time state push
  /dashboard/*         — web dashboard static files
"""

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and datetimes."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        return super().default(obj)


app = FastAPI(title="AI Forex Bot", version="2.0.0")


# ── No-cache middleware (dashboard files only) ──
#     Ensures browser always fetches latest HTML/CSS from disk.

class _NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/dashboard"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(_NoCacheMiddleware)


# ── Heartbeat state (used by main.py for /health, /status, /metrics) ──

class BotState(BaseModel):
    is_running: bool = False
    is_connected: bool = False
    account_balance: float = 0
    account_equity: float = 0
    open_positions: int = 0
    emergency_level: str = "NORMAL"
    trading_paused: bool = False
    last_heartbeat: Optional[str] = None
    total_retrains: int = 0
    skill_level: str = "Newborn"
    uptime_seconds: float = 0
    mode: str = "simulation"


_state = BotState()
_start_time = datetime.now()


def update_state(**kwargs):
    for k, v in kwargs.items():
        if hasattr(_state, k):
            setattr(_state, k, v)
    _state.last_heartbeat = datetime.now().isoformat()


# ── Full state for web dashboard (richer dict, written by main.py) ──

_latest_full_state: dict = {}
_candles_cache: Dict[str, List[dict]] = {}

# Cross-thread notification: main.py (sync) → websocket (async)
_state_updated = threading.Event()


def update_full_state(state: dict):
    """Called by main.py (sync) to publish latest full state."""
    global _latest_full_state
    _latest_full_state = state
    _state_updated.set()  # wake up websocket poller


def update_candles(symbol: str, candles: List[dict]):
    """Called by main.py (sync) to cache OHLC data."""
    _candles_cache[symbol] = candles


# ── REST endpoints (heartbeat) ──

@app.get("/health")
def health():
    return {
        "status": "ok" if _state.is_running else "starting",
        "timestamp": datetime.now().isoformat(),
        "uptime": (datetime.now() - _start_time).total_seconds(),
    }


@app.get("/status")
def status():
    return _state


@app.get("/metrics")
def metrics():
    uptime = (datetime.now() - _start_time).total_seconds()
    return {
        "bot_uptime_seconds": uptime,
        "bot_connected": 1 if _state.is_connected else 0,
        "bot_running": 1 if _state.is_running else 0,
        "bot_open_positions": _state.open_positions,
        "bot_retrains": _state.total_retrains,
        "bot_balance": _state.account_balance,
        "bot_equity": _state.account_equity,
        "bot_trading_paused": 1 if _state.trading_paused else 0,
    }


# ── REST endpoints (web dashboard) ──

@app.get("/api/state")
def get_full_state():
    payload = json.dumps(_latest_full_state, cls=NumpyEncoder)
    return Response(content=payload, media_type="application/json")


@app.get("/api/candles/{symbol}")
def get_candles(symbol: str):
    return _candles_cache.get(symbol.upper(), [])


_TRADE_DB = Path(__file__).resolve().parent.parent / "learning" / "trade_history" / "trade_memory.db"


@app.get("/api/stats")
def get_stats():
    """Daily, weekly, monthly win rates from trade_memory.db."""
    if not _TRADE_DB.is_file():
        return {"daily": None, "weekly": None, "monthly": None}
    try:
        import sqlite3
        conn = sqlite3.connect(str(_TRADE_DB))
        cur = conn.execute("SELECT exit_time, result FROM trades WHERE result IN ('WIN','LOSS')")
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return {"daily": None, "weekly": None, "monthly": None}

    from datetime import datetime, timedelta
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)

    def _calc(marker):
        wins = 0
        total = 0
        for et, res in rows:
            try:
                t = datetime.fromisoformat(et)
            except Exception:
                continue
            if marker(t):
                total += 1
                if res == "WIN":
                    wins += 1
        return round(wins / total * 100, 1) if total > 0 else None

    return {
        "daily": _calc(lambda t: t >= today_start),
        "weekly": _calc(lambda t: t >= week_start),
        "monthly": _calc(lambda t: t >= month_start),
    }


@app.get("/api/monthly_pnl")
def get_monthly_pnl():
    """Total closed P&L this month (bot trades only)."""
    if not _TRADE_DB.is_file():
        return {"profit": 0.0}
    try:
        import sqlite3
        conn = sqlite3.connect(str(_TRADE_DB))
        cur = conn.execute(
            "SELECT exit_time, profit, exit_reason FROM trades WHERE result IN ('WIN','LOSS')"
        )
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return {"profit": 0.0}

    from datetime import datetime
    now = datetime.now()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total = 0.0
    for et, p, reason in rows:
        if reason == "MT5":
            continue
        try:
            t = datetime.fromisoformat(et)
        except Exception:
            continue
        if t >= month_start:
            total += p
    return {"profit": round(total, 2)}


# ── WebSocket real-time push ──

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_state = None
    try:
        while True:
            # Poll every 250ms for new state (cross-thread safe)
            if _state_updated.is_set() or last_state is None:
                _state_updated.clear()
                current = _latest_full_state
                if current != last_state:
                    last_state = current
                    try:
                        payload = json.dumps(current, cls=NumpyEncoder)
                        await websocket.send_text(payload)
                    except Exception:
                        break
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ── Static files (web dashboard) ──

_static_dir = Path(__file__).resolve().parent.parent / "web_dashboard"
if _static_dir.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(_static_dir), html=True), name="dashboard")


# ── Server starter ──

def start_server(host: str = "127.0.0.1", port: int = 9090):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
