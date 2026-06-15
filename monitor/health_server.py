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
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="AI Forex Bot", version="2.0.0")


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
    return _latest_full_state


@app.get("/api/candles/{symbol}")
def get_candles(symbol: str):
    return _candles_cache.get(symbol.upper(), [])


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
                        await websocket.send_json(current)
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
