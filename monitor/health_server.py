"""
FastAPI health check server for AI Forex Bot.

Provides REST API for external monitoring:
  GET  /health          — basic liveness probe
  GET  /status          — full bot state snapshot
  GET  /metrics         — Prometheus-style key metrics
"""

from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Forex Bot Health", version="2.0.0")


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


def start_server(host: str = "127.0.0.1", port: int = 9090):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
