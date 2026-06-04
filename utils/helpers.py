import math
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def round_to_pip(value: float, digits: int = 5) -> float:
    return round(value, digits)


def pip_value(pair: str) -> float:
    if "JPY" in pair.upper():
        return 0.01
    return 0.0001


def compute_lot_size(
    balance: float,
    risk_pct: float,
    stop_loss_pips: float,
    pip_val: float,
    leverage: int = 100
) -> float:
    risk_amount = balance * risk_pct
    if stop_loss_pips <= 0 or pip_val <= 0:
        return 0.0
    lot = risk_amount / (stop_loss_pips * pip_val * 10)
    lot = max(min(lot, 100.0), 0.01)
    return round(lot, 2)


def compute_atr_based_sl(atr: float, atr_multiplier: float = 1.5) -> float:
    return atr * atr_multiplier


def compute_atr_based_tp(atr: float, rr_ratio: float = 2.0, atr_multiplier: float = 1.5) -> float:
    return atr * atr_multiplier * rr_ratio


def serialize_datetime(dt: datetime) -> str:
    return dt.isoformat() if dt else ""


def deserialize_datetime(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def compute_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    excess = arr.mean() - risk_free_rate / 252
    if arr.std() == 0:
        return 0.0
    return excess / arr.std() * np.sqrt(252)


def compute_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    excess = arr.mean() - risk_free_rate / 252
    downside = arr[arr < 0].std()
    if downside == 0:
        return 0.0
    return excess / downside * np.sqrt(252)


def compute_max_drawdown(equity_curve: List[float]) -> Tuple[float, int]:
    if len(equity_curve) < 2:
        return 0.0, 0
    arr = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    peak = np.where(peak == 0, 1, peak)
    dd = (arr - peak) / peak
    max_dd = dd.min()
    max_dd_idx = dd.argmin()
    return float(max_dd), int(max_dd_idx)


def compute_profit_factor(gross_profit: float, gross_loss: float) -> float:
    if abs(gross_loss) < 1e-10:
        return float("inf") if gross_profit > 0 else 0.0
    return abs(gross_profit / gross_loss)


def compute_expectancy(trades: List[Dict]) -> float:
    if not trades:
        return 0.0
    profits = [t.get("profit", 0) for t in trades]
    return np.mean(profits) if profits else 0.0


def compute_recovery_factor(net_profit: float, max_dd: float) -> float:
    if abs(max_dd) < 1e-10:
        return float("inf") if net_profit > 0 else 0.0
    return net_profit / abs(max_dd)


def dict_hash(d: Dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def timeframe_to_minutes(tf: str) -> int:
    mapping = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440, "W1": 10080
    }
    return mapping.get(tf.upper(), 60)


def timeframe_to_seconds(tf: str) -> int:
    return timeframe_to_minutes(tf) * 60


def get_timeframe_label(tf_minutes: int) -> str:
    mapping = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4"}
    return mapping.get(tf_minutes, f"{tf_minutes}min")


def safe_float_division(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-10:
        return default
    return a / b
