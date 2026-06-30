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


# ── Simulation-based trade outcome ──────────────────────────────────────────
# Used by OOSValidator and WalkForwardValidator to produce realistic P&L
# instead of simple return-based calculations.

SIM_TRADE_MAX_BARS = 72          # Max lookahead candles (12h at M5)
SIM_TRADE_VOLUME = 0.01          # Standard mini lot
SIM_TRADE_SL_ATR = 2.0           # SL = 2 × ATR (match live engine)
SIM_TRADE_TP_ATR = 2.0           # TP1 = 2 × ATR
SIM_TRADE_COMMISSION_RATE = 0.00006


def simulate_trade_outcome(
    df: pd.DataFrame,
    row_idx: int,
    predicted_dir: str,
    atr_col: str = "atr",
) -> Optional[Dict]:
    """Simulate a single trade with SL/TP scanning forward.
    
    Uses SL=1×ATR, TP1=2×ATR to calculate realistic P&L, matching the
    Simulator engine's logic. Returns a dict with keys:
        profit, entry_price, exit_price, exit_reason, win
    
    Returns None if trade cannot be simulated (invalid ATR, no future data).
    """
    if row_idx < 0 or row_idx >= len(df):
        return None

    entry_row = df.iloc[row_idx]
    entry_price = float(entry_row["close"])
    atr = float(entry_row.get(atr_col, 0))

    # Guard: ATR must be positive and reasonable
    if atr <= 0 or atr > entry_price * 0.1:
        return None

    # Calculate SL and TP (same as simulator.py:_execute_trade)
    if predicted_dir == "BUY":
        sl_price = entry_price - atr * SIM_TRADE_SL_ATR
        tp_price = entry_price + atr * SIM_TRADE_TP_ATR
    else:  # SELL
        sl_price = entry_price + atr * SIM_TRADE_SL_ATR
        tp_price = entry_price - atr * SIM_TRADE_TP_ATR

    # Scan forward up to SIM_TRADE_MAX_BARS to find which is hit first
    end_idx = min(len(df), row_idx + SIM_TRADE_MAX_BARS)
    exit_price = None
    exit_reason = None

    for j in range(row_idx + 1, end_idx):
        future_row = df.iloc[j]
        high = float(future_row["high"])
        low = float(future_row["low"])

        if predicted_dir == "BUY":
            if high >= tp_price:
                exit_price = tp_price
                exit_reason = "TP_HIT"
                break
            if low <= sl_price:
                exit_price = sl_price
                exit_reason = "SL_HIT"
                break
        else:  # SELL
            if low <= tp_price:
                exit_price = tp_price
                exit_reason = "TP_HIT"
                break
            if high >= sl_price:
                exit_price = sl_price
                exit_reason = "SL_HIT"
                break

    # If neither SL nor TP was hit within the window, close at last price
    if exit_price is None:
        exit_price = float(df.iloc[end_idx - 1]["close"])
        exit_reason = "TIME_EXIT"

    # Calculate P&L (matches VirtualPosition.calculate_pnl)
    if predicted_dir == "BUY":
        pnl = (exit_price - entry_price) * SIM_TRADE_VOLUME * 100000
    else:
        pnl = (entry_price - exit_price) * SIM_TRADE_VOLUME * 100000

    # Subtract commission (matches VirtualAccount._calc_commission)
    commission = SIM_TRADE_VOLUME * 100000 * entry_price * SIM_TRADE_COMMISSION_RATE
    net_pnl = pnl - commission

    # Determine if this was a winning trade
    if predicted_dir == "BUY":
        is_win = exit_price > entry_price
    else:
        is_win = exit_price < entry_price

    return {
        "profit": net_pnl,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "win": is_win,
    }
