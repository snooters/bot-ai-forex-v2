from datetime import datetime
from typing import Dict, List


def estimate_real_readiness(trades: List[Dict], current_dd: float = 0) -> Dict:
    if not trades:
        return {"score": 0, "eta_days": 0, "eta_str": "Insufficient data", "bar": "-" * 10, "detail": {}}

    closed = [t for t in trades if t.get("profit") is not None and t.get("entry_time")]
    if len(closed) < 10:
        return {"score": 0, "eta_days": 0, "eta_str": "Not enough trades", "bar": "-" * 10, "detail": {}}

    entry_times = []
    for t in closed:
        et = t.get("entry_time")
        if isinstance(et, str):
            entry_times.append(datetime.fromisoformat(et))
        elif isinstance(et, datetime):
            entry_times.append(et)

    if not entry_times:
        return {"score": 0, "eta_days": 0, "eta_str": "Insufficient data", "bar": "-" * 10, "detail": {}}

    first_trade = min(entry_times)
    days_trading = (datetime.now() - first_trade).days

    profits = [t["profit"] for t in closed if t["profit"] is not None]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]

    total = len(closed)
    win_rate = len(wins) / total if total > 0 else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0

    # Targets
    TARGET_DAYS = 90
    TARGET_PF = 1.5
    TARGET_WR = 0.40
    MAX_DD = 15.0

    # Scores per metric (0-1)
    time_score = min(days_trading / TARGET_DAYS, 1.0)
    pf_score = min(profit_factor / TARGET_PF, 1.0)
    wr_score = min(win_rate / TARGET_WR, 1.0)
    dd_score = 1.0 if current_dd < MAX_DD else max(0.0, 1.0 - (current_dd - MAX_DD) / MAX_DD)

    # Weights
    composite = time_score * 0.30 + pf_score * 0.30 + wr_score * 0.20 + dd_score * 0.20
    score = round(composite * 100)

    # ETA estimation
    missing_days = max(0, TARGET_DAYS - days_trading) if pf_score < 1.0 or wr_score < 1.0 else 0
    pf_gap = max(0, TARGET_PF - profit_factor)
    wr_gap = max(0, TARGET_WR - win_rate)
    perf_gap_days = int(max(pf_gap / 0.05 * 30, wr_gap / 0.02 * 30)) if pf_gap > 0 or wr_gap > 0 else 0

    eta_days = max(missing_days, perf_gap_days)

    if score >= 100:
        eta_str = "Ready"
    elif eta_days < 30:
        eta_str = f"~{eta_days} days"
    elif eta_days < 60:
        eta_str = f"~{eta_days // 7} weeks"
    elif eta_days < 365:
        eta_str = f"~{eta_days // 30} months"
    else:
        eta_str = ">1 year"

    bar_filled = score // 10
    bar = "#" * bar_filled + "-" * (10 - bar_filled)

    detail = {
        "days_trading": days_trading,
        "pf": round(profit_factor, 2),
        "wr": round(win_rate * 100, 1),
        "dd": round(current_dd, 1),
        "time_score": round(time_score, 2),
        "pf_score": round(pf_score, 2),
        "wr_score": round(wr_score, 2),
        "dd_score": round(dd_score, 2),
        "composite": composite,
    }

    return {"score": score, "eta_days": eta_days, "eta_str": eta_str, "bar": bar, "detail": detail}
