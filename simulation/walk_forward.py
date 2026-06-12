from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from simulation.performance_tracker import PerformanceTracker
from simulation.simulator import Simulator

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    label: str
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime


@dataclass
class WalkForwardResult:
    symbol: str
    windows: List[Dict[str, Any]] = field(default_factory=list)
    avg_train_wr: float = 0.0
    avg_val_wr: float = 0.0
    avg_train_pf: float = 0.0
    avg_val_pf: float = 0.0
    oos_decay_wr: float = 0.0
    oos_decay_pf: float = 0.0
    grade: str = "N/A"
    total_trades_train: int = 0
    total_trades_val: int = 0


def _generate_windows(years: int = 3, window_size_months: int = 6) -> List[WalkForwardWindow]:
    now = datetime.now()
    end = now
    start = now.replace(year=now.year - years)
    windows = []
    months_step = window_size_months // 2
    current_start = start
    idx = 0
    while True:
        train_end = current_start.replace(
            month=current_start.month + window_size_months % 12,
            year=current_start.year + (current_start.month + window_size_months - 1) // 12,
        )
        if train_end >= end:
            break
        val_end = train_end.replace(
            month=train_end.month + window_size_months % 12,
            year=train_end.year + (train_end.month + window_size_months - 1) // 12,
        )
        if val_end > end:
            val_end = end
        windows.append(WalkForwardWindow(
            label=f"W{idx + 1}",
            train_start=current_start,
            train_end=train_end,
            val_start=train_end,
            val_end=val_end,
        ))
        idx += 1
        current_start = current_start.replace(
            month=current_start.month + months_step,
            year=current_start.year + (current_start.month + months_step - 1) // 12,
        )
        if val_end >= end:
            break
    return windows


def _compute_oos_decay(train_wr: float, val_wr: float, train_pf: float, val_pf: float) -> float:
    wr_decay = train_wr - val_wr if train_wr > 0 else 0
    pf_decay = (train_pf - val_pf) / train_pf if train_pf > 0 else 0
    return (wr_decay + pf_decay) / 2


def _grade_oos(oos_decay: float) -> str:
    if oos_decay < 0.05:
        return "A"
    elif oos_decay < 0.10:
        return "B"
    elif oos_decay < 0.15:
        return "C"
    elif oos_decay < 0.25:
        return "D"
    elif oos_decay < 0.35:
        return "E"
    else:
        return "F"


async def run_walk_forward(
    symbol: str,
    years: int = 3,
    window_size_months: int = 6,
    initial_balance: float = 10000.0,
    volume_fixed: Optional[float] = None,
    pair_suffix: str = "",
) -> WalkForwardResult:
    windows = _generate_windows(years, window_size_months)
    logger.info("Walk-forward: %d windows for %s", len(windows), symbol)

    result = WalkForwardResult(symbol=symbol)
    tracker = PerformanceTracker()

    train_wrs: List[float] = []
    val_wrs: List[float] = []
    train_pfs: List[float] = []
    val_pfs: List[float] = []

    for w in windows:
        logger.info("Window %s: train %s-%s, val %s-%s",
                     w.label, w.train_start.date(), w.train_end.date(),
                     w.val_start.date(), w.val_end.date())

        train_sim = Simulator(
            symbol=symbol,
            from_date=w.train_start,
            to_date=w.train_end,
            initial_balance=initial_balance,
            volume_fixed=volume_fixed,
            pair_suffix=pair_suffix,
        )
        train_res = await train_sim.run()
        train_stats = train_res.get("stats", {})

        val_sim = Simulator(
            symbol=symbol,
            from_date=w.val_start,
            to_date=w.val_end,
            initial_balance=initial_balance,
            volume_fixed=volume_fixed,
            pair_suffix=pair_suffix,
        )
        val_res = await val_sim.run()
        val_stats = val_res.get("stats", {})

        train_wr = train_stats.get("win_rate", 0)
        val_wr = val_stats.get("win_rate", 0)
        train_pf = train_stats.get("profit_factor", 0)
        val_pf = val_stats.get("profit_factor", 0)

        train_wrs.append(train_wr)
        val_wrs.append(val_wr)
        train_pfs.append(train_pf)
        val_pfs.append(val_pf)

        window_result = {
            "label": w.label,
            "train_start": w.train_start.isoformat(),
            "train_end": w.train_end.isoformat(),
            "val_start": w.val_start.isoformat(),
            "val_end": w.val_end.isoformat(),
            "train_trades": train_stats.get("total_trades", 0),
            "val_trades": val_stats.get("total_trades", 0),
            "train_win_rate": round(train_wr, 4),
            "val_win_rate": round(val_wr, 4),
            "train_profit_factor": round(train_pf, 4),
            "val_profit_factor": round(val_pf, 4),
            "train_return_pct": train_stats.get("return_pct", 0),
            "val_return_pct": val_stats.get("return_pct", 0),
            "train_sharpe": train_stats.get("sharpe_ratio", 0),
            "val_sharpe": val_stats.get("sharpe_ratio", 0),
            "train_max_dd_pct": train_stats.get("max_drawdown_pct", 0),
            "val_max_dd_pct": val_stats.get("max_drawdown_pct", 0),
        }
        result.windows.append(window_result)
        result.total_trades_train += train_stats.get("total_trades", 0)
        result.total_trades_val += val_stats.get("total_trades", 0)

    result.avg_train_wr = round(sum(train_wrs) / len(train_wrs), 4) if train_wrs else 0
    result.avg_val_wr = round(sum(val_wrs) / len(val_wrs), 4) if val_wrs else 0
    result.avg_train_pf = round(sum(train_pfs) / len(train_pfs), 4) if train_pfs else 0
    result.avg_val_pf = round(sum(val_pfs) / len(val_pfs), 4) if val_pfs else 0

    oos_decay = _compute_oos_decay(result.avg_train_wr, result.avg_val_wr,
                                     result.avg_train_pf, result.avg_val_pf)
    result.oos_decay_wr = round(result.avg_train_wr - result.avg_val_wr, 4)
    result.oos_decay_pf = round(result.avg_train_pf - result.avg_val_pf, 4)
    result.grade = _grade_oos(oos_decay)

    logger.info("Walk-forward result for %s: grade=%s, avg_train_wr=%.2f%%, avg_val_wr=%.2f%%",
                symbol, result.grade, result.avg_train_wr * 100, result.avg_val_wr * 100)

    return result
