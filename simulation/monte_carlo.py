from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulation.performance_tracker import PerformanceTracker


def monte_carlo_simulation(
    trades: List[Dict],
    iterations: int = 1000,
    initial_balance: float = 10000.0,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    if not trades:
        return {
            "iterations": 0,
            "profit_probability": 0.0,
            "dd_probability": 0.0,
            "loss_streak_probability": 0.0,
            "pnl_ci_lower": 0.0,
            "pnl_ci_upper": 0.0,
            "avg_drawdown_pct": 0.0,
            "avg_profit": 0.0,
            "median_profit": 0.0,
        }

    pnls = [t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) is not None]
    if not pnls:
        return {
            "iterations": 0,
            "profit_probability": 0.0,
            "dd_probability": 0.0,
            "loss_streak_probability": 0.0,
            "pnl_ci_lower": 0.0,
            "pnl_ci_upper": 0.0,
            "avg_drawdown_pct": 0.0,
            "avg_profit": 0.0,
            "median_profit": 0.0,
        }

    n_trades = len(pnls)
    results: List[float] = []
    max_drawdowns: List[float] = []
    loss_streaks: List[int] = []
    profit_factors: List[float] = []
    win_rates: List[float] = []

    for _ in range(iterations):
        sample = random.choices(pnls, k=n_trades)

        cumulative = 0.0
        peak = initial_balance
        balance = initial_balance
        max_dd = 0.0
        max_loss_streak = 0
        current_loss_streak = 0
        wins = 0

        for pnl in sample:
            balance += pnl
            if balance > peak:
                peak = balance
            dd = peak - balance
            dd_pct = dd / peak if peak > 0 else 0
            if dd_pct > max_dd:
                max_dd = dd_pct

            if pnl > 0:
                wins += 1
                current_loss_streak = 0
            else:
                current_loss_streak += 1
                if current_loss_streak > max_loss_streak:
                    max_loss_streak = current_loss_streak

        total_pnl = balance - initial_balance
        results.append(total_pnl)
        max_drawdowns.append(max_dd)
        loss_streaks.append(max_loss_streak)
        win_rates.append(wins / n_trades)

        gross_profit = sum(p for p in sample if p > 0)
        gross_loss = abs(sum(p for p in sample if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        profit_factors.append(pf)

    results_sorted = sorted(results)
    lower_idx = int((1 - confidence_level) / 2 * iterations)
    upper_idx = int((1 + confidence_level) / 2 * iterations)
    lower_idx = max(0, min(lower_idx, iterations - 1))
    upper_idx = max(0, min(upper_idx, iterations - 1))

    profit_prob = sum(1 for r in results if r > 0) / iterations
    dd_prob = sum(1 for dd in max_drawdowns if dd > 0.20) / iterations
    loss_streak_prob = sum(1 for ls in loss_streaks if ls >= 5) / iterations

    return {
        "iterations": iterations,
        "n_trades_input": n_trades,
        "profit_probability": round(profit_prob, 4),
        "dd_gt_20pct_probability": round(dd_prob, 4),
        "loss_streak_ge_5_probability": round(loss_streak_prob, 4),
        "pnl_ci_lower": round(results_sorted[lower_idx], 2),
        "pnl_ci_upper": round(results_sorted[upper_idx], 2),
        "avg_profit": round(float(np.mean(results)), 2),
        "median_profit": round(float(np.median(results)), 2),
        "std_profit": round(float(np.std(results)), 2),
        "avg_drawdown_pct": round(float(np.mean(max_drawdowns)), 4),
        "median_drawdown_pct": round(float(np.median(max_drawdowns)), 4),
        "avg_profit_factor": round(float(np.mean(profit_factors)), 4),
        "avg_win_rate": round(float(np.mean(win_rates)), 4),
        "min_profit": round(min(results), 2),
        "max_profit": round(max(results), 2),
        "percentile_5": round(results_sorted[int(0.05 * iterations)], 2),
        "percentile_25": round(results_sorted[int(0.25 * iterations)], 2),
        "percentile_75": round(results_sorted[int(0.75 * iterations)], 2),
        "percentile_95": round(results_sorted[int(0.95 * iterations)], 2),
    }


def fast_monte_carlo(
    trades: List[Dict],
    iterations: int = 1000,
    initial_balance: float = 10000.0,
) -> Dict[str, Any]:
    pnls = np.array([t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) is not None])
    if len(pnls) == 0:
        return {"error": "No valid trades"}

    n = len(pnls)
    rng = np.random.default_rng()
    samples = rng.choice(pnls, size=(iterations, n))
    cumulative_pnls = np.cumsum(samples, axis=1)
    final_pnls = cumulative_pnls[:, -1]

    balances = initial_balance + cumulative_pnls
    peaks = np.maximum.accumulate(balances, axis=1)
    drawdowns = (peaks - balances) / np.where(peaks > 0, peaks, 1)
    max_dds = np.max(drawdowns, axis=1)

    return {
        "iterations": iterations,
        "n_trades_input": n,
        "profit_probability": round(float(np.mean(final_pnls > 0)), 4),
        "dd_gt_20pct_probability": round(float(np.mean(max_dds > 0.20)), 4),
        "avg_profit": round(float(np.mean(final_pnls)), 2),
        "median_profit": round(float(np.median(final_pnls)), 2),
        "std_profit": round(float(np.std(final_pnls)), 2),
        "avg_drawdown_pct": round(float(np.mean(max_dds)), 4),
        "percentile_5": round(float(np.percentile(final_pnls, 5)), 2),
        "percentile_95": round(float(np.percentile(final_pnls, 95)), 2),
    }
