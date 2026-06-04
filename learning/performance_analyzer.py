from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from core.constants import PERFORMANCE_LOG_DIR
from utils.logger import get_logger
from utils.helpers import (
    compute_sharpe_ratio, compute_sortino_ratio,
    compute_max_drawdown, compute_profit_factor,
    compute_expectancy, compute_recovery_factor,
)


class PerformanceAnalyzer:
    def __init__(self):
        self.logger = get_logger("performance_analyzer")
        self._history: List[Dict] = []

    def analyze_trades(self, trades: List[Dict], start_balance: float = 0) -> Dict:
        closed = [t for t in trades if t.get("profit") is not None]
        if not closed:
            return self._empty_result()

        profits = [t["profit"] for t in closed if t["profit"] is not None]
        if not profits:
            return self._empty_result()

        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        total_trades = len(closed)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = compute_profit_factor(gross_profit, gross_loss)

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        net_profit = gross_profit - gross_loss
        expectancy = compute_expectancy(closed)

        equity_curve = self._build_equity_curve(trades, start_balance)
        max_dd, max_dd_idx = compute_max_drawdown(equity_curve)

        sharpe = compute_sharpe_ratio(profits)
        sortino = compute_sortino_ratio(profits)
        recovery = compute_recovery_factor(net_profit, max_dd)

        result = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": float(win_rate * 100),
            "profit_factor": float(profit_factor),
            "net_profit": float(net_profit),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "avg_win_loss_ratio": float(avg_win_loss_ratio),
            "expectancy": float(expectancy),
            "max_drawdown": float(max_dd * 100),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "recovery_factor": float(recovery),
            "best_trade": float(max(profits)) if profits else 0,
            "worst_trade": float(min(profits)) if profits else 0,
            "by_timeframe": {},
        }

        by_tf: Dict[str, List[Dict]] = {}
        for t in closed:
            tf = t.get("timeframe", "M15")
            by_tf.setdefault(tf, []).append(t)

        for tf_name, tf_trades in by_tf.items():
            tf_profits = [t["profit"] for t in tf_trades if t["profit"] is not None]
            if not tf_profits:
                continue
            tf_wins = [p for p in tf_profits if p > 0]
            tf_losses = [p for p in tf_profits if p < 0]
            tf_win_rate = len(tf_wins) / len(tf_trades) if tf_trades else 0
            tf_gp = sum(tf_wins) if tf_wins else 0
            tf_gl = abs(sum(tf_losses)) if tf_losses else 0
            result["by_timeframe"][tf_name] = {
                "total_trades": len(tf_trades),
                "winning_trades": len(tf_wins),
                "losing_trades": len(tf_losses),
                "win_rate": float(tf_win_rate * 100),
                "profit_factor": float(compute_profit_factor(tf_gp, tf_gl)),
                "net_profit": float(tf_gp - tf_gl),
            }

        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": result,
        })

        return result

    def _build_equity_curve(self, trades: List[Dict], start_balance: float = 0) -> List[float]:
        closed = [t for t in trades if t.get("profit") is not None]
        closed.sort(key=lambda t: t.get("exit_time", ""))
        if not closed:
            return []
        balance = start_balance if start_balance > 0 else 10000
        equity = []
        for t in closed:
            balance += t.get("profit", 0)
            equity.append(balance)
        return equity

    def get_analysis_summary(self, trades: List[Dict]) -> str:
        analysis = self.analyze_trades(trades)
        lines = [
            f"Trades: {analysis['total_trades']}",
            f"Win Rate: {analysis['win_rate']:.1f}%",
            f"Profit Factor: {analysis['profit_factor']:.2f}",
            f"Net Profit: ${analysis['net_profit']:.2f}",
            f"Max DD: {analysis['max_drawdown']:.1f}%",
            f"Sharpe: {analysis['sharpe_ratio']:.2f}",
            f"Expectancy: ${analysis['expectancy']:.2f}",
        ]
        return " | ".join(lines)

    def _empty_result(self) -> Dict:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0, "profit_factor": 0, "net_profit": 0,
            "gross_profit": 0, "gross_loss": 0, "avg_win": 0, "avg_loss": 0,
            "avg_win_loss_ratio": 0, "expectancy": 0, "max_drawdown": 0,
            "sharpe_ratio": 0, "sortino_ratio": 0, "recovery_factor": 0,
            "best_trade": 0, "worst_trade": 0,
        }
