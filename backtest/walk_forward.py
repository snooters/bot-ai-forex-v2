from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from core.config import config
from backtest.backtest_engine import BacktestEngine
from utils.logger import get_logger


class WalkForwardAnalyzer:
    def __init__(self, backtest_engine: BacktestEngine):
        self.logger = get_logger("walk_forward")
        self.backtest = backtest_engine

    def run_walk_forward(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
        window_days: int = 90,
        step_days: int = 30,
        initial_balance: float = 1000.0,
    ) -> Dict:
        self.logger.info(f"Running walk-forward: {symbol} {window_days}d windows, {step_days}d steps")

        results = []
        current_start = start_date

        while current_start + timedelta(days=window_days + step_days) <= end_date:
            train_start = current_start
            train_end = train_start + timedelta(days=window_days)
            test_start = train_end
            test_end = min(test_start + timedelta(days=step_days), end_date)

            self.logger.info(f"Window: train={train_start.date()}->{train_end.date()}, "
                             f"test={test_start.date()}->{test_end.date()}")

            try:
                window_result = self.backtest.run_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=test_start,
                    end_date=test_end,
                    initial_balance=initial_balance,
                )
                results.append({
                    "train_period": {"start": train_start.isoformat(), "end": train_end.isoformat()},
                    "test_period": {"start": test_start.isoformat(), "end": test_end.isoformat()},
                    "metrics": window_result,
                })
            except Exception as e:
                self.logger.warning(f"Window failed: {e}")

            current_start += timedelta(days=step_days)

        combined = self._combine_results(results)
        combined["windows"] = results

        self.logger.info(f"Walk-forward complete: {len(results)} windows")
        return combined

    def _combine_results(self, results: List[Dict]) -> Dict:
        if not results:
            return {"total_trades": 0, "win_rate": 0, "profit_factor": 0}

        all_trades = []
        for r in results:
            all_trades.extend(r.get("metrics", {}).get("trades", []))

        from learning.performance_analyzer import PerformanceAnalyzer
        perf = PerformanceAnalyzer()
        combined_metrics = perf.analyze_trades(all_trades)
        combined_metrics["total_windows"] = len(results)
        combined_metrics["successful_windows"] = sum(
            1 for r in results if r.get("metrics", {}).get("net_profit", 0) > 0
        )
        combined_metrics["avg_window_return"] = np.mean([
            r.get("metrics", {}).get("total_return", 0) for r in results
        ]) if results else 0

        return combined_metrics
