from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import config
from utils.logger import get_logger
from utils.helpers import compute_max_drawdown


class MonteCarloSimulator:
    def __init__(self, num_simulations: int = 1000, confidence_level: float = 0.95):
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.logger = get_logger("monte_carlo")

    def simulate(self, trades: List[Dict], initial_balance: float = 1000.0) -> Dict:
        if not trades:
            return {"error": "No trades to simulate"}

        profits = [t.get("profit", 0) for t in trades if t.get("profit") is not None]
        if not profits:
            return {"error": "No closed trades"}

        profits = np.array(profits)
        n_trades = len(profits)

        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(self.num_simulations):
            sampled = np.random.choice(profits, size=n_trades, replace=True)
            equity = initial_balance + np.cumsum(sampled)
            final_equities.append(equity[-1])
            dd, _ = compute_max_drawdown(equity.tolist())
            max_drawdowns.append(dd)
            sharpe = np.mean(sampled) / np.std(sampled) * np.sqrt(252) if np.std(sampled) > 0 else 0
            sharpe_ratios.append(sharpe)

        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        sharpe_ratios = np.array(sharpe_ratios)

        sorted_equities = np.sort(final_equities)
        lower_idx = int(self.num_simulations * (1 - self.confidence_level) / 2)
        upper_idx = int(self.num_simulations * (1 + self.confidence_level) / 2)

        results = {
            "num_simulations": self.num_simulations,
            "confidence_level": self.confidence_level,
            "mean_final_balance": float(np.mean(final_equities)),
            "median_final_balance": float(np.median(final_equities)),
            "std_final_balance": float(np.std(final_equities)),
            "ci_lower": float(sorted_equities[lower_idx]),
            "ci_upper": float(sorted_equities[upper_idx]),
            "mean_max_drawdown": float(np.mean(max_drawdowns) * 100),
            "median_max_drawdown": float(np.median(max_drawdowns) * 100),
            "worst_max_drawdown": float(np.min(max_drawdowns) * 100),
            "mean_sharpe": float(np.mean(sharpe_ratios)),
            "probability_of_profit": float(np.mean(final_equities > initial_balance) * 100),
            "probability_of_ruin": float(np.mean(final_equities < initial_balance * 0.5) * 100),
            "expected_return": float((np.mean(final_equities) - initial_balance) / initial_balance * 100),
        }

        self.logger.info(f"Monte Carlo: {results['probability_of_profit']:.1f}% profitable, "
                         f"mean return {results['expected_return']:.1f}%")
        return results
