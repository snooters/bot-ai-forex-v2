from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from core.config import config
from utils.logger import get_logger


class OutOfSampleTester:
    def __init__(self, oos_ratio: float = 0.2):
        self.oos_ratio = oos_ratio
        self.logger = get_logger("oos_tester")

    def split_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        split_idx = int(len(df) * (1 - self.oos_ratio))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()

        self.logger.info(f"OOS split: train={len(train)} rows, test={len(test)} rows")
        return train, test

    def evaluate_oos(
        self,
        train_metrics: Dict,
        test_metrics: Dict,
    ) -> Dict:
        result = {
            "train_win_rate": train_metrics.get("win_rate", 0),
            "test_win_rate": test_metrics.get("win_rate", 0),
            "train_profit_factor": train_metrics.get("profit_factor", 0),
            "test_profit_factor": test_metrics.get("profit_factor", 0),
            "train_sharpe": train_metrics.get("sharpe_ratio", 0),
            "test_sharpe": test_metrics.get("sharpe_ratio", 0),
            "train_max_dd": train_metrics.get("max_drawdown", 0),
            "test_max_dd": test_metrics.get("max_drawdown", 0),
        }

        result["overfitting_ratio"] = (
            train_metrics.get("win_rate", 0) - test_metrics.get("win_rate", 0)
        ) if train_metrics.get("win_rate", 0) > 0 else 0

        result["overfit_detected"] = result["overfitting_ratio"] > 15

        if result["overfit_detected"]:
            self.logger.warning(f"Overfitting detected: train/test win rate gap "
                                f"{result['overfitting_ratio']:.1f}%")

        return result
