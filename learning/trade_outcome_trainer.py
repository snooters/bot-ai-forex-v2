from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import config
from learning.trade_memory import INDICATOR_FIELDS, TradeMemory
from utils.logger import get_logger

logger = get_logger("trade_outcome_trainer")


class TradeOutcomeTrainer:
    """Convert closed trade outcomes into labeled training samples.

    Closes the loop: trade outcome (WIN/LOSS) + entry-time feature snapshot
    -> (feature_vector, label) training sample.
    """

    LABEL_MAP = {
        ("BUY", "WIN"): 0,
        ("SELL", "WIN"): 1,
        ("BUY", "LOSS"): 2,
        ("SELL", "LOSS"): 2,
        ("BUY", "BREAK"): 2,
        ("SELL", "BREAK"): 2,
    }

    def __init__(self, trade_memory: Optional[TradeMemory] = None):
        self.trade_memory = trade_memory
        self._feature_cache: Dict[str, List[str]] = {}

    def convert_to_samples(
        self,
        trades: List[Dict],
        feature_columns: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert closed trade list to (X, y) training samples.

        Each trade's stored indicator snapshot is aligned to the full
        feature_columns list. Missing features default to 0.
        Labels: WIN_BUY=0, WIN_SELL=1, LOSS/HOLD/BREAK=2
        """
        if not trades:
            return np.empty((0, len(feature_columns))), np.empty(0, dtype=int)

        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        feature_set = set(feature_columns)
        indicator_set = set(INDICATOR_FIELDS)

        for trade in trades:
            indicators = self._extract_indicators(trade)
            if not indicators:
                continue

            label = self._trade_to_label(trade)
            if label is None:
                continue

            vec = np.zeros(len(feature_columns), dtype=np.float64)
            for i, col in enumerate(feature_columns):
                if col in indicators:
                    try:
                        vec[i] = float(indicators[col])
                    except (ValueError, TypeError):
                        vec[i] = 0.0

            X_list.append(vec)
            y_list.append(label)

        if not X_list:
            return np.empty((0, len(feature_columns))), np.empty(0, dtype=int)

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=int)
        return X, y

    def convert_from_simulation(
        self,
        sim_result: Dict,
        feature_columns: List[str],
        pair: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert simulation engine trades into training samples.

        sim_result is the dict returned by Simulator.run(), containing
        a 'trades' list where each trade has 'feature_vector' (JSON string
        of the full feature row at entry), 'side', and 'net_pnl'.
        """
        sim_trades = sim_result.get("trades", [])
        if not sim_trades:
            return np.empty((0, len(feature_columns))), np.empty(0, dtype=int)

        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for t in sim_trades:
            t_pair = t.get("pair", "")
            if pair and pair not in t_pair and t_pair not in pair:
                continue

            fv_raw = t.get("feature_vector")
            if not fv_raw:
                continue

            indicators = self._extract_feature_vector(fv_raw)
            if not indicators:
                continue

            label = self._sim_trade_to_label(t)
            if label is None:
                continue

            vec = np.zeros(len(feature_columns), dtype=np.float64)
            for i, col in enumerate(feature_columns):
                if col in indicators:
                    try:
                        vec[i] = float(indicators[col])
                    except (ValueError, TypeError):
                        vec[i] = 0.0

            X_list.append(vec)
            y_list.append(label)

        if not X_list:
            return np.empty((0, len(feature_columns))), np.empty(0, dtype=int)

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=int)
        logger.info(
            "Converted %d simulation trades to training samples. "
            "BUY=%d SELL=%d HOLD=%d",
            len(y), int((y == 0).sum()), int((y == 1).sum()), int((y == 2).sum()),
        )
        return X, y

    def merge_with_ohlc(
        self,
        X_ohlc: np.ndarray,
        y_ohlc: np.ndarray,
        X_trade: np.ndarray,
        y_trade: np.ndarray,
        upsample_wins: bool = True,
        win_weight: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge trade-outcome samples with OHLC-derived training data.

        Returns (X_merged, y_merged, sample_weights).
        Winning trade samples are optionally upsampled (duplicated).
        """
        if len(X_trade) == 0:
            return X_ohlc, y_ohlc, np.ones(len(y_ohlc))

        weights = np.ones(len(y_trade))
        if upsample_wins:
            for i in range(len(y_trade)):
                if y_trade[i] in (0, 1):
                    weights[i] = win_weight

        X_merged = np.vstack([X_ohlc, X_trade])
        y_merged = np.hstack([y_ohlc, y_trade])
        w_merged = np.hstack([
            np.ones(len(y_ohlc)),
            weights,
        ])

        perm = np.random.permutation(len(X_merged))
        X_merged = X_merged[perm]
        y_merged = y_merged[perm]
        w_merged = w_merged[perm]

        n_win = int((y_trade == 0).sum()) + int((y_trade == 1).sum())
        n_loss = int((y_trade == 2).sum())
        logger.info(
            "Trade samples merged: %d total (%d win, %d loss). "
            "OHLC: %d. Final: %d.",
            len(y_trade), n_win, n_loss, len(y_ohlc), len(y_merged),
        )
        return X_merged, y_merged, w_merged

    def get_recent_trades(
        self,
        pair: str,
        timeframe: Optional[int] = None,
        min_trades: int = 10,
        max_trades: int = 500,
    ) -> List[Dict]:
        """Fetch recent closed trades from TradeMemory suitable for training."""
        if self.trade_memory is None:
            logger.warning("No trade_memory provided, cannot fetch trades")
            return []

        try:
            trades = self.trade_memory.get_all_trades()
        except Exception as e:
            logger.warning("Failed to fetch trades: %s", e)
            return []

        filtered = []
        for t in trades:
            t_pair = t.get("pair", "")
            if pair not in t_pair and t_pair not in pair:
                continue
            if timeframe is not None:
                t_tf = t.get("timeframe", "")
                try:
                    if int(t_tf) != timeframe:
                        continue
                except (ValueError, TypeError):
                    pass
            result = t.get("result", "")
            if result in ("WIN", "LOSS", "BREAK"):
                filtered.append(t)

        filtered.sort(key=lambda x: x.get("exit_time", ""), reverse=True)
        filtered = filtered[:max_trades]

        valid = 0
        for t in filtered:
            indicators = self._extract_indicators(t)
            if indicators:
                valid += 1

        if valid < min_trades:
            logger.info(
                "Only %d/%d trades have indicator data (need %d), skipping",
                valid, len(filtered), min_trades,
            )
            return [] if valid < min_trades else filtered

        logger.info("Fetched %d recent trades for %s (TF=%s)", len(filtered), pair, timeframe)
        return filtered

    def get_trade_quality_stats(self, trades: List[Dict]) -> Dict:
        """Compute summary stats on trade outcomes for logging."""
        if not trades:
            return {"total": 0}
        total = len(trades)
        wins = sum(1 for t in trades if t.get("result") == "WIN")
        losses = sum(1 for t in trades if t.get("result") == "LOSS")
        breaks = sum(1 for t in trades if t.get("result") == "BREAK")
        avg_profit = sum(t.get("profit", 0) for t in trades) / total if total > 0 else 0
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "breaks": breaks,
            "win_rate": round(wins / total, 4) if total > 0 else 0,
            "avg_profit": round(avg_profit, 2),
        }

    def _extract_feature_vector(self, fv_raw) -> Dict:
        if isinstance(fv_raw, str):
            try:
                return json.loads(fv_raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        if isinstance(fv_raw, dict):
            return fv_raw
        return {}

    def _sim_trade_to_label(self, trade: Dict) -> Optional[int]:
        side = trade.get("side", "")
        net_pnl = trade.get("net_pnl", 0) or trade.get("pnl", 0)
        result = "WIN" if net_pnl > 0 else "LOSS"
        key = (side, result)
        return self.LABEL_MAP.get(key, None)

    def _extract_indicators(self, trade: Dict) -> Dict:
        indicators = trade.get("indicators")
        if indicators is None:
            return {}
        if isinstance(indicators, str):
            try:
                indicators = json.loads(indicators)
            except (json.JSONDecodeError, TypeError):
                return {}
        if isinstance(indicators, dict):
            return {k: v for k, v in indicators.items() if k in INDICATOR_FIELDS}
        return {}

    def _trade_to_label(self, trade: Dict) -> Optional[int]:
        direction = trade.get("direction", "")
        result = trade.get("result", "")
        key = (direction, result)
        if key in self.LABEL_MAP:
            return self.LABEL_MAP[key]
        return None
