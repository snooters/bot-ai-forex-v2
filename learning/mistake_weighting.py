import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from learning.trade_memory import TradeMemory
from utils.logger import get_logger


class MistakeWeighting:
    def __init__(self, trade_memory: TradeMemory):
        self.logger = get_logger("mistake_weighting")
        self.trade_memory = trade_memory
        self._feature_importance: Optional[Dict] = None

    def set_feature_importance(self, importance: Dict):
        self._feature_importance = importance

    def compute_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        feature_cols: List[str],
    ) -> np.ndarray:
        weights = sample_weight.copy()
        trades = self.trade_memory.get_all_trades()
        loss_trades = [t for t in trades if t.get("result") == "LOSS"]
        win_trades = [t for t in trades if t.get("result") == "WIN"]

        if len(loss_trades) < 3:
            return weights

        for i in range(len(X)):
            sample = X[i]
            similar_losses = self._find_similar(
                sample, loss_trades, feature_cols, top_k=5
            )
            similar_wins = self._find_similar(
                sample, win_trades, feature_cols, top_k=5
            )

            total_similar = len(similar_losses) + len(similar_wins)
            if total_similar < 3:
                continue

            loss_rate = len(similar_losses) / total_similar if total_similar > 0 else 0
            win_rate = len(similar_wins) / total_similar if total_similar > 0 else 0

            if loss_rate >= 0.7 and total_similar >= 3:
                weights[i] *= 1.8
            elif loss_rate >= 0.5 and total_similar >= 3:
                weights[i] *= 1.3
            elif win_rate >= 0.7 and total_similar >= 5:
                weights[i] *= 0.7
            elif win_rate >= 0.5 and total_similar >= 5:
                weights[i] *= 0.9

        self.logger.info(
            f"Mistake weights applied: {len(X)} samples, "
            f"weight_range=[{weights.min():.2f}, {weights.max():.2f}]"
        )
        return weights

    def adjust_labels(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str],
    ) -> np.ndarray:
        y_adjusted = y.copy()
        trades = self.trade_memory.get_all_trades()
        loss_trades = [t for t in trades if t.get("result") == "LOSS"]
        win_trades = [t for t in trades if t.get("result") == "WIN"]

        flips = 0
        for i in range(len(X)):
            sample = X[i]
            similar_losses = self._find_similar(
                sample, loss_trades, feature_cols, top_k=5
            )
            similar_wins = self._find_similar(
                sample, win_trades, feature_cols, top_k=5
            )

            total_similar = len(similar_losses) + len(similar_wins)
            if total_similar < 5:
                continue

            hist_loss_rate = len(similar_losses) / total_similar
            hist_win_rate = len(similar_wins) / total_similar

            if y[i] == 0 and hist_loss_rate > 0.7:
                y_adjusted[i] = 2
                flips += 1
            elif y[i] == 1 and hist_loss_rate > 0.7:
                y_adjusted[i] = 2
                flips += 1

        if flips > 0:
            self.logger.info(
                f"Label adjustment: {flips}/{len(X)} samples flipped to HOLD "
                f"(historically losing patterns)"
            )
        return y_adjusted

    def get_pattern_win_rate(
        self,
        direction: str,
        regime: str,
        timeframe: str,
    ) -> Optional[float]:
        trades = self.trade_memory.get_all_trades()
        matching = []
        for t in trades:
            if t.get("direction") == direction:
                mc = t.get("market_conditions", {})
                t_regime = mc.get("regime", "") if mc else ""
                t_tf = t.get("timeframe", "")
                if regime and regime not in t_regime:
                    continue
                if timeframe and timeframe != t_tf:
                    continue
                matching.append(t)

        closed = [t for t in matching if t.get("result") in ("WIN", "LOSS")]
        if len(closed) < 3:
            return None
        wins = sum(1 for t in closed if t["result"] == "WIN")
        return wins / len(closed)

    def _find_similar(
        self,
        sample: np.ndarray,
        trades: List[Dict],
        feature_cols: List[str],
        top_k: int = 5,
    ) -> List[Dict]:
        if not trades:
            return []

        weighted_importances = self._get_feature_weights(feature_cols)
        scored = []
        for t in trades:
            ind = t.get("indicators", {})
            if not ind:
                continue
            trade_vec = self._trade_to_vector(ind, feature_cols)
            if trade_vec is None:
                continue
            dist = self._weighted_distance(sample, trade_vec, weighted_importances)
            scored.append((dist, t))

        scored.sort(key=lambda x: x[0])
        return [t for _, t in scored[:top_k]]

    def _trade_to_vector(
        self, indicators: Dict, feature_cols: List[str]
    ) -> Optional[np.ndarray]:
        vec = []
        for col in feature_cols:
            val = indicators.get(col)
            if val is None:
                return None
            vec.append(float(val))
        return np.array(vec)

    def _weighted_distance(
        self, a: np.ndarray, b: np.ndarray, weights: np.ndarray
    ) -> float:
        diff = a - b
        return float(np.sqrt(np.sum(weights * (diff ** 2))))

    def _get_feature_weights(self, feature_cols: List[str]) -> np.ndarray:
        if self._feature_importance:
            weights = np.ones(len(feature_cols))
            for i, col in enumerate(feature_cols):
                if col in self._feature_importance:
                    weights[i] = 1.0 + self._feature_importance[col] * 0.5
            return weights
        return np.ones(len(feature_cols))
