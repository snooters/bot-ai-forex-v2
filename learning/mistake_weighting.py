import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

from core.constants import DECISION_LOG_DIR
from learning.trade_memory import TradeMemory
from utils.logger import get_logger


class MistakeWeighting:
    def __init__(self, trade_memory: TradeMemory):
        self.logger = get_logger("mistake_weighting")
        self.trade_memory = trade_memory
        self._feature_importance: Optional[Dict] = None

    def set_feature_importance(self, importance: Dict):
        self._feature_importance = importance

    def _load_incorrect_holds(self, lookback_hours: int = 168) -> List[Dict]:
        eval_dir = Path(DECISION_LOG_DIR) / "evaluated"
        if not eval_dir.exists():
            return []
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        results = []
        for f in eval_dir.glob("eval_*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                ts = data.get("timestamp", "")
                if ts and datetime.fromisoformat(ts) >= cutoff:
                    if data.get("outcome") == "INCORRECT_HOLD":
                        results.append(data)
            except Exception:
                continue
        return results

    def compute_no_trade_weights(
        self,
        X: np.ndarray,
        sample_weight: np.ndarray,
        feature_cols: List[str],
    ) -> np.ndarray:
        weights = sample_weight.copy()
        incorrect_holds = self._load_incorrect_holds(lookback_hours=168)
        if len(incorrect_holds) < 3:
            return weights

        hold_contexts = []
        for h in incorrect_holds:
            ctx = h.get("context", {})
            hold_contexts.append(ctx)

        flips = 0
        for i in range(len(X)):
            sample = X[i]
            similar_holds = self._find_similar_hold_contexts(
                sample, incorrect_holds, hold_contexts, feature_cols, top_k=5
            )
            if len(similar_holds) >= 3:
                weights[i] *= 1.5
                flips += 1

        if flips > 0:
            self.logger.info(
                f"No-trade weighting: {flips}/{len(X)} samples up-weighted "
                f"(similar to past INCORRECT_HOLD patterns)"
            )
        return weights

    def _find_similar_hold_contexts(
        self,
        sample: np.ndarray,
        holds: List[Dict],
        hold_contexts: List[Dict],
        feature_cols: List[str],
        top_k: int = 5,
    ) -> List[Dict]:
        if not holds:
            return []
        weighted_importances = self._get_feature_weights(feature_cols)
        scored = []
        for i, h in enumerate(holds):
            ctx = hold_contexts[i] if i < len(hold_contexts) else {}
            fsum = ctx.get("feature_summary", {})
            if not fsum:
                continue
            vec = []
            valid = True
            for col in feature_cols:
                val = fsum.get(col)
                if val is None:
                    valid = False
                    break
                try:
                    vec.append(float(val))
                except (ValueError, TypeError):
                    valid = False
                    break
            if not valid:
                continue
            trade_vec = np.array(vec)
            dist = self._weighted_distance(sample, trade_vec, weighted_importances)
            scored.append((dist, h))
        scored.sort(key=lambda x: x[0])
        return [t for _, t in scored[:top_k]]

    def get_incorrect_hold_rate(self) -> float:
        eval_dir = Path(DECISION_LOG_DIR) / "evaluated"
        if not eval_dir.exists():
            return 0.0
        total = 0
        incorrect = 0
        for f in eval_dir.glob("eval_*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                outcome = data.get("outcome")
                if outcome:
                    total += 1
                    if outcome == "INCORRECT_HOLD":
                        incorrect += 1
            except Exception:
                continue
        return incorrect / total if total > 0 else 0.0

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
