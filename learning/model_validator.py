import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from core.constants import (
    OOS_WR_THRESHOLD,
    OOS_PF_THRESHOLD,
    OOS_SHARPE_THRESHOLD,
    OOS_MIN_TRADES,
)
from ml.ensemble import VotingEnsemble
from utils.logger import get_logger


@dataclass
class ValidationResult:
    promoted: bool
    production_metrics: Dict
    candidate_metrics: Dict
    improvements: Dict[str, float]
    regressions: Dict[str, float]
    score: float
    reason: str


class ModelValidator:
    def __init__(self):
        self.logger = get_logger("model_validator")
        self._weights = {
            "profit_factor": 0.35,
            "sharpe_ratio": 0.25,
            "win_rate": 0.20,
            "max_drawdown": 0.10,
            "avg_return": 0.10,
        }

    def validate(
        self,
        production: VotingEnsemble,
        candidate: VotingEnsemble,
        timeframe: int,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict:
        self.logger.info(f"Validating candidate vs production for timeframe {timeframe}")

        if validation_data is not None:
            X_val, y_val = validation_data
            prod_preds = production.predict(X_val)
            cand_preds = candidate.predict(X_val)
            prod_metrics = self._compute_metrics(y_val, prod_preds)
            cand_metrics = self._compute_metrics(y_val, cand_preds)
        else:
            prod_metrics = {"win_rate": 0, "profit_factor": 0, "sharpe_ratio": 0, "max_drawdown": 0, "avg_return": 0}
            cand_metrics = {"win_rate": 0, "profit_factor": 0, "sharpe_ratio": 0, "max_drawdown": 0, "avg_return": 0}
            self.logger.warning("No validation data provided; using placeholder metrics")

        improvements = {}
        regressions = {}
        for key in self._weights:
            diff = cand_metrics.get(key, 0) - prod_metrics.get(key, 0)
            if diff > 0:
                improvements[key] = round(diff, 4)
            elif diff < 0:
                regressions[key] = round(abs(diff), 4)

        if not self._meets_minimum_thresholds(cand_metrics):
            return {
                "promote": False,
                "production_metrics": prod_metrics,
                "candidate_metrics": cand_metrics,
                "improvements": improvements,
                "regressions": regressions,
                "score": 0,
                "reject_reason": "Candidate does not meet minimum performance thresholds",
            }

        if improvements.get("profit_factor", 0) > 0 and regressions.get("sharpe_ratio", 0) > 0:
            pf_improvement = improvements["profit_factor"]
            sharpe_regression = regressions["sharpe_ratio"]
            if sharpe_regression > pf_improvement * 2:
                return {
                    "promote": False,
                    "production_metrics": prod_metrics,
                    "candidate_metrics": cand_metrics,
                    "improvements": improvements,
                    "regressions": regressions,
                    "score": 0,
                    "reject_reason": "Sharpe ratio regression outweighs profit factor improvement",
                }

        if regressions.get("max_drawdown", 0) > 0.05:
            if not improvements.get("profit_factor", 0) > regressions["max_drawdown"] * 2:
                return {
                    "promote": False,
                    "production_metrics": prod_metrics,
                    "candidate_metrics": cand_metrics,
                    "improvements": improvements,
                    "regressions": regressions,
                    "score": 0,
                    "reject_reason": "Drawdown increase not justified by profit factor improvement",
                }

        score = self._compute_composite_score(cand_metrics, prod_metrics)

        promoted = score > 0
        reason = ""
        if promoted:
            reason = f"Candidate outperforms production (score: {score:.2f})"
        else:
            reason = f"Production outperforms or matches candidate (score: {score:.2f})"

        return {
            "promote": promoted,
            "production_metrics": prod_metrics,
            "candidate_metrics": cand_metrics,
            "improvements": improvements,
            "regressions": regressions,
            "score": round(score, 4),
            "reject_reason": None if promoted else reason,
        }

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        tp = np.sum((y_pred > 0.5) & (y_true > 0.5))
        tn = np.sum((y_pred <= 0.5) & (y_true <= 0.5))
        fp = np.sum((y_pred > 0.5) & (y_true <= 0.5))
        fn = np.sum((y_pred <= 0.5) & (y_true > 0.5))
        total = len(y_true)

        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        returns = y_pred * y_true
        avg_return = np.mean(returns) if len(returns) > 0 else 0
        std_return = np.std(returns) if len(returns) > 0 else 1e-10

        win_mask = returns > 0
        loss_mask = returns < 0
        total_wins = np.sum(returns[win_mask]) if np.any(win_mask) else 0
        total_losses = abs(np.sum(returns[loss_mask])) if np.any(loss_mask) else 1e-10
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative) if len(cumulative) > 0 else np.array([0])
        drawdown = (cumulative - running_max) / (running_max + 1e-10) if len(cumulative) > 0 else np.array([0])
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

        return {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "win_rate": round(float(accuracy * 100), 2),
            "profit_factor": round(float(profit_factor), 4),
            "sharpe_ratio": round(float(sharpe_ratio), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "avg_return": round(float(avg_return), 6),
            "total_trades": total,
        }

    def _meets_minimum_thresholds(self, metrics: Dict) -> bool:
        if metrics.get("profit_factor", 0) < OOS_PF_THRESHOLD:
            return False
        if metrics.get("sharpe_ratio", 0) < OOS_SHARPE_THRESHOLD:
            return False
        return True

    def _compute_composite_score(self, candidate: Dict, production: Dict) -> float:
        score = 0.0
        for metric, weight in self._weights.items():
            cand_val = candidate.get(metric, 0)
            prod_val = production.get(metric, 0)
            if metric == "max_drawdown":
                diff = prod_val - cand_val
            else:
                diff = cand_val - prod_val
            score += diff * weight
        return score

    def get_promotion_criteria(self) -> Dict:
        return {
            "min_profit_factor": OOS_PF_THRESHOLD,
            "min_sharpe_ratio": OOS_SHARPE_THRESHOLD,
            "weights": self._weights,
        }
