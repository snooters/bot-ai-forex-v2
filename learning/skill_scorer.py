from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import numpy as np

from core.constants import TRADE_HISTORY_DIR
from utils.helpers import compute_profit_factor
from utils.logger import get_logger


class SkillScorer:
    def __init__(self):
        self.logger = get_logger("skill_scorer")
        self._score_dir = Path(TRADE_HISTORY_DIR)
        self._score_dir.mkdir(parents=True, exist_ok=True)
        self._score_file = self._score_dir / "skill_scores.json"

    def compute_global(
        self,
        retrain_count: int,
        oos_results: Optional[Dict] = None,
        val_accuracy: float = 0,
        version_history: Optional[List[Dict]] = None,
    ) -> Tuple[str, int]:
        score = 0

        score += self._score_retrain_count(retrain_count)

        if oos_results and oos_results.get("success"):
            score += self._score_oos(oos_results)

        score += self._score_val_accuracy(val_accuracy)

        if version_history:
            score += self._score_consistency(version_history)

        score = max(0, min(score, 100))
        skill = self._map_score_to_skill(score, retrain_count)

        return skill, score

    def compute_per_pair(self, trades_by_pair: Dict[str, List[Dict]]) -> Dict[str, int]:
        pair_scores = {}
        for pair, trades in trades_by_pair.items():
            score = self._score_pair(trades)
            pair_scores[pair] = score
        self._save(pair_scores)
        return pair_scores

    def get_pair_skills(self) -> Dict[str, int]:
        if self._score_file.exists():
            try:
                with open(self._score_file) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "scores" in data:
                    return data["scores"]
                return data
            except Exception:
                pass
        return {}

    def get_best_pair(self, pair_skills: Optional[Dict[str, int]] = None) -> Optional[str]:
        skills = pair_skills or self.get_pair_skills()
        if not skills:
            return None
        return max(skills, key=skills.get)

    def get_worst_pair(self, pair_skills: Optional[Dict[str, int]] = None) -> Optional[str]:
        skills = pair_skills or self.get_pair_skills()
        if not skills:
            return None
        return min(skills, key=skills.get)

    def _score_pair(self, trades: List[Dict]) -> int:
        closed = [t for t in trades if t.get("result") in ("WIN", "LOSS")]
        if len(closed) < 3:
            return 0

        wins = sum(1 for t in closed if t["result"] == "WIN")
        losses = len(closed) - wins
        win_rate = wins / len(closed)

        gross_profit = sum(t.get("profit", 0) for t in closed if t.get("profit", 0) > 0)
        gross_loss = abs(sum(t.get("profit", 0) for t in closed if t.get("profit", 0) < 0))
        pf = compute_profit_factor(gross_profit, gross_loss)

        profits = [t.get("profit", 0) for t in closed]
        avg_profit = np.mean(profits) if profits else 0
        std_profit = np.std(profits) if len(profits) > 1 else 1
        sharpe = avg_profit / std_profit if std_profit > 0 else 0

        equity = []
        running = 0.0
        for t in closed:
            running += t.get("profit", 0)
            equity.append(running)
        max_dd = 0
        if equity:
            peak = equity[0]
            for val in equity:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

        score = 0
        score += min(win_rate * 100 * 0.30, 30)
        score += min(pf * 8, 25) if pf < float("inf") else 25
        score += max(0, min(sharpe * 10, 20))
        score += max(0, 15 - max_dd * 100 * 0.15)
        score += min(len(closed) * 0.05, 10)

        return max(0, min(int(score), 100))

    def _score_retrain_count(self, count: int) -> int:
        if count >= 20:
            return 20
        elif count >= 10:
            return 16
        elif count >= 5:
            return 12
        elif count >= 3:
            return 8
        elif count >= 1:
            return 4
        return 0

    def _score_oos(self, oos: Dict) -> int:
        score = 0
        wr = oos.get("win_rate", 0)
        pf = oos.get("profit_factor", 0)
        sharpe = oos.get("sharpe_ratio", 0)
        trades = oos.get("total_trades", 0)

        if wr >= 65:
            score += 20
        elif wr >= 58:
            score += 14
        elif wr >= 50:
            score += 7
        elif wr >= 40:
            score += 2

        if pf >= 2.0:
            score += 18
        elif pf >= 1.5:
            score += 12
        elif pf >= 1.2:
            score += 7
        elif pf >= 0.8:
            score += 3

        if sharpe >= 1.5:
            score += 12
        elif sharpe >= 1.0:
            score += 8
        elif sharpe >= 0.5:
            score += 4
        elif sharpe > 0:
            score += 1

        if trades >= 200:
            score += 8
        elif trades >= 100:
            score += 5
        elif trades >= 50:
            score += 3
        elif trades >= 20:
            score += 1

        return min(score, 58)

    def _score_val_accuracy(self, val_acc: float) -> int:
        val_pct = val_acc * 100 if val_acc <= 1.0 else val_acc
        if val_pct >= 90:
            return 15
        elif val_pct >= 80:
            return 10
        elif val_pct >= 70:
            return 6
        elif val_pct >= 60:
            return 3
        return 0

    def _score_consistency(self, version_history: List[Dict]) -> int:
        if len(version_history) < 2:
            return 0
        recent = version_history[-3:] if len(version_history) >= 3 else version_history
        improvements = 0
        for i in range(1, len(recent)):
            prev_oos = recent[i - 1].get("oos_score", 0)
            curr_oos = recent[i].get("oos_score", 0)
            if curr_oos > prev_oos:
                improvements += 1
        if improvements == len(recent) - 1 and improvements >= 2:
            return 15
        elif improvements >= 2:
            return 10
        elif improvements >= 1:
            return 5
        return 0

    def _map_score_to_skill(self, score: int, retrain_count: int) -> str:
        if score >= 85:
            return "Expert"
        elif score >= 65:
            return "Advanced"
        elif score >= 45:
            return "Competent"
        elif score >= 25:
            return "Developing"
        elif score >= 10:
            return "Learning"
        elif retrain_count > 0:
            return "Learning"
        return "Newborn"

    def _save(self, pair_scores: Dict[str, int]):
        try:
            data = {
                "scores": pair_scores,
                "best_pair": self.get_best_pair(pair_scores),
                "worst_pair": self.get_worst_pair(pair_scores),
                "updated_at": __import__("datetime").datetime.now().isoformat(),
            }
            with open(self._score_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save skill scores: {e}")
