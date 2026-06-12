from __future__ import annotations

import math
from typing import Dict, List, Optional


class SkillManager:
    LEVELS = [
        (0, "Newborn"),
        (21, "Learning"),
        (41, "Intermediate"),
        (61, "Advanced"),
        (81, "Expert"),
    ]

    def compute_skill(self, trades: List[Dict], period_days: int = 90) -> Dict:
        if not trades:
            return {"score": 0, "level": "Newborn", "components": {}}

        recent_trades = trades
        total = len(recent_trades)
        if total == 0:
            return {"score": 0, "level": "Newborn", "components": {}}

        winning = [t for t in recent_trades if t.get("net_pnl", 0) > 0]
        win_rate = len(winning) / total

        gross_p = sum(t["net_pnl"] for t in winning) if winning else 0
        gross_l = abs(sum(t["net_pnl"] for t in recent_trades if t.get("net_pnl", 0) <= 0))
        pf = gross_p / gross_l if gross_l > 0 else (gross_p if gross_p > 0 else 0)

        stability = self._compute_stability(recent_trades)
        experience = min(math.log2(total + 1) * 10, 20)

        wr_score = win_rate * 100 * 0.4
        pf_score = min(pf / 2.0, 1.0) * 100 * 0.3
        stability_score = stability * 100 * 0.2
        exp_score = (experience / 20) * 100 * 0.1

        score = wr_score + pf_score + stability_score + exp_score
        score = max(0, min(100, round(score)))

        level = self._get_level(score)

        return {
            "score": score,
            "level": level,
            "components": {
                "win_rate": round(win_rate, 4),
                "win_rate_raw": round(wr_score, 2),
                "profit_factor": round(pf, 4),
                "pf_raw": round(pf_score, 2),
                "stability": round(stability, 4),
                "stability_raw": round(stability_score, 2),
                "experience": round(experience, 2),
                "exp_raw": round(exp_score, 2),
            },
        }

    def compute_pair_skills(self, trades: List[Dict]) -> Dict[str, int]:
        pairs: Dict[str, List[Dict]] = {}
        for t in trades:
            pair = t.get("pair", "UNKNOWN")
            if pair not in pairs:
                pairs[pair] = []
            pairs[pair].append(t)

        result = {}
        for pair, ptrades in pairs.items():
            skill = self.compute_skill(ptrades)
            result[pair] = skill["score"]
        return result

    def _compute_stability(self, trades: List[Dict], window: int = 20) -> float:
        if len(trades) < window:
            return 0.5
        win_rates = []
        for i in range(0, len(trades) - window + 1, window):
            chunk = trades[i:i + window]
            wins = sum(1 for t in chunk if t.get("net_pnl", 0) > 0)
            win_rates.append(wins / window)
        if len(win_rates) < 2:
            return 0.5
        mean = sum(win_rates) / len(win_rates)
        variance = sum((w - mean) ** 2 for w in win_rates) / (len(win_rates) - 1)
        std = math.sqrt(variance)
        stability = max(0, 1.0 - std / 0.5)
        return stability

    def _get_level(self, score: int) -> str:
        for threshold, level in reversed(self.LEVELS):
            if score >= threshold:
                return level
        return "Newborn"

    def classify_readiness(self, stats: Dict) -> str:
        wr = stats.get("win_rate", 0)
        pf = stats.get("profit_factor", 0)
        dd = stats.get("max_drawdown_pct", 1.0)
        total = stats.get("total_trades", 0)

        if total < 30:
            return "INSUFFICIENT_DATA"
        if wr < 0.35 or pf < 1.0:
            return "BLOCKED"
        if wr > 0.45 and pf > 1.2 and dd < 0.15:
            return "READY"
        return "CAUTION"
