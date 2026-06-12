from __future__ import annotations

from typing import Dict, List

from simulation.performance_tracker import PerformanceTracker
from simulation.skill_manager import SkillManager


class LearningEngine:
    def __init__(self):
        self.tracker = PerformanceTracker()
        self.skill_mgr = SkillManager()

    def analyze(self, trades: List[Dict], initial_balance: float) -> Dict:
        stats = self.tracker.compute(trades, initial_balance)
        skill = self.skill_mgr.compute_skill(trades)

        regime_analysis = self._analyze_regimes(stats)
        confidence_analysis = self._analyze_confidence(stats)
        volatility_analysis = self._analyze_key(stats, "volatility")
        hour_analysis = self._analyze_hours(trades)
        pair_analysis = self._analyze_key(stats, "pair")
        side_analysis = self._analyze_key(stats, "side")
        exit_analysis = self._analyze_key(stats, "exit_reason")

        recommended_confidence = self._find_optimal_confidence(stats)
        recommended_market_score = self._find_optimal_market_score(trades)
        blocked_regimes = self._find_blocked_regimes(regime_analysis)
        readiness = self.skill_mgr.classify_readiness(stats)

        return {
            "stats": stats,
            "skill": skill,
            "regime_analysis": regime_analysis,
            "confidence_analysis": confidence_analysis,
            "volatility_analysis": volatility_analysis,
            "hour_analysis": hour_analysis,
            "pair_analysis": pair_analysis,
            "side_analysis": side_analysis,
            "exit_analysis": exit_analysis,
            "recommended_confidence": recommended_confidence,
            "recommended_market_score": recommended_market_score,
            "blocked_regimes": blocked_regimes,
            "real_account_readiness": readiness,
        }

    def _analyze_regimes(self, stats: Dict) -> Dict:
        return stats.get("by_regime", {})

    def _analyze_confidence(self, stats: Dict) -> Dict:
        return stats.get("by_confidence", {})

    def _analyze_key(self, stats: Dict, key: str) -> Dict:
        return stats.get(f"by_{key}", {})

    def _analyze_hours(self, trades: List[Dict]) -> Dict:
        groups: Dict[str, List] = {}
        for t in trades:
            et = t.get("entry_time")
            if et is None:
                continue
            hour = et.hour if hasattr(et, "hour") else 0
            label = f"{hour:02d}:00"
            if label not in groups:
                groups[label] = []
            groups[label].append(t)

        result = {}
        for hour, g in groups.items():
            total = len(g)
            wins = sum(1 for t in g if t.get("net_pnl", 0) > 0)
            gross_p = sum(t["net_pnl"] for t in g if t.get("net_pnl", 0) > 0)
            gross_l = abs(sum(t["net_pnl"] for t in g if t.get("net_pnl", 0) <= 0))
            result[hour] = {
                "trades": total,
                "wins": wins,
                "win_rate": round(wins / total, 4) if total > 0 else 0,
                "profit_factor": round(gross_p / gross_l, 4) if gross_l > 0 else (gross_p if gross_p > 0 else 0),
                "net_pnl": round(gross_p - gross_l, 2),
            }
        return result

    def _find_optimal_confidence(self, stats: Dict) -> float:
        by_conf = stats.get("by_confidence", {})
        best_wr = 0.0
        best_conf = 0.0
        for label, data in by_conf.items():
            wr = data.get("win_rate", 0)
            trades = data.get("trades", 0)
            if wr > best_wr and trades >= 5:
                best_wr = wr
                parts = label.split("-")
                if parts:
                    try:
                        best_conf = int(parts[0]) / 100
                    except ValueError:
                        pass
        return best_conf

    def _find_optimal_market_score(self, trades: List[Dict]) -> int:
        buckets: Dict[str, List] = {}
        for t in trades:
            ms = t.get("market_score", 0)
            label = f"{ms // 10 * 10}-{ms // 10 * 10 + 10}"
            if label not in buckets:
                buckets[label] = []
            buckets[label].append(t)

        best_wr = 0.0
        best_ms = 0
        for label, g in buckets.items():
            if len(g) < 5:
                continue
            wins = sum(1 for t in g if t.get("net_pnl", 0) > 0)
            wr = wins / len(g)
            if wr > best_wr:
                best_wr = wr
                try:
                    best_ms = int(label.split("-")[0]) + 5
                except (ValueError, IndexError):
                    pass
        return best_ms

    def _find_blocked_regimes(self, regime_analysis: Dict) -> List[str]:
        blocked = []
        for regime, data in regime_analysis.items():
            wr = data.get("win_rate", 0)
            trades = data.get("trades", 0)
            if trades >= 5 and wr < 0.20:
                blocked.append(regime)
        return blocked
