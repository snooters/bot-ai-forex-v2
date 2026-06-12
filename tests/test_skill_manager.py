import pytest
from simulation.skill_manager import SkillManager


class TestSkillManager:
    def make_trade(self, net_pnl, side="BUY", pair="EURUSD"):
        return {
            "net_pnl": net_pnl,
            "pnl": net_pnl,
            "side": side,
            "pair": pair,
            "regime": "BULLISH",
            "confidence": 0.6,
            "exit_reason": "TP_HIT" if net_pnl > 0 else "SL_HIT",
            "holding_time_minutes": 60,
            "rr_ratio": 2.0,
            "market_score": 60,
        }

    def test_empty_trades(self):
        mgr = SkillManager()
        result = mgr.compute_skill([])
        assert result["score"] == 0
        assert result["level"] == "Newborn"

    def test_all_winners_high_skill(self):
        mgr = SkillManager()
        trades = [self.make_trade(10) for _ in range(50)]
        result = mgr.compute_skill(trades)
        assert result["score"] > 50

    def test_all_losers_low_skill(self):
        mgr = SkillManager()
        trades = [self.make_trade(-10) for _ in range(50)]
        result = mgr.compute_skill(trades)
        assert result["score"] <= 30

    def test_skill_levels(self):
        mgr = SkillManager()
        assert mgr._get_level(0) == "Newborn"
        assert mgr._get_level(10) == "Newborn"
        assert mgr._get_level(21) == "Learning"
        assert mgr._get_level(41) == "Intermediate"
        assert mgr._get_level(61) == "Advanced"
        assert mgr._get_level(81) == "Expert"
        assert mgr._get_level(100) == "Expert"

    def test_pair_skills(self):
        mgr = SkillManager()
        trades = [self.make_trade(10, pair="EURUSD") for _ in range(30)]
        trades += [self.make_trade(-5, pair="GBPUSD") for _ in range(30)]
        pair_skills = mgr.compute_pair_skills(trades)
        assert "EURUSD" in pair_skills
        assert "GBPUSD" in pair_skills

    def test_readiness_insufficient_data(self):
        mgr = SkillManager()
        status = mgr.classify_readiness({"win_rate": 0.5, "profit_factor": 1.2,
                                          "max_drawdown_pct": 0.1, "total_trades": 10})
        assert status == "INSUFFICIENT_DATA"

    def test_readiness_blocked(self):
        mgr = SkillManager()
        status = mgr.classify_readiness({"win_rate": 0.30, "profit_factor": 0.8,
                                          "max_drawdown_pct": 0.2, "total_trades": 50})
        assert status == "BLOCKED"

    def test_readiness_ready(self):
        mgr = SkillManager()
        status = mgr.classify_readiness({"win_rate": 0.50, "profit_factor": 1.5,
                                          "max_drawdown_pct": 0.1, "total_trades": 50})
        assert status == "READY"

    def test_readiness_caution(self):
        mgr = SkillManager()
        status = mgr.classify_readiness({"win_rate": 0.40, "profit_factor": 1.1,
                                          "max_drawdown_pct": 0.2, "total_trades": 50})
        assert status == "CAUTION"

    def test_stability(self):
        mgr = SkillManager()
        trades = [self.make_trade(10) for _ in range(40)]
        stability = mgr._compute_stability(trades)
        assert 0 <= stability <= 1
