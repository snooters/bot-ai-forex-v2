import pytest
from datetime import datetime
from simulation.learning_engine import LearningEngine


class TestLearningEngine:
    def make_trade(self, net_pnl, regime="STRONG_TRENDING_BULLISH",
                   confidence=0.6, side="BUY", pair="EURUSD",
                   entry_hour=8):
        return {
            "net_pnl": net_pnl,
            "pnl": net_pnl,
            "regime": regime,
            "confidence": confidence,
            "side": side,
            "pair": pair,
            "entry_time": datetime(2026, 6, 10, entry_hour, 0),
            "close_time": datetime(2026, 6, 10, entry_hour + 2, 0),
            "exit_reason": "TP_HIT" if net_pnl > 0 else "SL_HIT",
            "holding_time_minutes": 120,
            "rr_ratio": 2.0,
            "entry_price": 1.1000,
            "exit_price": 1.1050 if net_pnl > 0 else 1.0950,
            "commission": 0.5,
            "swap": 0.1,
            "market_score": 60,
            "atr_entry": 0.002,
            "spread": 0.5,
        }

    def test_analyze_empty(self):
        engine = LearningEngine()
        result = engine.analyze([], 10000)
        assert result["stats"]["total_trades"] == 0
        assert result["skill"]["score"] == 0

    def test_analyze_with_trades(self):
        engine = LearningEngine()
        trades = [self.make_trade(10) for _ in range(10)]
        result = engine.analyze(trades, 10000)
        assert result["stats"]["total_trades"] == 10
        assert result["stats"]["win_rate"] == 1.0
        assert result["real_account_readiness"] in ("READY", "CAUTION", "BLOCKED", "INSUFFICIENT_DATA")

    def test_blocked_regime(self):
        engine = LearningEngine()
        trades = [self.make_trade(-10, regime="SIDEWAYS") for _ in range(10)]
        trades += [self.make_trade(10, regime="BULLISH") for _ in range(10)]
        result = engine.analyze(trades, 10000)
        blocked = result["blocked_regimes"]
        assert "SIDEWAYS" in blocked
        assert "BULLISH" not in blocked if len(blocked) > 0 else True

    def test_regime_analysis(self):
        engine = LearningEngine()
        trades = [self.make_trade(10, regime="BULLISH") for _ in range(5)]
        trades += [self.make_trade(-5, regime="BEARISH") for _ in range(5)]
        result = engine.analyze(trades, 10000)
        regime = result["regime_analysis"]
        assert "BULLISH" in regime
        assert "BEARISH" in regime
        assert regime["BULLISH"]["win_rate"] > regime["BEARISH"]["win_rate"]

    def test_side_analysis(self):
        engine = LearningEngine()
        trades = [self.make_trade(10, side="BUY") for _ in range(5)]
        trades += [self.make_trade(-5, side="SELL") for _ in range(5)]
        result = engine.analyze(trades, 10000)
        sides = result["side_analysis"]
        assert "BUY" in sides
        assert "SELL" in sides

    def test_hour_analysis(self):
        engine = LearningEngine()
        trades = [self.make_trade(10, entry_hour=8) for _ in range(3)]
        trades += [self.make_trade(-5, entry_hour=14) for _ in range(3)]
        result = engine.analyze(trades, 10000)
        hours = result["hour_analysis"]
        assert "08:00" in hours
        assert "14:00" in hours

    def test_pair_analysis(self):
        engine = LearningEngine()
        trades = [self.make_trade(10, pair="EURUSD") for _ in range(5)]
        trades += [self.make_trade(-5, pair="GBPUSD") for _ in range(5)]
        result = engine.analyze(trades, 10000)
        pairs = result["pair_analysis"]
        assert "EURUSD" in pairs
        assert "GBPUSD" in pairs

    def test_exit_analysis(self):
        engine = LearningEngine()
        trades = [self.make_trade(10, side="BUY") for _ in range(5)]
        trades += [self.make_trade(-5, side="SELL") for _ in range(5)]
        result = engine.analyze(trades, 10000)
        exits = result["exit_analysis"]
        assert "TP_HIT" in exits
        assert "SL_HIT" in exits
