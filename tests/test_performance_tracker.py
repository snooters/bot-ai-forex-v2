import pytest
from simulation.performance_tracker import PerformanceTracker


class TestPerformanceTracker:
    def make_trade(self, net_pnl, side="BUY", regime="STRONG_TRENDING", 
                   confidence=0.6, entry_hour=8, exit_reason="TP_HIT",
                   holding=60, rr=2.0):
        return {
            "net_pnl": net_pnl,
            "pnl": net_pnl,
            "side": side,
            "regime": regime,
            "confidence": confidence,
            "entry_hour": entry_hour,
            "exit_reason": exit_reason,
            "holding_time_minutes": holding,
            "rr_ratio": rr,
            "pair": "EURUSD",
            "entry_price": 1.1000,
            "exit_price": 1.1050 if net_pnl > 0 else 1.0950,
            "commission": 0.5,
            "swap": 0.1,
            "market_score": 60,
            "atr_entry": 0.002,
            "spread": 0.5,
        }

    def test_empty_trades(self):
        tracker = PerformanceTracker()
        stats = tracker.compute([], 10000)
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0

    def test_all_winners(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(10) for _ in range(10)]
        stats = tracker.compute(trades, 10000)
        assert stats["total_trades"] == 10
        assert stats["win_count"] == 10
        assert stats["win_rate"] == 1.0

    def test_all_losers(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(-10) for _ in range(10)]
        stats = tracker.compute(trades, 10000)
        assert stats["total_trades"] == 10
        assert stats["loss_count"] == 10
        assert stats["win_rate"] == 0.0

    def test_mixed_trades(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(10) for _ in range(7)]
        trades += [self.make_trade(-10) for _ in range(3)]
        stats = tracker.compute(trades, 10000)
        assert stats["total_trades"] == 10
        assert stats["win_count"] == 7
        assert stats["loss_count"] == 3
        assert stats["win_rate"] == pytest.approx(0.7, rel=0.01)

    def test_profit_factor(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(10) for _ in range(5)]
        trades += [self.make_trade(-5) for _ in range(5)]
        stats = tracker.compute(trades, 10000)
        assert stats["profit_factor"] == pytest.approx(2.0, rel=0.01)

    def test_net_profit(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(10) for _ in range(5)]
        trades += [self.make_trade(-5) for _ in range(5)]
        stats = tracker.compute(trades, 10000)
        assert stats["net_profit"] == pytest.approx(25.0, rel=0.01)

    def test_return_pct(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(50) for _ in range(5)]
        stats = tracker.compute(trades, 10000)
        assert stats["return_pct"] == pytest.approx(2.5, rel=0.01)

    def test_consecutive_wins(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(10), self.make_trade(10), self.make_trade(-5),
                  self.make_trade(10), self.make_trade(10), self.make_trade(10)]
        stats = tracker.compute(trades, 10000)
        assert stats["consecutive_wins"] == 3

    def test_consecutive_losses(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(-5), self.make_trade(-5), self.make_trade(10),
                  self.make_trade(-5), self.make_trade(-5)]
        stats = tracker.compute(trades, 10000)
        assert stats["consecutive_losses"] == 2

    def test_avg_rr(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(10, rr=2.0), self.make_trade(10, rr=3.0)]
        stats = tracker.compute(trades, 10000)
        assert stats["avg_rr"] == pytest.approx(2.5, rel=0.01)

    def test_group_by_regime(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(10, regime="BULLISH") for _ in range(5)]
        trades += [self.make_trade(-5, regime="BEARISH") for _ in range(5)]
        stats = tracker.compute(trades, 10000)
        by_regime = stats["by_regime"]
        assert "BULLISH" in by_regime
        assert "BEARISH" in by_regime
        assert by_regime["BULLISH"]["trades"] == 5
        assert by_regime["BEARISH"]["trades"] == 5

    def test_sharpe_positive(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(10) for _ in range(10)]
        trades += [self.make_trade(-5) for _ in range(5)]
        trades += [self.make_trade(3) for _ in range(5)]
        stats = tracker.compute(trades, 10000)
        assert stats["sharpe_ratio"] != 0

    def test_expectancy(self):
        tracker = PerformanceTracker()
        trades = [self.make_trade(10) for _ in range(6)]
        trades += [self.make_trade(-5) for _ in range(4)]
        stats = tracker.compute(trades, 10000)
        assert stats["expectancy"] > 0
