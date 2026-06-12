import pytest
from simulation.monte_carlo import monte_carlo_simulation, fast_monte_carlo


class TestMonteCarlo:
    def make_trade(self, net_pnl):
        return {
            "net_pnl": net_pnl,
            "pnl": net_pnl,
            "pair": "EURUSD",
            "side": "BUY",
            "commission": 0,
            "swap": 0,
        }

    def test_empty_trades(self):
        result = monte_carlo_simulation([], iterations=100, initial_balance=10000)
        assert result["iterations"] == 0
        assert result["profit_probability"] == 0.0

    def test_all_winners(self):
        trades = [self.make_trade(10) for _ in range(100)]
        result = monte_carlo_simulation(trades, iterations=100, initial_balance=10000)
        assert result["iterations"] == 100
        assert result["profit_probability"] == 1.0
        assert result["avg_profit"] > 0

    def test_all_losers(self):
        trades = [self.make_trade(-10) for _ in range(100)]
        result = monte_carlo_simulation(trades, iterations=100, initial_balance=10000)
        assert result["profit_probability"] == 0.0
        assert result["avg_profit"] < 0

    def test_mixed_trades(self):
        trades = [self.make_trade(15) for _ in range(60)]
        trades += [self.make_trade(-10) for _ in range(40)]
        result = monte_carlo_simulation(trades, iterations=200, initial_balance=10000)
        assert result["iterations"] == 200
        assert result["avg_profit"] > 0

    def test_fast_monte_carlo(self):
        trades = [self.make_trade(10) for _ in range(100)]
        result = fast_monte_carlo(trades, iterations=100, initial_balance=10000)
        assert result["iterations"] == 100
        assert "profit_probability" in result

    def test_fast_monte_carlo_empty(self):
        result = fast_monte_carlo([], iterations=100)
        assert "error" in result

    def test_confidence_interval(self):
        trades = [self.make_trade(10) for _ in range(200)]
        result = monte_carlo_simulation(trades, iterations=200, initial_balance=10000)
        assert result["pnl_ci_lower"] <= result["pnl_ci_upper"]

    def test_dd_probability(self):
        trades = [self.make_trade(-50) for _ in range(30)]
        trades += [self.make_trade(10) for _ in range(70)]
        result = monte_carlo_simulation(trades, iterations=100, initial_balance=10000)
        assert 0 <= result["dd_gt_20pct_probability"] <= 1
