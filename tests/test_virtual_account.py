import pytest
from datetime import datetime
from simulation.virtual_account import VirtualAccount


class TestVirtualAccount:
    def test_init(self):
        acc = VirtualAccount(initial_balance=10000, leverage=100)
        assert acc.initial_balance == 10000
        assert acc.balance == 10000
        assert acc.equity == 10000
        assert acc.total_positions == 0

    def test_open_position(self):
        acc = VirtualAccount(10000)
        now = datetime(2026, 6, 10)
        pos = acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now)
        assert pos is not None
        assert pos.side == "BUY"
        assert acc.total_positions == 1
        assert acc.balance < 10000  # commission deducted

    def test_open_position_insufficient_margin(self):
        acc = VirtualAccount(100, leverage=1)
        now = datetime(2026, 6, 10)
        pos = acc.open_position("EURUSD", "BUY", 1.1000, 10.0, now)
        assert pos is None

    def test_close_position(self):
        acc = VirtualAccount(10000)
        now = datetime(2026, 6, 10)
        pos = acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now)
        close_time = datetime(2026, 6, 10, 12, 0)
        result = acc.close_position(pos.position_id, 1.1050, "TP_HIT", close_time)
        assert result is not None
        assert result["pnl"] == pytest.approx(5.0, rel=0.01)
        assert result["exit_reason"] == "TP_HIT"
        assert len(acc.closed_positions) == 1

    def test_close_nonexistent_position(self):
        acc = VirtualAccount(10000)
        now = datetime(2026, 6, 10)
        result = acc.close_position("nonexistent", 1.1000, "TEST", now)
        assert result is None

    def test_check_sl_tp(self):
        acc = VirtualAccount(10000)
        now = datetime(2026, 6, 10)
        acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now,
                          sl=1.0950, tp=1.1100)
        closed = acc.check_sl_tp_all(high=1.1110, low=1.1010, current_time=now)
        assert len(closed) == 1
        assert closed[0]["exit_reason"] == "TP_HIT"

    def test_check_sl_hit(self):
        acc = VirtualAccount(10000)
        now = datetime(2026, 6, 10)
        acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now,
                          sl=1.0950, tp=1.1100)
        closed = acc.check_sl_tp_all(high=1.1010, low=1.0940, current_time=now)
        assert len(closed) == 1
        assert closed[0]["exit_reason"] == "SL_HIT"

    def test_equity_property(self):
        acc = VirtualAccount(10000)
        now = datetime(2026, 6, 10)
        acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now)
        assert acc.equity < 10000  # commission

    def test_margin_level(self):
        acc = VirtualAccount(10000, leverage=100)
        now = datetime(2026, 6, 10)
        acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now)
        assert acc.margin_level > 0
        assert acc.used_margin > 0

    def test_close_all_positions(self):
        acc = VirtualAccount(10000)
        now = datetime(2026, 6, 10)
        acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now)
        acc.open_position("EURUSD", "SELL", 1.1000, 0.01, now)
        results = acc.close_all_positions(1.1050, "LIQUIDATION", now)
        assert len(results) == 2
        assert acc.total_positions == 0

    def test_reset(self):
        acc = VirtualAccount(10000)
        now = datetime(2026, 6, 10)
        acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now)
        acc.reset()
        assert acc.balance == 10000
        assert acc.total_positions == 0
        assert len(acc.closed_positions) == 0

    def test_apply_trailing_all(self):
        acc = VirtualAccount(10000)
        now = datetime(2026, 6, 10)
        pos = acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now,
                                trailing_activated=True)
        acc.apply_trailing_all(1.1200, atr=0.002, atr_multiplier=1.5)
        updated = acc.get_position_by_id(pos.position_id)
        assert updated is not None
        assert updated.trailing_stop_price is not None

    def test_check_time_exit_all(self):
        acc = VirtualAccount(10000)
        from datetime import timedelta
        now = datetime(2026, 6, 10, 8, 0)
        acc.open_position("EURUSD", "BUY", 1.1000, 0.01, now)
        later = now + timedelta(hours=13)
        closed = acc.check_time_exit_all(12, later)
        assert len(closed) == 1
