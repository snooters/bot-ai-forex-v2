import pytest
from datetime import datetime, timedelta
from simulation.virtual_position import VirtualPosition


class TestVirtualPosition:
    def test_init(self):
        now = datetime(2026, 6, 10)
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=now,
            open_price=1.1000, volume=0.01,
        )
        assert pos.pair == "EURUSD"
        assert pos.side == "BUY"
        assert pos.open_price == 1.1000
        assert pos.volume == 0.01
        assert pos.pnl == 0.0
        assert pos.close_price is None

    def test_calculate_pnl_buy(self):
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01,
        )
        pnl = pos.calculate_pnl(1.1050)
        assert pnl == pytest.approx(5.0, rel=0.01)

    def test_calculate_pnl_buy_negative(self):
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01,
        )
        pnl = pos.calculate_pnl(1.0950)
        assert pnl == pytest.approx(-5.0, rel=0.01)

    def test_calculate_pnl_sell(self):
        pos = VirtualPosition(
            pair="EURUSD", side="SELL", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01,
        )
        pnl = pos.calculate_pnl(1.0950)
        assert pnl == pytest.approx(5.0, rel=0.01)

    def test_calculate_pnl_sell_negative(self):
        pos = VirtualPosition(
            pair="EURUSD", side="SELL", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01,
        )
        pnl = pos.calculate_pnl(1.1050)
        assert pnl == pytest.approx(-5.0, rel=0.01)

    def test_check_sl_hit_buy(self):
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01, sl=1.0950,
        )
        assert pos.check_sl_hit(high=1.1020, low=1.0940) is True
        assert pos.check_sl_hit(high=1.1020, low=1.0960) is False

    def test_check_sl_hit_sell(self):
        pos = VirtualPosition(
            pair="EURUSD", side="SELL", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01, sl=1.1050,
        )
        assert pos.check_sl_hit(high=1.1060, low=1.0980) is True
        assert pos.check_sl_hit(high=1.1040, low=1.0980) is False

    def test_check_tp_hit_buy(self):
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01, tp=1.1100,
        )
        assert pos.check_tp_hit(high=1.1110, low=1.1010) is True
        assert pos.check_tp_hit(high=1.1090, low=1.1010) is False

    def test_check_tp_hit_sell(self):
        pos = VirtualPosition(
            pair="EURUSD", side="SELL", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01, tp=1.0900,
        )
        assert pos.check_tp_hit(high=1.1010, low=1.0890) is True
        assert pos.check_tp_hit(high=1.1010, low=1.0910) is False

    def test_no_sl_no_tp(self):
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01,
        )
        assert pos.check_sl_hit(1.2, 0.8) is False
        assert pos.check_tp_hit(1.2, 0.8) is False

    def test_trailing_update_buy(self):
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01, trailing_activated=True,
        )
        pos.update_trailing(1.1100, atr=0.002, atr_multiplier=1.5)
        expected_sl = 1.1100 - 0.002 * 1.5
        assert pos.sl == pytest.approx(expected_sl, rel=0.0001)
        assert pos.trailing_stop_price == pytest.approx(expected_sl, rel=0.0001)

    def test_trailing_update_buy_only_raises(self):
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01, trailing_activated=True,
        )
        pos.update_trailing(1.1100, atr=0.002, atr_multiplier=1.5)
        first_sl = pos.sl
        pos.update_trailing(1.1050, atr=0.002, atr_multiplier=1.5)
        assert pos.sl == pytest.approx(first_sl, rel=0.0001)

    def test_trailing_update_sell(self):
        pos = VirtualPosition(
            pair="EURUSD", side="SELL", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01, trailing_activated=True,
        )
        pos.update_trailing(1.0900, atr=0.002, atr_multiplier=1.5)
        expected_sl = 1.0900 + 0.002 * 1.5
        assert pos.sl == pytest.approx(expected_sl, rel=0.0001)

    def test_trailing_hit(self):
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=datetime(2026, 6, 10),
            open_price=1.1000, volume=0.01, trailing_activated=True,
        )
        pos.update_trailing(1.1100, atr=0.002, atr_multiplier=1.5)
        low = pos.trailing_stop_price - 0.0001
        assert pos.check_trailing_hit(high=1.1100, low=low) is True

    def test_check_time_exit(self):
        now = datetime(2026, 6, 10, 8, 0)
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=now,
            open_price=1.1000, volume=0.01,
        )
        hours_13 = now + timedelta(hours=13)
        assert pos.check_time_exit(12, hours_13) is True
        hours_11 = now + timedelta(hours=11)
        assert pos.check_time_exit(12, hours_11) is False

    def test_close(self):
        now = datetime(2026, 6, 10)
        pos = VirtualPosition(
            pair="EURUSD", side="BUY", open_time=now,
            open_price=1.1000, volume=0.01,
        )
        close_time = datetime(2026, 6, 10, 12, 0)
        pos.close(1.1050, "TP_HIT", close_time)
        assert pos.close_price == 1.1050
        assert pos.exit_reason == "TP_HIT"
        assert pos.close_time == close_time
        assert pos.pnl == pytest.approx(5.0, rel=0.01)
