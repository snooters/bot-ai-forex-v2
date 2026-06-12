import os
import tempfile
import pytest
from datetime import datetime
from simulation.trade_journal import TradeJournal


class TestTradeJournal:
    def make_trade(self, net_pnl, regime="BULLISH", side="BUY", pair="EURUSD",
                   confidence=0.6, entry_hour=8):
        return {
            "net_pnl": net_pnl,
            "pnl": net_pnl,
            "regime": regime,
            "side": side,
            "pair": pair,
            "confidence": confidence,
            "entry_time": datetime(2026, 6, 10, entry_hour, 0),
            "close_time": datetime(2026, 6, 10, entry_hour + 1, 0),
            "holding_time_minutes": 60,
            "rr_ratio": 2.0,
            "entry_price": 1.1000,
            "exit_price": 1.1050 if net_pnl > 0 else 1.0950,
            "exit_reason": "TP_HIT" if net_pnl > 0 else "SL_HIT",
            "commission": 0.5,
            "swap": 0.1,
            "signal": "BUY",
            "market_score": 60,
            "atr_entry": 0.002,
            "spread": 0.5,
            "magic": 1001,
            "multi_tf_agreement": 3,
            "skill_score": 65,
            "trade_quality_score": 7.5,
            "tp1": 1.1100,
            "tp2": 1.1200,
            "sl_price": 1.0950,
        }

    def test_record_trade(self):
        journal = TradeJournal()
        journal.record_trade(self.make_trade(10))
        assert journal.total_trades == 1
        assert journal.total_pnl > 0

    def test_multiple_trades(self):
        journal = TradeJournal()
        for _ in range(5):
            journal.record_trade(self.make_trade(10))
        assert journal.total_trades == 5

    def test_get_winning_trades(self):
        journal = TradeJournal()
        journal.record_trade(self.make_trade(10))
        journal.record_trade(self.make_trade(-5))
        assert len(journal.get_winning_trades()) == 1
        assert len(journal.get_losing_trades()) == 1

    def test_get_trades_by_regime(self):
        journal = TradeJournal()
        journal.record_trade(self.make_trade(10, regime="BULLISH"))
        journal.record_trade(self.make_trade(10, regime="BEARISH"))
        by_regime = journal.get_trades_by_regime()
        assert "BULLISH" in by_regime
        assert "BEARISH" in by_regime
        assert len(by_regime["BULLISH"]) == 1

    def test_get_trades_by_pair(self):
        journal = TradeJournal()
        journal.record_trade(self.make_trade(10, pair="EURUSD"))
        journal.record_trade(self.make_trade(10, pair="GBPUSD"))
        by_pair = journal.get_trades_by_pair()
        assert "EURUSD" in by_pair
        assert "GBPUSD" in by_pair

    def test_get_trades_by_side(self):
        journal = TradeJournal()
        journal.record_trade(self.make_trade(10, side="BUY"))
        journal.record_trade(self.make_trade(10, side="SELL"))
        by_side = journal.get_trades_by_side()
        assert len(by_side["BUY"]) == 1
        assert len(by_side["SELL"]) == 1

    def test_get_trades_by_hour(self):
        journal = TradeJournal()
        journal.record_trade(self.make_trade(10, entry_hour=8))
        journal.record_trade(self.make_trade(10, entry_hour=8))
        journal.record_trade(self.make_trade(10, entry_hour=14))
        by_hour = journal.get_trades_by_hour()
        assert len(by_hour[8]) == 2
        assert len(by_hour[14]) == 1

    def test_get_trades_by_confidence_range(self):
        journal = TradeJournal()
        journal.record_trade(self.make_trade(10, confidence=0.3))
        journal.record_trade(self.make_trade(10, confidence=0.6))
        journal.record_trade(self.make_trade(10, confidence=0.9))
        by_conf = journal.get_trades_by_confidence_range(bins=5)
        assert len(by_conf) >= 2

    def test_export_csv(self):
        journal = TradeJournal()
        journal.record_trade(self.make_trade(10))
        journal.record_trade(self.make_trade(-5))
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.close()
        try:
            journal.export_csv(tmp.name)
            with open(tmp.name) as f:
                content = f.read()
            assert "pair" in content
            assert "pnl" in content
        finally:
            os.unlink(tmp.name)

    def test_reset(self):
        journal = TradeJournal()
        journal.record_trade(self.make_trade(10))
        journal.reset()
        assert journal.total_trades == 0
        assert journal.total_pnl == 0.0

    def test_net_pnl_default(self):
        journal = TradeJournal()
        trade = self.make_trade(10)
        del trade["pnl"]
        journal.record_trade(trade)
        assert journal.total_trades == 1
