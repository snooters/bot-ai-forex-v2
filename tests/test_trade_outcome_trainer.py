import json
import numpy as np
import pytest

from learning.trade_outcome_trainer import TradeOutcomeTrainer
from learning.trade_memory import INDICATOR_FIELDS


class TestTradeOutcomeTrainer:
    @pytest.fixture
    def trainer(self):
        return TradeOutcomeTrainer(trade_memory=None)

    @pytest.fixture
    def feature_columns(self):
        base = list(INDICATOR_FIELDS)
        extra = [
            "session_asia", "session_london", "session_ny", "session_overlap",
            "ema_cross", "price_action", "candle_pattern",
            "hh", "hl", "lh", "ll", "market_structure",
            "realized_vol", "price_position", "mtf_alignment",
        ]
        return base + extra

    def make_trade(self, direction="BUY", result="WIN", indicators=None):
        default_indicators = {k: 0.5 for k in INDICATOR_FIELDS}
        default_indicators["rsi"] = 65.0 if result == "WIN" else 35.0
        default_indicators["macd"] = 0.002 if direction == "BUY" else -0.002
        if indicators:
            default_indicators.update(indicators)
        return {
            "pair": "EURUSD",
            "direction": direction,
            "result": result,
            "profit": 15.0 if result == "WIN" else -10.0,
            "entry_time": "2026-06-01T08:00:00",
            "exit_time": "2026-06-01T10:00:00",
            "timeframe": "5",
            "indicators": json.dumps(default_indicators),
        }

    def test_init(self, trainer):
        assert trainer is not None
        assert trainer.LABEL_MAP[("BUY", "WIN")] == 0
        assert trainer.LABEL_MAP[("SELL", "WIN")] == 1
        assert trainer.LABEL_MAP[("BUY", "LOSS")] == 2
        assert trainer.LABEL_MAP[("SELL", "LOSS")] == 2

    def test_trade_to_label(self, trainer):
        assert trainer._trade_to_label({"direction": "BUY", "result": "WIN"}) == 0
        assert trainer._trade_to_label({"direction": "SELL", "result": "WIN"}) == 1
        assert trainer._trade_to_label({"direction": "BUY", "result": "LOSS"}) == 2
        assert trainer._trade_to_label({"direction": "SELL", "result": "LOSS"}) == 2
        assert trainer._trade_to_label({"direction": "BUY", "result": "BREAK"}) == 2
        assert trainer._trade_to_label({"direction": "BUY", "result": "UNKNOWN"}) is None

    def test_extract_indicators_from_dict(self, trainer):
        trade = {"indicators": {"rsi": 65.0, "macd": 0.002, "unknown_field": 99}}
        extracted = trainer._extract_indicators(trade)
        assert "rsi" in extracted
        assert extracted["rsi"] == 65.0
        assert "unknown_field" not in extracted

    def test_extract_indicators_from_json(self, trainer):
        trade = {"indicators": '{"rsi": 65.0, "macd": 0.002}'}
        extracted = trainer._extract_indicators(trade)
        assert extracted["rsi"] == 65.0

    def test_extract_indicators_empty(self, trainer):
        assert trainer._extract_indicators({}) == {}
        assert trainer._extract_indicators({"indicators": None}) == {}
        assert trainer._extract_indicators({"indicators": "invalid json"}) == {}

    def test_convert_to_samples_empty(self, trainer, feature_columns):
        X, y = trainer.convert_to_samples([], feature_columns)
        assert X.shape == (0, len(feature_columns))
        assert len(y) == 0

    def test_convert_to_samples_single_buy_win(self, trainer, feature_columns):
        trade = self.make_trade("BUY", "WIN")
        X, y = trainer.convert_to_samples([trade], feature_columns)
        assert X.shape == (1, len(feature_columns))
        assert y[0] == 0  # BUY_WIN -> 0
        assert X[0, feature_columns.index("rsi")] == 65.0

    def test_convert_to_samples_single_sell_win(self, trainer, feature_columns):
        trade = self.make_trade("SELL", "WIN")
        X, y = trainer.convert_to_samples([trade], feature_columns)
        assert y[0] == 1  # SELL_WIN -> 1

    def test_convert_to_samples_single_buy_loss(self, trainer, feature_columns):
        trade = self.make_trade("BUY", "LOSS")
        X, y = trainer.convert_to_samples([trade], feature_columns)
        assert y[0] == 2  # BUY_LOSS -> 2 (HOLD)

    def test_convert_to_samples_multiple(self, trainer, feature_columns):
        trades = [
            self.make_trade("BUY", "WIN"),
            self.make_trade("SELL", "WIN"),
            self.make_trade("BUY", "LOSS"),
            self.make_trade("SELL", "LOSS"),
        ]
        X, y = trainer.convert_to_samples(trades, feature_columns)
        assert X.shape == (4, len(feature_columns))
        assert list(y) == [0, 1, 2, 2]

    def test_convert_to_samples_missing_indicators_skipped(self, trainer, feature_columns):
        trades = [
            {"direction": "BUY", "result": "WIN"},  # no indicators
            self.make_trade("BUY", "WIN"),
        ]
        X, y = trainer.convert_to_samples(trades, feature_columns)
        assert X.shape == (1, len(feature_columns))

    def test_merge_with_ohlc_no_trade_samples(self, trainer):
        X_ohlc = np.random.rand(100, 10)
        y_ohlc = np.random.randint(0, 3, 100)
        X, y, w = trainer.merge_with_ohlc(X_ohlc, y_ohlc, np.empty((0, 10)), np.empty(0))
        assert X.shape == X_ohlc.shape
        assert np.allclose(w, 1.0)

    def test_merge_with_ohlc_appends_and_shuffles(self, trainer, feature_columns):
        X_ohlc = np.random.rand(50, len(feature_columns))
        y_ohlc = np.zeros(50, dtype=int)
        X_trade = np.random.rand(10, len(feature_columns))
        y_trade = np.ones(10, dtype=int)
        X, y, w = trainer.merge_with_ohlc(X_ohlc, y_ohlc, X_trade, y_trade, upsample_wins=False)
        assert X.shape == (60, len(feature_columns))
        assert len(y) == 60
        assert len(w) == 60

    def test_merge_with_ohlc_upsamples_wins(self, trainer, feature_columns):
        X_ohlc = np.random.rand(50, len(feature_columns))
        y_ohlc = np.zeros(50, dtype=int)
        X_trade = np.random.rand(5, len(feature_columns))
        y_trade = np.array([0, 0, 1, 2, 2], dtype=int)  # 3 wins, 2 losses
        X, y, w = trainer.merge_with_ohlc(X_ohlc, y_ohlc, X_trade, y_trade, upsample_wins=True, win_weight=2.0)
        assert X.shape[0] == 55
        # Some trade-derived win samples should have weight 2.0
        assert 2.0 in np.unique(w)
        assert 1.0 in np.unique(w)

    def test_get_recent_trades_no_memory(self, trainer):
        trades = trainer.get_recent_trades("EURUSD")
        assert trades == []

    def test_get_trade_quality_stats_empty(self, trainer):
        stats = trainer.get_trade_quality_stats([])
        assert stats["total"] == 0

    def test_get_trade_quality_stats(self, trainer):
        trades = [
            self.make_trade("BUY", "WIN"),
            self.make_trade("BUY", "LOSS"),
            self.make_trade("SELL", "WIN"),
        ]
        stats = trainer.get_trade_quality_stats(trades)
        assert stats["total"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["win_rate"] == pytest.approx(2 / 3, rel=0.01)

    def test_feature_alignment_preserves_values(self, trainer, feature_columns):
        indicators = {"rsi": 70.0, "macd": -0.001, "atr": 0.002}
        trade = self.make_trade("SELL", "WIN", indicators=indicators)
        X, y = trainer.convert_to_samples([trade], feature_columns)
        assert X[0, feature_columns.index("rsi")] == 70.0
        assert X[0, feature_columns.index("macd")] == pytest.approx(-0.001, abs=1e-6)
        assert X[0, feature_columns.index("atr")] == 0.002
        assert X[0, feature_columns.index("session_asia")] == 0.0  # missing, default 0
        assert y[0] == 1  # SELL_WIN
