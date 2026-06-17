from unittest.mock import MagicMock

import pytest

from decision.decision_engine import DecisionEngine
from core.constants import TradeDirection


@pytest.fixture
def engine():
    ml_mock = MagicMock()
    ml_mock.get_buy_sell_hold.return_value = {
        "signal": "BUY", "confidence": 0.72,
        "buy_prob": 0.65, "sell_prob": 0.15, "hold_prob": 0.20,
    }
    scorer_mock = MagicMock()
    scorer_mock.compute_market_score.return_value = 70
    de = DecisionEngine(ml_predictor=ml_mock, market_scorer=scorer_mock)
    return de


def _trend_bullish():
    return {"direction": "BULLISH", "strength": 0.6, "score": 0.5}


def _trend_bearish():
    return {"direction": "BEARISH", "strength": 0.5, "score": -0.4}


def _regime(r: str):
    return {"regime": r, "confidence": 0.7, "volatility_score": 40, "trend": "BULLISH"}


def _sr():
    return {}


def _feat():
    return {"indicators": {"rsi": 50, "macd": 0, "macd_signal": 0}}


def _df():
    import pandas as pd
    return pd.DataFrame({"close": [1.05], "atr": [0.001]})


class TestDecisionEngineMTFFilter:
    def test_buy_allows_when_3_of_4_agree(self, engine):
        mtf = {"trend240": 1, "trend60": 1, "trend30": 0, "trend15": 1}
        d = engine.make_decision(
            symbol="EURUSD", df_entry=_df(),
            trend_result=_trend_bullish(), vol_result={}, momentum_result={},
            regime_result=_regime("STRONG_TRENDING_BULLISH"),
            sr_info=_sr(), feature_summary=_feat(),
            multi_tf_trends=mtf,
        )
        assert not d["no_trade"]

    def test_buy_with_2_agree_does_not_block(self, engine):
        # MTF is now BONUS, not blocker — 2/4 agree should still be allowed
        mtf = {"trend240": 1, "trend60": 1, "trend30": -1, "trend15": -1}
        d = engine.make_decision(
            symbol="EURUSD", df_entry=_df(),
            trend_result=_trend_bullish(), vol_result={}, momentum_result={},
            regime_result=_regime("STRONG_TRENDING_BULLISH"),
            sr_info=_sr(), feature_summary=_feat(),
            multi_tf_trends=mtf,
        )
        assert not d["no_trade"]
        assert any("MTF note" in r for r in d.get("reasons", []))

    def test_sell_allows_when_3_of_4_agree(self, engine):
        engine.ml_predictor.get_buy_sell_hold.return_value = {
            "signal": "SELL", "confidence": 0.72,
            "buy_prob": 0.15, "sell_prob": 0.65, "hold_prob": 0.20,
        }
        mtf = {"trend240": -1, "trend60": -1, "trend30": 0, "trend15": -1}
        d = engine.make_decision(
            symbol="EURUSD", df_entry=_df(),
            trend_result=_trend_bearish(), vol_result={}, momentum_result={},
            regime_result=_regime("STRONG_TRENDING_BEARISH"),
            sr_info=_sr(), feature_summary=_feat(),
            multi_tf_trends=mtf,
        )
        assert not d["no_trade"]

    def test_sell_with_2_agree_does_not_block(self, engine):
        # MTF is now BONUS, not blocker — 2/4 agree should still be allowed
        engine.ml_predictor.get_buy_sell_hold.return_value = {
            "signal": "SELL", "confidence": 0.72,
            "buy_prob": 0.15, "sell_prob": 0.65, "hold_prob": 0.20,
        }
        mtf = {"trend240": -1, "trend60": -1, "trend30": 1, "trend15": 1}
        d = engine.make_decision(
            symbol="EURUSD", df_entry=_df(),
            trend_result=_trend_bearish(), vol_result={}, momentum_result={},
            regime_result=_regime("STRONG_TRENDING_BEARISH"),
            sr_info=_sr(), feature_summary=_feat(),
            multi_tf_trends=mtf,
        )
        assert not d["no_trade"]
        assert any("MTF note" in r for r in d.get("reasons", []))

    def test_mtf_neutral_not_blocked(self, engine):
        mtf = {"trend240": 0, "trend60": 0, "trend30": 0, "trend15": 0}
        d = engine.make_decision(
            symbol="EURUSD", df_entry=_df(),
            trend_result=_trend_bullish(), vol_result={}, momentum_result={},
            regime_result=_regime("STRONG_TRENDING_BULLISH"),
            sr_info=_sr(), feature_summary=_feat(),
            multi_tf_trends=mtf,
        )
        assert not d["no_trade"]

    def test_mtf_not_provided_does_not_block(self, engine):
        d = engine.make_decision(
            symbol="EURUSD", df_entry=_df(),
            trend_result=_trend_bullish(), vol_result={}, momentum_result={},
            regime_result=_regime("STRONG_TRENDING_BULLISH"),
            sr_info=_sr(), feature_summary=_feat(),
        )
        assert not d["no_trade"]
