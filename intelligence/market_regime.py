import numpy as np
import pandas as pd
from typing import Dict, Optional

from core.constants import MarketRegime, TrendDirection
from utils.logger import get_logger


class MarketRegimeDetector:
    def __init__(self):
        self.logger = get_logger("market_regime")
        self._current_regime = MarketRegime.SIDEWAYS

    def detect_regime(
        self,
        trend_result: Dict,
        vol_result: Dict,
        momentum_result: Dict,
        df: pd.DataFrame
    ) -> Dict:
        if not trend_result or not vol_result:
            return self._default_regime()

        regime_scores = {}
        trend_dir = trend_result.get("direction", TrendDirection.SIDEWAYS.value)
        trend_strength = trend_result.get("strength", 0)
        vol_level = vol_result.get("level", "low")
        vol_score = vol_result.get("score", 0)

        strong_trending = trend_dir in [
            TrendDirection.STRONG_BULLISH.value,
            TrendDirection.STRONG_BEARISH.value
        ] and trend_strength > 0.5 and vol_score > 40

        weak_trending = trend_dir in [
            TrendDirection.BULLISH.value,
            TrendDirection.BEARISH.value,
            TrendDirection.WEAK_BULLISH.value,
            TrendDirection.WEAK_BEARISH.value,
        ] or (trend_strength > 0.2 and vol_score < 60)

        sideways = trend_dir in [
            TrendDirection.SIDEWAYS.value,
            TrendDirection.CONSOLIDATION.value
        ] and vol_score < 50

        high_vol = vol_level == "high" and vol_score > 70
        low_vol = vol_level == "low" and vol_score < 30

        news_regime = self._detect_news_regime(df)

        if news_regime:
            regime_scores["regime"] = MarketRegime.NEWS_DRIVEN.value
            regime_scores["confidence"] = 0.8
        elif strong_trending and high_vol:
            if "BULLISH" in trend_dir:
                regime_scores["regime"] = MarketRegime.STRONG_TRENDING_BULLISH.value
            else:
                regime_scores["regime"] = MarketRegime.STRONG_TRENDING_BEARISH.value
            regime_scores["confidence"] = 0.9
        elif strong_trending and not high_vol:
            if "BULLISH" in trend_dir:
                regime_scores["regime"] = MarketRegime.STRONG_TRENDING_BULLISH.value
            else:
                regime_scores["regime"] = MarketRegime.STRONG_TRENDING_BEARISH.value
            regime_scores["confidence"] = 0.7
        elif weak_trending:
            if "BULLISH" in trend_dir:
                regime_scores["regime"] = MarketRegime.WEAK_TRENDING_BULLISH.value
            else:
                regime_scores["regime"] = MarketRegime.WEAK_TRENDING_BEARISH.value
            regime_scores["confidence"] = 0.6
        elif sideways and low_vol:
            regime_scores["regime"] = MarketRegime.LOW_VOLATILITY.value
            regime_scores["confidence"] = 0.7
        elif sideways:
            regime_scores["regime"] = MarketRegime.SIDEWAYS.value
            regime_scores["confidence"] = 0.6
        elif high_vol and not strong_trending:
            regime_scores["regime"] = MarketRegime.HIGH_VOLATILITY.value
            regime_scores["confidence"] = 0.7
        else:
            regime_scores["regime"] = MarketRegime.SIDEWAYS.value
            regime_scores["confidence"] = 0.5

        regime_scores["trend"] = trend_dir
        regime_scores["trend_strength"] = trend_strength
        regime_scores["volatility"] = vol_level
        regime_scores["volatility_score"] = vol_score
        regime_scores["momentum_score"] = momentum_result.get("score", 0)
        regime_scores["is_trending"] = strong_trending or weak_trending
        regime_scores["is_volatile"] = high_vol

        self._current_regime = MarketRegime(regime_scores["regime"])

        return regime_scores

    def _detect_news_regime(self, df: pd.DataFrame) -> bool:
        if "volume" not in df.columns and "tick_volume" not in df.columns:
            return False
        vol_col = "volume" if "volume" in df.columns else "tick_volume"
        volume = df[vol_col]

        if len(volume) < 30:
            return False

        avg_vol = volume.tail(30).mean()
        recent_vol = volume.tail(3).mean()

        if avg_vol == 0:
            return False

        vol_spike = recent_vol > avg_vol * 2

        if "atr" in df.columns:
            atr = df["atr"].tail(5).mean()
            prev_atr = df["atr"].tail(20).head(15).mean()
            atr_spike = atr > prev_atr * 1.5 if prev_atr > 0 else False
        else:
            atr_spike = False

        return vol_spike or atr_spike

    def _default_regime(self) -> Dict:
        return {
            "regime": MarketRegime.SIDEWAYS.value,
            "confidence": 0.5,
            "trend": TrendDirection.SIDEWAYS.value,
            "trend_strength": 0,
            "volatility": "low",
            "volatility_score": 0,
            "momentum_score": 0,
            "is_trending": False,
            "is_volatile": False,
        }

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    def get_strategy_for_regime(self, regime: str) -> Dict:
        strategies = {
            MarketRegime.STRONG_TRENDING_BULLISH.value: {
                "action": "BUY",
                "aggressiveness": 0.8,
                "trailing_stop": True,
                "max_holding_bars": 50,
                "risk_multiplier": 1.0,
            },
            MarketRegime.STRONG_TRENDING_BEARISH.value: {
                "action": "SELL",
                "aggressiveness": 0.8,
                "trailing_stop": True,
                "max_holding_bars": 50,
                "risk_multiplier": 1.0,
            },
            MarketRegime.WEAK_TRENDING_BULLISH.value: {
                "action": "BUY",
                "aggressiveness": 0.5,
                "trailing_stop": False,
                "max_holding_bars": 30,
                "risk_multiplier": 0.8,
            },
            MarketRegime.WEAK_TRENDING_BEARISH.value: {
                "action": "SELL",
                "aggressiveness": 0.5,
                "trailing_stop": False,
                "max_holding_bars": 30,
                "risk_multiplier": 0.8,
            },
            MarketRegime.SIDEWAYS.value: {
                "action": "HOLD",
                "aggressiveness": 0.2,
                "trailing_stop": False,
                "max_holding_bars": 15,
                "risk_multiplier": 0.5,
            },
            MarketRegime.CONSOLIDATION.value: {
                "action": "HOLD",
                "aggressiveness": 0.1,
                "trailing_stop": False,
                "max_holding_bars": 10,
                "risk_multiplier": 0.3,
            },
            MarketRegime.HIGH_VOLATILITY.value: {
                "action": "HOLD",
                "aggressiveness": 0.3,
                "trailing_stop": True,
                "max_holding_bars": 20,
                "risk_multiplier": 0.5,
            },
            MarketRegime.LOW_VOLATILITY.value: {
                "action": "HOLD",
                "aggressiveness": 0.4,
                "trailing_stop": False,
                "max_holding_bars": 40,
                "risk_multiplier": 0.7,
            },
            MarketRegime.NEWS_DRIVEN.value: {
                "action": "HOLD",
                "aggressiveness": 0.0,
                "trailing_stop": False,
                "max_holding_bars": 5,
                "risk_multiplier": 0.0,
            },
        }

        return strategies.get(
            regime,
            {"action": "HOLD", "aggressiveness": 0.2, "trailing_stop": False,
             "max_holding_bars": 20, "risk_multiplier": 0.5},
        )
