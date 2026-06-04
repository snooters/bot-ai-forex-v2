import numpy as np
from typing import Dict, Optional

from core.config import config
from core.constants import TrendDirection, MarketRegime
from utils.logger import get_logger


class MarketScorer:
    def __init__(self):
        self.logger = get_logger("market_scorer")

    def compute_market_score(
        self,
        trend_result: Dict,
        vol_result: Dict,
        momentum_result: Dict,
        regime_result: Dict,
        sr_info: Dict,
        pattern_info: Dict,
    ) -> int:
        if not trend_result or not vol_result:
            return 0

        scores = []
        weights = []

        trend_score = self._score_trend(trend_result)
        scores.append(trend_score)
        weights.append(0.25)

        vol_score = self._score_volatility(vol_result, trend_result)
        scores.append(vol_score)
        weights.append(0.15)

        momentum_score_val = self._score_momentum(momentum_result, trend_result)
        scores.append(momentum_score_val)
        weights.append(0.20)

        regime_score_val = self._score_regime(regime_result)
        scores.append(regime_score_val)
        weights.append(0.15)

        sr_score_val = self._score_support_resistance(sr_info, trend_result)
        scores.append(sr_score_val)
        weights.append(0.15)

        pattern_score_val = self._score_patterns(pattern_info)
        scores.append(pattern_score_val)
        weights.append(0.10)

        total_score = sum(
            s * w for s, w in zip(scores, weights)
        )

        total_score = max(0, min(100, total_score))
        return int(round(total_score))

    def _score_trend(self, trend_result: Dict) -> float:
        direction = trend_result.get("direction", TrendDirection.SIDEWAYS.value)
        strength = trend_result.get("strength", 0)

        if direction in [TrendDirection.STRONG_BULLISH.value, TrendDirection.STRONG_BEARISH.value]:
            base = 80
        elif direction in [TrendDirection.BULLISH.value, TrendDirection.BEARISH.value]:
            base = 78
        elif direction in [TrendDirection.WEAK_BULLISH.value, TrendDirection.WEAK_BEARISH.value]:
            base = 75
        elif direction == TrendDirection.CONSOLIDATION.value:
            base = 30
        else:
            base = 20

        return base * (0.5 + 0.5 * strength)

    def _score_volatility(self, vol_result: Dict, trend_result: Dict) -> float:
        vol_level = vol_result.get("level", "low")
        trend_dir = trend_result.get("direction", TrendDirection.SIDEWAYS.value)

        if vol_level == "medium":
            return 70
        elif vol_level == "low":
            if trend_dir in [TrendDirection.STRONG_BULLISH.value, TrendDirection.STRONG_BEARISH.value]:
                return 60
            return 40
        else:
            if trend_dir in [TrendDirection.STRONG_BULLISH.value, TrendDirection.STRONG_BEARISH.value]:
                return 50
            return 20

    def _score_momentum(self, momentum_result: Dict, trend_result: Dict) -> float:
        mom_score = momentum_result.get("score", 50)
        mom_dir = momentum_result.get("direction", 0)
        trend_dir = trend_result.get("direction", 0)

        alignment = 1 if (mom_dir > 0 and "BULLISH" in trend_result.get("direction", "")) or \
                         (mom_dir < 0 and "BEARISH" in trend_result.get("direction", "")) else 0

        return min(mom_score * (0.7 + 0.3 * alignment), 100)

    def _score_regime(self, regime_result: Dict) -> float:
        regime = regime_result.get("regime", MarketRegime.SIDEWAYS.value)
        confidence = regime_result.get("confidence", 0.5)

        regime_scores = {
            MarketRegime.STRONG_TRENDING_BULLISH.value: 85,
            MarketRegime.STRONG_TRENDING_BEARISH.value: 85,
            MarketRegime.WEAK_TRENDING_BULLISH.value: 65,
            MarketRegime.WEAK_TRENDING_BEARISH.value: 65,
            MarketRegime.LOW_VOLATILITY.value: 50,
            MarketRegime.SIDEWAYS.value: 30,
            MarketRegime.CONSOLIDATION.value: 20,
            MarketRegime.HIGH_VOLATILITY.value: 30,
            MarketRegime.NEWS_DRIVEN.value: 10,
        }

        base = regime_scores.get(regime, 30)
        return base * confidence

    def _score_support_resistance(self, sr_info: Dict, trend_result: Dict) -> float:
        if not sr_info:
            return 50

        nearest_support = sr_info.get("nearest_support")
        nearest_resistance = sr_info.get("nearest_resistance")
        dist_to_support = sr_info.get("distance_to_support")
        dist_to_resistance = sr_info.get("distance_to_resistance")

        trend_dir = trend_result.get("direction", TrendDirection.SIDEWAYS.value)

        if "BULLISH" in trend_dir and nearest_resistance:
            distance_pct = dist_to_resistance / nearest_resistance if nearest_resistance > 0 else 0.01
            if distance_pct > 0.01:
                score = min(100 - distance_pct * 1000, 85)
            else:
                score = 45
        elif "BEARISH" in trend_dir and nearest_support:
            distance_pct = dist_to_support / nearest_support if nearest_support > 0 else 0.01
            if distance_pct > 0.01:
                score = min(100 - distance_pct * 1000, 85)
            else:
                score = 45
        else:
            score = 50

        return float(score)

    def _score_patterns(self, pattern_info: Dict) -> float:
        if not pattern_info:
            return 50

        candle_signal = pattern_info.get("candle_signal", 0)
        pa_pattern = pattern_info.get("price_action", "")

        score = 50

        if candle_signal == 1:
            score += 20
        elif candle_signal == -1:
            score += 20

        bullish_pa = pa_pattern in ["LIQUIDITY_GRAB", "REJECTION", "BREAKOUT"]
        bearish_pa = pa_pattern in ["FAKE_BREAKOUT", "FAKE_BREAKDOWN"]

        if bullish_pa:
            score += 15
        elif bearish_pa:
            score += 15

        return min(score, 100)
