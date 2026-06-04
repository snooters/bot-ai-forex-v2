import numpy as np
from typing import Dict, List, Optional

from core.config import config
from utils.logger import get_logger


class ConfidenceCalculator:
    def __init__(self):
        self.logger = get_logger("confidence_calculator")

    def calculate_confidence(
        self,
        ml_signal: Dict,
        market_score: int,
        trend_result: Dict,
        regime_result: Dict,
        sr_info: Dict,
        news_analysis: Optional[Dict] = None,
        llm_analysis: Optional[Dict] = None,
    ) -> float:
        if not ml_signal:
            return 0.0

        signal = ml_signal.get("signal", "HOLD")
        ml_confidence = ml_signal.get("confidence", 0)
        buy_prob = ml_signal.get("buy_prob", 0)
        sell_prob = ml_signal.get("sell_prob", 0)
        hold_prob = ml_signal.get("hold_prob", 0)

        ml_weight = 0.50
        market_weight = 0.30
        pattern_weight = 0.10
        news_weight = config.news["weight"]
        llm_weight = config.llm["weight"]

        ml_score = ml_confidence

        market_score_normalized = min(market_score / 100.0, 1.0)
        alignment = self._check_alignment(ml_signal, trend_result)

        pattern_score = self._score_patterns(sr_info)

        combined = (
            ml_score * ml_weight +
            market_score_normalized * market_weight +
            alignment * 0.05 +
            pattern_score * pattern_weight
        )

        if news_analysis:
            news_score = max(
                news_analysis.get("bullish", 0),
                news_analysis.get("bearish", 0),
                news_analysis.get("neutral", 0)
            )
            combined += news_score * news_weight

        if llm_analysis:
            llm_conf = llm_analysis.get("confidence", 0.5)
            combined += llm_conf * llm_weight

        if signal == "HOLD" or (hold_prob > 50 and buy_prob < 25 and sell_prob < 25):
            combined *= 0.85
        elif buy_prob < 20 and sell_prob < 20:
            combined *= 0.80
        elif max(buy_prob, sell_prob) > 60:
            combined *= 1.10

        if regime_result.get("regime") in ["NEWS_DRIVEN", "HIGH_VOLATILITY"]:
            combined *= 0.85

        if signal != "HOLD":
            spread = buy_prob - sell_prob if signal == "BUY" else sell_prob - buy_prob
            if spread > 15:
                combined *= 1.05

        confidence = min(max(combined, 0), 1.0)
        return confidence

    def _check_alignment(self, ml_signal: Dict, trend_result: Dict) -> float:
        signal = ml_signal.get("signal", "HOLD")
        trend_dir = trend_result.get("direction", "")

        if signal == "BUY" and "BULLISH" in trend_dir:
            return 1.0
        elif signal == "SELL" and "BEARISH" in trend_dir:
            return 1.0
        elif signal == "HOLD":
            return 0.5
        else:
            return 0.3

    def _score_patterns(self, sr_info: Dict) -> float:
        if not sr_info:
            return 0.5
        dist_to_support = sr_info.get("distance_to_support")
        dist_to_resistance = sr_info.get("distance_to_resistance")
        if dist_to_support and dist_to_resistance:
            total_dist = dist_to_support + dist_to_resistance
            if total_dist > 0:
                return min(dist_to_support / total_dist * 2, 1.0)
        return 0.5

    def is_tradeable(self, confidence: float) -> bool:
        return confidence >= config.ai_filter["min_confidence"]
