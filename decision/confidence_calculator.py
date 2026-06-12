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
        pair_skill_score: Optional[float] = None,
    ) -> float:
        if not ml_signal:
            return 0.0

        signal = ml_signal.get("signal", "HOLD")
        ml_confidence = ml_signal.get("confidence", 0)
        buy_prob = ml_signal.get("buy_prob", 0)
        sell_prob = ml_signal.get("sell_prob", 0)
        hold_prob = ml_signal.get("hold_prob", 0)

        ml_weight = 0.35
        market_weight = 0.45
        pattern_weight = 0.10
        news_weight = config.news["weight"]
        llm_weight = config.llm["weight"]

        ml_score = ml_confidence

        market_score_normalized = min(market_score / 100.0, 1.0)
        alignment = self._check_alignment(ml_signal, trend_result)

        pattern_score = self._score_patterns(sr_info, signal)

        combined = (
            ml_score * ml_weight +
            market_score_normalized * market_weight +
            alignment * 0.05 +
            pattern_score * pattern_weight
        )

        if news_analysis:
            if signal in ("BUY", "WEAK_BUY"):
                news_score = news_analysis.get("bullish", 0)
            elif signal in ("SELL", "WEAK_SELL"):
                news_score = news_analysis.get("bearish", 0)
            else:
                news_score = news_analysis.get("neutral", 0)
            combined += news_score * news_weight

        if llm_analysis:
            llm_conf = llm_analysis.get("confidence", 0.5)
            combined += llm_conf * llm_weight

        if signal == "HOLD" or (hold_prob > 0.50 and buy_prob < 0.25 and sell_prob < 0.25):
            combined *= 0.92
        elif buy_prob < 0.20 and sell_prob < 0.20:
            combined *= 0.80
        elif max(buy_prob, sell_prob) > 0.60:
            combined *= 1.10

        regime_str = regime_result.get("regime", "")
        regime_conf = regime_result.get("confidence", 0.5)

        if regime_str == "NEWS_SHOCK":
            return 0.0
        elif regime_str == "RANGING":
            combined *= 0.7 * regime_conf + 0.3
        elif regime_str == "VOLATILE":
            combined *= 0.8 * regime_conf + 0.2
        elif regime_str in ("NEWS_DRIVEN", "HIGH_VOLATILITY"):
            combined *= 0.85
        elif "TRENDING_BULLISH" in regime_str and signal in ("BUY", "WEAK_BUY"):
            buy_bonus = 0.05 * regime_conf
            combined = min(combined + buy_bonus, 1.0)
        elif "TRENDING_BEARISH" in regime_str and signal in ("SELL", "WEAK_SELL"):
            sell_bonus = 0.05 * regime_conf
            combined = min(combined + sell_bonus, 1.0)

        # ── Trend alignment boost ──
        trend_dir = trend_result.get("direction", "")
        if market_score_normalized >= 0.15:
            if "BULLISH" in trend_dir:
                if signal in ("BUY", "WEAK_BUY"):
                    combined += 0.10
                elif signal == "HOLD":
                    combined += 0.05
            elif "BEARISH" in trend_dir:
                if signal in ("SELL", "WEAK_SELL"):
                    combined += 0.10
                elif signal == "HOLD":
                    combined += 0.05

        if signal != "HOLD":
            spread = buy_prob - sell_prob if signal == "BUY" else sell_prob - buy_prob
            if spread > 0.15:
                combined *= 1.05

        # ── Pair skill score adjustment ──
        if pair_skill_score is not None:
            multiplier = 0.85 + (min(max(pair_skill_score, 0), 100) / 100.0) * 0.30
            combined *= multiplier

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

    def _score_patterns(self, sr_info: Dict, signal: str = "HOLD") -> float:
        if not sr_info:
            return 0.5
        dist_to_support = sr_info.get("distance_to_support")
        dist_to_resistance = sr_info.get("distance_to_resistance")
        if dist_to_support and dist_to_resistance:
            total_dist = dist_to_support + dist_to_resistance
            if total_dist > 0:
                raw = min(dist_to_support / total_dist * 2, 1.0)
                if signal in ("BUY", "WEAK_BUY"):
                    return 1.0 - raw
                elif signal in ("SELL", "WEAK_SELL"):
                    return raw
                return 0.5
        return 0.5

    def is_tradeable(self, confidence: float) -> bool:
        return confidence >= config.ai_filter["min_confidence"]
