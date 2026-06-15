from typing import Dict, Optional, List

from core.config import config
from utils.logger import get_logger


class NoTradeEngine:
    CRITICAL = 2
    WARNING = 1
    OK = 0

    def __init__(self):
        self.logger = get_logger("no_trade_engine")
        self._reasons: List[str] = []
        self._severity: int = 0

    def should_no_trade(
        self,
        confidence: float,
        market_score: int,
        spread: float,
        news_analysis: Optional[Dict] = None,
        regime_result: Optional[Dict] = None,
        existing_positions: List = None,
        balance: float = 0,
        trend_result: Optional[Dict] = None,
        signal: Optional[str] = None,
    ) -> int:
        self._reasons = []
        self._severity = 0

        if not config.ai_filter["allow_no_trade"]:
            return 0

        # P3: Quality gate — unified threshold for all trends
        QUALITY_MIN_CONF = 0.55
        QUALITY_MIN_SCORE = 35
        if confidence < QUALITY_MIN_CONF:
            self._severity = max(self._severity, 2)
            self._reasons.append(f"Quality gate: confidence {confidence:.0%} < {QUALITY_MIN_CONF:.0%}")
        if market_score < QUALITY_MIN_SCORE:
            self._severity = max(self._severity, 2)
            self._reasons.append(f"Quality gate: market score {market_score} < {QUALITY_MIN_SCORE}")

        dynamic_min = config.get_dynamic_min_confidence(balance)
        min_conf = max(config.ai_filter["min_confidence"], dynamic_min)
        if confidence < min_conf:
            gap = min_conf - confidence
            if gap > 0.20:
                self._severity = max(self._severity, 2)
                self._reasons.append(f"Critical low confidence: {confidence:.0%} < {min_conf:.0%}")
            else:
                self._severity = max(self._severity, 1)
                self._reasons.append(f"Low confidence: {confidence:.0%} < {min_conf:.0%}")

        min_score = config.ai_filter["min_market_score"]
        if market_score < min_score:
            self._severity = max(self._severity, 2)
            self._reasons.append(f"Critical low market score: {market_score} < {min_score}")

        max_spread = config.ai_filter["max_spread_pips"]
        if spread is not None and spread > max_spread:
            self._severity = max(self._severity, 1)
            self._reasons.append(f"Spread high: {spread:.1f} pips > {max_spread:.1f} pips")

        if news_analysis:
            if news_analysis.get("overall") == "high_impact":
                self._severity = max(self._severity, 2)
                self._reasons.append("High impact news event")
            if len(news_analysis.get("risk_events", [])) > 2:
                self._severity = max(self._severity, 1)
                self._reasons.append("Multiple risk events detected")

        if regime_result:
            regime = regime_result.get("regime", "")
            vol_score = regime_result.get("volatility_score", 0)
            regime_conf = regime_result.get("confidence", 0.5)

            if regime == "NEWS_DRIVEN":
                self._severity = max(self._severity, 2)
                self._reasons.append("News-driven market - block all trades")

            if regime == "HIGH_VOLATILITY":
                if vol_score > 80:
                    self._severity = max(self._severity, 2)
                    self._reasons.append(f"Extreme volatility (vol_score={vol_score})")
                elif confidence < 0.75:
                    self._severity = max(self._severity, 2)
                    self._reasons.append(f"High volatility + low confidence ({confidence:.0%})")

            if regime in ("SIDEWAYS", "LOW_VOLATILITY"):
                if signal in ("BUY", "SELL"):
                    self._severity = max(self._severity, 1)
                    self._reasons.append(f"{regime} regime - direction signal discouraged")

            # RANGING regime is a profit killer (PF=0.07 per audit) -- block immediately
            if regime == "RANGING":
                self._severity = max(self._severity, 2)
                self._reasons.append("RANGING regime - no clear direction, blocking all trades")

            if regime in ("STRONG_TRENDING_BULLISH",):
                if signal == "SELL":
                    self._severity = max(self._severity, 2)
                    self._reasons.append("SELL signal in STRONG_TRENDING_BULLISH regime")

            if regime in ("STRONG_TRENDING_BEARISH",):
                if signal == "BUY":
                    self._severity = max(self._severity, 2)
                    self._reasons.append("BUY signal in STRONG_TRENDING_BEARISH regime")

        if existing_positions:
            max_pos = config.get_dynamic_max_positions(balance)
            if len(existing_positions) >= max_pos:
                self._severity = max(self._severity, 2)
                self._reasons.append(f"Max positions reached ({max_pos})")

        if self._reasons:
            self.logger.debug(f"NO TRADE severity={self._severity}: {'; '.join(self._reasons)}")

        return self._severity

    @property
    def reasons(self) -> List[str]:
        return self._reasons

    @property
    def severity(self) -> int:
        return self._severity

    def get_no_trade_summary(self) -> str:
        if not self._reasons:
            return ""
        return " | ".join(self._reasons)
