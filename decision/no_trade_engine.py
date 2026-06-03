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
    ) -> int:
        self._reasons = []
        self._severity = 0

        if not config.ai_filter["allow_no_trade"]:
            return 0

        min_conf = config.ai_filter["min_confidence"] * 100
        if confidence < min_conf:
            gap = min_conf - confidence
            if gap > 20:
                self._severity = max(self._severity, 2)
                self._reasons.append(f"Critical low confidence: {confidence:.1f}% < {min_conf:.0f}%")
            else:
                self._severity = max(self._severity, 1)
                self._reasons.append(f"Low confidence: {confidence:.1f}% < {min_conf:.0f}%")

        min_score = config.ai_filter["min_market_score"]
        if market_score < min_score:
            gap = min_score - market_score
            if gap > 20:
                self._severity = max(self._severity, 2)
                self._reasons.append(f"Critical low market score: {market_score} < {min_score}")
            else:
                self._severity = max(self._severity, 1)
                self._reasons.append(f"Low market score: {market_score} < {min_score}")

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
            if regime == "NEWS_DRIVEN":
                self._severity = max(self._severity, 1)
                self._reasons.append("News-driven market - high uncertainty")
            if regime == "HIGH_VOLATILITY":
                vol_score = regime_result.get("volatility_score", 0)
                if vol_score > 85:
                    self._severity = max(self._severity, 2)
                    self._reasons.append("Extreme volatility detected")

        if existing_positions:
            max_pos = config.risk["max_open_positions"]
            if len(existing_positions) >= max_pos:
                self._severity = max(self._severity, 2)
                self._reasons.append(f"Max positions reached ({max_pos})")

        if self._reasons:
            self.logger.info(f"NO TRADE severity={self._severity}: {'; '.join(self._reasons)}")

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
