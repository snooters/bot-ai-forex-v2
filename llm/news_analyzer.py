import json
from typing import Dict, Optional, List
from datetime import datetime

from core.config import config
from utils.logger import get_logger


class NewsAnalyzer:
    def __init__(self):
        self.logger = get_logger("news_analyzer")
        self._enabled = config.news["enabled"]

    async def analyze_news(
        self,
        symbol: str,
    ) -> Dict:
        if not self._enabled:
            return self._neutral_result()

        try:
            impact = await self._fetch_news_impact(symbol)
            if impact:
                return impact
        except Exception as e:
            self.logger.debug(f"News analysis failed: {e}")

        return self._neutral_result()

    async def _fetch_news_impact(self, symbol: str) -> Optional[Dict]:
        base = symbol[:3]
        quote = symbol[3:]

        news_scores = {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
            "risk_events": [],
        }

        return news_scores

    def _neutral_result(self) -> Dict:
        return {
            "bullish": 0,
            "bearish": 0,
            "neutral": 1,
            "risk_events": [],
            "overall": "neutral",
            "score": 0,
        }

    def is_high_impact_news(self, news_analysis: Dict) -> bool:
        if not news_analysis:
            return False
        risk_events = news_analysis.get("risk_events", [])
        if len(risk_events) > 2:
            return True
        return False
