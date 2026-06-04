import json
from datetime import datetime
from typing import Dict, Optional

from llm.llm_client import LLMClient
from utils.logger import get_logger


class MarketAnalyst:
    def __init__(self, llm_client: LLMClient):
        self.logger = get_logger("market_analyst")
        self.llm = llm_client

    def _build_cache_key(self, symbol: str, market_data: Dict, features: Dict) -> str:
        now = datetime.now()
        slot = now.strftime("%Y-%m-%d-%H:%M")
        slot_minutes = (now.minute // 10) * 10
        time_slot = f"{now.strftime('%Y-%m-%d-%H')}:{slot_minutes:02d}"
        return f"{symbol}|{time_slot}"

    async def analyze_market(
        self,
        symbol: str,
        market_data: Dict,
        features: Dict,
    ) -> Optional[Dict]:
        if not self.llm.enabled:
            return None

        cache_key = self._build_cache_key(symbol, market_data, features)

        system_prompt = """You are a professional forex market analyst. Analyze the market data and provide:
1. Market structure assessment
2. Key support/resistance levels
3. Potential trading opportunities
4. Risk warnings
Be concise and data-driven. Output ONLY valid JSON."""

        user_prompt = f"""Analyze {symbol} forex pair:

Indicators:
- RSI: {features.get('rsi', 'N/A')}
- MACD: {features.get('macd', 'N/A')}
- ADX: {features.get('adx', 'N/A')}
- ATR: {features.get('atr', 'N/A')}

Trend: {market_data.get('trend', 'N/A')}
Regime: {market_data.get('regime', 'N/A')}
Volatility: {market_data.get('volatility', 'N/A')}

Recent Price: {market_data.get('price', 'N/A')}

Output JSON format:
{{
  "market_structure": "uptrend/downtrend/sideways",
  "key_levels": {{"support": X, "resistance": Y}},
  "bias": "bullish/bearish/neutral",
  "confidence": 0.0-1.0,
  "risk_warning": "string or null",
  "reasoning": "brief explanation"
}}"""

        try:
            content = await self.llm.query(system_prompt, user_prompt, cache_key=cache_key)
        except Exception as e:
            self.logger.warning(f"LLM market analysis skipped for {symbol}: {e}")
            return None
        if not content:
            return None

        try:
            json_start = content.index("{")
            json_end = content.rindex("}") + 1
            result = json.loads(content[json_start:json_end])
            self.llm.store_knowledge(symbol, cache_key, result)
            return result
        except (ValueError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to parse LLM market analysis: {e}")
            return None
