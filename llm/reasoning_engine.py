import json
from typing import Dict, Optional, List

from llm.llm_client import LLMClient
from utils.logger import get_logger


class ReasoningEngine:
    def __init__(self, llm_client: LLMClient):
        self.logger = get_logger("reasoning_engine")
        self.llm = llm_client

    async def explain_trade_decision(
        self,
        symbol: str,
        decision: Dict,
        market_analysis: Dict,
        risk_info: Dict,
    ) -> Optional[Dict]:
        if not self.llm.enabled:
            return None

        system_prompt = """You are a trade explanation engine. Explain the AI trading decision in clear terms.
Focus on the rationale, risk considerations, and market context."""

        user_prompt = f"""Explain this trade decision for {symbol}:

Decision: {json.dumps(decision, indent=2)}
Market Analysis: {json.dumps(market_analysis, indent=2)}
Risk Info: {json.dumps(risk_info, indent=2)}

Output JSON:
{{
  "summary": "2-3 sentence explanation",
  "key_factors": ["factor1", "factor2", ...],
  "risks": ["risk1", "risk2", ...],
  "confidence_assessment": "high/medium/low"
}}"""

        content = await self.llm.query(system_prompt, user_prompt)
        if not content:
            return None

        try:
            json_start = content.index("{")
            json_end = content.rindex("}") + 1
            return json.loads(content[json_start:json_end])
        except (ValueError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to parse reasoning: {e}")
            return None
