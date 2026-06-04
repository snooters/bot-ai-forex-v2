from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

from core.constants import REVERSAL_MAX_TREND_CHANGE_CANDLES, TF_VOTE_WEIGHTS
from utils.logger import get_logger


class TrendReversalDetector:
    NONE = 0
    WARNING = 1
    CRITICAL = 2

    def __init__(self):
        self.logger = get_logger("trend_reversal")
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

    def check(
        self,
        symbol: str,
        trend_results: Dict[int, Dict],
        current_price: float,
        sr_levels: Optional[Dict] = None,
    ) -> Dict:
        record = {
            "trends": {tf: trend_results.get(tf, {}).get("direction", "SIDEWAYS")
                       for tf in sorted(trend_results.keys())},
            "price": current_price,
        }
        self._history[symbol].append(record)

        severity = self.NONE
        reasons = []
        from_trend = ""
        to_trend = ""

        m5_flip = self._check_tf_flip(symbol, 5, lookback=3)
        m15_flip = self._check_tf_flip(symbol, 15, lookback=2)
        m30_flip = self._check_tf_flip(symbol, 30, lookback=1)
        price_break = self._check_price_break(symbol, current_price, sr_levels)

        if m5_flip["flipped"] and m15_flip["flipped"]:
            severity = self.CRITICAL
            from_trend = m15_flip["from"]
            to_trend = m15_flip["to"]
            reasons.append(f"M5+M15 trend reversal: {from_trend} -> {to_trend}")
        elif m5_flip["flipped"] and m30_flip["flipped"]:
            severity = self.CRITICAL
            from_trend = m30_flip["from"]
            to_trend = m30_flip["to"]
            reasons.append(f"M5+M30 trend reversal: {from_trend} -> {to_trend}")
        elif price_break:
            severity = self.CRITICAL
            reasons.append(f"Price broke S/R: {price_break}")
        elif m5_flip["flipped"]:
            severity = self.WARNING
            from_trend = m5_flip["from"]
            to_trend = m5_flip["to"]
            reasons.append(f"M5 trend change: {from_trend} -> {to_trend}")
        elif m15_flip["flipped"]:
            severity = self.WARNING
            from_trend = m15_flip["from"]
            to_trend = m15_flip["to"]
            reasons.append(f"M15 trend change: {from_trend} -> {to_trend}")

        return {
            "severity": severity,
            "reversal_detected": severity > self.NONE,
            "from_trend": from_trend,
            "to_trend": to_trend,
            "reason": "; ".join(reasons) if reasons else "",
        }

    def _check_tf_flip(self, symbol: str, tf: int, lookback: int) -> Dict:
        history = list(self._history.get(symbol, []))
        if len(history) < lookback + 1:
            return {"flipped": False, "from": "", "to": ""}

        current = history[-1]["trends"].get(tf, "SIDEWAYS")
        past = history[-(lookback + 1)]["trends"].get(tf, "SIDEWAYS")

        current_bull = "BULLISH" in current
        current_bear = "BEARISH" in current
        past_bull = "BULLISH" in past
        past_bear = "BEARISH" in past

        if (past_bull and current_bear) or (past_bear and current_bull):
            return {"flipped": True, "from": past, "to": current}
        return {"flipped": False, "from": past, "to": current}

    def _check_price_break(
        self,
        symbol: str,
        current_price: float,
        sr_levels: Optional[Dict],
    ) -> str:
        if not sr_levels:
            return ""
        nearest_support = sr_levels.get("nearest_support")
        nearest_resistance = sr_levels.get("nearest_resistance")
        if nearest_support and current_price < nearest_support:
            return f"Broke support {nearest_support:.5f}"
        if nearest_resistance and current_price > nearest_resistance:
            return f"Broke resistance {nearest_resistance:.5f}"
        return ""
