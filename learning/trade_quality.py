from typing import Dict, List, Optional
from datetime import datetime

import numpy as np

from utils.logger import get_logger


class TradeQualityScorer:
    def __init__(self):
        self.logger = get_logger("trade_quality")

    def score_trade(self, trade: Dict) -> int:
        score = 0
        score += self._score_entry_quality(trade)
        score += self._score_exit_quality(trade)
        score += self._score_risk_reward(trade)
        return max(0, min(int(score), 100))

    def _score_entry_quality(self, trade: Dict) -> int:
        score = 0
        confidence = trade.get("confidence", 0) or 0
        market_score = trade.get("market_score", 0) or 0

        if confidence >= 0.80:
            score += 20
        elif confidence >= 0.65:
            score += 14
        elif confidence >= 0.50:
            score += 8
        elif confidence >= 0.35:
            score += 3

        if market_score >= 80:
            score += 15
        elif market_score >= 60:
            score += 10
        elif market_score >= 40:
            score += 5
        elif market_score >= 20:
            score += 2

        mc = trade.get("market_conditions", {})
        trend = mc.get("trend", "")
        direction = trade.get("direction", "")
        if (direction == "BUY" and "BULLISH" in trend) or (direction == "SELL" and "BEARISH" in trend):
            score += 10
        elif "SIDEWAYS" in trend:
            score += 2

        regime = mc.get("regime", "")
        if "STRONG_TRENDING" in regime:
            score += 10
        elif "WEAK_TRENDING" in regime:
            score += 5
        elif "SIDEWAYS" in regime or "LOW_VOLATILITY" in regime:
            score += 2
        elif "NEWS" in regime or "HIGH_VOLATILITY" in regime:
            score += 0

        return min(score, 55)

    def _score_exit_quality(self, trade: Dict) -> int:
        score = 0
        exit_reason = trade.get("exit_reason", "")

        if exit_reason == "TP_HIT" or exit_reason.startswith("tp"):
            score += 25
        elif exit_reason == "TRAILING_STOP":
            score += 20
        elif exit_reason == "PARTIAL_CLOSE":
            score += 15
        elif exit_reason == "FULL_CLOSE":
            profit = trade.get("profit", 0) or 0
            if profit > 0:
                score += 12
            else:
                score += 5

        if exit_reason == "SL_HIT" or exit_reason.startswith("sl"):
            score += 2

        if exit_reason == "TIME_BASED":
            profit = trade.get("profit", 0) or 0
            if profit > 0:
                score += 8
            else:
                score += 0

        profit = trade.get("profit", 0) or 0
        profit_pips = trade.get("profit_pips", 0) or 0
        if profit > 0:
            score += 5

        if profit_pips > 0:
            if profit_pips >= 50:
                score += 10
            elif profit_pips >= 20:
                score += 7
            elif profit_pips >= 10:
                score += 4
            elif profit_pips > 0:
                score += 2

        return min(score, 40)

    def _score_risk_reward(self, trade: Dict) -> int:
        score = 0
        entry_price = trade.get("entry_price", 0) or 0
        exit_price = trade.get("exit_price", 0) or 0
        sl_price = trade.get("stop_loss", 0) or 0
        tp_price = trade.get("take_profit", 0) or 0
        direction = trade.get("direction", "")
        volume = trade.get("volume", 0) or 0

        if entry_price == 0 or sl_price == 0:
            return 5

        if direction == "BUY":
            sl_dist = entry_price - sl_price
            tp_dist = tp_price - entry_price if tp_price > 0 else exit_price - entry_price
        else:
            sl_dist = sl_price - entry_price
            tp_dist = entry_price - tp_price if tp_price > 0 else entry_price - exit_price

        if sl_dist <= 0:
            return 5

        realized_rr = tp_dist / sl_dist if sl_dist > 0 else 0

        if realized_rr >= 3.0:
            score += 5
        elif realized_rr >= 2.0:
            score += 4
        elif realized_rr >= 1.5:
            score += 3
        elif realized_rr >= 1.0:
            score += 2
        elif realized_rr > 0:
            score += 1

        if volume > 0:
            score += 2

        profit = trade.get("profit", 0) or 0
        if profit > 0 and sl_dist > 0:
            risk_amount = sl_dist * volume * 100000
            if risk_amount > 0:
                risk_ratio = profit / risk_amount
                if risk_ratio >= 3.0:
                    score += 5
                elif risk_ratio >= 2.0:
                    score += 4
                elif risk_ratio >= 1.0:
                    score += 3

        return min(score, 10)
