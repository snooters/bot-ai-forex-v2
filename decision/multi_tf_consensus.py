from typing import Dict, Optional

from core.constants import TF_VOTE_WEIGHTS, TradeDirection
from utils.logger import get_logger


class MultiTFConsensus:
    def __init__(self):
        self.logger = get_logger("multi_tf_consensus")
        self._prev_signal: Dict[str, str] = {}

    def compute(
        self,
        ml_signals: Dict[int, Dict],
        trend_results: Dict[int, Dict],
        symbol: str = "",
    ) -> Dict:
        if not ml_signals:
            return {"signal": "HOLD", "buy_prob": 0, "sell_prob": 0, "hold_prob": 1,
                    "confidence": 0, "agreement": "NONE", "dominant_trend": "SIDEWAYS"}

        weighted_buy = 0.0
        weighted_sell = 0.0
        weighted_hold = 0.0
        total_weight = 0.0
        tf_votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        tf_consensus = []

        for tf in sorted(ml_signals.keys()):
            sig = ml_signals[tf]
            if not sig:
                continue
            w = TF_VOTE_WEIGHTS.get(tf, 1.0)
            total_weight += w
            weighted_buy += sig.get("buy_prob", 0) * w
            weighted_sell += sig.get("sell_prob", 0) * w
            weighted_hold += sig.get("hold_prob", 0) * w

            vote = sig.get("signal", "HOLD")
            tf_votes[vote] = tf_votes.get(vote, 0) + 1
            tf_consensus.append((tf, vote))

        if total_weight == 0:
            return {"signal": "HOLD", "buy_prob": 0, "sell_prob": 0, "hold_prob": 1,
                    "confidence": 0, "agreement": "NONE", "dominant_trend": "SIDEWAYS"}

        consensus_buy = weighted_buy / total_weight
        consensus_sell = weighted_sell / total_weight
        consensus_hold = weighted_hold / total_weight

        if consensus_buy > consensus_sell and consensus_buy > consensus_hold:
            signal = "BUY"
            confidence = consensus_buy
        elif consensus_sell > consensus_buy and consensus_sell > consensus_hold:
            signal = "SELL"
            confidence = consensus_sell
        else:
            signal = "HOLD"
            confidence = consensus_hold

        # P1: minimum margin >= 10% to avoid noise trades
        probs = sorted([consensus_buy, consensus_sell, consensus_hold], reverse=True)
        margin = probs[0] - probs[1]
        if signal != "HOLD" and margin < 0.10:
            signal = "HOLD"
            confidence = probs[2]

        max_votes = max(tf_votes.values())
        total_votes = sum(tf_votes.values())
        if max_votes == total_votes and total_votes >= 3:
            agreement = "HIGH"
        elif max_votes >= 2:
            agreement = "MEDIUM"
        else:
            agreement = "LOW"

        dominant_trend = self._dominant_trend(trend_results)

        trend_aligned = self._trend_aligned(signal, dominant_trend)
        if not trend_aligned and signal != "HOLD":
            if agreement != "HIGH":
                signal = "HOLD"
                confidence = 0

        # P6: consecutive confirmation — same signal 2+ times in a row
        prev = self._prev_signal.get(symbol, "")
        if signal != "HOLD" and signal != prev:
            signal = "HOLD"
            confidence = 0
            self.logger.info(f"Consensus: {symbol} {signal} != prev {prev} — need consecutive confirmation")
        elif signal != "HOLD":
            self.logger.info(f"Consensus: {symbol} {signal} confirmed x2 — trade allowed")
        self._prev_signal[symbol] = signal

        return {
            "signal": signal,
            "buy_prob": consensus_buy,
            "sell_prob": consensus_sell,
            "hold_prob": consensus_hold,
            "confidence": confidence,
            "agreement": agreement,
            "dominant_trend": dominant_trend,
            "trend_aligned": trend_aligned,
            "tf_votes": tf_votes,
        }

    def _dominant_trend(self, trend_results: Dict[int, Dict]) -> str:
        bullish = 0
        bearish = 0
        sideways = 0
        for tf in sorted(trend_results.keys()):
            tr = trend_results.get(tf, {})
            d = tr.get("direction", "SIDEWAYS")
            w = TF_VOTE_WEIGHTS.get(tf, 1.0)
            if "BULLISH" in d:
                bullish += w
            elif "BEARISH" in d:
                bearish += w
            else:
                sideways += w
        if bullish > bearish and bullish > sideways:
            return "BULLISH"
        elif bearish > bullish and bearish > sideways:
            return "BEARISH"
        return "SIDEWAYS"

    def _trend_aligned(self, signal: str, dominant_trend: str) -> bool:
        if signal in ("BUY", "WEAK_BUY") and dominant_trend == "BULLISH":
            return True
        if signal in ("SELL", "WEAK_SELL") and dominant_trend == "BEARISH":
            return True
        if signal == "HOLD":
            return True
        return False
