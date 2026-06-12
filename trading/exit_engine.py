from datetime import datetime
from typing import Dict, Optional, List

import numpy as np

from core.config import config
from core.constants import PositionAction, TradeDirection
from data.market_data_engine import MarketDataEngine
from trading.execution_engine import ExecutionEngine
from utils.logger import get_logger


class ExitEngine:
    def __init__(
        self,
        execution_engine: ExecutionEngine,
        data_engine: MarketDataEngine,
    ):
        self.logger = get_logger("exit_engine")
        self.execution_engine = execution_engine
        self.data_engine = data_engine

    def evaluate_exit(
        self,
        position: Dict,
        current_price: float,
        trend_result: Dict,
        regime_result: Dict,
        confidence: float,
        market_structure: Dict,
        atr: float = 0,
        multi_tf_trends: Dict = None,
        momentum_result: Dict = None,
        vol_result: Dict = None,
        rsi: float = 0,
    ) -> Dict:
        action = PositionAction.HOLD
        reasons = []
        close_profit_score = 0

        is_buy = position["type"] == "BUY"
        entry = position.get("price_open", 0)
        current_profit = position.get("profit", 0)

        if is_buy:
            profit_pips = (current_price - entry) / 0.0001
        else:
            profit_pips = (entry - current_price) / 0.0001

        profit_pips = max(profit_pips, 0) if current_profit > 0 else profit_pips

        # ── Trend reversal: hard close (with multi-TF confirmation) ──
        trend_reversed = False
        m5_dir = trend_result.get("direction", "")
        is_bearish = m5_dir in ["STRONG_BEARISH", "BEARISH", "WEAK_BEARISH"]
        is_bullish = m5_dir in ["STRONG_BULLISH", "BULLISH", "WEAK_BULLISH"]

        if multi_tf_trends:
            h4 = multi_tf_trends.get("trend240", 0)
            h1 = multi_tf_trends.get("trend60", 0)
            m30 = multi_tf_trends.get("trend30", 0)
            tf_against = 0
            if is_buy:
                if h4 < 0: tf_against += 1
                if h1 < 0: tf_against += 1
                if m30 < 0: tf_against += 1
                if is_bearish: tf_against += 1
            else:
                if h4 > 0: tf_against += 1
                if h1 > 0: tf_against += 1
                if m30 > 0: tf_against += 1
                if is_bullish: tf_against += 1
            trend_reversed = tf_against >= 3
        else:
            trend_reversed = (is_buy and is_bearish) or (not is_buy and is_bullish)

        if trend_reversed:
            action = PositionAction.FULL_CLOSE
            reasons.append("Trend reversed against position")

        # ── Confidence crash: close ──
        if confidence < 0.30:
            if action != PositionAction.FULL_CLOSE:
                action = PositionAction.FULL_CLOSE
                reasons.append(f"Confidence dropped to {confidence:.0%}")

        # ── Break of structure: close ──
        if market_structure:
            if market_structure.get("has_bos"):
                if action != PositionAction.FULL_CLOSE:
                    action = PositionAction.FULL_CLOSE
                    reasons.append("Break of structure detected")

        # ── News-driven: close ──
        if regime_result.get("regime") == "NEWS_DRIVEN":
            action = PositionAction.FULL_CLOSE
            reasons.append("News driven market - closing positions")

        # ── TP reached: close ──
        tp = position.get("tp", 0) or 0
        if tp > 0:
            if (position["type"] == "BUY" and current_price >= tp) or \
               (position["type"] == "SELL" and current_price <= tp):
                action = PositionAction.FULL_CLOSE
                reasons.append("Take profit reached")

        # ── No-TP profit target close (ATR-based) ──
        if not tp and atr > 0:
            entry = position.get("price_open", 0)
            profit_distance = abs(current_price - entry)
            target_distance = atr * 2.0
            if profit_distance >= target_distance:
                action = PositionAction.FULL_CLOSE
                reasons.append(f"Profit target reached ({profit_distance/atr:.1f}x ATR)")

        # ── ATR-based trailing stop ──
        current_sl = position.get("sl", 0)
        if current_profit > 0 and atr > 0:
            trailing_dist = atr * config.risk.get("trailing_atr_multiplier", 1.5)
            if is_buy:
                new_sl = current_price - trailing_dist
                if new_sl > current_sl:
                    if action == PositionAction.HOLD:
                        action = PositionAction.TRAILING_STOP
                    reasons.append(f"Trailing stop: SL -> {new_sl:.5f}")
            else:
                new_sl = current_price + trailing_dist
                if new_sl < current_sl or current_sl == 0:
                    if action == PositionAction.HOLD:
                        action = PositionAction.TRAILING_STOP
                    reasons.append(f"Trailing stop: SL -> {new_sl:.5f}")

        # ── Time-based exit ──
        open_time = position.get("time")
        if open_time:
            max_hold_seconds = config.risk.get("max_hold_hours", 12) * 3600
            elapsed = (datetime.now() - open_time).total_seconds()
            if elapsed > max_hold_seconds:
                action = PositionAction.FULL_CLOSE
                reasons.append(f"Max hold time reached ({elapsed/3600:.1f}h)")

        # ── SECURE_PROFIT: Close profitable position when TP reach probability is low ──
        if current_profit > 0 and action not in [PositionAction.FULL_CLOSE]:
            secure = self._evaluate_secure_profit(
                position=position,
                current_price=current_price,
                current_profit=current_profit,
                profit_pips=profit_pips,
                trend_result=trend_result,
                regime_result=regime_result,
                confidence=confidence,
                multi_tf_trends=multi_tf_trends or {},
                momentum_result=momentum_result or {},
                vol_result=vol_result or {},
                rsi=rsi,
                atr=atr,
                is_buy=is_buy,
                entry=entry,
            )
            if secure["should_close"]:
                action = PositionAction.FULL_CLOSE
                reasons.append(secure["reason"])
                close_profit_score = secure["score"]

        return {
            "action": action,
            "reasons": reasons,
            "close_profit_score": close_profit_score,
        }

    def _evaluate_secure_profit(
        self,
        position: Dict,
        current_price: float,
        current_profit: float,
        profit_pips: float,
        trend_result: Dict,
        regime_result: Dict,
        confidence: float,
        multi_tf_trends: Dict,
        momentum_result: Dict,
        vol_result: Dict,
        rsi: float,
        atr: float,
        is_buy: bool,
        entry: float,
    ) -> Dict:
        tp = position.get("tp", 0) or 0
        sl = position.get("sl", 0) or 0
        tp_reach_prob = 50.0
        reversal_prob = 0.0
        close_score = 0.0
        should_close = False
        reason = ""

        # ── 1. Calculate profit progress ──
        if tp > 0 and sl > 0 and entry > 0:
            tp_dist = abs(tp - current_price)
            sl_dist = abs(current_price - sl)
            total_dist = abs(tp - entry)
            if total_dist > 0:
                progress = abs(current_price - entry) / total_dist
                if tp_dist > 0:
                    tp_reach_prob = max(10, min(90, (1.0 - progress) * 70 + sl_dist / (tp_dist + sl_dist) * 30))

        entry_dist = abs(current_price - entry) if entry > 0 else 0
        atr_multiple = entry_dist / atr if atr > 0 else 0

        # ── 2. Minimum profit threshold ──
        min_profit_atr = config.trading["min_profit_atr_exit"]
        min_profit_pips = config.trading["min_profit_pips_exit"]
        if atr > 0:
            min_profit_distance = atr * min_profit_atr
            min_profit_pips = max(min_profit_pips, min_profit_distance / 0.0001)

        profit_sufficient = profit_pips >= min_profit_pips
        if not profit_sufficient:
            return {"should_close": False, "score": 0, "reason": "Profit too small to secure",
                    "tp_reach_prob": tp_reach_prob, "reversal_prob": reversal_prob}

        # ── 3. Multi-TF trend analysis ──
        h4 = multi_tf_trends.get("trend240", 0)
        h1 = multi_tf_trends.get("trend60", 0)
        m30 = multi_tf_trends.get("trend30", 0)
        m15 = multi_tf_trends.get("trend15", 0)

        trend_against = 0
        if is_buy:
            if h4 < 0: trend_against += 3
            if h1 < 0: trend_against += 2
            if m30 < 0: trend_against += 2
            if m15 < 0: trend_against += 1
        else:
            if h4 > 0: trend_against += 3
            if h1 > 0: trend_against += 2
            if m30 > 0: trend_against += 2
            if m15 > 0: trend_against += 1

        # ── 4. Momentum check ──
        momentum_weakening = False
        mom_val = momentum_result.get("direction", 0) if momentum_result else 0
        if is_buy and mom_val < 0:
            momentum_weakening = True
        elif not is_buy and mom_val > 0:
            momentum_weakening = True

        # ── 5. RSI check ──
        rsi_extreme = False
        if is_buy and rsi > 70:
            rsi_extreme = True
            close_score += 15
        elif not is_buy and rsi < 30:
            rsi_extreme = True
            close_score += 15

        # ── 6. Regime check ──
        regime_str = regime_result.get("regime", "")
        if "SIDEWAYS" in regime_str or "CONSOLIDATION" in regime_str:
            close_score += 10
        if "NEWS_DRIVEN" in regime_str:
            close_score += 20

        # ── 7. Trend against position scoring ──
        if trend_against >= 3:
            close_score += 30
        elif trend_against >= 2:
            close_score += 20
        elif trend_against >= 1:
            close_score += 10

        # ── 8. Momentum weakening ──
        if momentum_weakening:
            close_score += 10

        # ── 9. Confidence drop ──
        if confidence < 0.5:
            close_score += 15
        elif confidence < 0.6:
            close_score += 5

        # ── 10. Progress-based: more progress = more reason to secure ──
        if atr > 0 and atr_multiple >= 1.5:
            close_score += 10
        if atr > 0 and atr_multiple >= 2.0:
            close_score += 10

        # ── 11. Reversal probability estimate ──
        reversal_prob = min(100, trend_against * 15 + (20 if momentum_weakening else 0) + (15 if rsi_extreme else 0) + (10 if confidence < 0.5 else 0))
        tp_reach_prob = max(0, 100 - reversal_prob - close_score * 0.3)

        # ── 12. Decision ──
        secure_threshold = config.trading["secure_close_threshold"]

        if close_score >= secure_threshold and profit_sufficient:
            should_close = True
            reason_parts = []
            if trend_against >= 2:
                reason_parts.append(f"{trend_against} TFs against position")
            if momentum_weakening:
                reason_parts.append("momentum fading")
            if rsi_extreme:
                reason_parts.append(f"RSI={rsi:.0f} extreme")
            if confidence < 0.5:
                reason_parts.append(f"low confidence ({confidence:.0%})")
            if atr_multiple > 0:
                reason_parts.append(f"profit={profit_pips:.0f}pips ({atr_multiple:.1f}x ATR)")
            reason = f"SECURE_PROFIT: {', '.join(reason_parts)} [score={close_score:.0f}, TP_prob={tp_reach_prob:.0f}%, reversal={reversal_prob:.0f}%]"

        elif profit_pips >= min_profit_pips * 3 and reversal_prob >= 60:
            should_close = True
            reason = f"SECURE_PROFIT: high reversal risk ({reversal_prob:.0f}%) with {profit_pips:.0f}pips profit"

        return {
            "should_close": should_close,
            "score": close_score,
            "reason": reason,
            "tp_reach_prob": tp_reach_prob,
            "reversal_prob": reversal_prob,
        }

    def close_position(self, position: Dict) -> bool:
        result = self.execution_engine.close_position(
            ticket=position["ticket"],
            symbol=position["symbol"],
            volume=position["volume"],
            position_type=position["type"],
        )
        if result:
            self.logger.info(f"Closed position {position['ticket']}: {position['type']} "
                             f"{position['symbol']} profit={position.get('profit', 0):.2f}")
        return result

    def modify_stop_loss(self, position: Dict, new_sl: float) -> bool:
        return self.execution_engine.modify_position(
            ticket=position["ticket"],
            sl=new_sl,
        )