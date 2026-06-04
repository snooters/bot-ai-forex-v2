from datetime import datetime
from typing import Dict, Optional, List

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
    ) -> PositionAction:
        action = PositionAction.HOLD
        reasons = []

        if trend_result.get("direction") in ["STRONG_BEARISH", "BEARISH", "WEAK_BEARISH"] and position["type"] == "BUY":
            action = PositionAction.FULL_CLOSE
            reasons.append("Trend reversed against position")

        if trend_result.get("direction") in ["STRONG_BULLISH", "BULLISH", "WEAK_BULLISH"] and position["type"] == "SELL":
            action = PositionAction.FULL_CLOSE
            reasons.append("Trend reversed against position")

        if confidence < 0.30:
            if action != PositionAction.FULL_CLOSE:
                action = PositionAction.FULL_CLOSE
                reasons.append(f"Confidence dropped to {confidence:.0%}")

        if market_structure:
            if market_structure.get("has_bos"):
                if action != PositionAction.FULL_CLOSE:
                    action = PositionAction.FULL_CLOSE
                    reasons.append("Break of structure detected")

        if regime_result.get("regime") == "NEWS_DRIVEN":
            action = PositionAction.FULL_CLOSE
            reasons.append("News driven market - closing positions")

        # ── TP-based auto close ──
        tp = position.get("tp", 0) or 0
        if tp > 0:
            if (position["type"] == "BUY" and current_price >= tp) or \
               (position["type"] == "SELL" and current_price <= tp):
                action = PositionAction.FULL_CLOSE
                reasons.append("Take profit reached")

        # ── No-TP profit target close ──
        if not tp and atr > 0:
            entry = position.get("price_open", 0)
            profit_distance = abs(current_price - entry)
            target_distance = atr * 2.0
            if profit_distance >= target_distance:
                action = PositionAction.FULL_CLOSE
                reasons.append(f"Profit target reached ({profit_distance/atr:.1f}× ATR)")

        # ── ATR-based trailing stop ──
        current_profit = position.get("profit", 0)
        if current_profit > 0:
            trailing_dist = atr * config.risk.get("trailing_atr_multiplier", 1.5) if atr > 0 else current_price * 0.001
            current_sl = position.get("sl", 0)
            if position["type"] == "BUY":
                new_sl = current_price - trailing_dist
                if new_sl > current_sl:
                    action = PositionAction.TRAILING_STOP
                    reasons.append(f"Trailing stop moved to {new_sl:.5f}")
            else:
                new_sl = current_price + trailing_dist
                if new_sl < current_sl or current_sl == 0:
                    action = PositionAction.TRAILING_STOP
                    reasons.append(f"Trailing stop moved to {new_sl:.5f}")

        # ── Time-based exit ──
        open_time = position.get("time")
        if open_time:
            max_hold_seconds = config.risk.get("max_hold_hours", 12) * 3600
            elapsed = (datetime.now() - open_time).total_seconds()
            if elapsed > max_hold_seconds:
                action = PositionAction.FULL_CLOSE
                reasons.append(f"Max hold time reached ({elapsed/3600:.1f}h)")

        return action

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
