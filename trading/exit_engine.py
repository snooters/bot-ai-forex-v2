from typing import Dict, Optional, List

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
    ) -> PositionAction:
        action = PositionAction.HOLD
        reasons = []

        if trend_result.get("direction") in ["STRONG_BEARISH", "WEAK_BEARISH"] and position["type"] == "BUY":
            action = PositionAction.FULL_CLOSE
            reasons.append("Trend reversed against position")

        if trend_result.get("direction") in ["STRONG_BULLISH", "WEAK_BULLISH"] and position["type"] == "SELL":
            action = PositionAction.FULL_CLOSE
            reasons.append("Trend reversed against position")

        if confidence < 30:
            if action != PositionAction.FULL_CLOSE:
                action = PositionAction.FULL_CLOSE
                reasons.append(f"Confidence dropped to {confidence:.1f}%")

        if market_structure:
            if market_structure.get("has_bos"):
                if action != PositionAction.FULL_CLOSE:
                    action = PositionAction.FULL_CLOSE
                    reasons.append("Break of structure detected")

        if regime_result.get("regime") == "NEWS_DRIVEN":
            action = PositionAction.FULL_CLOSE
            reasons.append("News driven market - closing positions")

        current_profit = position.get("profit", 0)
        if current_profit > 0:
            trailing_distance = current_price * 0.001
            current_sl = position.get("sl", 0)
            if position["type"] == "BUY":
                new_sl = current_price - trailing_distance
                if new_sl > current_sl:
                    action = PositionAction.TRAILING_STOP
                    reasons.append(f"Trailing stop moved to {new_sl:.5f}")
            else:
                new_sl = current_price + trailing_distance
                if new_sl < current_sl or current_sl == 0:
                    action = PositionAction.TRAILING_STOP
                    reasons.append(f"Trailing stop moved to {new_sl:.5f}")

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
