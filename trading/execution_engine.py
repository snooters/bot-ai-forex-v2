from typing import Dict, Optional, Tuple

from core.config import config
from core.exceptions import ExecutionError, OrderRejectedError
from data.mt5_connector import MT5Connector
from utils.logger import get_logger


class ExecutionEngine:
    def __init__(self):
        self.logger = get_logger("execution_engine")
        self.connector = MT5Connector()

    def execute_buy(
        self,
        symbol: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "AI_FOREX_V2",
    ) -> Optional[Dict]:
        try:
            result = self.connector.place_order(
                symbol=symbol,
                order_type="BUY",
                volume=volume,
                sl=sl,
                tp=tp,
                comment=comment,
            )
            self.logger.info(f"BUY executed: {volume} {symbol} at {result.get('price')}, "
                             f"ticket={result.get('ticket')}")
            return result
        except OrderRejectedError as e:
            self.logger.error(f"BUY rejected: {e}")
            return None
        except Exception as e:
            self.logger.error(f"BUY execution failed: {e}")
            return None

    def execute_sell(
        self,
        symbol: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "AI_FOREX_V2",
    ) -> Optional[Dict]:
        try:
            result = self.connector.place_order(
                symbol=symbol,
                order_type="SELL",
                volume=volume,
                sl=sl,
                tp=tp,
                comment=comment,
            )
            self.logger.info(f"SELL executed: {volume} {symbol} at {result.get('price')}, "
                             f"ticket={result.get('ticket')}")
            return result
        except OrderRejectedError as e:
            self.logger.error(f"SELL rejected: {e}")
            return None
        except Exception as e:
            self.logger.error(f"SELL execution failed: {e}")
            return None

    def close_position(
        self,
        ticket: int,
        symbol: str,
        volume: float,
        position_type: str,
    ) -> bool:
        return self.connector.close_position(ticket, symbol, volume, position_type)

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        return self.connector.modify_position(ticket, sl, tp)

    def close_all(self):
        self.connector.close_all_positions()

    def get_positions(self, symbol: str = "") -> list:
        return self.connector.get_positions(symbol)
