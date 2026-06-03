from typing import Dict, Optional, Tuple

from core.config import config
from core.constants import TradeDirection
from risk.risk_manager import RiskManager
from trading.execution_engine import ExecutionEngine
from data.market_data_engine import MarketDataEngine
from utils.logger import get_logger


class EntryEngine:
    def __init__(
        self,
        risk_manager: RiskManager,
        execution_engine: ExecutionEngine,
        data_engine: MarketDataEngine,
    ):
        self.logger = get_logger("entry_engine")
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.data_engine = data_engine

    def _resolve_direction(self, action: str) -> str:
        if action in (TradeDirection.BUY.value, "WEAK_BUY"):
            return TradeDirection.BUY.value
        elif action in (TradeDirection.SELL.value, "WEAK_SELL"):
            return TradeDirection.SELL.value
        return TradeDirection.HOLD.value

    def open_trade(
        self,
        symbol: str,
        decision: Dict,
        account_info: Dict,
        df_entry,
        atr: float,
        current_price: float,
        existing_positions: list,
    ) -> Optional[Dict]:
        raw_action = decision.get("action", TradeDirection.HOLD.value)
        direction = self._resolve_direction(raw_action)
        if direction == TradeDirection.HOLD.value:
            return None

        entry_price = current_price
        sl_price, tp_price = self._calculate_sl_tp(
            direction, entry_price, atr, df_entry
        )

        trade_eval = self.risk_manager.evaluate_trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl_price,
            take_profit=tp_price,
            atr=atr,
            account_info=account_info,
            existing_positions=existing_positions,
        )

        if not trade_eval["allowed"]:
            self.logger.info(f"Trade not allowed: {'; '.join(trade_eval['reasons'])}")
            return None

        lot_size = trade_eval["lot_size"]
        if lot_size <= 0:
            self.logger.warning("Invalid lot size")
            return None

        if direction == TradeDirection.BUY.value:
            result = self.execution_engine.execute_buy(
                symbol=symbol,
                volume=lot_size,
                sl=sl_price,
                tp=tp_price,
            )
        else:
            result = self.execution_engine.execute_sell(
                symbol=symbol,
                volume=lot_size,
                sl=sl_price,
                tp=tp_price,
            )

        if result:
            result["sl"] = sl_price
            result["tp"] = tp_price
            result["entry_price"] = entry_price
            result["direction"] = direction
            result["lot_size"] = lot_size
            result["risk"] = trade_eval
            result["decision"] = decision
            result["timeframe"] = decision.get("timeframe", "M15")
            return result

        return None

    def _calculate_sl_tp(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        df,
    ) -> Tuple[float, float]:
        use_dynamic_sl = config.risk["use_dynamic_sl"]
        use_dynamic_tp = config.risk["use_dynamic_tp"]

        if use_dynamic_sl and atr > 0:
            sl_distance = atr * 1.5
            sl_price = entry_price - sl_distance if direction == "BUY" else entry_price + sl_distance
        else:
            pip_size = 0.0001
            sl_pips = config.risk["sl_pips"]
            sl_distance = sl_pips * pip_size
            sl_price = entry_price - sl_distance if direction == "BUY" else entry_price + sl_distance

        if use_dynamic_tp and atr > 0:
            tp_distance = atr * 3.0
            tp_price = entry_price + tp_distance if direction == "BUY" else entry_price - tp_distance
        else:
            pip_size = 0.0001
            tp_pips = config.risk["tp_pips"]
            tp_distance = tp_pips * pip_size
            tp_price = entry_price + tp_distance if direction == "BUY" else entry_price - tp_distance

        return sl_price, tp_price
