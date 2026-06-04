from typing import Dict, Optional, List

from core.config import config
from core.constants import PositionAction
from trading.execution_engine import ExecutionEngine
from trading.exit_engine import ExitEngine
from data.market_data_engine import MarketDataEngine
from utils.logger import get_logger


class PositionManager:
    def __init__(
        self,
        execution_engine: ExecutionEngine,
        exit_engine: ExitEngine,
        data_engine: MarketDataEngine,
    ):
        self.logger = get_logger("position_manager")
        self.execution_engine = execution_engine
        self.exit_engine = exit_engine
        self.data_engine = data_engine
        self._open_positions: List[Dict] = []

    def refresh_positions(self, symbol: str = ""):
        self._open_positions = self.execution_engine.get_positions(symbol)

    def get_open_positions(self, symbol: str = "") -> List[Dict]:
        self.refresh_positions(symbol)
        return self._open_positions

    def has_open_positions(self, symbol: str = "") -> bool:
        positions = self.get_open_positions(symbol)
        return len(positions) > 0

    def manage_positions(
        self,
        symbol: str,
        trend_result: Dict,
        regime_result: Dict,
        confidence: float,
        market_structure: Dict,
        reversal_force_close: bool = False,
        atr: float = 0,
        balance: float = 0,
    ) -> List[Dict]:
        actions_taken = []
        positions = self.get_open_positions(symbol)

        for position in positions:
            current_price = self._get_current_price(symbol, position["type"])
            if current_price is None:
                continue

            if reversal_force_close:
                result = self.exit_engine.close_position(position)
                actions_taken.append({
                    "ticket": position["ticket"],
                    "action": "FULL_CLOSE",
                    "result": result,
                })
                continue

            # ── Breakeven SL (Langkah 4): SL = entry + spread after 1 ATR move ──
            if atr > 0:
                entry = position.get("price_open", 0)
                is_buy = position["type"] == "BUY"
                price_move = (current_price - entry) if is_buy else (entry - current_price)
                current_sl = position.get("sl")
                pip_size = 0.0001
                spread = self.data_engine.get_current_spread(symbol) or 10
                be_buffer = spread * pip_size * 0.5  # half spread as buffer
                be_price = entry + be_buffer if is_buy else entry - be_buffer

                if price_move >= atr:
                    if is_buy and (current_sl is None or current_sl < be_price):
                        self.execution_engine.modify_position(position["ticket"], sl=be_price)
                        self.logger.info(f"Breakeven SL set for ticket {position['ticket']} at {be_price:.5f}")
                    elif not is_buy and (current_sl is None or current_sl > be_price):
                        self.execution_engine.modify_position(position["ticket"], sl=be_price)
                        self.logger.info(f"Breakeven SL set for ticket {position['ticket']} at {be_price:.5f}")

                # ── Trailing stop for runner (Langkah 5, balance >= $500) ──
                if balance >= 500 and price_move >= atr * 2.0 and position.get("comment", "") == "AI_FOREX_V2_RUN":
                    trail_dist = atr * 1.5
                    if is_buy:
                        new_sl = current_price - trail_dist
                        if new_sl > (current_sl or 0):
                            self.execution_engine.modify_position(position["ticket"], sl=new_sl)
                            self.logger.info(f"Runner trail: ticket {position['ticket']} SL → {new_sl:.5f}")
                    else:
                        new_sl = current_price + trail_dist
                        if new_sl < (current_sl or float("inf")):
                            self.execution_engine.modify_position(position["ticket"], sl=new_sl)
                            self.logger.info(f"Runner trail: ticket {position['ticket']} SL → {new_sl:.5f}")

            action = self.exit_engine.evaluate_exit(
                position=position,
                current_price=current_price,
                trend_result=trend_result,
                regime_result=regime_result,
                confidence=confidence,
                market_structure=market_structure,
                atr=atr,
            )

            result = self._execute_action(position, action, current_price, atr)
            actions_taken.append({
                "ticket": position["ticket"],
                "action": action.value,
                "result": result,
            })

        self.refresh_positions(symbol)
        return actions_taken

    def _execute_action(self, position: Dict, action: PositionAction, current_price: float, atr: float = 0) -> bool:
        if action == PositionAction.FULL_CLOSE:
            return self.exit_engine.close_position(position)

        elif action == PositionAction.TRAILING_STOP:
            trailing_dist = atr * config.risk.get("trailing_atr_multiplier", 1.5) if atr > 0 else current_price * 0.001
            if position["type"] == "BUY":
                new_sl = current_price - trailing_dist
                if new_sl > (position.get("sl", 0) or 0):
                    return self.execution_engine.modify_position(position["ticket"], sl=new_sl)
            else:
                new_sl = current_price + trailing_dist
                if new_sl < (position.get("sl", float("inf")) or float("inf")):
                    return self.execution_engine.modify_position(position["ticket"], sl=new_sl)
            return False

        elif action == PositionAction.MOVE_STOP_LOSS:
            if position["type"] == "BUY":
                new_sl = position.get("sl", 0) + (current_price - position.get("price_open", current_price)) * 0.3
                return self.execution_engine.modify_position(position["ticket"], sl=new_sl)
            else:
                new_sl = position.get("sl", current_price) - (position.get("price_open", current_price) - current_price) * 0.3
                return self.execution_engine.modify_position(position["ticket"], sl=new_sl)

        return False

    def _get_current_price(self, symbol: str, position_type: str) -> Optional[float]:
        tick = self.data_engine.get_current_price(symbol)
        if tick:
            return tick["bid"] if position_type == "BUY" else tick["ask"]
        return None

    def close_all(self):
        self.execution_engine.close_all()
        self._open_positions = []

    @property
    def open_positions(self) -> List[Dict]:
        return self._open_positions
