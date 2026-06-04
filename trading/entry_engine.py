from typing import Dict, Optional

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
        balance = (account_info or {}).get("balance", 0)
        sl_tp = self._calculate_sl_tp(
            direction, entry_price, atr, df_entry, balance=balance
        )
        sl_price = sl_tp["sl_price"]
        tp1_price = sl_tp["tp1_price"]
        tp2_price = sl_tp["tp2_price"]
        use_scale_out = sl_tp["use_scale_out"]

        # Langkah 6: Min R:R gate — reward must be >= risk threshold
        # Skip gate for small accounts (< $500) — quick 1:1 profit lebih baik
        is_small = 0 < balance < 500
        if not is_small:
            sl_dist = abs(entry_price - sl_price)
            if use_scale_out:
                blended_tp = tp1_price * 0.5 + tp2_price * 0.5
                tp_dist = abs(blended_tp - entry_price)
                min_rr = 1.0
            else:
                tp_dist = abs(tp1_price - entry_price)
                min_rr = 1.2
            if sl_dist > 0 and (tp_dist / sl_dist) < min_rr:
                self.logger.info(
                    f"Min R:R gate: {tp_dist/sl_dist:.2f} < {min_rr} — skipping trade"
                )
                return None

        # For risk evaluation, use appropriate TP
        # Scale-out: pass runner TP2 (represents full potential)
        # Single: pass TP1
        eval_tp = tp2_price if use_scale_out and tp2_price > 0 else tp1_price

        trade_eval = self.risk_manager.evaluate_trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl_price,
            take_profit=eval_tp,
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

        # Scale-out: 2 orders for balance >= $500 with sufficient lot size
        if use_scale_out and lot_size >= 0.02:
            half_lot = round(lot_size / 2, 2)
            if direction == TradeDirection.BUY.value:
                r1 = self.execution_engine.execute_buy(
                    symbol, half_lot, sl=sl_price, tp=tp1_price,
                    comment="AI_FOREX_V2_TP1",
                )
                r2 = self.execution_engine.execute_buy(
                    symbol, half_lot, sl=sl_price, tp=tp2_price,
                    comment="AI_FOREX_V2_RUN",
                )
            else:
                r1 = self.execution_engine.execute_sell(
                    symbol, half_lot, sl=sl_price, tp=tp1_price,
                    comment="AI_FOREX_V2_TP1",
                )
                r2 = self.execution_engine.execute_sell(
                    symbol, half_lot, sl=sl_price, tp=tp2_price,
                    comment="AI_FOREX_V2_RUN",
                )
            results = [r for r in [r1, r2] if r]
            if not results:
                return None
            result = results[0]
            result["sl"] = sl_price
            result["tp"] = tp1_price
            result["tp2"] = tp2_price
            result["entry_price"] = entry_price
            result["direction"] = direction
            result["lot_size"] = lot_size
            result["risk"] = trade_eval
            result["decision"] = decision
            result["timeframe"] = decision.get("timeframe", "M15")
            result["use_scale_out"] = True
            result["tickets"] = [r.get("ticket") for r in results]
            return result

        # Single order mode
        if direction == TradeDirection.BUY.value:
            result = self.execution_engine.execute_buy(
                symbol=symbol,
                volume=lot_size,
                sl=sl_price,
                tp=tp1_price,
            )
        else:
            result = self.execution_engine.execute_sell(
                symbol=symbol,
                volume=lot_size,
                sl=sl_price,
                tp=tp1_price,
            )

        if result:
            result["sl"] = sl_price
            result["tp"] = tp1_price
            result["entry_price"] = entry_price
            result["direction"] = direction
            result["lot_size"] = lot_size
            result["risk"] = trade_eval
            result["decision"] = decision
            result["timeframe"] = decision.get("timeframe", "M15")
            result["use_scale_out"] = False
            return result

        return None

    def _calculate_sl_tp(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        df,
        balance: float = 0,
    ) -> Dict:
        use_dynamic_sl = config.risk["use_dynamic_sl"]
        use_dynamic_tp = config.risk["use_dynamic_tp"]
        pip_size = 0.0001
        is_small = 0 < balance < 500

        # ── SL ──
        if use_dynamic_sl and atr > 0:
            atr_mult = 2.0 if is_small else 1.5
            sl_distance = atr * atr_mult
        else:
            sl_distance = config.risk["sl_pips"] * pip_size

        if is_small:
            min_sl_pips = 25.0
            max_sl_pips = 40.0
            sl_pips = sl_distance / pip_size
            if sl_pips < min_sl_pips:
                sl_distance = min_sl_pips * pip_size
                self.logger.info(f"SL floor: raised to {min_sl_pips}pips (was {sl_pips:.0f}pips)")
            elif sl_pips > max_sl_pips:
                sl_distance = max_sl_pips * pip_size
                self.logger.info(f"SL ceiling: capped to {max_sl_pips}pips (was {sl_pips:.0f}pips)")

        sl_price = entry_price - sl_distance if direction == "BUY" else entry_price + sl_distance

        # ── TP ──
        sl_final = abs(entry_price - sl_price)

        if use_dynamic_tp and atr > 0:
            if is_small:
                # Langkah 3: Full close cepat — 1:1 R:R
                tp1_distance = sl_final
                tp1_price = entry_price + tp1_distance if direction == "BUY" else entry_price - tp1_distance
                return {
                    "sl_price": sl_price,
                    "tp1_price": tp1_price,
                    "tp2_price": 0,
                    "use_scale_out": False,
                }
            else:
                # Langkah 2: Scale-out — 50% at 1:0.67, 50% at 1:1.67
                tp1_price = entry_price + atr * 1.0 if direction == "BUY" else entry_price - atr * 1.0
                tp2_price = entry_price + atr * 2.5 if direction == "BUY" else entry_price - atr * 2.5
                return {
                    "sl_price": sl_price,
                    "tp1_price": tp1_price,
                    "tp2_price": tp2_price,
                    "use_scale_out": True,
                }

        # Fallback: fixed TP from config
        tp_distance = config.risk["tp_pips"] * pip_size
        tp_price = entry_price + tp_distance if direction == "BUY" else entry_price - tp_distance
        return {
            "sl_price": sl_price,
            "tp1_price": tp_price,
            "tp2_price": 0,
            "use_scale_out": False,
        }
