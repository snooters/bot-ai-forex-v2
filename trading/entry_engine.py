from typing import Dict, Optional

from core.config import config
from core.constants import TradeDirection
from risk.risk_manager import RiskManager
from trading.execution_engine import ExecutionEngine
from data.market_data_engine import MarketDataEngine
from data.mt5_connector import MT5Connector
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

    def _check_candle_confirmation(self, df_entry, direction: str) -> bool:
        """Candle confirmation filter — mencegah candle trap / fake breakout.

        Memeriksa apakah candle TERAKHIR (confirm candle) bergerak SEARAH
        dengan sinyal dari candle SEBELUMNYA (signal candle).

        Untuk BUY:  confirm candle harus BULL (close > open) DAN close > signal close
        Untuk SELL: confirm candle harus BEAR (close < open) DAN close < signal close

        Jika hanya ada 1 candle (belum ada konfirmasi), trade tetap diproses.
        Jika konfirmasi gagal, entry dibatalkan (HOLD).
        """
        try:
            if df_entry is None or len(df_entry) < 2:
                return True  # Tidak cukup data untuk konfirmasi

            signal_candle = df_entry.iloc[-2]
            confirm_candle = df_entry.iloc[-1]

            sig_close = float(signal_candle.get("close", 0))
            sig_open = float(signal_candle.get("open", 0))
            con_close = float(confirm_candle.get("close", 0))
            con_open = float(confirm_candle.get("open", 0))

            if direction == TradeDirection.BUY.value:
                # BUY: confirm candle harus bullish & close lebih tinggi dari signal close
                if con_close <= con_open:
                    return False  # Bearish candle — tidak konfirmasi
                if con_close <= sig_close:
                    return False  # Tidak membuat higher close
                return True

            elif direction == TradeDirection.SELL.value:
                # SELL: confirm candle harus bearish & close lebih rendah dari signal close
                if con_close >= con_open:
                    return False  # Bullish candle — tidak konfirmasi
                if con_close >= sig_close:
                    return False  # Tidak membuat lower close
                return True

            return True
        except Exception as e:
            self.logger.warning(f"Candle confirmation check error: {e}")
            return True  # Fallback: proceed with trade

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

        # ── Candle confirmation: pastikan candle setelah signal tidak membantah ──
        if not self._check_candle_confirmation(df_entry, direction):
            self.logger.info(
                f"Candle confirmation failed for {direction} on {symbol} — "
                f"next candle reversed direction, skipping entry"
            )
            return None

        entry_price = current_price
        balance = (account_info or {}).get("balance", 0)
        sl_tp = self._calculate_sl_tp(
            direction, entry_price, atr, df_entry, balance=balance, symbol=symbol
        )
        sl_price = sl_tp["sl_price"]
        tp1_price = sl_tp["tp1_price"]
        tp2_price = sl_tp["tp2_price"]
        use_scale_out = sl_tp["use_scale_out"]

        # Dynamic R:R gate based on confidence
        # Higher confidence → lower R:R requirement (we trust the signal more)
        # Lower confidence → higher R:R requirement (need more buffer)
        confidence_val = decision.get("confidence", 0.5)
        # Map: confidence 1.0 → min_rr 1.0, confidence 0.4 → min_rr 2.0
        dynamic_min_rr = max(1.0, min(2.0, 2.5 - confidence_val * 1.5))
        sl_dist = abs(entry_price - sl_price)
        if use_scale_out:
            blended_tp = tp1_price * 0.5 + tp2_price * 0.5
            tp_dist_val = abs(blended_tp - entry_price)
        else:
            tp_dist_val = abs(tp1_price - entry_price)
        if sl_dist > 0 and (tp_dist_val / sl_dist) < dynamic_min_rr:
            self.logger.info(
                f"Dynamic R:R gate: {tp_dist_val/sl_dist:.2f} < {dynamic_min_rr:.2f} "
                f"(confidence={confidence_val:.0%}) — skipping trade"
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
            from risk.position_sizing import PositionSizer
            sizer = PositionSizer()
            sym_info = sizer._get_symbol_info(symbol)
            min_vol = sym_info.get("min_volume", 0.01) if sym_info else 0.01
            half_lot = sizer._normalize_volume(round(lot_size / 2, 2), symbol)
            if half_lot < min_vol:
                use_scale_out = False
            else:
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
        symbol: str = "",
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

        # ── Enforce minimum stops level from broker ──
        broker_min_stop_pts = 0
        if symbol:
            try:
                conn = MT5Connector()
                sym_info = conn.get_symbol_info(symbol)
                if sym_info:
                    stops_level = sym_info.get("stops_level", 0)
                    point = sym_info.get("point", 0.00001)
                    spread = sym_info.get("spread", 0)
                    broker_min_stop_pts = stops_level + spread
                    if broker_min_stop_pts > 0:
                        min_stop_dist = broker_min_stop_pts * point
                        current_sl_dist = abs(entry_price - sl_price)
                        if current_sl_dist < min_stop_dist:
                            self.logger.info(f"SL too close ({current_sl_dist/point:.0f} pts < "
                                            f"broker min {broker_min_stop_pts} pts "
                                            f"[stops={stops_level}+spread={spread}]), adjusting SL")
                            sl_price = entry_price - min_stop_dist if direction == "BUY" else entry_price + min_stop_dist
            except Exception as broker_e:
                self.logger.debug(f"Broker symbol info unavailable: {broker_e}")

        # ── TP ──
        sl_final = abs(entry_price - sl_price)

        # If broker forced SL wider, ensure TP >= SL distance (min 1:1 RR)
        min_tp_dist = sl_final if broker_min_stop_pts > 0 else 0

        if use_dynamic_tp and atr > 0:
            if is_small:
                # Langkah 3: Full close cepat — 1:1 R:R
                tp1_distance = max(sl_final, min_tp_dist)
                tp1_price = entry_price + tp1_distance if direction == "BUY" else entry_price - tp1_distance
                return {
                    "sl_price": sl_price,
                    "tp1_price": tp1_price,
                    "tp2_price": 0,
                    "use_scale_out": False,
                }
            else:
                # Scale-out: TP1 at 1:1.33, TP2 at 1:2.67, blended >= 1:2.0
                tp1_distance = max(atr * 2.0, min_tp_dist)
                tp2_distance = max(atr * 4.0, min_tp_dist * 2.5)
                tp1_price = entry_price + tp1_distance if direction == "BUY" else entry_price - tp1_distance
                tp2_price = entry_price + tp2_distance if direction == "BUY" else entry_price - tp2_distance
                return {
                    "sl_price": sl_price,
                    "tp1_price": tp1_price,
                    "tp2_price": tp2_price,
                    "use_scale_out": True,
                }

        # Fallback: fixed TP from config
        tp_distance = max(config.risk["tp_pips"] * pip_size, min_tp_dist)
        tp_price = entry_price + tp_distance if direction == "BUY" else entry_price - tp_distance
        return {
            "sl_price": sl_price,
            "tp1_price": tp_price,
            "tp2_price": 0,
            "use_scale_out": False,
        }
