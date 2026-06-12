from datetime import datetime
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

from core.config import config
from core.exceptions import MT5ConnectionError, MT5DataError
from core.constants import MT5_TIMEOUT
from utils.logger import get_logger


class MT5Connector:
    _instance = None
    _connected = False
    _mt5 = None

    SYMBOL_SUFFIXES = ["", ".fl", ".raw", ".ecn", ".pro", ".m", ".fx", ".i", ".MBE"]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.logger = get_logger("mt5_connector")
        self._account_info = None
        self._mt5_available = False
        self._symbol_map: dict = {}
        self._try_import_mt5()

    def _try_import_mt5(self):
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            self._mt5_available = True
        except ImportError:
            self._mt5_available = False
            self.logger.warning("MetaTrader5 not installed. Running in simulation mode.")

    def resolve_symbol(self, symbol: str) -> str:
        if symbol in self._symbol_map:
            return self._symbol_map[symbol]

        if not self._mt5_available:
            return symbol

        for suffix in self.SYMBOL_SUFFIXES:
            candidate = f"{symbol}{suffix}" if suffix else symbol
            if symbol.endswith(suffix[1:]) and suffix:
                continue
            info = self._mt5.symbol_info(candidate)
            if info is not None:
                if candidate != symbol:
                    self.logger.info(f"Symbol '{symbol}' resolved to '{candidate}'")
                self._symbol_map[symbol] = candidate
                self._mt5.symbol_select(candidate, True)
                return candidate

        base = symbol.split(".")[0] if "." in symbol else symbol
        if base != symbol:
            info = self._mt5.symbol_info(base)
            if info is not None:
                self.logger.info(f"Symbol '{symbol}' resolved to '{base}'")
                self._symbol_map[symbol] = base
                self._mt5.symbol_select(base, True)
                return base

        self._symbol_map[symbol] = symbol
        return symbol

    def connect(self, max_retries: int = 3, retry_delay: float = 5.0) -> bool:
        if self._connected:
            return True

        if not self._mt5_available:
            self.logger.info("MT5 not available - running in simulation mode")
            self._connected = True
            return True

        import time as _time
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Connecting to MetaTrader 5 (attempt {attempt}/{max_retries})...")
                if not self._mt5.initialize():
                    error = self._mt5.last_error()
                    raise MT5ConnectionError(f"MT5 initialize failed: {error}")

                acct = config.account
                if acct["server"]:
                    authorized = self._mt5.login(
                        login=int(acct["user_id"]),
                        password=acct["password"],
                        server=acct["server"]
                    )
                else:
                    authorized = True

                if not authorized:
                    error = self._mt5.last_error()
                    raise MT5ConnectionError(f"MT5 login failed: {error}")

                self._account_info = self._mt5.account_info()
                self.logger.info(f"Logged in. Account: {self._account_info.login}, "
                                 f"Balance: {self._account_info.balance:.2f}")

                self._enable_symbols()
                self._wait_terminal_ready()

                self._connected = True
                return True

            except MT5ConnectionError as e:
                if attempt < max_retries:
                    self.logger.warning(f"Connection attempt {attempt} failed: {e}. "
                                       f"Retrying in {retry_delay}s...")
                    _time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    def _enable_symbols(self):
        symbols = config.trading["pairs"]
        resolved = []
        for symbol in symbols:
            real = self.resolve_symbol(symbol)
            resolved.append(real)
            if not self._mt5.symbol_select(real, True):
                self.logger.warning(f"Failed to enable symbol {real}: {self._mt5.last_error()}")
        config._config["trading"]["pairs"] = resolved

    def _wait_terminal_ready(self, max_wait: int = 30):
        self.logger.info("Waiting for MT5 terminal data feed...")
        import time as _time
        test_symbols = config.trading["pairs"]
        test_sym = self.resolve_symbol(test_symbols[0]) if test_symbols else "EURUSD"
        for sec in range(max_wait):
            try:
                tinfo = self._mt5.terminal_info()
                if tinfo and tinfo.connected:
                    rates = self._mt5.copy_rates_from_pos(test_sym, 5, 0, 1)
                    if rates is not None and len(rates) > 0:
                        self.logger.info(f"MT5 terminal ready after {sec+1}s")
                        return
            except Exception as e:
                if sec == 0:
                    self.logger.debug(f"MT5 not ready yet: {e}")
            if sec % 5 == 0 or sec == 0:
                self.logger.info(f"Waiting for terminal data feed... ({sec+1}s/{max_wait}s)")
            _time.sleep(1)
        if config.account["trading_mode"] == "live":
            raise MT5ConnectionError(f"LIVE mode: MT5 terminal not ready after {max_wait}s")
        self.logger.warning(f"MT5 terminal data feed not ready after {max_wait}s, will use simulation fallback")

    def disconnect(self):
        if self._connected and self._mt5_available:
            self._mt5.shutdown()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        if not self._mt5_available:
            return True
        if not self._connected:
            return False
        return self._mt5.terminal_info() is not None

    def ensure_connected(self):
        if not self.is_connected:
            self._connected = False
            self.connect()

    def get_rates(
        self, symbol: str, timeframe: int, count: int = 100
    ) -> pd.DataFrame:
        is_live = config.account["trading_mode"] == "live"
        resolved = self.resolve_symbol(symbol)

        if not self._mt5_available:
            if is_live:
                raise MT5ConnectionError(
                    f"LIVE mode: MT5 not available — cannot fetch {symbol} tf={timeframe}"
                )
            return self._simulate_rates(symbol, timeframe, count)

        self.ensure_connected()
        rates = self._mt5.copy_rates_from_pos(resolved, timeframe, 0, count)

        if rates is None or len(rates) == 0:
            error = self._mt5.last_error()
            error_str = str(error) if error else "Unknown"
            if any(x in error_str for x in ["Terminal: Call failed", "Terminal: Invalid params"]):
                self._mt5.symbol_select(resolved, True)
                rates = self._mt5.copy_rates_from_pos(resolved, timeframe, 0, count)
                if rates is not None and len(rates) > 0:
                    self.logger.info(f"MT5 data recovered after symbol_select for {resolved} tf={timeframe}")
                    return self._rates_to_df(symbol, timeframe, rates)

                if timeframe > 5:
                    m5_count = count * (timeframe // 5) + 10
                    m5_rates = self._mt5.copy_rates_from_pos(resolved, 5, 0, m5_count)
                    if m5_rates is not None and len(m5_rates) > count * (timeframe // 5):
                        m5_df = self._rates_to_df(symbol, 5, m5_rates)
                        resampled = self._resample_ohlc(m5_df, timeframe)
                        self.logger.debug(f"Resampled {symbol} tf={timeframe} from M5 ({len(resampled)} candles)")
                        return resampled

                if is_live:
                    raise MT5DataError(
                        f"LIVE mode: MT5 terminal not ready for {symbol} tf={timeframe}"
                    )
                self.logger.warning(f"MT5 terminal not ready for {symbol} tf={timeframe}, using simulation")
                return self._simulate_rates(symbol, timeframe, count)
            raise MT5DataError(f"Failed to get rates for {symbol} tf={timeframe}: {error_str}")

        return self._rates_to_df(symbol, timeframe, rates)

    async def async_get_rates(self, symbol: str, timeframe: int, count: int = 100) -> pd.DataFrame:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_rates, symbol, timeframe, count)

    def _rates_to_df(self, symbol: str, timeframe: int, rates) -> pd.DataFrame:

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "tick_volume": "volume", "spread": "spread"
        }, inplace=True)
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        df.sort_values("time", inplace=True)
        df.drop_duplicates(subset=["time"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _resample_ohlc(m5_df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
        rule = f"{tf_minutes}min"
        resampled = m5_df.resample(rule, on="time").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "spread": "mean",
        })
        resampled.dropna(subset=["open", "close"], inplace=True)
        resampled.reset_index(inplace=True)
        resampled["timeframe"] = tf_minutes
        return resampled

    def _simulate_rates(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        import numpy as np
        periods = count
        freq_map = {1: "1min", 5: "5min", 15: "15min", 30: "30min", 60: "1h", 240: "4h"}
        freq = freq_map.get(timeframe, "5min")
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        base_price = 1.1000
        noise = np.random.randn(periods) * 0.0005
        trend = np.linspace(0, 0.001 * np.random.choice([-1, 1]), periods)
        prices = base_price + np.cumsum(noise) + trend
        df = pd.DataFrame({
            "time": dates, "open": prices, "high": prices + abs(np.random.randn(periods) * 0.0003),
            "low": prices - abs(np.random.randn(periods) * 0.0003),
            "close": prices + np.random.randn(periods) * 0.0002,
            "volume": np.random.randint(100, 10000, periods),
            "spread": np.random.randint(1, 10, periods),
            "symbol": symbol, "timeframe": timeframe,
        })
        df["close"] = df["close"].where(df["close"] > 0, 1.0)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_last_rates(self, symbol: str, timeframe: int, count: int = 10) -> pd.DataFrame:
        return self.get_rates(symbol, timeframe, count=count)

    def get_account_info(self) -> Optional[Dict]:
        if not self._mt5_available:
            return {
                "login": 0, "balance": config.account["balance"],
                "equity": config.account["balance"], "margin": 0,
                "margin_free": config.account["balance"], "margin_level": 0,
                "profit": 0, "leverage": config.account["leverage"],
                "currency": config.account["currency"], "server": "simulation",
            }
        if not self._connected:
            return None
        info = self._mt5.account_info()
        if info is None:
            return None
        return {
            "login": info.login, "balance": info.balance, "equity": info.equity,
            "margin": info.margin, "margin_free": info.margin_free,
            "margin_level": info.margin_level, "profit": info.profit,
            "leverage": info.leverage, "currency": info.currency, "server": info.server,
        }

    def get_account_balance(self) -> float:
        info = self.get_account_info()
        return info["balance"] if info else config.account["balance"]

    def get_account_equity(self) -> float:
        info = self.get_account_info()
        return info["equity"] if info else config.account["balance"]

    def get_positions(self, symbol: str = "") -> List[Dict]:
        if not self._mt5_available:
            return []
        self.ensure_connected()
        if symbol:
            resolved = self.resolve_symbol(symbol)
            positions = self._mt5.positions_get(symbol=resolved)
        else:
            positions = self._mt5.positions_get()
        if positions is None:
            return []
        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket, "symbol": pos.symbol,
                "type": "BUY" if pos.type == 0 else "SELL",
                "volume": pos.volume, "price_open": pos.price_open,
                "sl": pos.sl, "tp": pos.tp, "profit": pos.profit,
                "swap": pos.swap, "time": datetime.fromtimestamp(pos.time),
                "comment": pos.comment,
            })
        return result

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        if not self._mt5_available:
            return {"symbol": symbol, "digits": 5, "point": 0.00001, "spread": 2,
                    "spread_float": False, "trade_mode": 0, "min_volume": 0.01,
                    "max_volume": 100.0, "volume_step": 0.01, "contract_size": 100000,
                    "stops_level": 0, "freeze_level": 0}
        self.ensure_connected()
        resolved = self.resolve_symbol(symbol)
        info = self._mt5.symbol_info(resolved)
        if info is None:
            return None
        return {
            "symbol": info.name, "digits": info.digits, "point": info.point,
            "spread": info.spread, "spread_float": info.spread_float,
            "trade_mode": info.trade_mode, "min_volume": info.volume_min,
            "max_volume": info.volume_max, "volume_step": info.volume_step,
            "contract_size": info.trade_contract_size,
            "stops_level": info.trade_stops_level, "freeze_level": info.trade_freeze_level,
        }

    def get_symbol_tick(self, symbol: str) -> Optional[Dict]:
        if not self._mt5_available:
            return {"bid": 1.1000, "ask": 1.1002, "last": 1.1000,
                    "volume": 0, "time": datetime.now()}
        self.ensure_connected()
        resolved = self.resolve_symbol(symbol)
        tick = self._mt5.symbol_info_tick(resolved)
        if tick is None:
            return None
        return {
            "bid": tick.bid, "ask": tick.ask, "last": tick.last,
            "volume": tick.volume, "time": datetime.fromtimestamp(tick.time),
        }

    def _get_filling_type(self, symbol: str):
        if not self._mt5_available:
            return self._mt5.ORDER_FILLING_IOC
        resolved = self.resolve_symbol(symbol)
        info = self._mt5.symbol_info(resolved)
        if info is not None:
            mode = info.filling_mode
            if mode & 2:
                return self._mt5.ORDER_FILLING_IOC
            if mode & 1:
                return self._mt5.ORDER_FILLING_FOK
            if mode & 4:
                return self._mt5.ORDER_FILLING_RETURN
        return self._mt5.ORDER_FILLING_FOK

    def place_order(self, symbol: str, order_type: str, volume: float,
                    price: Optional[float] = None, sl: Optional[float] = None,
                    tp: Optional[float] = None, comment: str = "AI_FOREX_V2",
                    magic: int = 2024001) -> Optional[Dict]:
        if not self._mt5_available:
            tick = self.get_symbol_tick(symbol)
            return {
                "ticket": hash(f"{symbol}{datetime.now().timestamp()}") % 1000000,
                "price": tick["ask"] if order_type.upper() == "BUY" else tick["bid"],
                "volume": volume, "type": order_type, "symbol": symbol,
            }
        self.ensure_connected()
        resolved = self.resolve_symbol(symbol)
        filling = self._get_filling_type(symbol)
        order_type_mt5 = (
            self._mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY"
            else self._mt5.ORDER_TYPE_SELL
        )
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL, "symbol": resolved,
            "volume": volume, "type": order_type_mt5, "magic": magic,
            "comment": comment, "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        if order_type.upper() == "BUY":
            tick = self._mt5.symbol_info_tick(resolved)
            request["price"] = tick.ask
        else:
            tick = self._mt5.symbol_info_tick(resolved)
            request["price"] = tick.bid

        sym_info = self._mt5.symbol_info(resolved)
        if sym_info is not None:
            stops_level = sym_info.trade_stops_level
            point = sym_info.point
            digits = sym_info.digits
            min_stop_dist = stops_level * point
            if min_stop_dist > 0:
                is_buy = order_type.upper() == "BUY"
                ref_price = tick.bid if is_buy else tick.ask
                if sl is not None:
                    sl_dist = abs(ref_price - sl)
                    if sl_dist < min_stop_dist:
                        sl = ref_price - min_stop_dist if is_buy else ref_price + min_stop_dist
                        sl = round(sl, digits)
                        self.logger.info(f"Adjusted SL to {sl} (min stop {stops_level} pts from {ref_price})")
                if tp is not None:
                    tp_dist = abs(tp - ref_price)
                    if tp_dist < min_stop_dist:
                        tp = ref_price + min_stop_dist if is_buy else ref_price - min_stop_dist
                        tp = round(tp, digits)
                        self.logger.info(f"Adjusted TP to {tp} (min stop {stops_level} pts from {ref_price})")

        if sl is not None:
            request["sl"] = round(sl, 5)
        if tp is not None:
            request["tp"] = round(tp, 5)
        self.logger.debug(f"Order: {order_type} {resolved} vol={volume} "
                         f"price={request.get('price')} sl={request.get('sl')} "
                         f"tp={request.get('tp')} fill={filling}")
        result = self._mt5.order_send(request)
        if result is None:
            raise MT5DataError(f"Order send failed: {self._mt5.last_error()}")
        if result.retcode != self._mt5.TRADE_RETCODE_DONE:
            raise MT5DataError(f"Order rejected: {result.comment}")
        return {"ticket": result.order, "price": result.price, "volume": volume,
                "type": order_type, "symbol": symbol}

    def modify_position(self, ticket: int, sl: Optional[float] = None,
                        tp: Optional[float] = None) -> bool:
        if not self._mt5_available:
            return True
        self.ensure_connected()
        request = {"action": self._mt5.TRADE_ACTION_SLTP, "position": ticket}
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            return False
        return True

    def get_history_deals(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        magic: int = 2024001,
    ) -> List[Dict]:
        if not self._mt5_available or not self._connected:
            return []
        try:
            from datetime import timedelta
            if from_date is None:
                from_date = datetime.now() - timedelta(days=365)
            if to_date is None:
                to_date = datetime.now() + timedelta(days=1)
            deals = self._mt5.history_deals_get(from_date, to_date)
            if deals is None:
                return []
            result = []
            for d in deals:
                deal = d._asdict() if hasattr(d, '_asdict') else {}
                if deal.get("magic", 0) == magic:
                    result.append(deal)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get history deals: {e}")
            return []

    def close_position(self, ticket: int, symbol: str, volume: float, position_type: str) -> bool:
        if not self._mt5_available:
            return True
        self.ensure_connected()
        resolved = self.resolve_symbol(symbol)
        filling = self._get_filling_type(symbol)
        order_type = (
            self._mt5.ORDER_TYPE_SELL if position_type.upper() == "BUY"
            else self._mt5.ORDER_TYPE_BUY
        )
        tick = self._mt5.symbol_info_tick(resolved)
        price = tick.bid if position_type.upper() == "BUY" else tick.ask
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL, "symbol": resolved,
            "volume": volume, "type": order_type, "position": ticket,
            "price": price, "magic": 2024001, "comment": "AI_CLOSE",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        result = self._mt5.order_send(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            return False
        return True

    def close_all_positions(self):
        positions = self.get_positions()
        for pos in positions:
            self.close_position(pos["ticket"], pos["symbol"], pos["volume"], pos["type"])
