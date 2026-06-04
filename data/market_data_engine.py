import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable

import pandas as pd

from core.config import config
from core.exceptions import DataError, MT5DataError
from core.constants import Timeframe
from data.mt5_connector import MT5Connector
from data.data_cache import RateCache
from data.data_storage import ParquetStorage
from utils.logger import get_logger
from utils.decorators import safe_execute


TIMEFRAME_TO_CANDLES = {
    1: 525600,
    5: 105120,
    15: 35040,
    30: 17520,
    60: 8760,
    240: 2190,
}


class MarketDataEngine:
    def __init__(self):
        self.logger = get_logger("market_data_engine")
        self.connector = MT5Connector()
        self.cache = RateCache()
        self.storage = ParquetStorage()
        self._listeners: Dict[str, List[Callable]] = {}
        self._last_update: Dict[str, datetime] = {}
        self._running = False

    @safe_execute(default_return=False, raise_on_error=True)
    def initialize(self) -> bool:
        self.logger.info("Initializing Market Data Engine...")
        self.connector.connect()
        self._running = True
        self.logger.info("Market Data Engine initialized successfully")
        return True

    def shutdown(self):
        self._running = False
        self.connector.disconnect()
        self.logger.info("Market Data Engine shutdown")

    @safe_execute(default_return=pd.DataFrame())
    def get_rates(
        self, symbol: str, timeframe: int, count: int = 100,
        use_cache: bool = True, force_refresh: bool = False,
    ) -> pd.DataFrame:
        if use_cache and not force_refresh:
            cached = self.cache.get(symbol, timeframe)
            if cached is not None and len(cached) >= count:
                return cached.iloc[-count:].reset_index(drop=True)

        try:
            df = self.connector.get_rates(symbol, timeframe, count=count)
            if not df.empty:
                self.storage.append_data(symbol, timeframe, df)
                self.cache.set(symbol, timeframe, df)
            return df
        except MT5DataError as e:
            is_live = config.account["trading_mode"] == "live"
            if is_live:
                self.logger.warning(f"LIVE: data unavailable for {symbol} tf={timeframe} — {e}")
                return pd.DataFrame()
            self.logger.warning(f"get_rates failed for {symbol} tf={timeframe}: {e}")
            stored = self.storage.load_data(symbol, timeframe)
            if not stored.empty and len(stored) >= count:
                self.logger.info(f"Using cached data for {symbol} tf={timeframe}")
                return stored.iloc[-count:].reset_index(drop=True)
            return pd.DataFrame()

    async def get_historical_data_async(
        self, symbol: str, timeframe: int, years: int = 2,
        max_rows: int = 10000,
    ) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        stored = self.storage.load_data(symbol, timeframe, start_date, end_date)

        max_candles = 5000
        self.logger.info(f"Downloading {max_candles} candles for {symbol} tf={timeframe}...")
        try:
            df = self.connector.get_rates(symbol, timeframe, count=max_candles)
            if df is None or df.empty:
                self.logger.info(f"No new data for {symbol} tf={timeframe}, using stored")
                if not stored.empty:
                    return stored.iloc[-max_rows:].reset_index(drop=True) if len(stored) > max_rows else stored
                return pd.DataFrame()
            self.storage.append_data(symbol, timeframe, df)
            self.cache.set(symbol, timeframe, df)
            accumulated = self.storage.load_data(symbol, timeframe, start_date)
            self.logger.info(f"Downloaded {len(df)} new, total accumulated: {len(accumulated)} candles for {symbol} tf={timeframe}")
            if len(accumulated) > max_rows:
                accumulated = accumulated.iloc[-max_rows:].reset_index(drop=True)
                self.logger.info(f"Using last {max_rows} candles for training")
            return accumulated
        except Exception as e:
            self.logger.warning(f"Historical download failed for {symbol} tf={timeframe}: {e}")
            if not stored.empty:
                return stored.iloc[-max_rows:].reset_index(drop=True) if len(stored) > max_rows else stored
            return pd.DataFrame()

    @safe_execute(default_return=pd.DataFrame())
    def get_historical_data(
        self, symbol: str, timeframe: int, years: int = 2,
    ) -> pd.DataFrame:
        return self._sync_get_historical(symbol, timeframe, years)

    def _sync_get_historical(self, symbol: str, timeframe: int, years: int,
                              max_rows: int = 10000) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        stored = self.storage.load_data(symbol, timeframe, start_date, end_date)

        max_candles = 5000
        self.logger.info(f"Downloading {max_candles} candles for {symbol} tf={timeframe}...")
        try:
            df = self.connector.get_rates(symbol, timeframe, count=max_candles)
            if df is None or df.empty:
                self.logger.info(f"No new data for {symbol} tf={timeframe}, using stored")
                if not stored.empty:
                    return stored.iloc[-max_rows:].reset_index(drop=True) if len(stored) > max_rows else stored
                return pd.DataFrame()
            self.storage.append_data(symbol, timeframe, df)
            self.cache.set(symbol, timeframe, df)
            accumulated = self.storage.load_data(symbol, timeframe, start_date)
            self.logger.info(f"Downloaded {len(df)} new, total accumulated: {len(accumulated)} candles for {symbol} tf={timeframe}")
            if len(accumulated) > max_rows:
                accumulated = accumulated.iloc[-max_rows:].reset_index(drop=True)
                self.logger.info(f"Using last {max_rows} candles for training")
            return accumulated
        except Exception:
            if not stored.empty:
                return stored.iloc[-max_rows:].reset_index(drop=True) if len(stored) > max_rows else stored
            return pd.DataFrame()

    @safe_execute(default_return=0)
    def refresh_stored_data(self, symbol: str, timeframe: int, count: int = 5000) -> int:
        try:
            df = self.connector.get_rates(symbol, timeframe, count=count)
            if df is None or df.empty:
                return 0
            self.storage.append_data(symbol, timeframe, df)
            self.cache.set(symbol, timeframe, df)
            self.logger.info(f"Refreshed {len(df)} candles for {symbol} tf={timeframe}")
            return len(df)
        except Exception as e:
            self.logger.warning(f"Refresh failed for {symbol} tf={timeframe}: {e}")
            return 0

    def get_latest_candles(self, symbol: str, timeframe: int, count: int = 10) -> pd.DataFrame:
        return self.get_rates(symbol, timeframe, count=count, force_refresh=True)

    def check_last_candle(self, symbol: str, timeframe: int) -> Optional[float]:
        try:
            df = self.connector.get_rates(symbol, timeframe, count=1)
            if df is not None and not df.empty:
                return df["time"].iloc[-1].timestamp()
        except Exception as e:
            self.logger.debug(f"check_last_candle via connector failed: {e}")
        try:
            self.connector.ensure_connected()
            rates = self.connector._mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
            if rates is not None and len(rates) > 0:
                return float(rates[0][0])
        except Exception:
            pass
        return None

    def get_multi_timeframe_data(
        self, symbol: str, timeframes: Optional[List[int]] = None, count: int = 100,
    ) -> Dict[int, pd.DataFrame]:
        if timeframes is None:
            timeframes = Timeframe.ALL
        return {tf: self.get_rates(symbol, tf, count=count) for tf in timeframes}

    def get_current_price(self, symbol: str) -> Optional[Dict]:
        return self.connector.get_symbol_tick(symbol)

    def get_current_spread(self, symbol: str) -> Optional[float]:
        info = self.connector.get_symbol_info(symbol)
        if info:
            return info["spread"] * info.get("point", 0.00001)
        return None

    def get_account_info(self) -> Optional[Dict]:
        return self.connector.get_account_info()

    def load_training_data(self, symbol: str, timeframe: int, days: int = 180) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stored = self.storage.load_data(symbol, timeframe, start_date, end_date)
        if stored.empty:
            self.logger.warning(f"No stored data for {symbol} tf={timeframe}, downloading...")
            return self.get_historical_data(symbol, timeframe, years=max(1, days // 365))
        self.logger.info(f"Loaded {len(stored)} candles from storage for {symbol} tf={timeframe}")
        return stored

    def get_account_balance(self) -> float:
        return self.connector.get_account_balance()

    def get_account_equity(self) -> float:
        return self.connector.get_account_equity()

    def subscribe(self, event: str, callback: Callable):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)

    def _emit(self, event: str, data=None):
        for cb in self._listeners.get(event, []):
            try:
                cb(data)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
