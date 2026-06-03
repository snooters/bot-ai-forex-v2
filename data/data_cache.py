from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Tuple

import pandas as pd

from core.exceptions import CacheError
from utils.logger import get_logger


class RateCache:
    def __init__(self, max_size: int = 50, default_ttl_minutes: int = 5):
        self._max_size = max_size
        self._default_ttl = timedelta(minutes=default_ttl_minutes)
        self._cache: OrderedDict[str, Tuple[pd.DataFrame, datetime]] = OrderedDict()
        self.logger = get_logger("data_cache")

    def _make_key(self, symbol: str, timeframe: int) -> str:
        return f"{symbol}_{timeframe}"

    def get(self, symbol: str, timeframe: int) -> Optional[pd.DataFrame]:
        key = self._make_key(symbol, timeframe)
        if key not in self._cache:
            return None
        df, cached_at = self._cache[key]
        if datetime.now() - cached_at > self._default_ttl:
            del self._cache[key]
            return None
        return df.copy()

    def set(self, symbol: str, timeframe: int, df: pd.DataFrame):
        key = self._make_key(symbol, timeframe)
        self._cache[key] = (df.copy(), datetime.now())
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, symbol: str, timeframe: int):
        key = self._make_key(symbol, timeframe)
        self._cache.pop(key, None)

    def invalidate_all(self):
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)

    def get_cached_symbols(self) -> list:
        return list(set(k.split("_")[0] for k in self._cache.keys()))

    def has(self, symbol: str, timeframe: int) -> bool:
        return self.get(symbol, timeframe) is not None


class FeatureCache:
    def __init__(self, max_size: int = 100, default_ttl_seconds: int = 30):
        self._max_size = max_size
        self._default_ttl = timedelta(seconds=default_ttl_seconds)
        self._cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self.logger = get_logger("feature_cache")

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        value, cached_at = self._cache[key]
        if datetime.now() - cached_at > self._default_ttl:
            del self._cache[key]
            return None
        return value

    def set(self, key: str, value: Any):
        self._cache[key] = (value, datetime.now())
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, key: str):
        self._cache.pop(key, None)
