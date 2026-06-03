import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import pandas as pd

from core.constants import HISTORICAL_DIR, CACHE_DIR
from utils.logger import get_logger


class ParquetStorage:
    def __init__(self):
        self.logger = get_logger("data_storage")
        self._base_dir = Path(HISTORICAL_DIR)
        self._cache_dir = Path(CACHE_DIR)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._parquet_available = False
        self._try_import()

    def _try_import(self):
        try:
            import pyarrow
            self._parquet_available = True
        except ImportError:
            self._parquet_available = False
            self.logger.info("pyarrow not installed, using CSV fallback")

    def _get_path(self, symbol: str, timeframe: int, ext: str = "parquet") -> Path:
        pair_dir = self._base_dir / symbol
        pair_dir.mkdir(exist_ok=True)
        return pair_dir / f"tf_{timeframe}.{ext}"

    def save_data(self, symbol: str, timeframe: int, df: pd.DataFrame):
        if df.empty:
            return
        df = df.copy()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.sort_values("time", inplace=True)
            df.drop_duplicates(subset=["time"], keep="last", inplace=True)

        if self._parquet_available:
            filepath = self._get_path(symbol, timeframe, "parquet")
            df.to_parquet(str(filepath), index=False)
        else:
            filepath = self._get_path(symbol, timeframe, "csv")
            df.to_csv(str(filepath), index=False)
        self.logger.debug(f"Saved {len(df)} rows to {filepath}")

    def load_data(
        self, symbol: str, timeframe: int,
        from_date: Optional[datetime] = None, to_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        if self._parquet_available:
            filepath = self._get_path(symbol, timeframe, "parquet")
        else:
            filepath = self._get_path(symbol, timeframe, "csv")

        if not filepath.exists():
            return pd.DataFrame()

        try:
            if self._parquet_available:
                df = pd.read_parquet(str(filepath))
            else:
                df = pd.read_csv(str(filepath), parse_dates=["time"] if "time" in pd.read_csv(str(filepath), nrows=0).columns else [])
        except Exception as e:
            self.logger.warning(f"Failed to read {filepath}: {e}")
            return pd.DataFrame()

        if df.empty:
            return df
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            if from_date:
                df = df[df["time"] >= from_date]
            if to_date:
                df = df[df["time"] <= to_date]
            df.sort_values("time", inplace=True)
        return df

    def append_data(self, symbol: str, timeframe: int, new_df: pd.DataFrame):
        if new_df.empty:
            return
        existing = self.load_data(symbol, timeframe)
        if existing.empty:
            self.save_data(symbol, timeframe, new_df)
            return
        combined = pd.concat([existing, new_df], ignore_index=True)
        if "time" in combined.columns:
            combined["time"] = pd.to_datetime(combined["time"])
            combined.sort_values("time", inplace=True)
            combined.drop_duplicates(subset=["time"], keep="last", inplace=True)
        self.save_data(symbol, timeframe, combined)

    def get_date_range(self, symbol: str, timeframe: int) -> tuple:
        df = self.load_data(symbol, timeframe)
        if df.empty or "time" not in df.columns:
            return (None, None)
        return (df["time"].min(), df["time"].max())

    def has_data(self, symbol: str, timeframe: int) -> bool:
        parquet_path = self._get_path(symbol, timeframe, "parquet")
        csv_path = self._get_path(symbol, timeframe, "csv")
        return parquet_path.exists() or csv_path.exists()

    def get_all_symbols(self) -> List[str]:
        return [d.name for d in self._base_dir.iterdir() if d.is_dir()]

    def get_storage_stats(self) -> dict:
        stats = {}
        for symbol_dir in self._base_dir.iterdir():
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            stats[symbol] = {}
            for f in list(symbol_dir.glob("*.parquet")) + list(symbol_dir.glob("*.csv")):
                tf = f.stem.replace("tf_", "")
                if f.stat().st_size > 0:
                    try:
                        if f.suffix == ".parquet":
                            df = pd.read_parquet(str(f))
                        else:
                            df = pd.read_csv(str(f))
                        stats[symbol][tf] = {
                            "rows": len(df), "size_mb": f.stat().st_size / 1_048_576,
                            "from": str(df["time"].min()) if "time" in df.columns else "",
                            "to": str(df["time"].max()) if "time" in df.columns else "",
                        }
                    except Exception:
                        pass
        return stats

    def delete_old_data(self, symbol: str, timeframe: int, before: datetime):
        df = self.load_data(symbol, timeframe)
        if df.empty:
            return
        df = df[df["time"] >= before]
        self.save_data(symbol, timeframe, df)
