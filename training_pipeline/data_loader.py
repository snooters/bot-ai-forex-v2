from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .utils import setup_logger


class DataLoader:
    REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = setup_logger("data_loader", config.log_dir, config.log_level)

    def load(self, path: Optional[str] = None) -> pd.DataFrame:
        data_path = Path(path or self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        if data_path.is_file():
            return self._load_single(data_path)
        return self._load_directory(data_path)

    def _load_directory(self, directory: Path) -> pd.DataFrame:
        parquet_files = sorted(directory.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {directory}")

        self.logger.info(f"Found {len(parquet_files)} parquet files in {directory}")
        dfs = []
        for fpath in parquet_files:
            try:
                df = self._load_single(fpath)
                if df is not None and not df.empty:
                    dfs.append(df)
                    self.logger.info(f"  Loaded {fpath.name}: {len(df)} rows")
            except Exception as e:
                self.logger.warning(f"  Skipped {fpath.name}: {e}")

        if not dfs:
            raise ValueError("No data loaded from any parquet file")

        combined = pd.concat(dfs, ignore_index=True)
        combined = self._clean(combined)
        self.logger.info(f"Combined dataset: {len(combined)} rows")
        return combined

    def _load_single(self, path: Path) -> pd.DataFrame:
        if not path.suffix == ".parquet":
            raise ValueError(f"Not a parquet file: {path}")

        df = pd.read_parquet(path)
        if df.empty:
            return df

        df = self._normalize_columns(df)
        err = self._validate(df)
        if err:
            raise ValueError(err)

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {
            "time": "timestamp",
            "date": "timestamp",
            "datetime": "timestamp",
            "tick_volume": "volume",
            "vol": "volume",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        return df

    def _validate(self, df: pd.DataFrame) -> Optional[str]:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            return f"Missing required columns: {missing}"
        return None

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = df.drop_duplicates(subset=["timestamp"], keep="last")

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"])
        invalid = (df["high"] < df["low"]) | (df["close"] < 0) | (df["open"] < 0)
        if invalid.any():
            self.logger.warning(f"Dropping {invalid.sum()} rows with invalid OHLC")
            df = df[~invalid]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        return df.reset_index(drop=True)

    def train_val_test_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        test_start = int(n * (1 - self.config.test_split))
        val_start = int(test_start * (1 - self.config.val_split))

        train = df.iloc[:val_start].reset_index(drop=True)
        val = df.iloc[val_start:test_start].reset_index(drop=True)
        test = df.iloc[test_start:].reset_index(drop=True)

        self.logger.info(
            f"Split: train={len(train)}, val={len(val)}, test={len(test)}"
        )
        return train, val, test
