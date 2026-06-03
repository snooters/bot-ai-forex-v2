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

        files = sorted(data_path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")

        self.logger.info(f"Found {len(files)} parquet files in {data_path}")

        if len(files) > 1:
            timeframes = set()
            for f in files:
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    timeframes.add(parts[-1])
            if len(timeframes) > 1:
                self.logger.warning(
                    f"Multiple timeframe files detected: {sorted(timeframes)}. "
                    f"Mixing them will corrupt feature engineering. "
                    f"Use --data-path <single_file.parquet> instead."
                )

        dfs = []
        for fpath in files:
            try:
                df = self._load_single(fpath)
                if df is not None and not df.empty:
                    dfs.append(df)
                    self.logger.info(f"  Loaded {fpath.name}: {len(df)} rows, "
                                    f"range={df['timestamp'].min()} to {df['timestamp'].max()}")
            except Exception as e:
                self.logger.warning(f"  Skipped {fpath.name}: {e}")

        if not dfs:
            raise ValueError("No data loaded from any parquet file")

        combined = pd.concat(dfs, ignore_index=True)
        combined = self._clean(combined)

        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        if before_dedup != len(combined):
            self.logger.warning(f"Dropped {before_dedup - len(combined)} duplicate timestamps "
                                f"(likely from mixed timeframe files)")

        self.logger.info(
            f"Combined dataset: {len(combined)} rows, "
            f"range={combined['timestamp'].min()} to {combined['timestamp'].max()}"
        )

        interval = combined["timestamp"].diff().dropna()
        if not interval.empty:
            median_interval = interval.median()
            self.logger.info(f"Median interval: {median_interval}")
            if len(files) > 1 and median_interval > pd.Timedelta(minutes=1):
                self.logger.warning(
                    f"Median interval is {median_interval}. Data appears to be "
                    f"mixed timeframe. Strongly recommend using a single file."
                )

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
