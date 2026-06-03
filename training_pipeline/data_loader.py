from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TrainingConfig
from .utils import setup_logger


TF_LABELS = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4"}
TF_SORT = {v: k for k, v in TF_LABELS.items()}


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
                    f"Use --multi-tf to train on all timeframes, or "
                    f"--data-path <single_file.parquet> for single timeframe."
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
            self.logger.warning(f"Dropped {before_dedup - len(combined)} duplicate timestamps")

        self.logger.info(
            f"Combined dataset: {len(combined)} rows, "
            f"range={combined['timestamp'].min()} to {combined['timestamp'].max()}"
        )
        return combined

    def load_multi_timeframe(self, path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        data_path = Path(path or self.config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        if data_path.is_file():
            single = self._load_single(data_path)
            tf = self._detect_timeframe(single)
            return {tf: single}

        files = sorted(data_path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")

        self.logger.info(f"Multi-timeframe mode: loading {len(files)} files from {data_path}")

        result = {}
        for fpath in files:
            try:
                df = self._load_single(fpath)
                if df is not None and not df.empty:
                    tf = self._detect_timeframe(df)
                    if tf in result:
                        self.logger.warning(f"Duplicate timeframe {tf} from {fpath.name}, skipping")
                        continue
                    result[tf] = df
                    self.logger.info(f"  {tf}: {fpath.name} — {len(df)} rows, "
                                    f"range={df['timestamp'].min()} to {df['timestamp'].max()}")
            except Exception as e:
                self.logger.warning(f"  Skipped {fpath.name}: {e}")

        if not result:
            raise ValueError("No data loaded from any parquet file")

        self.logger.info(f"Loaded {len(result)} timeframes: {sorted(result.keys())}")
        return result

    def _detect_timeframe(self, df: pd.DataFrame) -> str:
        if "timeframe" in df.columns:
            tf_min = int(df["timeframe"].iloc[0])
            return TF_LABELS.get(tf_min, f"TF{tf_min}")

        ts = df["timestamp"].dropna().sort_values()
        if len(ts) < 10:
            return "UNK"
        diffs = ts.diff().dropna()
        if diffs.empty:
            return "UNK"
        median_sec = diffs.median().total_seconds()
        if median_sec < 90:
            return "M1"
        elif median_sec < 300:
            return "M5"
        elif median_sec < 600:
            return "M10"
        elif median_sec < 900:
            return "M15"
        elif median_sec < 1800:
            return "M30"
        elif median_sec < 3600:
            return "H1"
        elif median_sec < 7200:
            return "H2"
        elif median_sec < 14400:
            return "H4"
        else:
            return "D1"

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
            "real_volume": "volume",
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

    def align_timeframes(
        self, tf_data: Dict[str, pd.DataFrame], fast_tf: str = "M5"
    ) -> pd.DataFrame:
        fast_df = tf_data.pop(fast_tf, None)
        if fast_df is None:
            fast_tf = sorted(tf_data.keys(), key=lambda k: TF_SORT.get(k, 999))[0]
            fast_df = tf_data.pop(fast_tf)

        fast_df = fast_df.sort_values("timestamp").set_index("timestamp")
        fast_df.index = pd.to_datetime(fast_df.index)
        fast_df = fast_df[~fast_df.index.duplicated(keep="last")]

        combined = fast_df[["open", "high", "low", "close", "volume"]].copy()
        combined.columns = [f"{c}_{fast_tf}" for c in combined.columns]

        for tf, df in tf_data.items():
            df = df.sort_values("timestamp").set_index("timestamp")
            df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep="last")]
            df = df[["open", "high", "low", "close", "volume"]]
            df.columns = [f"{c}_{tf}" for c in df.columns]

            df_aligned = df.reindex(combined.index, method="ffill", tolerance=pd.Timedelta(hours=4))
            combined = combined.join(df_aligned)

        combined = combined.reset_index()
        combined = combined.dropna()
        self.logger.info(
            f"Aligned multi-timeframe data: {len(combined)} rows, "
            f"columns={list(combined.columns[:8])}..."
        )
        return combined
