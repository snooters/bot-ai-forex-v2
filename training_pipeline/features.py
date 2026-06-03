import numpy as np
import pandas as pd
from typing import List, Optional

from .config import TrainingConfig
from .utils import setup_logger


class FeatureEngineer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = setup_logger("features", config.log_dir, config.log_level)

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self.logger.warning("Empty dataframe, skipping feature engineering")
            return df

        df = df.copy()
        required = {"close", "high", "low"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required - set(df.columns)}")

        self.logger.info(f"Computing features on {len(df)} rows...")

        df = self._compute_rsi(df, 14)
        df = self._compute_ema(df, "close", 20)
        df = self._compute_ema(df, "close", 50)
        df = self._compute_macd(df, 12, 26, 9)
        df = self._compute_atr(df, 14)
        df = self._compute_returns(df, [1, 5, 10])
        df = self._compute_volatility(df, 20)

        df = self._compute_extra_features(df)

        before = len(df)
        df = self._safe_clean(df)
        after = len(df)
        if before != after:
            self.logger.warning(f"Dropped {before - after} rows with NaN features")

        self.logger.info(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} cols")
        return df

    def _compute_rsi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def _compute_ema(self, df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
        df[f"ema_{period}"] = df[col].ewm(span=period, adjust=False).mean()
        return df

    def _compute_macd(self, df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = macd_line - signal_line
        return df

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=period).mean()
        return df

    def _compute_returns(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        close = df["close"]
        for p in periods:
            df[f"return_{p}"] = close.pct_change(periods=p)
        return df

    def _compute_volatility(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        log_returns = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = log_returns.rolling(window=period).std()
        return df

    def _compute_extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close, high, low = df["close"], df["high"], df["low"]

        df["high_low_ratio"] = (high - low) / close.replace(0, np.nan)
        df["close_position"] = (close - low) / (high - low).replace(0, np.nan)

        vol = df.get("volume", pd.Series(0, index=df.index))
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        df["volume_ma"] = vol.rolling(20).mean()
        df["volume_ratio"] = vol / df["volume_ma"].replace(0, np.nan)

        return df

    def _safe_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [c for c in df.columns if c not in
                        {"timestamp", "open", "high", "low", "close", "volume"}]
        if not feature_cols:
            return df
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feature_cols, how="any").reset_index(drop=True)
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude = {
            "timestamp", "time", "open", "high", "low", "close", "volume",
            "label", "label_encoded", "spread", "symbol", "timeframe",
            "tick_volume",
        }
        base_exclude = {"open", "high", "low", "close", "volume"}
        full_exclude = exclude.copy()
        for suffix in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
            for b in base_exclude:
                full_exclude.add(f"{b}_{suffix}")

        cols = [c for c in df.columns if c not in full_exclude]
        numeric_types = df[cols].select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = set(cols) - set(numeric_types)
        if non_numeric:
            cols = [c for c in cols if c not in non_numeric]
        return cols

    def validate_features(self, df: pd.DataFrame) -> bool:
        features = self.get_feature_columns(df)
        if not features:
            self.logger.error("No feature columns found")
            return False
        nan_count = df[features].isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"{nan_count} NaN values remain in features")
        inf_count = np.isinf(df[features].select_dtypes(include=[np.number]).values).sum()
        if inf_count > 0:
            self.logger.warning(f"{inf_count} inf values remain in features")
        return True
