import numpy as np
import pandas as pd
from typing import Optional, Tuple

from .config import TrainingConfig
from .utils import setup_logger


class LabelEngine:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = setup_logger("labeling", config.log_dir, config.log_level)

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self.logger.warning("Empty dataframe, skipping labeling")
            return df

        df = df.copy()
        horizon = self.config.prediction_horizon
        threshold = self.config.threshold
        close = df["close"].values

        future_returns = np.full(len(df), np.nan, dtype=np.float64)
        if len(df) > horizon:
            future_close = np.roll(close, -horizon)
            future_close[-horizon:] = np.nan
            future_returns = (future_close - close) / close

        df["future_return"] = future_returns

        labels = np.full(len(df), "HOLD", dtype=object)
        labels[future_returns > threshold] = "BUY"
        labels[future_returns < -threshold] = "SELL"
        df["label"] = labels

        encoding = self.config.label_encoding
        df["label_encoded"] = df["label"].map(encoding).astype(np.int64)

        df = df.dropna(subset=["future_return"]).reset_index(drop=True)
        df = df.drop(columns=["future_return"])

        counts = df["label"].value_counts()
        self.logger.info(
            f"Labels: "
            f"BUY={counts.get('BUY', 0)}, "
            f"SELL={counts.get('SELL', 0)}, "
            f"HOLD={counts.get('HOLD', 0)}"
        )
        return df

    def compute_class_weights(self, df: pd.DataFrame) -> dict:
        counts = df["label_encoded"].value_counts().sort_index()
        n = len(df)
        n_classes = len(counts)
        weights = {}
        for cls, cnt in counts.items():
            weights[int(cls)] = n / (n_classes * cnt)
        return weights
