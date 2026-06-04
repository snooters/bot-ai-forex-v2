import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from utils.logger import get_logger

CONTEXT_TFS = [15, 30, 60, 240]


class MultiTFFeatureEngine:
    def __init__(self):
        self.logger = get_logger("multi_tf_features")

    def compute(self, df_aligned: pd.DataFrame) -> pd.DataFrame:
        if df_aligned.empty:
            return df_aligned

        df = df_aligned.copy()

        for tf in CONTEXT_TFS:
            suffix = f"_tf{tf}"
            close_col = f"close{suffix}"
            high_col = f"high{suffix}"
            low_col = f"low{suffix}"
            vol_col = f"volume{suffix}"

            if close_col not in df.columns:
                continue

            # ── Required EMAs per TF spec ──
            # M15 (15): EMA20, EMA50, EMA200
            # M30 (30): EMA20, EMA50, EMA200
            # H1  (60): EMA50, EMA200
            # H4 (240): EMA50, EMA200
            ema20_col = f"ema_20{suffix}" if tf <= 30 else None
            ema50_col = f"ema_50{suffix}"
            ema200_col = f"ema_200{suffix}"

            if ema20_col and ema20_col not in df.columns:
                df[ema20_col] = df[close_col].ewm(span=20, adjust=False).mean()
            if ema50_col not in df.columns:
                df[ema50_col] = df[close_col].ewm(span=50, adjust=False).mean()
            if ema200_col not in df.columns:
                df[ema200_col] = df[close_col].ewm(span=200, adjust=False).mean()

            # trend direction: close vs EMA20 (for TFs with EMA20) or EMA50
            if tf <= 30:
                df[f"trend{tf}"] = np.where(
                    df[close_col] > df[ema20_col], 1,
                    np.where(df[close_col] < df[ema20_col], -1, 0),
                )
            else:
                df[f"trend{tf}"] = np.where(
                    df[close_col] > df[ema50_col], 1,
                    np.where(df[close_col] < df[ema50_col], -1, 0),
                )

            df[f"ema_cross{tf}"] = (
                df[close_col].ewm(span=20, adjust=False).mean()
                - df[close_col].ewm(span=50, adjust=False).mean()
            )

            df[f"momentum{tf}"] = df[close_col].pct_change(periods=3)

            atr_raw = (df[high_col] - df[low_col]).rolling(14).mean()
            df[f"atr{tf}"] = atr_raw

            df[f"volatility{tf}"] = atr_raw / df[close_col].rolling(20).mean()

            df[f"rsi{tf}"] = self._compute_rsi(df[close_col], 14)

            # MACD only for M15
            if tf <= 15:
                df[f"macd{tf}"] = (
                    df[close_col].ewm(span=12, adjust=False).mean()
                    - df[close_col].ewm(span=26, adjust=False).mean()
                )

            if tf <= 30:
                df[f"close_vs_ema20{tf}"] = (
                    (df[close_col] - df[ema20_col]) / df[ema20_col]
                )

            min_di = df[low_col].diff().clip(upper=0).abs().rolling(14).mean()
            plus_di = df[high_col].diff().clip(lower=0).rolling(14).mean()
            tr = pd.concat(
                [
                    df[high_col] - df[low_col],
                    (df[high_col] - df[close_col].shift(1)).abs(),
                    (df[low_col] - df[close_col].shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr14 = tr.rolling(14).mean()
            df[f"adx{tf}"] = (
                100
                * ((plus_di - min_di).abs() / (plus_di + min_di).replace(0, np.nan))
                .rolling(14)
                .mean()
            )

            n_sma = df[f"adx{tf}"].notna().sum()
            if n_sma > 0:
                df[f"adx_strong{tf}"] = (df[f"adx{tf}"] > 25).astype(int)

        # alignment score: how many context TFs agree with primary
        primary_trend = np.where(
            df["close"] > df["close"].ewm(span=20, adjust=False).mean(), 1, -1
        )
        for tf in CONTEXT_TFS:
            trend_col = f"trend{tf}"
            if trend_col in df.columns:
                df[f"align{tf}"] = (primary_trend == df[trend_col]).astype(int)

        align_cols = [f"align{tf}" for tf in CONTEXT_TFS if f"align{tf}" in df.columns]
        if align_cols:
            df["mtf_alignment"] = df[align_cols].sum(axis=1) / len(align_cols)
        else:
            df["mtf_alignment"] = 0.0

        return df

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
