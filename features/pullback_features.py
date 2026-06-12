"""
Pullback Detection — Price Pullback to Higher TF EMAs

Detects when price pulls back to key EMA levels on higher timeframes (H1, H4).
In an uptrend, price pulling back to EMA20/50 = potential buy opportunity.
In a downtrend, price pulling back to EMA20/50 = potential sell opportunity.

Features computed AFTER multi_tf_features (which provides higher TF EMA columns).
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from utils.logger import get_logger

# Higher TFs for pullback detection (H1, H4)
PULLBACK_TFS = [60, 240]


class PullbackEngine:
    """Detect price pullbacks to higher timeframe EMAs."""

    def __init__(self, threshold_pct: float = 0.003):
        """
        Args:
            threshold_pct: Distance threshold to consider price "near" EMA (0.003 = 0.3%)
        """
        self.threshold_pct = threshold_pct
        self.logger = get_logger("pullback")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pullback features to dataframe.

        Requires columns from multi_tf_features:
          close_vs_ema20{tf}, trend{tf}, ema_20_tf{tf}, ema_50_tf{tf}

        Adds columns:
          pullback_to_ema20_{tf}, pullback_to_ema50_{tf},
          pullback_dist_{tf}, pullback_active_{tf},
          pullback_quality
        """
        if df.empty or len(df) < 50:
            return df

        # For each higher TF, compute pullback metrics
        for tf in PULLBACK_TFS:
            close_vs_ema20_col = f"close_vs_ema20{tf}"
            trend_col = f"trend{tf}"
            ema20_col = f"ema_20_tf{tf}"
            ema50_col = f"ema_50_tf{tf}"

            # Distance to EMA20 (%)
            if close_vs_ema20_col in df.columns:
                df[f"pullback_to_ema20_{tf}"] = df[close_vs_ema20_col].abs()
            else:
                df[f"pullback_to_ema20_{tf}"] = 1.0  # Default far away

            # Distance to EMA50 (compute if ema50 available)
            if ema50_col in df.columns and "close" in df.columns:
                ema50 = df[ema50_col]
                close = df["close"]
                dist_to_ema50 = ((close - ema50) / ema50.replace(0, np.nan)).abs()
                df[f"pullback_to_ema50_{tf}"] = dist_to_ema50.fillna(1.0)
            else:
                df[f"pullback_to_ema50_{tf}"] = 1.0

            # Is pullback active? (price near EMA + trend in same direction)
            if trend_col in df.columns and close_vs_ema20_col in df.columns:
                trend = df[trend_col]
                close_vs_ema20 = df[close_vs_ema20_col]

                # Active pullback BUY: uptrend (trend > 0) + price near or below EMA20
                pullback_buy = np.where(
                    (trend > 0) & (close_vs_ema20 < self.threshold_pct),
                    1, 0
                ).astype(np.float64)

                # Active pullback SELL: downtrend (trend < 0) + price near or above EMA20
                pullback_sell = np.where(
                    (trend < 0) & (close_vs_ema20 > -self.threshold_pct),
                    1, 0
                ).astype(np.float64)

                df[f"pullback_active_buy_{tf}"] = pullback_buy
                df[f"pullback_active_sell_{tf}"] = pullback_sell

                # Combined: +1 for buy pullback, -1 for sell pullback
                df[f"pullback_active_{tf}"] = np.where(
                    pullback_buy > 0, 1,
                    np.where(pullback_sell > 0, -1, 0)
                ).astype(np.float64)
            else:
                df[f"pullback_active_buy_{tf}"] = 0.0
                df[f"pullback_active_sell_{tf}"] = 0.0
                df[f"pullback_active_{tf}"] = 0.0

        # Aggregate pullback quality score (0-5)
        # Higher = more TFs agree on pullback + deeper pullback + RSI confirmation
        quality = np.zeros(len(df), dtype=np.float64)

        for tf in PULLBACK_TFS:
            active_col = f"pullback_active_{tf}"
            dist_ema20_col = f"pullback_to_ema20_{tf}"
            dist_ema50_col = f"pullback_to_ema50_{tf}"

            if active_col in df.columns:
                active = df[active_col].values
                dist20 = df.get(dist_ema20_col, pd.Series(1.0, index=df.index)).values
                dist50 = df.get(dist_ema50_col, pd.Series(1.0, index=df.index)).values

                # Score 1 per TF if pullback active
                quality += np.abs(active)

                # Bonus for near EMA50 (deeper pullback = better reversal)
                near_ema50 = (dist50 < self.threshold_pct * 2).astype(np.float64)
                quality += near_ema50 * 0.5

                # Bonus for near EMA20 (very close)
                near_ema20 = (dist20 < self.threshold_pct).astype(np.float64)
                quality += near_ema20 * 0.5

        # RSI confirmation: if pullback direction matches RSI extreme
        if "rsi" in df.columns:
            rsi = df["rsi"].values
            for tf in PULLBACK_TFS:
                active_col = f"pullback_active_{tf}"
                if active_col in df.columns:
                    active = df[active_col].values
                    # Buy pullback + RSI < 30 = stronger signal
                    rsi_buy_conf = (active > 0) & (rsi < 35)
                    quality += rsi_buy_conf.astype(np.float64) * 1.0
                    # Sell pullback + RSI > 70 = stronger signal
                    rsi_sell_conf = (active < 0) & (rsi > 65)
                    quality += rsi_sell_conf.astype(np.float64) * 1.0

        df["pullback_quality"] = quality.clip(0, 5)

        # Pullback direction: +1 buy, -1 sell, 0 none
        pb60 = df.get("pullback_active_60", pd.Series(0.0, index=df.index))
        pb240 = df.get("pullback_active_240", pd.Series(0.0, index=df.index))
        df["pullback_direction"] = np.sign(pb60 + pb240).astype(np.float64)

        return df
