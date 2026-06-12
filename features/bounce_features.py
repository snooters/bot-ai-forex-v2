"""
Support/Resistance Bounce Detection

Detects when price bounces off support/resistance levels with confirmation
from reversal candle patterns and volume.

Computed AFTER support_resistance features are available.
"""

import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import get_logger


class BounceEngine:
    """Detect price bounces at support/resistance levels."""

    def __init__(self, bounce_threshold_pct: float = 0.002):
        """
        Args:
            bounce_threshold_pct: Max distance to S/R level to consider a bounce
                (0.002 = 0.2% ≈ 2-3 pips for EURUSD)
        """
        self.bounce_threshold_pct = bounce_threshold_pct
        self.logger = get_logger("bounce")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bounce features to dataframe.

        Requires columns:
          'close', 'high', 'low', 'open',
          'nearest_support', 'nearest_resistance',
          'dist_to_support', 'dist_to_resistance',
          'pattern_bullish', 'pattern_bearish', (from candle_patterns)

        Adds columns:
          bounce_support, bounce_resistance,
          bounce_strength (0-3), bounce_direction (+1/-1/0)
        """
        if df.empty or len(df) < 5:
            return df

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        _open = df["open"].values.astype(np.float64)
        n = len(close)

        nearest_support = df.get("nearest_support", pd.Series(0.0, index=df.index)).values
        nearest_resistance = df.get("nearest_resistance", pd.Series(0.0, index=df.index)).values
        dist_to_support = df.get("dist_to_support", pd.Series(1.0, index=df.index)).values
        dist_to_resistance = df.get("dist_to_resistance", pd.Series(1.0, index=df.index)).values

        # Candle pattern signals
        pattern_bullish = df.get("pattern_bullish", pd.Series(0, index=df.index)).values
        pattern_bearish = df.get("pattern_bearish", pd.Series(0, index=df.index)).values

        # Volume (if available)
        volume = None
        if "volume" in df.columns:
            volume = df["volume"].values.astype(np.float64)
        elif "tick_volume" in df.columns:
            volume = df["tick_volume"].values.astype(np.float64)

        # Pre-allocate
        bounce_support = np.zeros(n, dtype=np.float64)
        bounce_resistance = np.zeros(n, dtype=np.float64)
        bounce_strength = np.zeros(n, dtype=np.float64)

        # Volume moving average for confirmation
        vol_ma = None
        if volume is not None and n > 20:
            vol_ma = pd.Series(volume).rolling(20).mean().values

        for i in range(1, n):
            # --- Support Bounce ---
            if nearest_support[i] > 0:
                # Distance from low to support (price touched support?)
                price_vs_support = (low[i] - nearest_support[i]) / max(nearest_support[i], 0.0001)
                dist_pct = abs(price_vs_support)

                # Also check if close is near support
                close_vs_support = (close[i] - nearest_support[i]) / max(nearest_support[i], 0.0001)

                # Did the low touch or come very close to support?
                touched_support = dist_pct < self.bounce_threshold_pct

                # Did the close bounce off support? (close above low + near support)
                closed_above_support = close_vs_support > 0 and close_vs_support < self.bounce_threshold_pct * 3

                if touched_support or closed_above_support:
                    bounce_support[i] = 1.0

                    # Strength calculation
                    strength = 1.0

                    # 1. Bullish candle pattern confirmation
                    if pattern_bullish[i] > 0:
                        strength += 1.0

                    # 2. Bullish candlestick body (close > open, green candle)
                    if close[i] > _open[i]:
                        body_pct = (close[i] - _open[i]) / max(_open[i], 0.0001)
                        if body_pct > self.bounce_threshold_pct * 0.5:
                            strength += 0.5

                    # 3. Volume confirmation (higher than 20-period avg)
                    if volume is not None and vol_ma is not None and vol_ma[i] > 0:
                        if volume[i] > vol_ma[i] * 1.2:
                            strength += 0.5

                    # 4. Lower wick (rejection of lower prices)
                    lower_wick = min(_open[i], close[i]) - low[i]
                    candle_range = high[i] - low[i]
                    if candle_range > 0 and lower_wick / candle_range > 0.4:
                        strength += 0.5

                    bounce_strength[i] = min(strength, 3.0)

            # --- Resistance Bounce ---
            if nearest_resistance[i] > 0:
                # Distance from high to resistance (price touched resistance?)
                price_vs_resistance = (high[i] - nearest_resistance[i]) / max(nearest_resistance[i], 0.0001)
                dist_pct = abs(price_vs_resistance)

                # Also check if close is near resistance
                close_vs_resistance = (close[i] - nearest_resistance[i]) / max(nearest_resistance[i], 0.0001)

                # Did the high touch or come very close to resistance?
                touched_resistance = dist_pct < self.bounce_threshold_pct

                # Did the close reject resistance? (close below high + near resistance)
                closed_below_resistance = close_vs_resistance < 0 and abs(close_vs_resistance) < self.bounce_threshold_pct * 3

                if touched_resistance or closed_below_resistance:
                    bounce_resistance[i] = 1.0

                    # Strength calculation
                    strength = 1.0

                    # 1. Bearish candle pattern confirmation
                    if pattern_bearish[i] > 0:
                        strength += 1.0

                    # 2. Bearish candlestick body (close < open, red candle)
                    if close[i] < _open[i]:
                        body_pct = (_open[i] - close[i]) / max(_open[i], 0.0001)
                        if body_pct > self.bounce_threshold_pct * 0.5:
                            strength += 0.5

                    # 3. Volume confirmation
                    if volume is not None and vol_ma is not None and vol_ma[i] > 0:
                        if volume[i] > vol_ma[i] * 1.2:
                            strength += 0.5

                    # 4. Upper wick (rejection of higher prices)
                    upper_wick = high[i] - max(_open[i], close[i])
                    candle_range = high[i] - low[i]
                    if candle_range > 0 and upper_wick / candle_range > 0.4:
                        strength += 0.5

                    bounce_strength[i] = min(strength, 3.0)

        df["bounce_support"] = bounce_support
        df["bounce_resistance"] = bounce_resistance
        df["bounce_strength"] = bounce_strength

        # Bounce direction: +1 = support bounce (bullish), -1 = resistance bounce (bearish)
        df["bounce_direction"] = np.where(
            bounce_support > 0, 1,
            np.where(bounce_resistance > 0, -1, 0)
        ).astype(np.float64)

        # Combined: support bounce + strong confirmation
        df["bounce_buy_signal"] = np.where(
            (bounce_support > 0) & (bounce_strength >= 2.0), 1, 0
        ).astype(np.float64)

        df["bounce_sell_signal"] = np.where(
            (bounce_resistance > 0) & (bounce_strength >= 2.0), 1, 0
        ).astype(np.float64)

        return df
