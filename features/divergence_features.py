"""
RSI Divergence Detection — Regular + Hidden Divergence

Detects 4 types of RSI divergence:
  - Regular Bullish:   price lower low, RSI higher low   → trend reversal UP
  - Hidden Bullish:    price higher low, RSI lower low    → trend continuation UP
  - Regular Bearish:   price higher high, RSI lower high  → trend reversal DOWN
  - Hidden Bearish:    price lower high, RSI higher high  → trend continuation DOWN

Integrated into feature pipeline as binary + continuous features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from utils.logger import get_logger


class DivergenceEngine:
    """Detect RSI divergence patterns for reversal signals."""

    def __init__(self, swing_window: int = 5, max_lookback: int = 50):
        """
        Args:
            swing_window: Half-window for finding pivot highs/lows (total = 2*swing_window+1)
            max_lookback: Maximum bars to look back for divergence detection
        """
        self.swing_window = swing_window
        self.max_lookback = max_lookback
        self.logger = get_logger("divergence")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI divergence features to dataframe.

        Requires columns: 'close', 'rsi'
        Adds columns: 'div_reg_bull', 'div_hid_bull', 'div_reg_bear', 'div_hid_bear', 'div_strength'
        """
        if df.empty or len(df) < 30:
            return df

        close = df["close"].values.astype(np.float64)
        rsi = df["rsi"].values.astype(np.float64)

        n = len(close)

        # Pre-allocate outputs
        div_reg_bull = np.zeros(n, dtype=np.float64)
        div_hid_bull = np.zeros(n, dtype=np.float64)
        div_reg_bear = np.zeros(n, dtype=np.float64)
        div_hid_bear = np.zeros(n, dtype=np.float64)

        w = self.swing_window
        lb = min(self.max_lookback, n)

        # Find swing highs and lows
        swing_highs = self._find_swing_highs(close, w)
        swing_lows = self._find_swing_lows(close, w)

        for i in range(lb, n):
            # Only detect divergence at swing points or recent bars
            # Look back from i to find the last 2 swing highs and lows

            # --- Regular Bullish Divergence ---
            # Price makes lower low, RSI makes higher low
            lows_idx = np.where(swing_lows[max(0, i-lb):i+1])[0] + max(0, i-lb)
            if len(lows_idx) >= 2:
                last2 = lows_idx[-2:]
                p0, p1 = last2[0], last2[1]
                price_lower = close[p1] < close[p0]
                rsi_higher = rsi[p1] > rsi[p0]
                if price_lower and rsi_higher and rsi[p1] < 50:
                    div_reg_bull[i] = 1.0

            # --- Hidden Bullish Divergence ---
            # Price makes higher low, RSI makes lower low
            if len(lows_idx) >= 2:
                last2 = lows_idx[-2:]
                p0, p1 = last2[0], last2[1]
                price_higher = close[p1] > close[p0]
                rsi_lower = rsi[p1] < rsi[p0]
                if price_higher and rsi_lower and rsi[p1] < 50:
                    div_hid_bull[i] = 1.0

            # --- Regular Bearish Divergence ---
            # Price makes higher high, RSI makes lower high
            highs_idx = np.where(swing_highs[max(0, i-lb):i+1])[0] + max(0, i-lb)
            if len(highs_idx) >= 2:
                last2 = highs_idx[-2:]
                p0, p1 = last2[0], last2[1]
                price_higher = close[p1] > close[p0]
                rsi_lower = rsi[p1] < rsi[p0]
                if price_higher and rsi_lower and rsi[p1] > 50:
                    div_reg_bear[i] = 1.0

            # --- Hidden Bearish Divergence ---
            # Price makes lower high, RSI makes higher high
            if len(highs_idx) >= 2:
                last2 = highs_idx[-2:]
                p0, p1 = last2[0], last2[1]
                price_lower = close[p1] < close[p0]
                rsi_higher = rsi[p1] > rsi[p0]
                if price_lower and rsi_higher and rsi[p1] > 50:
                    div_hid_bear[i] = 1.0

        df["div_reg_bull"] = div_reg_bull
        df["div_hid_bull"] = div_hid_bull
        df["div_reg_bear"] = div_reg_bear
        df["div_hid_bear"] = div_hid_bear

        # Aggregate strength: count how many divergence types are active
        df["div_strength"] = (
            df["div_reg_bull"]
            + df["div_hid_bull"]
            + df["div_reg_bear"]
            + df["div_hid_bear"]
        )

        # Divergence direction: +1 for bullish, -1 for bearish, 0 for none
        df["div_direction"] = np.where(
            (df["div_reg_bull"] > 0) | (df["div_hid_bull"] > 0), 1,
            np.where(
                (df["div_reg_bear"] > 0) | (df["div_hid_bear"] > 0), -1, 0
            ),
        )

        return df

    def _find_swing_highs(self, series: np.ndarray, window: int) -> np.ndarray:
        """Find swing highs: points higher than `window` neighbours on each side."""
        n = len(series)
        result = np.zeros(n, dtype=bool)
        for i in range(window, n - window):
            left = series[i - window:i]
            right = series[i + 1:i + window + 1]
            if series[i] > np.max(left) and series[i] > np.max(right):
                result[i] = True
        return result

    def _find_swing_lows(self, series: np.ndarray, window: int) -> np.ndarray:
        """Find swing lows: points lower than `window` neighbours on each side."""
        n = len(series)
        result = np.zeros(n, dtype=bool)
        for i in range(window, n - window):
            left = series[i - window:i]
            right = series[i + 1:i + window + 1]
            if series[i] < np.min(left) and series[i] < np.min(right):
                result[i] = True
        return result
