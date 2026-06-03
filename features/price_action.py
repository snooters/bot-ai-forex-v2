import numpy as np
import pandas as pd
from typing import Dict, List

from core.constants import PriceActionPattern
from utils.logger import get_logger


class PriceActionEngine:
    def __init__(self):
        self.logger = get_logger("price_action")

    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 10:
            return df

        df["price_action"] = PriceActionPattern.NONE.value

        self._detect_liquidity_grab(df)
        self._detect_rejection(df)
        self._detect_retest(df)
        self._detect_momentum_candle(df)
        self._detect_breakout(df)
        self._detect_fake_breakout(df)

        return df

    def _detect_liquidity_grab(self, df: pd.DataFrame):
        for i in range(3, len(df)):
            prev_high = df["high"].iloc[i - 1]
            prev_low = df["low"].iloc[i - 1]
            curr_high = df["high"].iloc[i]
            curr_low = df["low"].iloc[i]
            curr_close = df["close"].iloc[i]
            curr_open = df["open"].iloc[i] if "open" in df.columns else df["close"].iloc[i]

            if i >= 2:
                high_2 = df["high"].iloc[i - 2]
                low_2 = df["low"].iloc[i - 2]

                if (
                    curr_high > high_2 and
                    curr_high > prev_high and
                    curr_close < prev_high and
                    curr_low < curr_open
                ):
                    df.at[df.index[i], "price_action"] = PriceActionPattern.LIQUIDITY_GRAB.value

                if (
                    curr_low < low_2 and
                    curr_low < prev_low and
                    curr_close > prev_low and
                    curr_high > curr_open
                ):
                    df.at[df.index[i], "price_action"] = PriceActionPattern.LIQUIDITY_GRAB.value

    def _detect_rejection(self, df: pd.DataFrame):
        for i in range(1, len(df)):
            body = abs(df["close"].iloc[i] - df["open"].iloc[i]) if "open" in df.columns else 0
            upper_wick = df["high"].iloc[i] - max(df["close"].iloc[i], df["open"].iloc[i]) if "open" in df.columns else 0
            lower_wick = min(df["close"].iloc[i], df["open"].iloc[i]) - df["low"].iloc[i] if "open" in df.columns else 0
            total_range = df["high"].iloc[i] - df["low"].iloc[i]

            if total_range == 0:
                continue

            if upper_wick > 2 * body and upper_wick > 0.5 * total_range:
                df.at[df.index[i], "price_action"] = PriceActionPattern.REJECTION.value

            if lower_wick > 2 * body and lower_wick > 0.5 * total_range:
                df.at[df.index[i], "price_action"] = PriceActionPattern.REJECTION.value

    def _detect_retest(self, df: pd.DataFrame):
        for i in range(3, len(df)):
            prev_high = df["high"].iloc[i - 1]
            prev_low = df["low"].iloc[i - 1]
            curr_close = df["close"].iloc[i]

            if abs(curr_close - prev_high) / prev_high < 0.001:
                df.at[df.index[i], "price_action"] = PriceActionPattern.RETEST.value

            if abs(curr_close - prev_low) / prev_low < 0.001:
                df.at[df.index[i], "price_action"] = PriceActionPattern.RETEST.value

    def _detect_momentum_candle(self, df: pd.DataFrame):
        for i in range(1, len(df)):
            if "open" not in df.columns:
                continue
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            avg_body = df["close"].diff().abs().rolling(20).mean().iloc[i]

            if avg_body == 0 or pd.isna(avg_body):
                continue

            if body > avg_body * 2:
                df.at[df.index[i], "price_action"] = PriceActionPattern.MOMENTUM_CANDLE.value

    def _detect_breakout(self, df: pd.DataFrame):
        for i in range(5, len(df)):
            range_high = df["high"].iloc[i - 5:i].max()
            range_low = df["low"].iloc[i - 5:i].min()
            curr_close = df["close"].iloc[i]
            curr_open = df["open"].iloc[i] if "open" in df.columns else df["close"].iloc[i]

            if curr_close > range_high and curr_open < range_high:
                df.at[df.index[i], "price_action"] = PriceActionPattern.BREAKOUT.value

            if curr_close < range_low and curr_open > range_low:
                df.at[df.index[i], "price_action"] = PriceActionPattern.BREAKOUT.value

    def _detect_fake_breakout(self, df: pd.DataFrame):
        for i in range(6, len(df)):
            range_high = df["high"].iloc[i - 5:i].max()
            range_low = df["low"].iloc[i - 5:i].min()

            prev_close = df["close"].iloc[i - 1]
            prev_open = df["open"].iloc[i - 1] if "open" in df.columns else df["close"].iloc[i - 1]
            curr_close = df["close"].iloc[i]

            if prev_close > range_high and curr_close < range_high:
                df.at[df.index[i - 1], "price_action"] = PriceActionPattern.FAKE_BREAKOUT.value

            if prev_close < range_low and curr_close > range_low:
                df.at[df.index[i - 1], "price_action"] = PriceActionPattern.FAKE_BREAKDOWN.value

    def get_current_pattern(self, df: pd.DataFrame) -> str:
        if "price_action" not in df.columns:
            return PriceActionPattern.NONE.value
        recent = df["price_action"].dropna().tail(5)
        if recent.empty:
            return PriceActionPattern.NONE.value
        return recent.iloc[-1]
