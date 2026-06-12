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
        if len(df) < 3:
            return
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        open_ = df["open"].values if "open" in df.columns else close
        pa = df["price_action"].values

        for i in range(3, len(df)):
            if (high[i] > high[i - 2] and high[i] > high[i - 1] and
                close[i] < high[i - 1] and low[i] < open_[i]):
                pa[i] = PriceActionPattern.LIQUIDITY_GRAB.value
            elif (low[i] < low[i - 2] and low[i] < low[i - 1] and
                  close[i] > low[i - 1] and high[i] > open_[i]):
                pa[i] = PriceActionPattern.LIQUIDITY_GRAB.value

    def _detect_rejection(self, df: pd.DataFrame):
        if len(df) < 1 or "open" not in df.columns:
            return
        open_ = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        pa = df["price_action"].values

        body = np.abs(close - open_)
        upper_wick = high - np.maximum(close, open_)
        lower_wick = np.minimum(close, open_) - low
        total_range = high - low

        mask = total_range > 0
        up_mask = mask & (upper_wick > 2 * body) & (upper_wick > 0.5 * total_range)
        lo_mask = mask & (lower_wick > 2 * body) & (lower_wick > 0.5 * total_range)
        pa[up_mask | lo_mask] = PriceActionPattern.REJECTION.value

    def _detect_retest(self, df: pd.DataFrame):
        if len(df) < 3:
            return
        close = df["close"].values
        prev_high = np.roll(df["high"].values, 1)
        prev_low = np.roll(df["low"].values, 1)
        prev_high[:1] = np.nan
        prev_low[:1] = np.nan
        pa = df["price_action"].values

        mask_high = (np.abs(close - prev_high) / np.where(prev_high == 0, 1e-10, prev_high)) < 0.001
        mask_low = (np.abs(close - prev_low) / np.where(prev_low == 0, 1e-10, prev_low)) < 0.001
        pa[mask_high | mask_low] = PriceActionPattern.RETEST.value

    def _detect_momentum_candle(self, df: pd.DataFrame):
        if "open" not in df.columns:
            return
        body = (df["close"] - df["open"]).abs()
        avg_body = body.rolling(20, min_periods=1).mean()
        cond = (body > avg_body * 2) & (avg_body > 0)
        df.loc[cond, "price_action"] = PriceActionPattern.MOMENTUM_CANDLE.value

    def _detect_breakout(self, df: pd.DataFrame):
        range_high = df["high"].rolling(5, min_periods=5).max()
        range_low = df["low"].rolling(5, min_periods=5).min()
        open_ = df["open"] if "open" in df.columns else df["close"]
        cond_buy = (df["close"] > range_high) & (open_ < range_high)
        cond_sell = (df["close"] < range_low) & (open_ > range_low)
        df.loc[cond_buy | cond_sell, "price_action"] = PriceActionPattern.BREAKOUT.value

    def _detect_fake_breakout(self, df: pd.DataFrame):
        range_high = df["high"].rolling(5, min_periods=5).max()
        range_low = df["low"].rolling(5, min_periods=5).min()
        prev_close = df["close"].shift(1)
        curr_close = df["close"]

        cond_fake_buy = (prev_close > range_high) & (curr_close < range_high)
        cond_fake_sell = (prev_close < range_low) & (curr_close > range_low)
        df.loc[cond_fake_buy, "price_action"] = PriceActionPattern.FAKE_BREAKOUT.value
        df.loc[cond_fake_sell, "price_action"] = PriceActionPattern.FAKE_BREAKDOWN.value

    def get_current_pattern(self, df: pd.DataFrame) -> str:
        if "price_action" not in df.columns:
            return PriceActionPattern.NONE.value
        recent = df["price_action"].dropna().tail(5)
        if recent.empty:
            return PriceActionPattern.NONE.value
        return recent.iloc[-1]
