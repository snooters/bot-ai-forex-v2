import numpy as np
import pandas as pd
from typing import List, Dict

from core.constants import CandlePattern
from utils.logger import get_logger


class CandlePatternEngine:
    def __init__(self):
        self.logger = get_logger("candle_patterns")

    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 5 or "open" not in df.columns:
            return df

        df["candle_pattern"] = CandlePattern.NONE.value
        df["pattern_bullish"] = 0
        df["pattern_bearish"] = 0

        self._detect_hammer(df)
        self._detect_inverted_hammer(df)
        self._detect_pin_bar(df)
        self._detect_doji(df)
        self._detect_engulfing(df)
        self._detect_morning_evening_star(df)
        self._detect_shooting_star(df)
        self._detect_three_soldiers(df)
        self._detect_three_crows(df)

        return df

    def _detect_hammer(self, df: pd.DataFrame):
        o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
        body = np.abs(c - o)
        lower_wick = np.minimum(c, o) - l
        upper_wick = h - np.maximum(c, o)
        total_range = h - l

        mask = (total_range > 0) & (lower_wick >= 2 * body) & (upper_wick <= 0.3 * body) & (body > 0) & (c > o)
        idx = df.index[mask]
        if len(idx):
            df.loc[idx, "candle_pattern"] = CandlePattern.HAMMER.value
            df.loc[idx, "pattern_bullish"] = 1

    def _detect_inverted_hammer(self, df: pd.DataFrame):
        o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
        body = np.abs(c - o)
        lower_wick = np.minimum(c, o) - l
        upper_wick = h - np.maximum(c, o)
        total_range = h - l

        mask = (total_range > 0) & (upper_wick >= 2 * body) & (lower_wick <= 0.3 * body) & (body > 0)
        idx = df.index[mask]
        if len(idx):
            df.loc[idx, "candle_pattern"] = CandlePattern.INVERTED_HAMMER.value
            df.loc[idx, "pattern_bullish"] = 1

    def _detect_pin_bar(self, df: pd.DataFrame):
        o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
        body = np.abs(c - o)
        lower_wick = np.minimum(c, o) - l
        upper_wick = h - np.maximum(c, o)
        total_range = h - l

        mask = (total_range > 0) & (body <= 0.3 * total_range) & (body > 0)
        long_lower = lower_wick >= 2 * body
        long_upper = upper_wick >= 2 * body
        pin_mask = mask & (long_lower | long_upper)
        idx = df.index[pin_mask]
        if len(idx):
            df.loc[idx, "candle_pattern"] = CandlePattern.PIN_BAR.value
            bull_idx = df.index[pin_mask & long_lower]
            bear_idx = df.index[pin_mask & ~long_lower & long_upper]
            if len(bull_idx):
                df.loc[bull_idx, "pattern_bullish"] = 1
            if len(bear_idx):
                df.loc[bear_idx, "pattern_bearish"] = 1

    def _detect_doji(self, df: pd.DataFrame):
        o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
        body = np.abs(c - o)
        total_range = h - l
        mask = (total_range > 0) & (body <= 0.1 * total_range)
        idx = df.index[mask]
        if len(idx):
            df.loc[idx, "candle_pattern"] = CandlePattern.DOJI.value

    def _detect_engulfing(self, df: pd.DataFrame):
        o, c = df["open"].values, df["close"].values
        body = np.abs(c - o)
        bullish = c > o

        prev_o = np.roll(o, 1); prev_o[:1] = np.nan
        prev_c = np.roll(c, 1); prev_c[:1] = np.nan
        prev_body = np.abs(prev_c - prev_o)
        prev_bullish = prev_c > prev_o

        mask = (prev_body > 0) & (body > 0)
        bull_mask = mask & ~prev_bullish & bullish & (c > prev_o) & (o < prev_c)
        bear_mask = mask & prev_bullish & ~bullish & (c < prev_o) & (o > prev_c)

        bull_idx = df.index[bull_mask]
        bear_idx = df.index[bear_mask]
        if len(bull_idx):
            df.loc[bull_idx, "candle_pattern"] = CandlePattern.BULLISH_ENGULFING.value
            df.loc[bull_idx, "pattern_bullish"] = 1
        if len(bear_idx):
            df.loc[bear_idx, "candle_pattern"] = CandlePattern.BEARISH_ENGULFING.value
            df.loc[bear_idx, "pattern_bearish"] = 1

    def _detect_morning_evening_star(self, df: pd.DataFrame):
        o, c = df["open"].values, df["close"].values
        body = np.abs(c - o)
        bullish = c > o

        o2 = np.roll(o, 2); o2[:2] = np.nan
        c2 = np.roll(c, 2); c2[:2] = np.nan
        o1 = np.roll(o, 1); o1[:1] = np.nan
        c1 = np.roll(c, 1); c1[:1] = np.nan

        body2 = np.abs(c2 - o2)
        body1 = np.abs(c1 - o1)
        bullish2 = c2 > o2

        mask = (body2 > 0) & (body > 0)
        min_body = np.minimum(body2, body)
        mid_avg = (o2 + c2) / 2

        morning_mask = mask & ~bullish2 & bullish & (body1 <= min_body * 0.5) & (c > mid_avg)
        evening_mask = mask & bullish2 & ~bullish & (body1 <= min_body * 0.5) & (c < mid_avg)

        m_idx = df.index[morning_mask]
        e_idx = df.index[evening_mask]
        if len(m_idx):
            df.loc[m_idx, "candle_pattern"] = CandlePattern.MORNING_STAR.value
            df.loc[m_idx, "pattern_bullish"] = 1
        if len(e_idx):
            df.loc[e_idx, "candle_pattern"] = CandlePattern.EVENING_STAR.value
            df.loc[e_idx, "pattern_bearish"] = 1

    def _detect_shooting_star(self, df: pd.DataFrame):
        o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
        body = np.abs(c - o)
        lower_wick = np.minimum(c, o) - l
        upper_wick = h - np.maximum(c, o)
        total_range = h - l

        mask = (total_range > 0) & (upper_wick >= 2 * body) & (lower_wick <= 0.3 * body) & (body > 0) & (c < o)
        idx = df.index[mask]
        if len(idx):
            df.loc[idx, "candle_pattern"] = CandlePattern.SHOOTING_STAR.value
            df.loc[idx, "pattern_bearish"] = 1

    def _detect_three_soldiers(self, df: pd.DataFrame):
        o, c = df["open"].values, df["close"].values
        body = np.abs(c - o)
        bullish = c > o

        o2 = np.roll(o, 2); o2[:2] = np.nan
        c2 = np.roll(c, 2); c2[:2] = np.nan
        o1 = np.roll(o, 1); o1[:1] = np.nan
        c1 = np.roll(c, 1); c1[:1] = np.nan
        b1 = np.roll(bullish, 2); b1[:2] = False
        b2 = np.roll(bullish, 1); b2[:1] = False
        body2 = np.abs(c2 - o2)
        body1 = np.abs(c1 - o1)

        mask = (body2 > 0) & (body1 > 0) & (body > 0) & b1 & b2 & bullish
        cond = (mask &
                (body1 <= body2 * 1.5) & (body <= body1 * 1.5) &
                (c1 > c2) & (c > c1) &
                (o1 > o2) & (o > o1) &
                (o1 > c2 * 0.98) & (o > c1 * 0.98))
        idx = df.index[cond]
        if len(idx):
            df.loc[idx, "candle_pattern"] = CandlePattern.THREE_WHITE_SOLDIERS.value
            df.loc[idx, "pattern_bullish"] = 1

    def _detect_three_crows(self, df: pd.DataFrame):
        o, c = df["open"].values, df["close"].values
        body = np.abs(c - o)
        bearish = c < o

        o2 = np.roll(o, 2); o2[:2] = np.nan
        c2 = np.roll(c, 2); c2[:2] = np.nan
        o1 = np.roll(o, 1); o1[:1] = np.nan
        c1 = np.roll(c, 1); c1[:1] = np.nan
        b1 = np.roll(bearish, 2); b1[:2] = False
        b2 = np.roll(bearish, 1); b2[:1] = False
        body2 = np.abs(c2 - o2)
        body1 = np.abs(c1 - o1)

        mask = (body2 > 0) & (body1 > 0) & (body > 0) & b1 & b2 & bearish
        cond = (mask &
                (body1 <= body2 * 1.5) & (body <= body1 * 1.5) &
                (c1 < c2) & (c < c1) &
                (o1 < o2) & (o < o1) &
                (o1 < c2 * 1.02) & (o < c1 * 1.02))
        idx = df.index[cond]
        if len(idx):
            df.loc[idx, "candle_pattern"] = CandlePattern.THREE_BLACK_CROWS.value
            df.loc[idx, "pattern_bearish"] = 1

    def get_current_pattern(self, df: pd.DataFrame) -> str:
        if "candle_pattern" not in df.columns:
            return CandlePattern.NONE.value
        recent = df["candle_pattern"].dropna().tail(3)
        if recent.empty:
            return CandlePattern.NONE.value
        return recent.iloc[-1]

    def get_pattern_signal(self, df: pd.DataFrame) -> int:
        if "candle_pattern" not in df.columns or df.empty:
            return 0
        last = df.iloc[-1]
        if last.get("pattern_bullish", 0) == 1:
            return 1
        if last.get("pattern_bearish", 0) == 1:
            return -1
        return 0

    def count_bullish_patterns(self, df: pd.DataFrame, lookback: int = 10) -> int:
        if "pattern_bullish" not in df.columns:
            return 0
        recent = df["pattern_bullish"].tail(lookback).sum()
        return int(recent)

    def count_bearish_patterns(self, df: pd.DataFrame, lookback: int = 10) -> int:
        if "pattern_bearish" not in df.columns:
            return 0
        recent = df["pattern_bearish"].tail(lookback).sum()
        return int(recent)
