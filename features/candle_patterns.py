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
        for i in range(1, len(df)):
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            lower_wick = min(df["close"].iloc[i], df["open"].iloc[i]) - df["low"].iloc[i]
            upper_wick = df["high"].iloc[i] - max(df["close"].iloc[i], df["open"].iloc[i])
            total_range = df["high"].iloc[i] - df["low"].iloc[i]

            if total_range == 0:
                continue

            if (
                lower_wick >= 2 * body
                and upper_wick <= 0.3 * body
                and body > 0
                and df["close"].iloc[i] > df["open"].iloc[i]
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.HAMMER.value
                df.at[df.index[i], "pattern_bullish"] = 1

    def _detect_inverted_hammer(self, df: pd.DataFrame):
        for i in range(1, len(df)):
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            lower_wick = min(df["close"].iloc[i], df["open"].iloc[i]) - df["low"].iloc[i]
            upper_wick = df["high"].iloc[i] - max(df["close"].iloc[i], df["open"].iloc[i])
            total_range = df["high"].iloc[i] - df["low"].iloc[i]

            if total_range == 0:
                continue

            if (
                upper_wick >= 2 * body
                and lower_wick <= 0.3 * body
                and body > 0
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.INVERTED_HAMMER.value
                df.at[df.index[i], "pattern_bullish"] = 1

    def _detect_pin_bar(self, df: pd.DataFrame):
        for i in range(1, len(df)):
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            lower_wick = min(df["close"].iloc[i], df["open"].iloc[i]) - df["low"].iloc[i]
            upper_wick = df["high"].iloc[i] - max(df["close"].iloc[i], df["open"].iloc[i])
            total_range = df["high"].iloc[i] - df["low"].iloc[i]

            if total_range == 0:
                continue

            if (
                (lower_wick >= 2 * body or upper_wick >= 2 * body)
                and body <= 0.3 * total_range
                and body > 0
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.PIN_BAR.value
                if lower_wick >= 2 * body:
                    df.at[df.index[i], "pattern_bullish"] = 1
                else:
                    df.at[df.index[i], "pattern_bearish"] = 1

    def _detect_doji(self, df: pd.DataFrame):
        for i in range(1, len(df)):
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            total_range = df["high"].iloc[i] - df["low"].iloc[i]

            if total_range == 0:
                continue

            if body <= 0.1 * total_range:
                df.at[df.index[i], "candle_pattern"] = CandlePattern.DOJI.value

    def _detect_engulfing(self, df: pd.DataFrame):
        for i in range(1, len(df)):
            prev_body = abs(df["close"].iloc[i - 1] - df["open"].iloc[i - 1])
            curr_body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            prev_bullish = df["close"].iloc[i - 1] > df["open"].iloc[i - 1]
            curr_bullish = df["close"].iloc[i] > df["open"].iloc[i]

            if prev_body == 0 or curr_body == 0:
                continue

            if (
                not prev_bullish and curr_bullish
                and df["close"].iloc[i] > df["open"].iloc[i - 1]
                and df["open"].iloc[i] < df["close"].iloc[i - 1]
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.BULLISH_ENGULFING.value
                df.at[df.index[i], "pattern_bullish"] = 1

            if (
                prev_bullish and not curr_bullish
                and df["close"].iloc[i] < df["open"].iloc[i - 1]
                and df["open"].iloc[i] > df["close"].iloc[i - 1]
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.BEARISH_ENGULFING.value
                df.at[df.index[i], "pattern_bearish"] = 1

    def _detect_morning_evening_star(self, df: pd.DataFrame):
        for i in range(2, len(df)):
            c1, c2, c3 = (
                df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
            )

            body1 = abs(c1["close"] - c1["open"])
            body2 = abs(c2["close"] - c2["open"])
            body3 = abs(c3["close"] - c3["open"])

            if body1 == 0 or body3 == 0:
                continue

            bullish1 = c1["close"] > c1["open"]
            bullish3 = c3["close"] > c3["open"]

            if (
                not bullish1
                and bullish3
                and body2 <= min(body1, body3) * 0.5
                and c3["close"] > (c1["open"] + c1["close"]) / 2
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.MORNING_STAR.value
                df.at[df.index[i], "pattern_bullish"] = 1

            if (
                bullish1
                and not bullish3
                and body2 <= min(body1, body3) * 0.5
                and c3["close"] < (c1["open"] + c1["close"]) / 2
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.EVENING_STAR.value
                df.at[df.index[i], "pattern_bearish"] = 1

    def _detect_shooting_star(self, df: pd.DataFrame):
        for i in range(1, len(df)):
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            lower_wick = min(df["close"].iloc[i], df["open"].iloc[i]) - df["low"].iloc[i]
            upper_wick = df["high"].iloc[i] - max(df["close"].iloc[i], df["open"].iloc[i])
            total_range = df["high"].iloc[i] - df["low"].iloc[i]

            if total_range == 0:
                continue

            if (
                upper_wick >= 2 * body
                and lower_wick <= 0.3 * body
                and body > 0
                and df["close"].iloc[i] < df["open"].iloc[i]
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.SHOOTING_STAR.value
                df.at[df.index[i], "pattern_bearish"] = 1

    def _detect_three_soldiers(self, df: pd.DataFrame):
        for i in range(2, len(df)):
            c1, c2, c3 = (
                df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
            )
            b1 = c1["close"] > c1["open"]
            b2 = c2["close"] > c2["open"]
            b3 = c3["close"] > c3["open"]
            if not (b1 and b2 and b3):
                continue
            body1 = abs(c1["close"] - c1["open"])
            body2 = abs(c2["close"] - c2["open"])
            body3 = abs(c3["close"] - c3["open"])
            if body1 == 0 or body2 == 0 or body3 == 0:
                continue
            if (
                body2 <= body1 * 1.5
                and body3 <= body2 * 1.5
                and c2["close"] > c1["close"]
                and c3["close"] > c2["close"]
                and c2["open"] > c1["open"]
                and c3["open"] > c2["open"]
                and c2["open"] > c1["close"] * 0.98
                and c3["open"] > c2["close"] * 0.98
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.THREE_WHITE_SOLDIERS.value
                df.at[df.index[i], "pattern_bullish"] = 1

    def _detect_three_crows(self, df: pd.DataFrame):
        for i in range(2, len(df)):
            c1, c2, c3 = (
                df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
            )
            b1 = c1["close"] < c1["open"]
            b2 = c2["close"] < c2["open"]
            b3 = c3["close"] < c3["open"]
            if not (b1 and b2 and b3):
                continue
            body1 = abs(c1["close"] - c1["open"])
            body2 = abs(c2["close"] - c2["open"])
            body3 = abs(c3["close"] - c3["open"])
            if body1 == 0 or body2 == 0 or body3 == 0:
                continue
            if (
                body2 <= body1 * 1.5
                and body3 <= body2 * 1.5
                and c2["close"] < c1["close"]
                and c3["close"] < c2["close"]
                and c2["open"] < c1["open"]
                and c3["open"] < c2["open"]
                and c2["open"] < c1["close"] * 1.02
                and c3["open"] < c2["close"] * 1.02
            ):
                df.at[df.index[i], "candle_pattern"] = CandlePattern.THREE_BLACK_CROWS.value
                df.at[df.index[i], "pattern_bearish"] = 1

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
