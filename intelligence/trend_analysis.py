import numpy as np
import pandas as pd
from typing import Dict, Tuple

from core.constants import TrendDirection, EME_FAST, EME_MEDIUM, EME_SLOW
from utils.logger import get_logger


class TrendAnalyzer:
    def __init__(self):
        self.logger = get_logger("trend_analysis")

    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < EME_SLOW:
            return {"direction": TrendDirection.SIDEWAYS.value, "strength": 0, "score": 0}

        result = {"direction": TrendDirection.SIDEWAYS.value, "strength": 0.0, "score": 0}
        signals = []

        ema_alignment = self._check_ema_alignment(df)
        signals.append(ema_alignment)
        result["ema_alignment"] = ema_alignment

        slope_score = self._check_slope(df)
        signals.append(slope_score)
        result["slope_score"] = slope_score

        price_position = self._check_price_position(df)
        signals.append(price_position)
        result["price_position"] = price_position

        adx_score = self._check_adx(df)
        signals.append(adx_score)
        result["adx_score"] = adx_score

        direction_scores = [s.get("direction", 0) for s in signals]
        strength_scores = [s.get("strength", 0) for s in signals]

        avg_direction = np.mean(direction_scores) if direction_scores else 0
        avg_strength = np.mean(strength_scores) if strength_scores else 0

        avg_direction = np.clip(avg_direction, -1, 1)
        avg_strength = np.clip(avg_strength, 0, 1)

        if avg_direction > 0.2:
            if avg_strength > 0.6:
                result["direction"] = TrendDirection.STRONG_BULLISH.value
            else:
                result["direction"] = TrendDirection.WEAK_BULLISH.value
        elif avg_direction < -0.2:
            if avg_strength > 0.6:
                result["direction"] = TrendDirection.STRONG_BEARISH.value
            else:
                result["direction"] = TrendDirection.WEAK_BEARISH.value
        else:
            if avg_strength < 0.3:
                result["direction"] = TrendDirection.SIDEWAYS.value
            else:
                result["direction"] = TrendDirection.CONSOLIDATION.value

        result["strength"] = float(avg_strength)
        result["score"] = float(avg_direction * 100)
        return result

    def _check_ema_alignment(self, df: pd.DataFrame) -> Dict:
        ema20 = df["ema_20"].iloc[-1]
        ema50 = df["ema_50"].iloc[-1]
        ema200 = df["ema_200"].iloc[-1]
        close = df["close"].iloc[-1]

        if close > ema20 > ema50 > ema200:
            return {"direction": 1, "strength": 1.0}
        elif close > ema20 > ema50:
            return {"direction": 1, "strength": 0.7}
        elif close > ema20:
            return {"direction": 0.5, "strength": 0.4}
        elif close < ema20 < ema50 < ema200:
            return {"direction": -1, "strength": 1.0}
        elif close < ema20 < ema50:
            return {"direction": -1, "strength": 0.7}
        elif close < ema20:
            return {"direction": -0.5, "strength": 0.4}
        else:
            return {"direction": 0, "strength": 0.2}

    def _check_slope(self, df: pd.DataFrame) -> Dict:
        ema20 = df["ema_20"]
        slope_20 = (ema20.iloc[-1] - ema20.iloc[-5]) / ema20.iloc[-5] if ema20.iloc[-5] != 0 else 0
        slope_50 = (df["ema_50"].iloc[-1] - df["ema_50"].iloc[-10]) / df["ema_50"].iloc[-10] if df["ema_50"].iloc[-10] != 0 else 0

        avg_slope = (slope_20 + slope_50) / 2

        direction = np.sign(avg_slope)
        strength = min(abs(avg_slope) * 1000, 1.0)

        return {"direction": float(direction), "strength": float(strength)}

    def _check_price_position(self, df: pd.DataFrame) -> Dict:
        close = df["close"]
        recent = close.tail(20)

        if len(recent) < 2:
            return {"direction": 0, "strength": 0}

        highs = df["high"].tail(50).max()
        lows = df["low"].tail(50).min()

        if highs == lows:
            return {"direction": 0, "strength": 0}

        position = (close.iloc[-1] - lows) / (highs - lows)
        position = np.clip(position, 0, 1)

        direction = (position - 0.5) * 2
        strength = abs(position - 0.5) * 2

        return {"direction": float(direction), "strength": float(strength)}

    def _check_adx(self, df: pd.DataFrame) -> Dict:
        if "adx" not in df.columns or "plus_di" not in df.columns or "minus_di" not in df.columns:
            return {"direction": 0, "strength": 0}

        adx_val = df["adx"].iloc[-1]
        plus_di = df["plus_di"].iloc[-1]
        minus_di = df["minus_di"].iloc[-1]

        if pd.isna(adx_val) or pd.isna(plus_di) or pd.isna(minus_di):
            return {"direction": 0, "strength": 0}

        strength = min(adx_val / 50, 1.0)
        direction = 1 if plus_di > minus_di else -1

        return {"direction": float(direction), "strength": float(strength)}

    def is_bullish(self, trend_result: Dict) -> bool:
        return trend_result["direction"] in [
            TrendDirection.STRONG_BULLISH.value,
            TrendDirection.WEAK_BULLISH.value
        ]

    def is_bearish(self, trend_result: Dict) -> bool:
        return trend_result["direction"] in [
            TrendDirection.STRONG_BEARISH.value,
            TrendDirection.WEAK_BEARISH.value
        ]

    def is_sideways(self, trend_result: Dict) -> bool:
        return trend_result["direction"] in [
            TrendDirection.SIDEWAYS.value,
            TrendDirection.CONSOLIDATION.value
        ]
