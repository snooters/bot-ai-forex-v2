import numpy as np
import pandas as pd
from typing import Dict

from utils.logger import get_logger


class MomentumAnalyzer:
    def __init__(self):
        self.logger = get_logger("momentum_analysis")

    def analyze_momentum(self, df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < 20:
            return {"score": 0, "direction": 0, "strength": "neutral"}

        result = {"score": 0, "direction": 0, "strength": "neutral"}
        signals = []

        rsi_signal = self._check_rsi(df)
        signals.append(rsi_signal)
        result["rsi"] = rsi_signal

        macd_signal = self._check_macd(df)
        signals.append(macd_signal)
        result["macd"] = macd_signal

        stochastic_signal = self._check_stochastic(df)
        signals.append(stochastic_signal)
        result["stochastic"] = stochastic_signal

        mom_signal = self._check_momentum(df)
        signals.append(mom_signal)
        result["momentum"] = mom_signal

        obv_signal = self._check_obv(df)
        signals.append(obv_signal)
        result["obv"] = obv_signal

        avg_score = np.mean([s.get("score", 0) for s in signals]) if signals else 0
        avg_direction = np.mean([s.get("direction", 0) for s in signals]) if signals else 0

        result["score"] = float(avg_score)
        result["direction"] = float(avg_direction)

        if avg_score > 60:
            result["strength"] = "strong"
        elif avg_score > 35:
            result["strength"] = "moderate"
        else:
            result["strength"] = "weak"

        return result

    def _check_rsi(self, df: pd.DataFrame) -> Dict:
        if "rsi" not in df.columns:
            return {"score": 50, "direction": 0}

        rsi_val = df["rsi"].iloc[-1]
        if pd.isna(rsi_val):
            return {"score": 50, "direction": 0}

        rsi_score = min(abs(rsi_val - 50) * 2, 100)
        direction = 1 if rsi_val > 50 else -1

        return {"score": float(rsi_score), "direction": float(direction)}

    def _check_macd(self, df: pd.DataFrame) -> Dict:
        if "macd" not in df.columns or "macd_signal" not in df.columns:
            return {"score": 50, "direction": 0}

        macd = df["macd"].iloc[-1]
        signal = df["macd_signal"].iloc[-1]
        hist = df["macd_histogram"].iloc[-1]

        if pd.isna(macd) or pd.isna(signal):
            return {"score": 50, "direction": 0}

        cross_diff = abs(macd - signal)
        score = min(cross_diff / 0.001 * 50, 100)
        direction = 1 if macd > signal else -1

        macd_positive = 1 if macd > 0 else -1
        combined_dir = direction if hist > 0 else direction * 0.5

        return {"score": float(score), "direction": float(combined_dir)}

    def _check_stochastic(self, df: pd.DataFrame) -> Dict:
        if "stoch_k" not in df.columns or "stoch_d" not in df.columns:
            return {"score": 50, "direction": 0}

        k = df["stoch_k"].iloc[-1]
        d = df["stoch_d"].iloc[-1]
        prev_k = df["stoch_k"].iloc[-2] if len(df) >= 2 else k

        if pd.isna(k) or pd.isna(d):
            return {"score": 50, "direction": 0}

        if k < 20 and k > d:
            score = 80
            direction = 1
        elif k > 80 and k < d:
            score = 80
            direction = -1
        else:
            score = min(abs(k - 50) * 1.5, 60)
            direction = 1 if k > d else -1

        return {"score": float(score), "direction": float(direction)}

    def _check_momentum(self, df: pd.DataFrame) -> Dict:
        if "momentum_10" not in df.columns:
            return {"score": 50, "direction": 0}

        mom10 = df["momentum_10"].iloc[-1]
        mom20 = df["momentum_20"].iloc[-1]

        if pd.isna(mom10):
            return {"score": 50, "direction": 0}

        avg_mom = (mom10 + (mom20 if not pd.isna(mom20) else 0)) / 2
        score = min(abs(avg_mom) * 1000, 100)
        direction = 1 if avg_mom > 0 else -1

        return {"score": float(score), "direction": float(direction)}

    def _check_obv(self, df: pd.DataFrame) -> Dict:
        if "obv" not in df.columns:
            return {"score": 50, "direction": 0}

        obv = df["obv"]
        close = df["close"]

        obv_trend = obv.iloc[-1] - obv.iloc[-5]
        price_trend = close.iloc[-1] - close.iloc[-5]

        if abs(price_trend) < 1e-10:
            return {"score": 50, "direction": 0}

        if (obv_trend > 0 and price_trend > 0) or (obv_trend < 0 and price_trend < 0):
            score = min(abs(obv_trend) / abs(price_trend) * 10, 80)
            direction = 1 if price_trend > 0 else -1
        else:
            score = 40
            direction = 0

        return {"score": float(min(score, 100)), "direction": float(direction)}
