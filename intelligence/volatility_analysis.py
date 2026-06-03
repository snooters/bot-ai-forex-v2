import numpy as np
import pandas as pd
from typing import Dict

from utils.logger import get_logger


class VolatilityAnalyzer:
    def __init__(self):
        self.logger = get_logger("volatility_analysis")

    def analyze_volatility(self, df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < 20:
            return {"level": "low", "score": 0, "expanding": False}

        close = df["close"]
        atr = df.get("atr", pd.Series(index=df.index))
        log_returns = np.log(close / close.shift(1))

        if "high" in df.columns and "low" in df.columns:
            hl_range = df["high"] - df["low"]
            avg_range = hl_range.tail(50).mean()
            recent_range = hl_range.tail(5).mean()
        else:
            avg_range = close.tail(50).std()
            recent_range = close.tail(5).std()

        hv = log_returns.tail(20).std() if len(log_returns) >= 20 else 0

        atr_value = atr.iloc[-1] if "atr" in df.columns and not atr.empty else 0
        avg_atr = atr.tail(50).mean() if not atr.empty else 0
        atr_ratio = atr_value / avg_atr if avg_atr > 0 else 1

        range_ratio = recent_range / avg_range if avg_range > 0 else 1

        expanding = (
            atr_ratio > 1.1 or
            range_ratio > 1.2
        )

        vol_score = min(max(atr_ratio * 50, range_ratio * 30), 100)

        if vol_score > 70:
            level = "high"
        elif vol_score > 40:
            level = "medium"
        else:
            level = "low"

        bb_width = df["bb_width"].iloc[-1] if "bb_width" in df.columns and not df["bb_width"].empty else 0
        avg_bb_width = df["bb_width"].tail(50).mean() if "bb_width" in df.columns and not df["bb_width"].empty else 0
        bb_expanding = bb_width > avg_bb_width * 1.1 if avg_bb_width > 0 else False

        return {
            "level": level,
            "score": float(vol_score),
            "atr": float(atr_value),
            "expanding": expanding,
            "contracting": atr_ratio < 0.8,
            "hv": float(hv * 100),
            "bb_expanding": bb_expanding,
            "range_ratio": float(range_ratio),
            "atr_ratio": float(atr_ratio),
        }

    def is_high_volatility(self, vol_result: Dict) -> bool:
        return vol_result.get("level") == "high" or vol_result.get("score", 0) > 70

    def is_low_volatility(self, vol_result: Dict) -> bool:
        return vol_result.get("level") == "low" or vol_result.get("score", 0) < 30
