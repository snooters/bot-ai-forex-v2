import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

from utils.logger import get_logger


class SupportResistanceEngine:
    def __init__(self, window: int = 20, threshold_pct: float = 0.002):
        self.window = window
        self.threshold_pct = threshold_pct
        self.logger = get_logger("support_resistance")

    def _find_extrema(self, high: pd.Series, low: pd.Series) -> Tuple[List[int], List[int]]:
        highs_idx = []
        lows_idx = []

        for i in range(self.window, len(high) - self.window):
            if high.iloc[i] == high.iloc[i - self.window:i + self.window + 1].max():
                highs_idx.append(i)
            if low.iloc[i] == low.iloc[i - self.window:i + self.window + 1].min():
                lows_idx.append(i)

        return highs_idx, lows_idx

    def _cluster_levels(
        self, levels: List[float], prices: pd.Series
    ) -> List[Dict]:
        if not levels:
            return []

        levels = sorted(set(levels))
        clusters = []
        current_cluster = [levels[0]]
        current_sum = levels[0]

        for level in levels[1:]:
            avg = current_sum / len(current_cluster)
            if abs(level - avg) / avg < self.threshold_pct * 5:
                current_cluster.append(level)
                current_sum += level
            else:
                avg_level = current_sum / len(current_cluster)
                touch_count = sum(1 for l in current_cluster)
                strength = min(touch_count / 3, 1.0)
                clusters.append({
                    "level": avg_level,
                    "strength": strength,
                    "touches": touch_count,
                    "type": self._classify_level(avg_level, prices)
                })
                current_cluster = [level]
                current_sum = level

        if current_cluster:
            avg_level = current_sum / len(current_cluster)
            touch_count = len(current_cluster)
            strength = min(touch_count / 3, 1.0)
            clusters.append({
                "level": avg_level,
                "strength": strength,
                "touches": touch_count,
                "type": self._classify_level(avg_level, prices)
            })

        return clusters

    def _classify_level(self, level: float, prices: pd.Series) -> str:
        current_price = prices.iloc[-1]
        pct_diff = abs(current_price - level) / current_price

        if pct_diff < 0.002:
            return "current"
        elif level < current_price:
            return "support"
        else:
            return "resistance"

    def detect_levels(self, df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < self.window * 3:
            return {"support": [], "resistance": [], "major": [], "minor": []}

        high = df["high"]
        low = df["low"]
        close = df["close"]

        highs_idx, lows_idx = self._find_extrema(high, low)

        resistance_levels = [high.iloc[i] for i in highs_idx]
        support_levels = [low.iloc[i] for i in lows_idx]

        resistance_clusters = self._cluster_levels(resistance_levels, close)
        support_clusters = self._cluster_levels(support_levels, close)

        supports = [c for c in support_clusters if c["type"] in ("support", "current")]
        resistances = [c for c in resistance_clusters if c["type"] in ("resistance", "current")]

        major_support = [s for s in supports if s["strength"] >= 0.6]
        minor_support = [s for s in supports if s["strength"] < 0.6]
        major_resistance = [r for r in resistances if r["strength"] >= 0.6]
        minor_resistance = [r for r in resistances if r["strength"] < 0.6]

        current_price = close.iloc[-1]
        nearest_support = None
        nearest_support_dist = None
        for s in supports:
            dist = current_price - s["level"]
            if dist > 0:
                if nearest_support is None or dist < nearest_support_dist:
                    nearest_support = s["level"]
                    nearest_support_dist = dist

        nearest_resistance = None
        nearest_resistance_dist = None
        for r in resistances:
            dist = r["level"] - current_price
            if dist > 0:
                if nearest_resistance is None or dist < nearest_resistance_dist:
                    nearest_resistance = r["level"]
                    nearest_resistance_dist = dist

        return {
            "support": supports,
            "resistance": resistances,
            "major_support": major_support,
            "minor_support": minor_support,
            "major_resistance": major_resistance,
            "minor_resistance": minor_resistance,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "distance_to_support": nearest_support_dist,
            "distance_to_resistance": nearest_resistance_dist,
        }

    def detect_supply_demand(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        if df.empty or len(df) < lookback:
            return {"supply_zones": [], "demand_zones": []}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        supply_zones = []
        demand_zones = []

        for i in range(2, min(lookback, len(df) - 1)):
            if (
                high.iloc[i] > high.iloc[i - 1] and
                high.iloc[i] > high.iloc[i + 1]
            ):
                base = min(close.iloc[i - 1], close.iloc[i + 1])
                supply_zones.append({
                    "top": high.iloc[i],
                    "bottom": base,
                    "strength": 0.5,
                })

            if (
                low.iloc[i] < low.iloc[i - 1] and
                low.iloc[i] < low.iloc[i + 1]
            ):
                top = max(close.iloc[i - 1], close.iloc[i + 1])
                demand_zones.append({
                    "top": top,
                    "bottom": low.iloc[i],
                    "strength": 0.5,
                })

        return {
            "supply_zones": supply_zones,
            "demand_zones": demand_zones,
        }
