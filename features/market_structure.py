import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from core.constants import MarketStructure
from utils.logger import get_logger


class MarketStructureEngine:
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.logger = get_logger("market_structure")

    def _find_pivots(self, high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
        highs = high.copy()
        lows = low.copy()

        pivot_high = pd.Series(False, index=highs.index)
        pivot_low = pd.Series(False, index=lows.index)

        for i in range(self.lookback, len(highs) - self.lookback):
            if highs.iloc[i] == highs.iloc[i - self.lookback:i + self.lookback + 1].max():
                pivot_high.iloc[i] = True
            if lows.iloc[i] == lows.iloc[i - self.lookback:i + self.lookback + 1].min():
                pivot_low.iloc[i] = True

        return pivot_high, pivot_low

    def detect_structures(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < self.lookback * 3:
            return df

        high = df["high"]
        low = df["low"]

        pivot_high, pivot_low = self._find_pivots(high, low)

        df["pivot_high"] = pivot_high
        df["pivot_low"] = pivot_low

        ph_idx = df[pivot_high].index
        pl_idx = df[pivot_low].index

        df["hh"] = 0
        df["hl"] = 0
        df["lh"] = 0
        df["ll"] = 0
        df["market_structure"] = MarketStructure.UNDEFINED.value

        prev_ph = None
        prev_pl = None
        prev_ph_idx = None
        prev_pl_idx = None

        for i, idx in enumerate(ph_idx):
            if prev_ph is not None:
                if high[idx] > prev_ph:
                    df.at[idx, "hh"] = 1
                    df.at[idx, "market_structure"] = MarketStructure.HIGHER_HIGH.value
                else:
                    df.at[idx, "lh"] = 1
                    df.at[idx, "market_structure"] = MarketStructure.LOWER_HIGH.value

                if prev_pl is not None and prev_pl_idx is not None:
                    if low[idx] > prev_pl if idx > prev_pl_idx else True:
                        pass

            prev_ph = high[idx]
            prev_ph_idx = idx

            if prev_pl is not None and prev_pl_idx is not None and idx > prev_pl_idx:
                if low[idx] > prev_pl if prev_pl_idx is not None else True:
                    pass

        for i, idx in enumerate(pl_idx):
            if prev_pl is not None:
                if low[idx] > prev_pl:
                    df.at[idx, "hl"] = 1
                    if df.at[idx, "market_structure"] == MarketStructure.UNDEFINED.value:
                        df.at[idx, "market_structure"] = MarketStructure.HIGHER_LOW.value
                else:
                    df.at[idx, "ll"] = 1
                    if df.at[idx, "market_structure"] == MarketStructure.UNDEFINED.value:
                        df.at[idx, "market_structure"] = MarketStructure.LOWER_LOW.value

            prev_pl = low[idx]
            prev_pl_idx = idx

        self._detect_bos_choch(df, ph_idx, pl_idx)
        df.fillna({"market_structure": MarketStructure.UNDEFINED.value}, inplace=True)
        return df

    def _detect_bos_choch(self, df: pd.DataFrame, ph_idx, pl_idx):
        if len(ph_idx) < 2 or len(pl_idx) < 2:
            return

        high = df["high"]
        low = df["low"]

        for i in range(2, len(ph_idx)):
            if (
                high[ph_idx[i]] > high[ph_idx[i - 1]] and
                low[ph_idx[i]] < low[ph_idx[i - 1]]
            ):
                df.at[ph_idx[i], "market_structure"] = MarketStructure.BREAK_OF_STRUCTURE.value

        for i in range(2, len(pl_idx)):
            if (
                low[pl_idx[i]] > low[pl_idx[i - 1]] and
                high[pl_idx[i]] > high[pl_idx[i - 1]]
            ):
                df.at[pl_idx[i], "market_structure"] = MarketStructure.CHANGE_OF_CHARACTER.value

    def get_current_structure(self, df: pd.DataFrame) -> str:
        if "market_structure" not in df.columns:
            return MarketStructure.UNDEFINED.value
        recent = df["market_structure"].dropna().tail(10)
        if recent.empty:
            return MarketStructure.UNDEFINED.value
        value_counts = recent.value_counts()
        if value_counts.empty:
            return MarketStructure.UNDEFINED.value
        return value_counts.index[0]

    def is_uptrend(self, df: pd.DataFrame) -> bool:
        return self.get_current_structure(df) in [
            MarketStructure.HIGHER_HIGH.value,
            MarketStructure.HIGHER_LOW.value,
        ]

    def is_downtrend(self, df: pd.DataFrame) -> bool:
        return self.get_current_structure(df) in [
            MarketStructure.LOWER_HIGH.value,
            MarketStructure.LOWER_LOW.value,
        ]

    def has_bos(self, df: pd.DataFrame) -> bool:
        if "market_structure" not in df.columns:
            return False
        recent = df["market_structure"].tail(20)
        return (recent == MarketStructure.BREAK_OF_STRUCTURE.value).any()

    def has_choch(self, df: pd.DataFrame) -> bool:
        if "market_structure" not in df.columns:
            return False
        recent = df["market_structure"].tail(20)
        return (recent == MarketStructure.CHANGE_OF_CHARACTER.value).any()

    def get_last_hh_ll(self, df: pd.DataFrame) -> dict:
        hh_rows = df[df["hh"] == 1]
        ll_rows = df[df["ll"] == 1]
        hl_rows = df[df["hl"] == 1]
        lh_rows = df[df["lh"] == 1]

        return {
            "last_hh": hh_rows["high"].iloc[-1] if not hh_rows.empty else None,
            "last_ll": ll_rows["low"].iloc[-1] if not ll_rows.empty else None,
            "last_hl": hl_rows["low"].iloc[-1] if not hl_rows.empty else None,
            "last_lh": lh_rows["high"].iloc[-1] if not lh_rows.empty else None,
        }
