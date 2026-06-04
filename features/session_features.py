import numpy as np
import pandas as pd
from typing import Dict

from utils.logger import get_logger


class SessionFeatureEngine:
    def __init__(self):
        self.logger = get_logger("session_features")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "time" not in df.columns:
            return df

        df = df.copy()

        dt = pd.to_datetime(df["time"])
        hour = dt.dt.hour
        minute = dt.dt.minute
        day_of_week = dt.dt.dayofweek

        df["hour"] = hour
        df["day_of_week"] = day_of_week
        df["is_weekend"] = (day_of_week >= 5).astype(int)

        session = np.full(len(df), "OTHER", dtype=object)

        asia_mask = (hour >= 0) & (hour < 7)
        london_mask = ((hour >= 7) & (hour < 12)) | ((hour == 7) & (minute >= 0))
        ny_mask = ((hour >= 12) & (hour < 20)) | ((hour == 12) & (minute >= 0))
        overlap_mask = (hour >= 12) & (hour < 15)

        session[asia_mask] = "ASIA"
        session[london_mask & ~overlap_mask] = "LONDON"
        session[overlap_mask] = "OVERLAP"
        session[ny_mask & ~overlap_mask] = "NEW_YORK"

        df["session"] = session
        df["session_asia"] = (session == "ASIA").astype(int)
        df["session_london"] = (session == "LONDON").astype(int)
        df["session_ny"] = (session == "NEW_YORK").astype(int)
        df["session_overlap"] = (session == "OVERLAP").astype(int)

        df["session_binary"] = np.where(
            session == "ASIA", 0,
            np.where(session == "LONDON", 1,
                     np.where(session == "OVERLAP", 2,
                              np.where(session == "NEW_YORK", 3, -1))),
        )

        for s_name in ["ASIA", "LONDON", "NEW_YORK", "OVERLAP"]:
            col = f"session_{s_name.lower()}_vol"
            s_mask = session == s_name
            df[col] = 0.0
            if s_mask.any():
                vol = df["close"] / df["close"].shift(1) - 1
                avg_vol = vol[s_mask].rolling(100, min_periods=20).std()
                df.loc[s_mask, col] = avg_vol.reindex(df.index[s_mask]).values

        df["is_monday"] = (day_of_week == 0).astype(int)
        df["is_friday"] = (day_of_week == 4).astype(int)
        df["is_midweek"] = ((day_of_week >= 1) & (day_of_week <= 3)).astype(int)

        market_hours = ((hour >= 0) & (hour < 23)).astype(int)
        df["is_market_hours"] = market_hours

        return df
