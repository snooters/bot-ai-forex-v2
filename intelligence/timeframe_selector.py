import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from core.constants import Timeframe
from utils.logger import get_logger


class TimeframeSelector:
    def __init__(self):
        self.logger = get_logger("timeframe_selector")
        self._trend_tf = Timeframe.H1
        self._confirmation_tf = Timeframe.M30
        self._entry_tf = Timeframe.M15

    def select_timeframes(
        self, multi_data: Dict[int, pd.DataFrame],
        performance_data: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, int]:
        if not multi_data:
            return {
                "trend": self._trend_tf,
                "confirmation": self._confirmation_tf,
                "entry": self._entry_tf,
            }

        scores = {}
        for tf, df in multi_data.items():
            if df.empty or len(df) < 50:
                continue
            s = self._score_timeframe(df)
            tf_label = Timeframe.LABELS.get(tf, str(tf))
            if performance_data and tf_label in performance_data:
                wr = performance_data[tf_label].get("win_rate", 50)
                perf_bonus = (wr - 50) * 0.2
                s["total"] += max(perf_bonus, -10)
                s["performance_bonus"] = round(perf_bonus, 2)
            scores[tf] = s

        if not scores:
            return {
                "trend": self._trend_tf,
                "confirmation": self._confirmation_tf,
                "entry": self._entry_tf,
            }

        sorted_tfs = sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True)

        higher_tfs = [tf for tf in sorted(Timeframe.ALL, reverse=True) if tf in multi_data]
        lower_tfs = [tf for tf in sorted(Timeframe.ALL) if tf in multi_data]

        trend_tf = higher_tfs[0] if higher_tfs else Timeframe.H1
        entry_tf = lower_tfs[0] if lower_tfs else Timeframe.M15

        confirmation_tf = None
        for tf in Timeframe.ALL:
            if trend_tf > tf > entry_tf:
                confirmation_tf = tf
                break
        if confirmation_tf is None:
            for tf in Timeframe.ALL:
                if tf != trend_tf and tf != entry_tf:
                    confirmation_tf = tf
                    break

        self._trend_tf = trend_tf
        self._confirmation_tf = confirmation_tf or Timeframe.M30
        self._entry_tf = entry_tf

        return {
            "trend": trend_tf,
            "confirmation": confirmation_tf or Timeframe.M30,
            "entry": entry_tf,
            "scores": {tf: s["total"] for tf, s in scores.items()},
            "best_timeframe": sorted_tfs[0][0],
        }

    def _score_timeframe(self, df: pd.DataFrame) -> Dict:
        score = {"trend": 0, "momentum": 0, "volatility": 0, "structure": 0,
                 "sr": 0, "pattern": 0, "total": 0}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        if len(close) < 50:
            return score

        ema_short = close.ewm(span=20).mean()
        ema_long = close.ewm(span=50).mean()

        trend_aligned = abs(ema_short.iloc[-1] - ema_long.iloc[-1]) / close.iloc[-1]
        score["trend"] = min(trend_aligned * 1000, 100)

        rsi_series = self._compute_rsi(close)
        if not rsi_series.empty:
            rsi_val = rsi_series.iloc[-1]
            score["momentum"] = min(abs(rsi_val - 50) * 2, 100)

        log_ret = np.log(close / close.shift(1))
        vol = log_ret.tail(20).std() if len(log_ret) >= 20 else 0
        score["volatility"] = min(vol * 500, 100)

        score["structure"] = self._score_structure(high, low)

        score["sr"] = self._score_sr(high, low, close)

        score["total"] = (
            score["trend"] * 0.25 +
            score["momentum"] * 0.20 +
            score["volatility"] * 0.15 +
            score["structure"] * 0.20 +
            score["sr"] * 0.20
        )

        return score

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _score_structure(self, high: pd.Series, low: pd.Series) -> float:
        if len(high) < 20:
            return 0
        recent_highs = high.tail(10)
        recent_lows = low.tail(10)

        hh = sum(1 for i in range(1, len(recent_highs))
                 if recent_highs.iloc[i] > recent_highs.iloc[i - 1])
        ll = sum(1 for i in range(1, len(recent_lows))
                 if recent_lows.iloc[i] < recent_lows.iloc[i - 1])

        structure_score = (hh + ll) / (len(recent_highs) * 2) * 100
        return float(structure_score)

    def _score_sr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        if len(close) < 20:
            return 0
        current = close.iloc[-1]
        recent_high = high.tail(20).max()
        recent_low = low.tail(20).min()

        if recent_high == recent_low:
            return 0

        position = (current - recent_low) / (recent_high - recent_low)
        return float(min(abs(position - 0.5) * 200, 100))

    @property
    def trend_timeframe(self) -> int:
        return self._trend_tf

    @property
    def confirmation_timeframe(self) -> int:
        return self._confirmation_tf

    @property
    def entry_timeframe(self) -> int:
        return self._entry_tf
