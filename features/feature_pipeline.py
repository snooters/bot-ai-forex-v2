import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple

from core.config import config
from core.constants import Timeframe
from core.exceptions import FeatureError
from features.indicators import IndicatorEngine
from features.market_structure import MarketStructureEngine
from features.support_resistance import SupportResistanceEngine
from features.price_action import PriceActionEngine
from features.candle_patterns import CandlePatternEngine
from features.session_features import SessionFeatureEngine
from features.multi_tf_features import MultiTFFeatureEngine
from data.data_cache import FeatureCache
from utils.logger import get_logger
from utils.decorators import measure_time, safe_execute


class FeaturePipeline:
    def __init__(self):
        self.logger = get_logger("feature_pipeline")
        self.indicator_engine = IndicatorEngine()
        self.market_structure = MarketStructureEngine()
        self.support_resistance = SupportResistanceEngine()
        self.price_action = PriceActionEngine()
        self.candle_patterns = CandlePatternEngine()
        self.session_features = SessionFeatureEngine()
        self.multi_tf_features = MultiTFFeatureEngine()
        self.cache = FeatureCache()

    @measure_time
    def compute_all(self, df: pd.DataFrame, cache_key: Optional[str] = None) -> pd.DataFrame:
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        if df.empty or len(df) < 50:
            self.logger.warning(f"Insufficient data for feature computation: {len(df)} rows")
            return df

        df = self.session_features.compute(df)

        df = self.indicator_engine.compute(df)

        df = self.market_structure.detect_structures(df)

        sr_features = self.support_resistance.detect_levels(df)
        df = self._add_sr_features(df, sr_features)

        df = self.price_action.detect_patterns(df)
        df = self.candle_patterns.detect_patterns(df)

        df = self._add_derived_features(df)

        df = self.multi_tf_features.compute(df)

        df = self._clean_data(df)

        if cache_key:
            self.cache.set(cache_key, df)

        return df

    def compute_features_summary(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {}

        sr = self.support_resistance.detect_levels(df)
        sd = self.support_resistance.detect_supply_demand(df)

        summary = {
            "indicators": {
                "ema_20": float(df["ema_20"].iloc[-1]) if "ema_20" in df.columns else 0,
                "ema_50": float(df["ema_50"].iloc[-1]) if "ema_50" in df.columns else 0,
                "ema_200": float(df["ema_200"].iloc[-1]) if "ema_200" in df.columns else 0,
                "rsi": float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50,
                "macd": float(df["macd"].iloc[-1]) if "macd" in df.columns else 0,
                "macd_signal": float(df["macd_signal"].iloc[-1]) if "macd_signal" in df.columns else 0,
                "adx": float(df["adx"].iloc[-1]) if "adx" in df.columns else 0,
                "atr": float(df["atr"].iloc[-1]) if "atr" in df.columns else 0,
            },
            "market_structure": {
                "current": self.market_structure.get_current_structure(df),
                "is_uptrend": self.market_structure.is_uptrend(df),
                "is_downtrend": self.market_structure.is_downtrend(df),
                "has_bos": self.market_structure.has_bos(df),
                "has_choch": self.market_structure.has_choch(df),
            },
            "support_resistance": {
                "nearest_support": sr.get("nearest_support"),
                "nearest_resistance": sr.get("nearest_resistance"),
                "dist_to_support": sr.get("distance_to_support"),
                "dist_to_resistance": sr.get("distance_to_resistance"),
            },
            "price_action": {
                "current": self.price_action.get_current_pattern(df),
            },
            "candle_pattern": {
                "current": self.candle_patterns.get_current_pattern(df),
                "signal": self.candle_patterns.get_pattern_signal(df),
                "bullish_count": self.candle_patterns.count_bullish_patterns(df),
                "bearish_count": self.candle_patterns.count_bearish_patterns(df),
            },
        }
        return summary

    def _add_sr_features(self, df: pd.DataFrame, sr: Dict) -> pd.DataFrame:
        df["nearest_support"] = sr.get("nearest_support") or 0.0
        df["nearest_resistance"] = sr.get("nearest_resistance") or 0.0
        df["dist_to_support"] = sr.get("distance_to_support") or 0.0
        df["dist_to_resistance"] = sr.get("distance_to_resistance") or 0.0
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ema_20" in df.columns and "ema_50" in df.columns:
            df["ema_cross"] = df["ema_20"] - df["ema_50"]
            df["ema_slope_20"] = df["ema_20"].diff(3) / df["ema_20"].shift(3).replace(0, np.nan)
            df["ema_slope_50"] = df["ema_50"].diff(5) / df["ema_50"].shift(5).replace(0, np.nan)

        if "close" in df.columns and "ema_200" in df.columns:
            df["price_vs_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"].replace(0, np.nan)

        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            df["bb_position"] = np.where(
                df["close"] > df["bb_upper"], 2,
                np.where(df["close"] < df["bb_lower"], -2, 0)
            )

        if "rsi" in df.columns:
            df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
            df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

        if "stoch_k" in df.columns and "stoch_d" in df.columns:
            df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]

        if "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_cross"] = df["macd"] - df["macd_signal"]
            df["macd_above_zero"] = (df["macd"] > 0).astype(int)

        if "adx" in df.columns:
            df["adx_strong"] = (df["adx"] > 25).astype(int)

        if "close" in df.columns:
            df["returns"] = df["close"].pct_change()
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
            df["realized_vol"] = df["log_return"].rolling(20).std()

        if "high" in df.columns and "low" in df.columns:
            df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)

        if "volume" in df.columns or "tick_volume" in df.columns:
            vol_col = "volume" if "volume" in df.columns else "tick_volume"
            df["volume_trend"] = df[vol_col] / df[vol_col].rolling(50).mean().replace(0, np.nan)

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            for col in nan_cols:
                n_nan = df[col].isna().sum()
                if n_nan == len(df):
                    df[col] = 0.0
                elif df[col].dtype in (np.float64, np.float32):
                    df[col] = df[col].ffill()
                    remaining = df[col].isna().sum()
                    if remaining > 0:
                        df[col] = df[col].bfill()
                    if df[col].isna().sum() > 0:
                        df[col] = df[col].fillna(0.0)
            self.logger.debug(f"Cleaned NaN in {len(nan_cols)} columns: {nan_cols}")

        return df

    def get_feature_columns(self) -> List[str]:
        base = [
            "ema_20", "ema_50", "ema_200",
            "ema_cross", "ema_slope_20", "ema_slope_50",
            "price_vs_ema200",
            "rsi", "rsi_oversold", "rsi_overbought",
            "macd", "macd_signal", "macd_histogram",
            "macd_cross", "macd_above_zero",
            "adx", "plus_di", "minus_di", "adx_strong",
            "atr",
            "bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_pct", "bb_position",
            "stoch_k", "stoch_d", "stoch_cross",
            "momentum_10", "momentum_20",
            "volatility", "realized_vol",
            "williams_r", "cci", "mass_index",
            "obv",
            "returns", "log_return",
            "volume_ratio", "volume_trend",
            "spread_pips",
            "hl_range", "body", "upper_wick", "lower_wick",
            "price_position",
            "nearest_support", "nearest_resistance",
            "dist_to_support", "dist_to_resistance",
            "pattern_bullish", "pattern_bearish",
            "vwap",
            "hour", "day_of_week", "is_weekend",
            "session_asia", "session_london", "session_ny", "session_overlap",
            "is_monday", "is_friday", "is_midweek", "is_market_hours",
            "mtf_alignment",
        ]
        for tf in [15, 30, 60, 240]:
            cols = [
                f"trend{tf}", f"momentum{tf}", f"volatility{tf}",
                f"atr{tf}", f"rsi{tf}",
                f"ema_cross{tf}", f"adx{tf}", f"adx_strong{tf}",
                f"align{tf}",
                f"ema_50_tf{tf}", f"ema_200_tf{tf}",
            ]
            if tf <= 30:
                cols += [f"ema_20_tf{tf}", f"close_vs_ema20{tf}"]
            if tf <= 15:
                cols += [f"macd{tf}"]
            base += cols
        return base
