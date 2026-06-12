import numpy as np
import pandas as pd
from typing import Dict, Tuple

from core.constants import ATR_PERIOD, ADX_PERIOD, BB_PERIOD, RSI_PERIOD
from features.indicators import compute_atr, compute_adx
from utils.logger import get_logger


class RegimeClassifier:
    def __init__(self):
        self.logger = get_logger("regime_classifier")

    def classify(self, df: pd.DataFrame) -> Dict:
        if df.empty or len(df) < 30:
            return {"regime": "RANGING", "confidence": 0.5, "scores": {}}

        features = self._compute_features(df)
        regime, confidence = self._fuzzy_decision(features)

        self.logger.debug(
            f"Regime: {regime} (conf={confidence:.2f}) "
            f"ADX={features['adx']:.1f} ATR_pct={features['atr_percentile']:.2f} "
            f"BBW={features['bb_width']:.4f} EMA20_50={features['ema20_50_dist']:.6f}"
        )

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "scores": features,
        }

    def _compute_features(self, df: pd.DataFrame) -> Dict:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        adx = compute_adx(high, low, close, ADX_PERIOD)
        if isinstance(adx, tuple):
            adx = adx[0]
        adx_val = float(adx.iloc[-1]) if hasattr(adx, "iloc") else 0.0

        atr = compute_atr(high, low, close, ATR_PERIOD)
        atr_vals = atr.dropna().values
        atr_current = float(atr.iloc[-1]) if hasattr(atr, "iloc") else 0.0
        atr_percentile = 0.5
        if len(atr_vals) > 20:
            atr_percentile = np.sum(atr_vals <= atr_current) / len(atr_vals)

        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()

        ema20_50_dist = float((ema_20.iloc[-1] - ema_50.iloc[-1]) / close.iloc[-1])
        ema50_200_dist = float((ema_50.iloc[-1] - ema_200.iloc[-1]) / close.iloc[-1])

        rsi_val = self._compute_rsi(close)
        bb_width = self._compute_bb_width(close)

        log_ret = np.log(close / close.shift(1)).dropna()
        realized_vol = float(log_ret.tail(20).std())

        mtf_align = float(df.get("mtf_alignment", pd.Series([0])).iloc[-1]) if "mtf_alignment" in df.columns else 0.0

        vol_col = "volume" if "volume" in df.columns else "tick_volume"
        volume = df[vol_col] if vol_col in df.columns else pd.Series([0] * len(df))
        avg_vol = float(volume.tail(30).mean())
        recent_vol = float(volume.tail(3).mean())
        vol_spike = (recent_vol / avg_vol) if avg_vol > 0 else 1.0

        atr_prev = float(atr.tail(20).head(15).mean()) if hasattr(atr, "iloc") else 0.0
        atr_spike = (atr_current / atr_prev) if atr_prev > 0 else 1.0

        return {
            "adx": adx_val,
            "atr": atr_current,
            "atr_percentile": atr_percentile,
            "bb_width": bb_width,
            "realized_vol": realized_vol,
            "ema20_50_dist": ema20_50_dist,
            "ema50_200_dist": ema50_200_dist,
            "rsi": rsi_val,
            "mtf_alignment": mtf_align,
            "vol_spike": vol_spike,
            "atr_spike": atr_spike,
        }

    def _compute_rsi(self, close: pd.Series, period: int = RSI_PERIOD) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        avg_loss = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    def _compute_bb_width(self, close: pd.Series, period: int = BB_PERIOD, std_dev: float = 2.0) -> float:
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        mid = sma
        width = (upper - lower) / mid
        return float(width.iloc[-1]) if not width.empty else 0.0

    def _fuzzy_decision(self, f: Dict) -> Tuple[str, float]:
        scores = {
            "trending_bullish": 0.0,
            "trending_bearish": 0.0,
            "ranging": 0.0,
            "volatile": 0.0,
            "news_shock": 0.0,
        }

        if f["vol_spike"] > 2.0 and f["atr_spike"] > 1.5:
            scores["news_shock"] = min((f["vol_spike"] - 2.0) / 3.0, 1.0) * 0.9
            scores["volatile"] = max(scores["volatile"], scores["news_shock"] * 0.5)

        is_high_adx = f["adx"] > 25
        is_strong_adx = f["adx"] > 35
        is_low_adx = f["adx"] < 20
        is_bullish = f["ema20_50_dist"] > 0.0005 and f["rsi"] > 50
        is_bearish = f["ema20_50_dist"] < -0.0005 and f["rsi"] < 50
        high_vol = f["atr_percentile"] > 0.80 or f["bb_width"] > 0.05
        low_bb = f["bb_width"] < 0.015
        near_neutral = abs(f["ema20_50_dist"]) < 0.0003 and abs(f["ema50_200_dist"]) < 0.001

        if scores["news_shock"] > 0.3:
            return ("NEWS_SHOCK", scores["news_shock"])

        # Strong ADX + EMA alignment = trending regardless of RSI
        if is_strong_adx and f["ema20_50_dist"] > 0 and f["mtf_alignment"] >= -0.5:
            adx_score = min(f["adx"] / 50, 1.0)
            ema_score = min(abs(f["ema20_50_dist"]) * 500, 1.0)
            rsi_score = (f["rsi"] - 50) / 30 if f["rsi"] > 50 else 0.0
            scores["trending_bullish"] = adx_score * 0.4 + ema_score * 0.4 + rsi_score * 0.2
            return ("TRENDING_BULLISH", round(max(scores["trending_bullish"], 0.5), 2))

        if is_strong_adx and f["ema20_50_dist"] < 0 and f["mtf_alignment"] <= 0.5:
            adx_score = min(f["adx"] / 50, 1.0)
            ema_score = min(abs(f["ema20_50_dist"]) * 500, 1.0)
            rsi_score = (50 - f["rsi"]) / 30 if f["rsi"] < 50 else 0.0
            scores["trending_bearish"] = adx_score * 0.4 + ema_score * 0.4 + rsi_score * 0.2
            return ("TRENDING_BEARISH", round(max(scores["trending_bearish"], 0.5), 2))

        if is_high_adx and is_bullish and f["mtf_alignment"] >= -0.5:
            adx_score = min(f["adx"] / 50, 1.0)
            ema_score = min(abs(f["ema20_50_dist"]) * 500, 1.0)
            rsi_score = (f["rsi"] - 50) / 30 if f["rsi"] > 50 else 0.0
            scores["trending_bullish"] = adx_score * 0.4 + ema_score * 0.3 + rsi_score * 0.3
            if scores["trending_bullish"] > 0.4:
                return ("TRENDING_BULLISH", round(scores["trending_bullish"], 2))

        if is_high_adx and is_bearish and f["mtf_alignment"] <= 0.5:
            adx_score = min(f["adx"] / 50, 1.0)
            ema_score = min(abs(f["ema20_50_dist"]) * 500, 1.0)
            rsi_score = (50 - f["rsi"]) / 30 if f["rsi"] < 50 else 0.0
            scores["trending_bearish"] = adx_score * 0.4 + ema_score * 0.3 + rsi_score * 0.3
            if scores["trending_bearish"] > 0.4:
                return ("TRENDING_BEARISH", round(scores["trending_bearish"], 2))

        if high_vol:
            vol_score = max(f["atr_percentile"], f["bb_width"] * 20)
            scores["volatile"] = min(vol_score, 1.0)
            if scores["volatile"] > 0.55:
                return ("VOLATILE", round(scores["volatile"], 2))

        if is_low_adx and low_bb and near_neutral:
            scores["ranging"] = min((25 - f["adx"]) / 15 * 0.7 + (1 - f["bb_width"] * 50) * 0.3, 0.85)
            if scores["ranging"] > 0.3:
                return ("RANGING", round(scores["ranging"], 2))

        if f["atr_percentile"] > 0.70:
            scores["volatile"] = min((f["atr_percentile"] - 0.7) * 2, 0.6)
            if scores["volatile"] > 0.4:
                return ("VOLATILE", round(scores["volatile"], 2))

        return ("RANGING", 0.35)
