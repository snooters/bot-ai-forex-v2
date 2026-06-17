"""Ensemble integration with the forex bot.

Bridges the 3-model ensemble (H4/H1/M5) with the bot's execution pipeline.
When ENSEMBLE_MODE=true, this replaces the old decision engine.
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

from ensemble.config import CONFIG as ENS_CONFIG
from ensemble.ensemble import EnsembleDecision
from ensemble.data import add_technical_features, load_tf_data
from core.config import config


class EnsembleIntegration:
    """Integrates ensemble models with live bot execution."""
    
    def __init__(self):
        self.ensemble = EnsembleDecision()
        self._loaded = False
        self._last_h4_features: Optional[np.ndarray] = None
        self._last_h1_features: Optional[np.ndarray] = None
        self._last_m5_features: Optional[np.ndarray] = None
        self._last_trade_times: list = []
        self._daily_trade_count = 0
        self._last_reset_day = datetime.now().day
    
    def initialize(self) -> bool:
        """Load all ensemble models from disk."""
        self._loaded = self.ensemble.load_all()
        # Override weights from config
        ENS_CONFIG.H4_WEIGHT = config.ensemble.get("h4_weight", 0.50)
        ENS_CONFIG.H1_WEIGHT = config.ensemble.get("h1_weight", 0.35)
        ENS_CONFIG.M5_WEIGHT = config.ensemble.get("m5_weight", 0.15)
        ENS_CONFIG.MIN_CONFIDENCE = config.ensemble.get("min_confidence", 0.55)
        return self._loaded
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def _reset_daily_counter(self):
        """Reset trade counter at day boundary."""
        today = datetime.now().day
        if today != self._last_reset_day:
            self._daily_trade_count = 0
            self._last_reset_day = today
    
    def _compute_h4_features(self) -> Optional[np.ndarray]:
        """Compute H4 features from latest MT5 data."""
        try:
            df = load_tf_data(240)
            if df is None or len(df) < 100:
                return None
            
            closes = df["close"].values.astype(np.float64)
            highs = df["high"].values.astype(np.float64)
            lows = df["low"].values.astype(np.float64)
            volumes = df.get("volume", df.get("tick_volume", 
                                              pd.Series(np.ones(len(df))))).values.astype(np.float64)
            
            feat = add_technical_features(df, closes, highs, lows, volumes)
            
            # Get latest feature values
            feature_cols = ENS_CONFIG.H4_FEATURES
            latest = np.array([feat.get(col, [0])[-1] for col in feature_cols], dtype=np.float64)
            
            # Handle NaN
            latest = np.nan_to_num(latest, nan=0.0)
            
            return latest
        except Exception as e:
            print(f"[Ensemble] H4 feature error: {e}")
            return None
    
    def _compute_h1_features(self) -> Optional[np.ndarray]:
        """Compute H1 features from latest MT5 data."""
        try:
            df = load_tf_data(60)
            if df is None or len(df) < 100:
                return None
            
            closes = df["close"].values.astype(np.float64)
            highs = df["high"].values.astype(np.float64)
            lows = df["low"].values.astype(np.float64)
            volumes = df.get("volume", df.get("tick_volume",
                                              pd.Series(np.ones(len(df))))).values.astype(np.float64)
            
            feat = add_technical_features(df, closes, highs, lows, volumes)
            
            # Compute H1-specific features
            n = len(closes)
            
            # H4 trend
            h4_df = load_tf_data(240)
            if h4_df is not None:
                h4_close = h4_df["close"].values.astype(np.float64)
                h4_ema20 = pd.Series(h4_close).ewm(span=20).mean().values
                h4_trend = 1.0 if h4_close[-1] > h4_ema20[-1] else 0.0
            else:
                h4_trend = 0.0
            feat["h4_trend"] = np.full(n, h4_trend)
            
            # EMA5/20 cross
            ema5 = pd.Series(closes).ewm(span=5).mean().values
            ema20 = feat["ema20"]
            feat["ema5_20_cross"] = np.sign(ema5 - ema20)
            
            # Stochastics
            if n > 14:
                low_14 = pd.Series(lows).rolling(14).min().values
                high_14 = pd.Series(highs).rolling(14).max().values
                feat["stoch_k"] = np.where(high_14 != low_14,
                                          (closes - low_14) / (high_14 - low_14) * 100, 50)
            else:
                feat["stoch_k"] = np.full(n, 50)
            
            # Volume spike
            vol_avg = pd.Series(volumes).rolling(20).mean().values
            feat["volume_spike"] = np.where(vol_avg > 0, volumes / vol_avg, 1.0)
            
            # M5 price position
            m5_df = load_tf_data(5)
            if m5_df is not None:
                m5_high = m5_df["high"].resample("1h").max()
                m5_low = m5_df["low"].resample("1h").min()
                m5_close = m5_df["close"].resample("1h").last()
                price_pos_s = pd.Series(
                    np.where(m5_high.values != m5_low.values,
                            (m5_close.values - m5_low.values) / (m5_high.values - m5_low.values + 1e-10), 0.5),
                    index=m5_high.index
                )
                feat["m5_price_pos_12bar"] = price_pos_s.reindex(df.index, method="ffill").values[:n]
            else:
                feat["m5_price_pos_12bar"] = np.full(n, 0.5)
            
            # Previous bar direction
            feat["prev_bar_dir"] = np.concatenate([[0], np.sign(closes[1:] - closes[:-1])])
            
            # Range vs ATR
            feat["range_vs_atr"] = np.where(feat["atr"] > 0, (highs - lows) / feat["atr"], 1.0)
            
            # Session
            hour = pd.Series(df.index.hour, index=df.index)
            feat["session_asia"] = ((hour >= 0) & (hour < 8)).astype(np.float64).values
            feat["session_london"] = ((hour >= 8) & (hour < 16)).astype(np.float64).values
            feat["session_overlap"] = ((hour >= 12) & (hour < 16)).astype(np.float64).values
            
            # Get latest values
            feature_cols = ENS_CONFIG.H1_FEATURES
            latest = np.array([feat.get(col, [0])[-1] for col in feature_cols], dtype=np.float64)
            latest = np.nan_to_num(latest, nan=0.0)
            
            return latest
        except Exception as e:
            print(f"[Ensemble] H1 feature error: {e}")
            return None
    
    def _compute_m5_features(self) -> Optional[np.ndarray]:
        """Compute M5 features from latest MT5 data."""
        try:
            df = load_tf_data(5)
            if df is None or len(df) < 100:
                return None
            
            closes = df["close"].values.astype(np.float64)
            highs = df["high"].values.astype(np.float64)
            lows = df["low"].values.astype(np.float64)
            volumes = df.get("volume", df.get("tick_volume",
                                              pd.Series(np.ones(len(df))))).values.astype(np.float64)
            
            feat = add_technical_features(df, closes, highs, lows, volumes)
            
            # RSI direction
            rsi = feat["rsi14"]
            feat["rsi_direction"] = np.concatenate([[0], np.diff(rsi)])
            
            # Price vs MA20
            feat["price_vs_ma20"] = feat["price_pos_vs_ema20"]
            
            # Volume spike
            vol_avg = pd.Series(volumes).rolling(20).mean().values
            feat["volume_spike"] = np.where(vol_avg > 0, volumes / vol_avg, 1.0)
            
            # H1 trend 
            h1_df = load_tf_data(60)
            if h1_df is not None:
                h1_close = h1_df["close"].values.astype(np.float64)
                h1_ema20 = pd.Series(h1_close).ewm(span=20).mean().values if len(h1_close) > 20 else np.full(len(h1_close), h1_close[-1])
                h1_trend_v = 1.0 if h1_close[-1] > h1_ema20[-1] else 0.0
            else:
                h1_trend_v = 0.0
            feat["h1_trend"] = np.full(len(closes), h1_trend_v)
            
            # Spread
            feat["spread"] = df.get("spread", pd.Series(np.full(len(closes), 16.6))).values.astype(np.float64)
            
            # Get latest values
            feature_cols = ENS_CONFIG.M5_FEATURES
            latest = np.array([feat.get(col, [0])[-1] for col in feature_cols], dtype=np.float64)
            latest = np.nan_to_num(latest, nan=0.0)
            
            return latest
        except Exception as e:
            print(f"[Ensemble] M5 feature error: {e}")
            return None
    
    def compute_all_features(self) -> bool:
        """Compute features for all 3 models."""
        h4 = self._compute_h4_features()
        h1 = self._compute_h1_features()
        m5 = self._compute_m5_features()
        
        if h4 is None or h1 is None or m5 is None:
            return False
        
        self._last_h4_features = h4
        self._last_h1_features = h1
        self._last_m5_features = m5
        return True
    
    def get_decision(self) -> Dict:
        """Get ensemble decision with current features."""
        self._reset_daily_counter()
        
        if not self._loaded:
            return {"action": "HOLD", "confidence": 0.0, 
                    "reason": "ensemble_not_loaded",
                    "no_trade": True, "market_score": 0}
        
        # Compute features if needed
        if self._last_h4_features is None:
            if not self.compute_all_features():
                return {"action": "HOLD", "confidence": 0.0,
                        "reason": "feature_compute_failed",
                        "no_trade": True, "market_score": 0}
        
        decision = self.ensemble.decide(
            self._last_h4_features,
            self._last_h1_features,
            self._last_m5_features,
        )
        
        # Add keys expected by main.py
        decision["no_trade"] = decision["action"] not in ("BUY", "SELL")
        decision["market_score"] = int(decision["confidence"] * 100)
        decision["no_trade_reasons"] = [] if decision["no_trade"] else []
        decision["reasons"] = [f"Ensemble: {decision.get('reason', '')}"]
        
        # Check daily trade limit
        max_trades = config.ensemble.get("entertain_max_per_day", 50)
        if self._daily_trade_count >= max_trades:
            decision["action"] = "HOLD"
            decision["no_trade"] = True
            decision["reasons"].append("daily_limit_reached")
        
        return decision
    
    def record_trade(self):
        """Record a trade for daily counting."""
        self._daily_trade_count += 1
        self._last_trade_times.append(datetime.now())
        # Keep only last 100 timestamps
        if len(self._last_trade_times) > 100:
            self._last_trade_times = self._last_trade_times[-100:]
    
    def get_stats(self) -> Dict:
        """Get ensemble statistics for dashboard."""
        summary = self.ensemble.get_model_summary() if self._loaded else {}
        return {
            "loaded": self._loaded,
            "daily_trades": self._daily_trade_count,
            "h4_acc": summary.get("h4", {}).get("val_accuracy", 0),
            "h1_acc": summary.get("h1", {}).get("val_accuracy", 0),
            "m5_acc": summary.get("m5", {}).get("val_accuracy", 0),
        }
