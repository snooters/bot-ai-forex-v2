"""Ensemble configuration."""
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class EnsembleConfig:
    """Configuration for the 3-model ensemble system."""
    
    # Model weights (must sum to 1.0)
    H4_WEIGHT: float = 0.50
    H1_WEIGHT: float = 0.35
    M5_WEIGHT: float = 0.15
    
    # Minimum confidence to execute a trade
    MIN_CONFIDENCE: float = 0.55
    
    # Model training
    H4_MIN_SAMPLES: int = 1000
    H1_MIN_SAMPLES: int = 3000
    M5_MIN_SAMPLES: int = 10000
    
    # Validation
    OOS_SPLIT: float = 0.20
    WALK_FORWARD_WINDOWS: int = 4
    
    # Paths
    MODEL_DIR: str = "models/ensemble_v2"
    
    # Features per model
    H4_FEATURES: List[str] = field(default_factory=lambda: [
        "ema20_slope", "ema50_slope", "ema200_slope",
        "adx", "rsi14", "macd_line_minus_signal",
        "volume_ratio", "atr", "range_pct",
        "prev_3_dir", "h1_trend_slope",
        "m30_momentum", "price_pos_vs_ema20",
        "m30_close_vs_h4_ema20", "h1_rsi",
    ])
    
    H1_FEATURES: List[str] = field(default_factory=lambda: [
        "h4_trend", "ema5_20_cross", "rsi14",
        "stoch_k", "volume_spike", "atr",
        "m5_price_pos_12bar", "macd",
        "prev_bar_dir", "range_vs_atr",
        "session_asia", "session_london",
        "session_overlap", "adx",
    ])
    
    M5_FEATURES: List[str] = field(default_factory=lambda: [
        "rsi_direction", "price_vs_ma20",
        "volume_spike", "h1_trend",
        "consecutive_bars", "atr",
        "spread", "body_wick_ratio",
    ])
    
    @property
    def weights_ok(self) -> bool:
        return abs(self.H4_WEIGHT + self.H1_WEIGHT + self.M5_WEIGHT - 1.0) < 0.01
    
    @property
    def model_dir_path(self) -> Path:
        return Path(self.MODEL_DIR)


CONFIG = EnsembleConfig()
assert CONFIG.weights_ok, "Ensemble weights must sum to 1.0"
