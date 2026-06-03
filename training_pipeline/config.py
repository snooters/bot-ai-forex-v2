from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    data_path: str = "./data"
    output_dir: str = "./models/xgboost_model"
    log_dir: str = "./logs"
    log_level: str = "INFO"

    window_days: int = 730
    step_days: int = 30
    prediction_horizon: int = 5
    threshold: float = 0.001

    val_split: float = 0.15
    test_split: float = 0.15

    rolling: bool = False
    multi_timeframe: bool = False
    num_rolls: int = 4
    retrain_frequency_days: int = 30

    feature_list: List[str] = field(default_factory=lambda: [
        "rsi", "ema_20", "ema_50", "macd", "macd_signal", "macd_histogram",
        "atr", "return_1", "return_5", "return_10", "volatility"
    ])

    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
        "early_stopping_rounds": 50,
    })

    early_stopping_rounds: int = 50
    eval_metric: str = "mlogloss"

    label_encoding: dict = field(default_factory=lambda: {
        "SELL": 0,
        "HOLD": 1,
        "BUY": 2,
    })

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "TrainingConfig":
        return cls()
