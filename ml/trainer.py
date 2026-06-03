import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from core.config import config
from core.constants import LOOKAHEAD_5, MODEL_DIR
from core.exceptions import ModelTrainingError
from features.feature_pipeline import FeaturePipeline
from ml.ensemble import VotingEnsemble
from learning.trade_memory import TradeMemory
from learning.mistake_weighting import MistakeWeighting
from utils.logger import get_logger
from utils.decorators import measure_time, safe_execute


class ModelTrainer:
    def __init__(self, trade_memory: Optional[TradeMemory] = None):
        self.logger = get_logger("model_trainer")
        self.feature_pipeline = FeaturePipeline()
        self.ensemble = VotingEnsemble()
        self.trade_memory = trade_memory
        self.mistake_weighting = MistakeWeighting(trade_memory) if trade_memory else None

    def _create_model_instance(self, name: str):
        try:
            if name == "xgboost":
                from ml.xgboost_model import XGBoostModel
                return XGBoostModel()
            elif name == "random_forest":
                from ml.random_forest_model import RandomForestModel
                return RandomForestModel()
            elif name == "lightgbm":
                from ml.lightgbm_model import LightGBMModel
                return LightGBMModel()
            elif name == "lstm":
                from ml.lstm_model import LSTMModel
                return LSTMModel()
        except Exception as e:
            self.logger.warning(f"Cannot create {name} model: {e}")
            return None

    @measure_time
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        lookahead: int = LOOKAHEAD_5,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if df.empty or len(df) < 250:
            raise ModelTrainingError(f"Insufficient data: {len(df)} rows")

        df = self.feature_pipeline.compute_all(df)
        feature_cols = self.feature_pipeline.get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]
        self.ensemble.feature_cols = available_cols

        df = df.dropna(subset=available_cols).copy()
        if df.empty or len(df) < 200:
            raise ModelTrainingError("Insufficient data after cleaning")

        future_close = df["close"].shift(-lookahead)
        current_close = df["close"]
        future_return = (future_close - current_close) / current_close

        y = np.zeros(len(df), dtype=int)
        y[future_return > 0.001] = 0
        y[future_return < -0.001] = 1
        y[(future_return >= -0.001) & (future_return <= 0.001)] = 2

        X = df[available_cols].values
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            raise ModelTrainingError(f"Too few training samples: {len(X)}")

        self.logger.info(f"Prepared {len(X)} training samples with {len(available_cols)} features. "
                         f"BUY: {(y==0).sum()}, SELL: {(y==1).sum()}, HOLD: {(y==2).sum()}")
        return X, y, available_cols

    def _compute_sample_weights(self, y: np.ndarray, multiplier: float = 1.0) -> np.ndarray:
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        weights = np.zeros(n_samples, dtype=float)
        for cls, count in zip(classes, counts):
            cls_mask = y == cls
            base_weight = n_samples / (n_classes * count)
            if cls in (0, 1):
                base_weight *= multiplier
            weights[cls_mask] = base_weight
        self.logger.info(f"Sample weights computed (multiplier={multiplier}): "
                         f"classes={dict(zip(classes, counts))}, "
                         f"weight_range=[{weights.min():.2f}, {weights.max():.2f}]")
        return weights

    @measure_time
    def train_all_models(self, X: np.ndarray, y: np.ndarray,
                         sample_weight_multiplier: float = 1.0,
                         model_params: Optional[Dict] = None,
                         feature_cols: Optional[List[str]] = None) -> Dict:
        y_effective = y.copy()
        if self.mistake_weighting and feature_cols:
            y_effective = self.mistake_weighting.adjust_labels(X, y_effective, feature_cols)
            flips = int((y_effective != y).sum())
            if flips > 0:
                self.logger.info(f"Labels adjusted: {flips} samples flipped based on trade memory")

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_effective[:split_idx], y_effective[split_idx:]

        sample_weight = self._compute_sample_weights(y_train, multiplier=sample_weight_multiplier)

        if self.mistake_weighting and feature_cols:
            sample_weight = self.mistake_weighting.compute_weights(
                X_train, y_train, sample_weight, feature_cols
            )

        model_params = model_params or {}

        results = {"models": {}, "ensemble": {}}
        ml_config = config.ml

        model_names = []
        if ml_config["enable_xgboost"]:
            model_names.append("xgboost")
        if ml_config["enable_random_forest"]:
            model_names.append("random_forest")
        if ml_config["enable_lightgbm"]:
            model_names.append("lightgbm")
        if ml_config.get("enable_lstm", True):
            model_names.append("lstm")

        for name in model_names:
            try:
                self.logger.info(f"Training {name}...")
                model = self._create_model_instance(name)
                if model is None:
                    self.logger.warning(f"Skipping {name}: not available")
                    results["models"][name] = {"error": "not available"}
                    continue

                params = model_params.get(name, {})
                model.create_model(**params)
                if hasattr(model, "_available") and not model._available:
                    self.logger.warning(f"Skipping {name}: not available")
                    results["models"][name] = {"error": "not available"}
                    continue

                result = model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
                results["models"][name] = result
                self.ensemble.register_model(name, model)

                if self.mistake_weighting and hasattr(model.model, "feature_importances_"):
                    try:
                        importances = model.model.feature_importances_
                        if feature_cols and len(importances) == len(feature_cols):
                            fi_dict = {feature_cols[i]: float(imp) for i, imp in enumerate(importances)}
                            self.mistake_weighting.set_feature_importance(fi_dict)
                    except Exception:
                        pass

                self.logger.info(f"{name} trained: {result}")
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {e}")
                results["models"][name] = {"error": str(e)}

        if self.ensemble.get_num_models() > 0:
            results["ensemble"]["num_models"] = self.ensemble.get_num_models()
            results["ensemble"]["active_models"] = self.ensemble.get_active_models()
            self.logger.info(f"Ensemble ready with {self.ensemble.get_num_models()} models")

        return results

    def get_ensemble(self) -> VotingEnsemble:
        return self.ensemble
