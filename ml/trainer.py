import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from core.config import config
from core.constants import LOOKAHEAD_5, MODEL_DIR, RESULTS_DIR
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
        buy_threshold: float = 0.001,
        sell_threshold: float = 0.001,
        target_type: str = "class",
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
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

        if target_type == "regression":
            y = future_return.values
        else:
            y = np.zeros(len(df), dtype=int)
            y[future_return > buy_threshold] = 0
            y[future_return < -sell_threshold] = 1
            y[(future_return >= -sell_threshold) & (future_return <= buy_threshold)] = 2

        X = df[available_cols].values
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]
        df_clean = df[mask].copy() if hasattr(df, 'iloc') else df

        if len(X) < 100:
            raise ModelTrainingError(f"Too few training samples: {len(X)}")

        if target_type != "regression":
            self.logger.info(f"Prepared {len(X)} training samples with {len(available_cols)} features. "
                             f"BUY: {(y==0).sum()}, SELL: {(y==1).sum()}, HOLD: {(y==2).sum()}")
        else:
            self.logger.info(f"Prepared {len(X)} training samples with {len(available_cols)} features (regression). "
                             f"return range=[{y.min():.4f}, {y.max():.4f}]")
        return X, y, available_cols, df_clean

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



    @staticmethod
    def compute_recency_weights(time_col: pd.Series, now: Optional[datetime] = None) -> np.ndarray:
        """Weight by recency: newer data gets higher weight.
        0-30d=1.0, 31-90d=0.8, 91-180d=0.6, 181-365d=0.4, >365d=0.2.
        """
        if now is None:
            now = datetime.now()
        age_days = (now - time_col).dt.total_seconds() / 86400.0
        weights = np.ones(len(age_days), dtype=float)
        weights[age_days > 365] = 0.2
        for lo, hi, val in [(180, 365, 0.4), (90, 180, 0.6), (30, 90, 0.8)]:
            weights[(age_days > lo) & (age_days <= hi)] = val
        return weights

    @measure_time
    def train_all_models(self, X: np.ndarray, y: np.ndarray,
                         sample_weight_multiplier: float = 1.0,
                         model_params: Optional[Dict] = None,
                         feature_cols: Optional[List[str]] = None,
                         progress=None,
                         tf_label=None,
                         target_type: str = "class",
                         recency_weights: Optional[np.ndarray] = None) -> Dict:
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
        if recency_weights is not None:
            rw_train = recency_weights[:split_idx]
            sample_weight = sample_weight * rw_train
            self.logger.info(f"Recency weights applied: "
                             f"range=[{rw_train.min():.2f}, {rw_train.max():.2f}]")

        if self.mistake_weighting and feature_cols:
            sample_weight = self.mistake_weighting.compute_weights(
                X_train, y_train, sample_weight, feature_cols
            )
            sample_weight = self.mistake_weighting.compute_no_trade_weights(
                X_train, sample_weight, feature_cols
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

                if progress and tf_label:
                    n_est = model.model.get_params().get("n_estimators", 200) if hasattr(model, "model") and model.model else 200
                    if hasattr(model, "model") and model.model and hasattr(model.model, "get_params"):
                        try:
                            n_est = model.model.get_params().get("n_estimators", 200)
                        except Exception:
                            pass
                    if name == "random_forest":
                        cb_total = n_est
                    else:
                        cb_total = n_est
                    progress.begin_model(name, total=cb_total)

                result = model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight,
                                     progress_callback=progress.make_model_callback(name) if (progress and tf_label) else None)
                results["models"][name] = result
                self.ensemble.register_model(name, model)

                if progress and tf_label:
                    progress.end_model(name)

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

    def save_feature_importance(self, ensemble: VotingEnsemble, feature_cols: List[str],
                                 pair: str, timeframe: int, version: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available, skipping feature importance plot")
            return

        importance_dir = Path(RESULTS_DIR) / pair / str(timeframe) / version
        importance_dir.mkdir(parents=True, exist_ok=True)

        all_fi = {}
        for name, model in ensemble.get_active_models().items():
            if hasattr(model.model, "feature_importances_"):
                fi = model.model.feature_importances_
                if len(fi) == len(feature_cols):
                    all_fi[name] = {feature_cols[i]: float(f) for i, f in enumerate(fi)}

        if not all_fi:
            self.logger.warning("No feature importances available from any model")
            return

        avg_fi = {}
        for col in feature_cols:
            vals = [all_fi[m][col] for m in all_fi if col in all_fi[m]]
            avg_fi[col] = sum(vals) / len(vals) if vals else 0.0

        fi_df = pd.DataFrame(avg_fi.items(), columns=["feature", "importance"])
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_path = importance_dir / "feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        self.logger.info(f"Feature importance saved to {fi_path}")

        try:
            top_n = min(30, len(fi_df))
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(top_n), fi_df.iloc[:top_n]["importance"][::-1])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(fi_df.iloc[:top_n]["feature"][::-1])
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance — {pair} ({timeframe}m) v{version}")
            plt.tight_layout()
            png_path = importance_dir / "feature_importance.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            self.logger.info(f"Feature importance plot saved to {png_path}")
        except Exception as e:
            self.logger.warning(f"Failed to create importance plot: {e}")

        json_path = importance_dir / "feature_importance.json"
        with open(json_path, "w") as f:
            json.dump(all_fi, f, indent=2)

    def get_ensemble(self) -> VotingEnsemble:
        return self.ensemble
