import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

from core.exceptions import ModelError, ModelTrainingError
from utils.logger import get_logger
from utils.decorators import safe_execute


class XGBoostModel:
    def __init__(self):
        self.logger = get_logger("xgboost_model")
        self.model = None
        self.feature_importance: Optional[Dict] = None
        self._trained = False
        self._available = False
        self._import_error = None
        self._try_import()

    def _try_import(self):
        try:
            from xgboost import XGBClassifier
            self._xgboost = XGBClassifier
            self._available = True
        except ImportError as e:
            self._available = False
            self._import_error = str(e)
            self.logger.warning(f"XGBoost not available: {e}")

    def create_model(self, **kwargs):
        if not self._available:
            raise ModelError(f"XGBoost not installed: {self._import_error}")
        params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "min_child_weight": kwargs.get("min_child_weight", 3),
            "gamma": kwargs.get("gamma", 0.1),
            "reg_alpha": kwargs.get("reg_alpha", 0.1),
            "reg_lambda": kwargs.get("reg_lambda", 1.0),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": kwargs.get("n_jobs", -1),
            "eval_metric": kwargs.get("eval_metric", "mlogloss"),
            "objective": "multi:softprob",
            "num_class": 3,
            "early_stopping_rounds": kwargs.get("early_stopping_rounds", 20),
            "verbosity": 0,
        }
        self.model = self._xgboost(**params)
        self.logger.info("XGBoost model created")
        return self.model

    @safe_execute(default_return=None, raise_on_error=True)
    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None,
              progress_callback=None):
        if self.model is None:
            self.create_model()
        if progress_callback is not None:
            try:
                from xgboost.callback import TrainingCallback
                class _ProgressCB(TrainingCallback):
                    def after_iteration(self, bst, epoch, evals_log):
                        progress_callback(epoch + 1)
                        return False
                self.model.callbacks = [_ProgressCB()]
            except Exception:
                pass
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        fit_kwargs = {"eval_set": eval_set, "verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self.model.fit(X_train, y_train, **fit_kwargs)
        self.model.callbacks = None
        self._trained = True
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val) if X_val is not None else None
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = {
                f"feature_{i}": float(imp)
                for i, imp in enumerate(self.model.feature_importances_)
            }
        result = {"train_accuracy": float(train_score)}
        if val_score is not None:
            result["val_accuracy"] = float(val_score)
        self.logger.info(f"XGBoost trained. Train acc: {train_score:.4f}")
        return result

    @safe_execute(default_return=None, raise_on_error=True)
    def predict_proba(self, X):
        if self.model is None or not self._trained:
            raise ModelError("XGBoost model not trained")
        return self.model.predict_proba(X)

    @safe_execute(default_return=None, raise_on_error=True)
    def predict(self, X):
        if self.model is None or not self._trained:
            raise ModelError("XGBoost model not trained")
        return self.model.predict(X)

    def save(self, path: str):
        if self.model is None:
            raise ModelError("No model to save")
        self.model.save_model(path)

    def load(self, path: str):
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier()
            self.model.load_model(path)
            self._trained = True
        except Exception as e:
            raise ModelError(f"Failed to load XGBoost model: {e}")

    @property
    def is_trained(self) -> bool:
        return self._trained and self._available

    def get_params(self) -> Dict:
        return self.model.get_params() if self.model else {}
