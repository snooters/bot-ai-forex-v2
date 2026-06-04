import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

from core.exceptions import ModelError
from utils.logger import get_logger
from utils.decorators import safe_execute


class LightGBMModel:
    def __init__(self):
        self.logger = get_logger("lightgbm_model")
        self.model = None
        self.feature_importance: Optional[Dict] = None
        self._trained = False
        self._available = False
        self._import_error = None
        self._try_import()

    def _try_import(self):
        try:
            import lightgbm as lgb
            self._lgb = lgb
            self._lgb_classifier = lgb.LGBMClassifier
            self._available = True
        except ImportError as e:
            self._available = False
            self._import_error = str(e)
            self.logger.warning(f"LightGBM not available: {e}")

    def create_model(self, **kwargs):
        if not self._available:
            raise ModelError(f"LightGBM not installed: {self._import_error}")
        params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "num_leaves": kwargs.get("num_leaves", 31),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "min_child_samples": kwargs.get("min_child_samples", 20),
            "reg_alpha": kwargs.get("reg_alpha", 0.1),
            "reg_lambda": kwargs.get("reg_lambda", 1.0),
            "class_weight": kwargs.get("class_weight", "balanced"),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": kwargs.get("n_jobs", -1),
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbosity": -1,
        }
        self.model = self._lgb_classifier(**params)
        self.logger.info("LightGBM model created")
        return self.model

    @safe_execute(default_return=None, raise_on_error=True)
    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None,
              progress_callback=None):
        if self.model is None:
            self.create_model()
        _X_train = X_train
        _X_val = X_val
        if not isinstance(X_train, pd.DataFrame):
            cols = [f"f{i}" for i in range(X_train.shape[1])]
            _X_train = pd.DataFrame(X_train, columns=cols)
            if X_val is not None:
                _X_val = pd.DataFrame(X_val, columns=cols)
        eval_set = [(_X_train, y_train)]
        if _X_val is not None and y_val is not None:
            eval_set.append((_X_val, y_val))
        fit_kwargs = {"eval_set": eval_set}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if progress_callback is not None:
            n_estimators = self.model.get_params().get("n_estimators", 200)
            def _lgb_callback(env):
                progress_callback(env.iteration + 1)
            fit_kwargs["callbacks"] = [_lgb_callback]
        self.model.fit(_X_train, y_train, **fit_kwargs)
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
        self.logger.info(f"LightGBM trained. Train acc: {train_score:.4f}")
        return result

    @safe_execute(default_return=None, raise_on_error=True)
    def predict_proba(self, X):
        if self.model is None or not self._trained:
            raise ModelError("LightGBM model not trained")
        if not isinstance(X, pd.DataFrame) and hasattr(self.model, "feature_names_in_"):
            X = pd.DataFrame(X, columns=self.model.feature_names_in_)
        return self.model.predict_proba(X)

    @safe_execute(default_return=None, raise_on_error=True)
    def predict(self, X):
        if self.model is None or not self._trained:
            raise ModelError("LightGBM model not trained")
        if not isinstance(X, pd.DataFrame) and hasattr(self.model, "feature_names_in_"):
            X = pd.DataFrame(X, columns=self.model.feature_names_in_)
        return self.model.predict(X)

    def save(self, path: str):
        if self.model is None:
            raise ModelError("No model to save")
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str):
        try:
            import joblib
            self.model = joblib.load(path)
            self._trained = True
        except Exception as e:
            raise ModelError(f"Failed to load LightGBM model: {e}")

    @property
    def is_trained(self) -> bool:
        return self._trained and self._available

    def get_params(self) -> Dict:
        return self.model.get_params() if self.model else {}
