import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

from core.exceptions import ModelError
from utils.logger import get_logger
from utils.decorators import safe_execute


class RandomForestModel:
    def __init__(self):
        self.logger = get_logger("random_forest_model")
        self.model = None
        self.feature_importance: Optional[Dict] = None
        self._trained = False
        self._available = False
        self._import_error = None
        self._try_import()

    def _try_import(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            self._rf = RandomForestClassifier
            self._available = True
        except ImportError as e:
            self._available = False
            self._import_error = str(e)
            self.logger.warning(f"RandomForest not available: {e}")

    def create_model(self, **kwargs):
        if not self._available:
            raise ModelError(f"scikit-learn not installed: {self._import_error}")
        params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "max_depth": kwargs.get("max_depth", 8),
            "min_samples_split": kwargs.get("min_samples_split", 10),
            "min_samples_leaf": kwargs.get("min_samples_leaf", 5),
            "max_features": kwargs.get("max_features", "sqrt"),
            "bootstrap": kwargs.get("bootstrap", True),
            "oob_score": kwargs.get("oob_score", True),
            "class_weight": kwargs.get("class_weight", "balanced_subsample"),
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": kwargs.get("n_jobs", -1),
        }
        self.model = self._rf(**params)
        self.logger.info("Random Forest model created")
        return self.model

    @safe_execute(default_return=None, raise_on_error=True)
    def train(self, X_train, y_train, X_val=None, y_val=None, sample_weight=None,
              progress_callback=None):
        if self.model is None:
            self.create_model()
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if progress_callback is not None:
            n_estimators = self.model.get_params().get("n_estimators", 200)
            chunk = 25
            self.model.set_params(warm_start=True)
            for i in range(chunk, n_estimators + 1, chunk):
                self.model.set_params(n_estimators=i)
                self.model.fit(X_train, y_train, **fit_kwargs)
                progress_callback(i)
            self.model.set_params(warm_start=False)
            self.model.n_estimators = n_estimators
        else:
            self.model.fit(X_train, y_train, **fit_kwargs)
        self._trained = True
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val) if X_val is not None and y_val is not None else None
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = {
                f"feature_{i}": float(imp)
                for i, imp in enumerate(self.model.feature_importances_)
            }
        oob_score = self.model.oob_score_ if hasattr(self.model, "oob_score_") else None
        result = {"train_accuracy": float(train_score)}
        if val_score is not None:
            result["val_accuracy"] = float(val_score)
        if oob_score is not None:
            result["oob_score"] = float(oob_score)
        self.logger.info(f"Random Forest trained. Train acc: {train_score:.4f}")
        return result

    @safe_execute(default_return=None, raise_on_error=True)
    def predict_proba(self, X):
        if self.model is None or not self._trained:
            raise ModelError("Random Forest model not trained")
        return self.model.predict_proba(X)

    @safe_execute(default_return=None, raise_on_error=True)
    def predict(self, X):
        if self.model is None or not self._trained:
            raise ModelError("Random Forest model not trained")
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
            raise ModelError(f"Failed to load Random Forest model: {e}")

    @property
    def is_trained(self) -> bool:
        return self._trained and self._available

    def get_params(self) -> Dict:
        return self.model.get_params() if self.model else {}
