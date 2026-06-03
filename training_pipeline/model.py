import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .config import TrainingConfig
from .utils import setup_logger, safe_json_dump


class XGBoostModel:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = setup_logger("model", config.log_dir, config.log_level)
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []

    def build(self, **overrides) -> "XGBoostModel":
        params = dict(self.config.xgb_params)
        params.update(overrides)
        self.model = xgb.XGBClassifier(**params)
        return self

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            self.build()

        self.feature_names = list(X_train.columns)
        eval_set = [(X_train, y_train)]
        sample_weight = None

        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            weights = self._compute_sample_weights(y_train)
            if weights is not None:
                sample_weight = weights

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            sample_weight=sample_weight,
            verbose=False,
        )

        result = {"best_iteration": self.model.best_iteration}
        if hasattr(self.model, "best_score"):
            result["best_score"] = float(self.model.best_score)
        self.logger.info(f"Training complete: best_iteration={result.get('best_iteration', 'N/A')}")
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        X_aligned = self._align_features(X)
        return self.model.predict_proba(X_aligned)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            return X
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            self.logger.warning(f"Missing features: {missing}, filling with 0")
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.feature_names]

    def _compute_sample_weights(self, y: pd.Series) -> Optional[np.ndarray]:
        try:
            counts = y.value_counts()
            n = len(y)
            n_classes = len(counts)
            weights = np.ones(n, dtype=np.float64)
            for cls, cnt in counts.items():
                mask = y.values == cls
                weights[mask] = n / (n_classes * cnt)
            return weights
        except Exception as e:
            self.logger.warning(f"Could not compute sample weights: {e}")
            return None

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, map(float, importance)))

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        model_path = save_path / "model.json"
        self.model.save_model(str(model_path))
        joblib.dump(self.feature_names, save_path / "features.joblib")
        safe_json_dump(self.get_feature_importance(), str(save_path / "feature_importance.json"))
        self.logger.info(f"Model saved to {save_path}")

    def load(self, path: str) -> "XGBoostModel":
        load_path = Path(path)
        model_file = load_path / "model.json"
        features_file = load_path / "features.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_file))
        if features_file.exists():
            self.feature_names = joblib.load(features_file)
        self.logger.info(f"Model loaded from {load_path}")
        return self
