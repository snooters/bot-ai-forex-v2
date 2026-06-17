"""M5 Timing Model — Random Forest for pullback prediction.

Predicts whether M5 price will drop >3 pips within next 6 bars (~30 min).
This acts as a timing filter: if pullback predicted, wait before entering.
"""
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ensemble.config import CONFIG


class M5TimingModel:
    """Random Forest model for M5 pullback timing."""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir or CONFIG.model_dir_path / "m5")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[RandomForestClassifier] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        self.oos_score: float = 0.0
        self.val_accuracy: float = 0.0
    
    def train(self, X: np.ndarray, y: np.ndarray,
              times: pd.DatetimeIndex,
              test_size: float = 0.20) -> Dict[str, float]:
        """Train Random Forest with chronological split.
        
        Uses a subset of data (max 50k samples) for faster training.
        """
        n = len(X)
        split_idx = int(n * (1 - test_size))
        
        # Use subset for faster training
        max_train = 50000
        if n > max_train:
            # Take most recent data
            X = X[-max_train:]
            y = y[-max_train:]
            times = times[-max_train:]
            n = len(X)
            split_idx = int(n * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Handle class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        class_weight = {0: 1.0, 1: neg_count / max(pos_count, 1)}
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=50,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.val_accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results = {
            "val_accuracy": self.val_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "pos_ratio_train": float(y_train.mean()),
            "pos_ratio_test": float(y_test.mean()),
        }
        
        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = {
                CONFIG.M5_FEATURES[i]: float(self.model.feature_importances_[i])
                for i in range(len(CONFIG.M5_FEATURES))
            }
            results["feature_importance"] = self.feature_importance
        
        self.oos_score = self.val_accuracy
        print(f"  [M5] val_acc={self.val_accuracy:.4f}, precision={precision:.4f}, "
              f"recall={recall:.4f}, f1={f1:.4f}")
        return results
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict pullback timing.
        
        Returns:
            (will_pullback, confidence)
            will_pullback: 1 = pullback predicted (wait), 0 = no pullback (ok to enter)
            confidence: probability (0-1)
        """
        if self.model is None:
            return 0, 0.38  # Default to no pullback (baseline rate)
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        pred = int(self.model.predict(features.reshape(1, -1))[0])
        confidence = float(max(proba))
        return pred, confidence
    
    def save(self) -> Path:
        """Save model and metadata."""
        if self.model is None:
            raise ValueError("No model to save")
        
        import joblib
        model_path = self.model_dir / "model.joblib"
        joblib.dump(self.model, model_path)
        
        meta = {
            "val_accuracy": self.val_accuracy,
            "oos_score": self.oos_score,
            "feature_importance": self.feature_importance,
            "features": CONFIG.M5_FEATURES,
        }
        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"  [M5] Model saved to {model_path}")
        return model_path
    
    def load(self) -> bool:
        """Load saved model."""
        import joblib
        model_path = self.model_dir / "model.joblib"
        if not model_path.exists():
            return False
        
        self.model = joblib.load(model_path)
        
        meta_path = self.model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.val_accuracy = meta.get("val_accuracy", 0)
            self.oos_score = meta.get("oos_score", 0)
            self.feature_importance = meta.get("feature_importance")
        
        return True
