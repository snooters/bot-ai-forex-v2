import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from core.constants import MODEL_DIR
from utils.logger import get_logger


class ProbabilityCalibrator:
    def __init__(self, method: str = "platt"):
        self.logger = get_logger("prob_calibrator")
        self.method = method
        self.calibrators = {}
        self.brier_score = None
        self.ece = None
        self._fitted = False

    def train(self, y_true: np.ndarray, y_pred_proba: np.ndarray):
        n_classes = y_pred_proba.shape[1]
        self.calibrators = {}

        for i in range(n_classes):
            y_binary = (y_true == i).astype(float)
            probs_i = y_pred_proba[:, i]

            if self.method == "isotonic":
                from sklearn.isotonic import IsotonicRegression
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(probs_i, y_binary)
                self.calibrators[i] = ir
            else:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(C=1.0, solver="lbfgs")
                probs_2d = probs_i.reshape(-1, 1)
                lr.fit(probs_2d, y_binary)
                self.calibrators[i] = lr

        self._fitted = True
        self.brier_score = self.compute_brier_score(y_true, y_pred_proba)
        self.ece = self.compute_ece(y_true, y_pred_proba)
        self.logger.info(
            f"Calibrator trained ({self.method}): "
            f"Brier={self.brier_score:.4f} ECE={self.ece:.4f}"
        )

    def calibrate(self, raw_probas: np.ndarray) -> np.ndarray:
        if not self._fitted or not self.calibrators:
            return raw_probas

        n_classes = raw_probas.shape[1]
        calibrated = np.zeros_like(raw_probas)

        for i in range(n_classes):
            if i in self.calibrators:
                probs_i = raw_probas[:, i]
                if self.method == "isotonic":
                    cal = self.calibrators[i].predict(probs_i)
                else:
                    probs_2d = probs_i.reshape(-1, 1)
                    cal = self.calibrators[i].predict_proba(probs_2d)[:, 1]
                calibrated[:, i] = cal
            else:
                calibrated[:, i] = raw_probas[:, i]

        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        calibrated = calibrated / row_sums

        return calibrated

    def compute_brier_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        n = len(y_true)
        y_onehot = np.zeros((n, y_pred_proba.shape[1]))
        y_onehot[np.arange(n), y_true] = 1.0
        return float(np.mean((y_pred_proba - y_onehot) ** 2))

    def compute_ece(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        confidences = np.max(y_pred_proba, axis=1)
        predictions = np.argmax(y_pred_proba, axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > lo) & (confidences <= hi)
            if in_bin.sum() == 0:
                continue
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (in_bin.sum() / len(y_true)) * abs(bin_acc - bin_conf)
        return float(ece)

    def reliability_diagram(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> Dict:
        confidences = np.max(y_pred_proba, axis=1)
        predictions = np.argmax(y_pred_proba, axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins = []
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > lo) & (confidences <= hi)
            if in_bin.sum() == 0:
                continue
            bins.append({
                "bin_center": (lo + hi) / 2,
                "accuracy": float(accuracies[in_bin].mean()),
                "confidence": float(confidences[in_bin].mean()),
                "count": int(in_bin.sum()),
            })
        return {
            "bins": bins,
            "brier_score": self.compute_brier_score(y_true, y_pred_proba),
            "ece": self.compute_ece(y_true, y_pred_proba, n_bins),
        }

    def save(self, path: str):
        import joblib
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        for i, cal in self.calibrators.items():
            joblib.dump(cal, p / f"calibrator_{i}.pkl")
        meta = {
            "method": self.method,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "num_classes": len(self.calibrators),
        }
        with open(p / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        self.logger.info(f"Calibrator saved to {p}")

    def load(self, path: str):
        import joblib
        p = Path(path)
        if not p.exists():
            self.logger.warning(f"No calibrator found at {p}")
            return
        meta_path = p / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.method = meta.get("method", "platt")
            self.brier_score = meta.get("brier_score")
            self.ece = meta.get("ece")
        self.calibrators = {}
        for f in p.glob("calibrator_*.pkl"):
            idx = int(f.stem.split("_")[1])
            self.calibrators[idx] = joblib.load(f)
        self._fitted = len(self.calibrators) > 0
        self.logger.info(f"Calibrator loaded from {p} ({len(self.calibrators)} classes)")

    @property
    def is_fitted(self) -> bool:
        return self._fitted
