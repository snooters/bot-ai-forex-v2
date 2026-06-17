"""Ensemble Decision Engine — weighted voting from 3 models.

Combines H4, H1, and M5 model predictions into a final decision.
"""
from typing import Dict, Optional, Tuple
import numpy as np
from ensemble.config import CONFIG
from ensemble.h4_model import H4TrendModel
from ensemble.h1_model import H1EntryModel
from ensemble.m5_model import M5TimingModel


class EnsembleDecision:
    """Weighted ensemble of H4 + H1 + M5 models."""
    
    def __init__(self):
        self.h4_model = H4TrendModel()
        self.h1_model = H1EntryModel()
        self.m5_model = M5TimingModel()
        self._loaded = False
    
    def load_all(self) -> bool:
        """Load all 3 models from disk."""
        h4_ok = self.h4_model.load()
        h1_ok = self.h1_model.load()
        m5_ok = self.m5_model.load()
        self._loaded = h4_ok and h1_ok and m5_ok
        if self._loaded:
            print(f"[Ensemble] All 3 models loaded")
            print(f"  H4: val_acc={self.h4_model.val_accuracy:.4f}")
            print(f"  H1: val_acc={self.h1_model.val_accuracy:.4f}")
            print(f"  M5: val_acc={self.m5_model.val_accuracy:.4f}")
        else:
            missing = []
            if not h4_ok: missing.append("H4")
            if not h1_ok: missing.append("H1")
            if not m5_ok: missing.append("M5")
            print(f"[Ensemble] Missing models: {', '.join(missing)}")
        return self._loaded
    
    def decide(self, h4_features: np.ndarray,
               h1_features: np.ndarray,
               m5_features: np.ndarray) -> Dict:
        """Make ensemble decision.
        
        Args:
            h4_features: 15 features for H4 model
            h1_features: 14 features for H1 model
            m5_features: 8 features for M5 model
        
        Returns:
            decision dict with:
                - action: "BUY" | "SELL" | "HOLD"
                - confidence: float 0-1
                - details: per-model breakdown
        """
        if not self._loaded:
            return {"action": "HOLD", "confidence": 0.0, 
                    "details": {"error": "Models not loaded"}}
        
        # Get predictions from each model
        h4_dir, h4_conf = self.h4_model.predict(h4_features)
        h1_sig, h1_conf = self.h1_model.predict(h1_features)
        m5_pull, m5_conf = self.m5_model.predict(m5_features)
        
        # H4: 1 = BULLISH, 0 = BEARISH
        # H1: 1 = BUY signal, 0 = HOLD
        # M5: 1 = pullback expected (wait), 0 = no pullback (ok to enter)
        
        # Calculate weighted direction score
        # Positive = BUY bias, Negative = SELL bias
        h4_score = (h4_dir * 2 - 1) * CONFIG.H4_WEIGHT * h4_conf  # -1 to +1
        h1_score = (h1_sig * 2 - 1) * CONFIG.H1_WEIGHT * h1_conf  # -1 to +1
        
        # M5: if pullback predicted, reduce confidence (need to wait)
        wait_penalty = 1.0
        if m5_pull == 1 and m5_conf > 0.5:
            wait_penalty = 0.7  # Reduce confidence by 30% if pullback expected
        
        # Combined score
        raw_score = h4_score + h1_score
        confidence = abs(raw_score) * wait_penalty
        # Normalize confidence to 0-1
        max_possible = CONFIG.H4_WEIGHT * 1.0 + CONFIG.H1_WEIGHT * 1.0
        confidence = min(confidence / max_possible, 1.0) if max_possible > 0 else 0
        
        # Determine action
        if raw_score > 0.1 and confidence >= CONFIG.MIN_CONFIDENCE:
            action = "BUY"
        elif raw_score < -0.1 and confidence >= CONFIG.MIN_CONFIDENCE:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Apply M5 timing: if pullback expected and confidence still good,
        # suggest HOLD with reason "wait for pullback"
        if m5_pull == 1 and m5_conf > 0.55 and action != "HOLD":
            action = "HOLD"
            reason = f"wait_for_pullback (M5 confidence={m5_conf:.0%})"
        else:
            reason = "ok_to_enter"
        
        decision = {
            "action": action,
            "confidence": round(confidence, 4),
            "raw_score": round(raw_score, 4),
            "wait_penalty": round(wait_penalty, 4),
            "details": {
                "h4": {
                    "direction": "BULLISH" if h4_dir == 1 else "BEARISH",
                    "confidence": round(h4_conf, 4),
                    "weight": CONFIG.H4_WEIGHT,
                },
                "h1": {
                    "signal": "BUY" if h1_sig == 1 else "HOLD",
                    "confidence": round(h1_conf, 4),
                    "weight": CONFIG.H1_WEIGHT,
                },
                "m5": {
                    "pullback_predicted": bool(m5_pull),
                    "confidence": round(m5_conf, 4),
                    "weight": CONFIG.M5_WEIGHT,
                },
            },
            "reason": reason,
        }
        
        return decision
    
    def get_model_summary(self) -> Dict:
        """Get summary of all model performances."""
        return {
            "loaded": self._loaded,
            "h4": {
                "val_accuracy": self.h4_model.val_accuracy,
                "oos_score": self.h4_model.oos_score,
            },
            "h1": {
                "val_accuracy": self.h1_model.val_accuracy,
                "oos_score": self.h1_model.oos_score,
            },
            "m5": {
                "val_accuracy": self.m5_model.val_accuracy,
                "oos_score": self.m5_model.oos_score,
            },
        }
