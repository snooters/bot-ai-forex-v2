from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import config
from utils.logger import get_logger


DRIFT_WINDOW = 30
VOLATILITY_MULTIPLIER = 2.0
SPREAD_MULTIPLIER = 2.0
MIN_TRADES_FOR_DRIFT = 20


class ConceptDriftDetector:
    def __init__(self, window_size: int = 30):
        self.logger = get_logger("concept_drift")
        self.window_size = window_size
        self._baseline_win_rate: Optional[float] = None
        self._baseline_profit_factor: Optional[float] = None
        self._baseline_volatility: Optional[float] = None
        self._baseline_spread: Optional[float] = None
        self._baseline_win_loss_ratio: Optional[float] = None
        self._drift_detected = False
        self._drift_history: List[Dict] = []

    def detect_drift(self, trades: List[Dict]) -> Dict:
        if not trades or len(trades) < DRIFT_WINDOW:
            return self._no_drift_result("insufficient trades")

        closed_trades = [t for t in trades if t.get("profit") is not None]
        if len(closed_trades) < DRIFT_WINDOW:
            return self._no_drift_result("insufficient closed trades")

        closed_trades.sort(key=lambda t: t.get("exit_time", ""))

        recent = closed_trades[-self.window_size:]
        older = closed_trades[:-self.window_size]

        recent_win_rate = self._compute_win_rate(recent)
        older_win_rate = self._compute_win_rate(older) if older else recent_win_rate

        recent_pf = self._compute_profit_factor(recent)
        older_pf = self._compute_profit_factor(older) if older else recent_pf

        recent_wlr = self._compute_win_loss_ratio(recent)
        older_wlr = self._compute_win_loss_ratio(older) if older else recent_wlr

        win_rate_drop = older_win_rate - recent_win_rate if older_win_rate > 0 else 0
        pf_drop = older_pf - recent_pf if older_pf > 0 else 0
        wlr_drop = older_wlr - recent_wlr if older_wlr > 0 else 0

        pf_change_score = pf_drop * 10 if pf_drop > 0 else 0
        wlr_change_score = wlr_drop * 5 if wlr_drop > 0 else 0

        drift_score = max(win_rate_drop, pf_change_score, wlr_change_score)
        drift_score = max(0, min(drift_score, 1))

        if self._baseline_win_rate is None:
            self._baseline_win_rate = older_win_rate
            self._baseline_profit_factor = older_pf
            self._baseline_win_loss_ratio = older_wlr

        drift_type = "none"
        if drift_score > 0.15:
            drift_type = "performance"
        elif win_rate_drop > 0.10:
            drift_type = "win_rate_degradation"

        self._drift_detected = drift_score > 0.15
        if self._drift_detected:
            self._drift_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": drift_type,
                "score": float(drift_score),
                "recent_win_rate": float(recent_win_rate),
                "older_win_rate": float(older_win_rate),
            })

        return {
            "drift_detected": self._drift_detected,
            "type": drift_type,
            "score": float(drift_score),
            "confidence": float(min(drift_score * 2, 1.0)),
            "recent_win_rate": float(recent_win_rate * 100),
            "older_win_rate": float(older_win_rate * 100),
            "recent_profit_factor": float(recent_pf) if recent_pf != float("inf") else 999.0,
            "older_profit_factor": float(older_pf) if older_pf != float("inf") else 999.0,
            "win_rate_drop": float(win_rate_drop * 100),
            "drift_history_count": len(self._drift_history),
        }

    def detect_volatility_drift(self, recent_volatility: float, older_volatility: float) -> Dict:
        if self._baseline_volatility is None:
            self._baseline_volatility = older_volatility if older_volatility > 0 else recent_volatility

        vol_change = recent_volatility / self._baseline_volatility if self._baseline_volatility > 0 else 1.0
        spread_change = 1.0

        volatility_drift = vol_change > VOLATILITY_MULTIPLIER
        extreme_volatility = vol_change > VOLATILITY_MULTIPLIER * 1.5

        drift_detected = volatility_drift or extreme_volatility

        if drift_detected:
            self._drift_detected = True
            self._drift_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "volatility",
                "vol_change": float(vol_change),
                "spread_change": float(spread_change),
            })

        return {
            "drift_detected": drift_detected,
            "type": "volatility" if drift_detected else "none",
            "volatility_change": float(vol_change),
            "spread_change": float(spread_change),
            "score": float(min(max(vol_change - 1, 0) / VOLATILITY_MULTIPLIER, 1.0)),
            "confidence": float(min(max(vol_change - 1, 0) / VOLATILITY_MULTIPLIER, 1.0)),
            "baseline_volatility": float(self._baseline_volatility),
        }

    def detect_spread_drift(self, recent_spread: float, older_spread: float) -> Dict:
        if self._baseline_spread is None:
            self._baseline_spread = older_spread if older_spread > 0 else recent_spread

        spread_change = recent_spread / self._baseline_spread if self._baseline_spread > 0 else 1.0
        spread_drift = spread_change > SPREAD_MULTIPLIER

        if spread_drift:
            self._drift_detected = True
            self._drift_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "spread",
                "spread_change": float(spread_change),
            })

        return {
            "drift_detected": spread_drift,
            "score": float(min(max(spread_change - 1, 0) / SPREAD_MULTIPLIER, 1.0)),
            "spread_change": float(spread_change),
        }

    def should_retrain(self, trades: List[Dict]) -> Tuple[bool, str]:
        if len(trades) < MIN_TRADES_FOR_DRIFT:
            return False, "insufficient trades for drift detection"

        drift = self.detect_drift(trades)
        if drift["drift_detected"]:
            return True, f"Performance drift: score={drift['score']:.2f} WR_drop={drift['win_rate_drop']:.1f}%"

        win_rate_drop = config.learning.get("win_rate_drop_threshold", 0.10)
        if drift.get("win_rate_drop", 0) > win_rate_drop * 100:
            return True, f"Win rate drop: {drift['win_rate_drop']:.1f}%"

        return False, "no drift detected"

    def get_drift_summary(self) -> Dict:
        return {
            "drift_detected": self._drift_detected,
            "total_drift_events": len(self._drift_history),
            "last_drift": self._drift_history[-1] if self._drift_history else None,
            "drift_history": self._drift_history[-5:],
        }

    def _compute_win_rate(self, trades: List[Dict]) -> float:
        if not trades:
            return 0
        wins = sum(1 for t in trades if t.get("profit", 0) > 0)
        return wins / len(trades)

    def _compute_profit_factor(self, trades: List[Dict]) -> float:
        gross_profit = sum(t.get("profit", 0) for t in trades if t.get("profit", 0) > 0)
        gross_loss = abs(sum(t.get("profit", 0) for t in trades if t.get("profit", 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def _compute_win_loss_ratio(self, trades: List[Dict]) -> float:
        wins = [t for t in trades if t.get("profit", 0) > 0]
        losses = [t for t in trades if t.get("profit", 0) < 0]
        avg_win = np.mean([t["profit"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["profit"] for t in losses])) if losses else 0
        return avg_win / avg_loss if avg_loss > 0 else 0

    def reset_baseline(self):
        self._baseline_win_rate = None
        self._baseline_profit_factor = None
        self._baseline_volatility = None
        self._baseline_spread = None
        self._baseline_win_loss_ratio = None
        self._drift_detected = False

    @property
    def drift_status(self) -> bool:
        return self._drift_detected

    def _no_drift_result(self, reason: str = "") -> Dict:
        return {
            "drift_detected": False,
            "type": "none",
            "score": 0,
            "confidence": 0,
            "recent_win_rate": 0,
            "older_win_rate": 0,
            "win_rate_drop": 0,
            "reason": reason,
        }
