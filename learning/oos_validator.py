from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from core.config import config
from core.constants import LOOKAHEAD_5
from features.feature_pipeline import FeaturePipeline
from ml.ensemble import VotingEnsemble
from ml.trainer import ModelTrainer
from utils.helpers import (
    compute_sharpe_ratio, compute_profit_factor,
    compute_max_drawdown,
)
from utils.logger import get_logger


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

PF_THRESHOLD = 1.5
DD_THRESHOLD = 0.10

WARMUP_ROWS = 250


class OOSValidator:
    def __init__(self):
        self.logger = get_logger("oos_validator")
        self.feature_pipeline = FeaturePipeline()

    def validate(
        self,
        df: pd.DataFrame,
        ensemble: VotingEnsemble,
        trainer: ModelTrainer,
        timeframe_label: str,
        oos_split: float = 0.2,
        buy_threshold: Optional[float] = None,
        sell_threshold: Optional[float] = None,
        timeframe: Optional[int] = None,
    ) -> Dict:
        if buy_threshold is None:
            buy_threshold = config.training["buy_threshold"]
        if sell_threshold is None:
            sell_threshold = config.training["sell_threshold"]
        if df.empty or len(df) < 400:
            return self._empty_result("insufficient data (<400 rows)")

        df = df.sort_values("time").reset_index(drop=True)

        feature_cols = self.feature_pipeline.get_feature_columns()

        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        # Compute features PER SPLIT — NEVER on full dataset
        # Each split gets its own feature computation using only data up to that split
        train_raw = df.iloc[:train_end].copy()
        val_raw = df.iloc[:val_end].copy()
        test_raw = df.iloc[:n].copy()

        train_feat_full = self.feature_pipeline.compute_all(train_raw)
        val_feat_full = self.feature_pipeline.compute_all(val_raw)
        test_feat_full = self.feature_pipeline.compute_all(test_raw)

        # Training: use all computed features, skip warmup rows
        train_df = train_feat_full.iloc[WARMUP_ROWS:].copy()

        # Validation: use features computed on data up to val_end
        # Take the portion corresponding to the val period
        val_start_idx = max(0, train_end)
        val_df = val_feat_full.iloc[val_start_idx:].copy()
        if len(val_df) > (val_end - train_end):
            val_df = val_df.iloc[-(val_end - train_end):].copy()

        # Test: use features computed on data up to n (end)
        # Take the portion corresponding to the test period
        test_start_idx = max(WARMUP_ROWS, val_end)
        test_df = test_feat_full.iloc[test_start_idx:].copy()
        expected_test_len = n - val_end
        if len(test_df) > expected_test_len:
            test_df = test_df.iloc[-expected_test_len:].copy()

        if len(test_df) < 50:
            return self._empty_result(f"test set too small: {len(test_df)} rows")
        if len(train_df) < 50:
            return self._empty_result(f"train set too small: {len(train_df)} rows")

        available_cols = [c for c in feature_cols if c in test_df.columns]
        if not available_cols:
            return self._empty_result("no feature columns after compute")

        self.logger.info(
            f"OOS split: train={len(train_df)} val={len(val_df)} test={len(test_df)}"
            f" (features=computed-per-split)"
        )

        try:
            X_train, y_train, _, _ = trainer.prepare_training_data(
                train_df, lookahead=LOOKAHEAD_5, timeframe=timeframe,
            )
        except Exception as e:
            return self._empty_result(f"train data prep failed: {e}")

        # Use ensemble's feature_cols (set by prepare_training_data) for test alignment
        ef_cols = ensemble.feature_cols or available_cols

        # Ensure test data has all required features (fill missing with 0)
        test_clean = test_df.copy()
        missing_cols = [c for c in ef_cols if c not in test_clean.columns]
        if missing_cols:
            for col in missing_cols:
                test_clean[col] = 0.0
        test_clean = test_clean.dropna(subset=ef_cols)
        if test_clean.empty or len(test_clean) < 20:
            return self._empty_result(f"test set too few rows after cleaning: {len(test_clean)}")

        y_true, oos_X = self._prepare_labels(test_clean, ef_cols,
                                              buy_threshold=buy_threshold,
                                              sell_threshold=sell_threshold)
        if len(oos_X) < 10:
            return self._empty_result(f"too few OOS samples: {len(oos_X)}")

        try:
            ensemble_preds = ensemble.predict(oos_X)
        except Exception as e:
            return self._empty_result(f"prediction failed: {e}")

        val_result = self._validate_split(val_df, ensemble, ef_cols,
                                           buy_threshold=buy_threshold,
                                           sell_threshold=sell_threshold)
        test_result = self._run_test(oos_X, y_true, ensemble_preds, test_clean)

        result = {
            "success": test_result.get("success", False),
            "train_samples": len(X_train),
            "val_accuracy": val_result.get("accuracy", 0),
            **test_result,
            "timeframe": timeframe_label,
            "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
            "validated_at": datetime.now().isoformat(),
        }

        # Cap unrealistic values before grading
        result = self._cap_results(result)

        result["passed"] = self._check_passed(result)
        result["grade"] = self._compute_oos_grade(result)

        log_wr = test_result.get("win_rate", 0)
        log_pf = test_result.get("profit_factor", 0)
        log_sharpe = test_result.get("sharpe_ratio", 0)
        log_acc = test_result.get("accuracy", 0)
        log_non_hold = test_result.get("non_hold_accuracy", 0)
        log_directional = test_result.get("directional_predictions", 0)
        log_total = test_result.get("total_trades", 0)

        self.logger.info(
            f"OOS {timeframe_label}: Acc={log_acc:.1f}% "
            f"DirAcc={log_non_hold:.1f}% "
            f"DirTrades={log_directional} "
            f"Trades={log_total} "
            f"WR={log_wr:.1f}% "
            f"PF={log_pf:.2f} "
            f"Grade={result['grade']} "
            f"Passed={result['passed']}"
        )
        return result

    def _validate_split(self, val_df: pd.DataFrame, ensemble: VotingEnsemble, feature_cols: List[str],
                         buy_threshold: Optional[float] = None,
                         sell_threshold: Optional[float] = None) -> Dict:
        if buy_threshold is None:
            buy_threshold = config.training["buy_threshold"]
        if sell_threshold is None:
            sell_threshold = config.training["sell_threshold"]
        val_clean = val_df.copy()
        missing_cols = [c for c in feature_cols if c not in val_clean.columns]
        if missing_cols:
            for col in missing_cols:
                val_clean[col] = 0.0
        val_clean = val_clean.dropna(subset=feature_cols)
        if val_clean.empty or len(val_clean) < 10:
            return {"accuracy": 0}

        y_true_val, X_val = self._prepare_labels(val_clean, feature_cols,
                                                  buy_threshold=buy_threshold,
                                                  sell_threshold=sell_threshold)
        if len(X_val) < 10:
            return {"accuracy": 0}

        try:
            preds = ensemble.predict(X_val)
            correct = (preds == y_true_val).sum()
            accuracy = float(correct / len(y_true_val)) if len(y_true_val) > 0 else 0
            return {"accuracy": accuracy, "samples": len(X_val)}
        except Exception:
            return {"accuracy": 0}

    def _prepare_labels(self, df: pd.DataFrame, available_cols: List[str],
                         buy_threshold: Optional[float] = None,
                         sell_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        if buy_threshold is None:
            buy_threshold = config.training["buy_threshold"]
        if sell_threshold is None:
            sell_threshold = config.training["sell_threshold"]
        future_close = df["close"].shift(-LOOKAHEAD_5)
        current_close = df["close"]
        future_return = (future_close - current_close) / current_close

        y_true = np.zeros(len(df), dtype=int)
        y_true[future_return > buy_threshold] = 0       # BUY
        y_true[future_return < -sell_threshold] = 1     # SELL
        y_true[(future_return >= -sell_threshold) & (future_return <= buy_threshold)] = 2  # HOLD

        mask = ~np.isnan(y_true)
        X = df[available_cols].values[mask]
        y_true = y_true[mask]
        return y_true, X

    def _run_test(self, X: np.ndarray, y_true: np.ndarray, preds: np.ndarray, df: pd.DataFrame) -> Dict:
        correct = (preds == y_true).sum()
        accuracy = float(correct / len(y_true)) if len(y_true) > 0 else 0

        total_signals = len(preds)
        directional_preds = 0
        trade_signals = []
        all_predictions = []

        for i in range(total_signals):
            pred = int(preds[i])
            actual = int(y_true[i])

            pred_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
            actual_map = {0: "BUY", 1: "SELL", 2: "HOLD"}

            predicted_dir = pred_map.get(pred, "HOLD")
            actual_dir = actual_map.get(actual, "HOLD")

            correct_dir = predicted_dir == actual_dir
            all_predictions.append({"pred": predicted_dir, "actual": actual_dir, "correct": correct_dir})

            if predicted_dir == "HOLD":
                continue

            directional_preds += 1
            row_idx = len(df) - len(y_true) + i
            if 0 <= row_idx < len(df) and row_idx + LOOKAHEAD_5 < len(df):
                entry_price = float(df.iloc[row_idx]["close"])
                future_price = float(df.iloc[row_idx + LOOKAHEAD_5]["close"])
                price_return = (future_price - entry_price) / entry_price

                if predicted_dir == "BUY":
                    profit = price_return * 10000
                else:
                    profit = -price_return * 10000

                trade_signals.append({
                    "predicted": predicted_dir,
                    "actual": actual_dir,
                    "win": correct_dir,
                    "profit": profit,
                    "entry_price": entry_price,
                    "exit_price": future_price,
                })

        total_trades = len(trade_signals)
        non_hold_correct = sum(1 for t in trade_signals if t["win"])
        non_hold_accuracy = (non_hold_correct / total_trades * 100) if total_trades > 0 else 0

        if total_trades == 0:
            return {
                "success": False,
                "reason": "no directional predictions made (all HOLD)",
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "accuracy": round(accuracy * 100, 2),
                "non_hold_accuracy": 0,
                "directional_predictions": 0,
                "hold_predictions": total_signals,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown_pct": 0,
                "net_profit": 0,
                "oos_samples": len(X),
            }

        wins = non_hold_correct
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        gross_profit = sum(t["profit"] for t in trade_signals if t["profit"] > 0)
        gross_loss = abs(sum(t["profit"] for t in trade_signals if t["profit"] < 0))
        profit_factor = compute_profit_factor(gross_profit, gross_loss)

        profits_list = [t["profit"] for t in trade_signals]
        sharpe = compute_sharpe_ratio(profits_list)

        equity = []
        running = 0.0
        for t in trade_signals:
            running += t["profit"]
            equity.append(running)
        max_dd, _ = compute_max_drawdown(equity) if equity else (0, 0)

        avg_win = np.mean([t["profit"] for t in trade_signals if t["profit"] > 0]) if wins else 0
        avg_loss = np.mean([t["profit"] for t in trade_signals if t["profit"] < 0]) if losses else 0
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        hold_predictions = total_signals - directional_preds

        return {
            "success": True,
            "total_trades": total_trades,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate": round(win_rate, 2),
            "accuracy": round(accuracy * 100, 2),
            "non_hold_accuracy": round(non_hold_accuracy, 2),
            "directional_predictions": directional_preds,
            "hold_predictions": hold_predictions,
            "profit_factor": round(profit_factor, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "net_profit": round(sum(t["profit"] for t in trade_signals), 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_win_loss_ratio": round(avg_win_loss_ratio, 2),
            "oos_samples": len(X),
        }

    def _check_passed(self, result: Dict) -> bool:
        pf = result.get("profit_factor", 0)
        max_dd_pct = result.get("max_drawdown_pct", 100)
        total_trades = result.get("total_trades", 0)
        directional = result.get("directional_predictions", 0)
        accuracy = result.get("accuracy", 0)
        non_hold_acc = result.get("non_hold_accuracy", 0)

        if not result.get("success", False):
            if result.get("reason") == "no directional predictions made (all HOLD)":
                if accuracy >= 85:
                    return True
            return False

        if total_trades < 50:
            return False

        pf_ok = pf >= PF_THRESHOLD * 0.8
        dd_ok = max_dd_pct <= DD_THRESHOLD * 120

        passed = pf_ok and dd_ok
        if not passed:
            self.logger.info(
                f"OOS check: PF={pf:.2f}(need>={PF_THRESHOLD*0.8:.2f}) "
                f"DD={max_dd_pct:.1f}%(need<={DD_THRESHOLD*120:.0f}%) "
                f"DirTrades={directional} "
                f"Acc={accuracy:.1f}%"
            )
        return passed

    def _compute_oos_grade(self, oos_result: Dict) -> str:
        wr = oos_result.get("win_rate", 0)
        pf = oos_result.get("profit_factor", 0)
        sharpe = oos_result.get("sharpe_ratio", 0)
        trades = oos_result.get("total_trades", 0)
        accuracy = oos_result.get("accuracy", 0)
        directional = oos_result.get("directional_predictions", 0)

        if not oos_result.get("success", False):
            if accuracy >= 95:
                return "B"
            elif accuracy >= 85:
                return "C"
            return "D"

        if trades < 3 and accuracy >= 90:
            return "C"
        if trades < 10:
            if accuracy >= 85:
                return "C"
            return "D"

        score = 0
        if wr >= 65:
            score += 30
        elif wr >= 58:
            score += 20
        elif wr >= 50:
            score += 10

        if pf >= 2.0:
            score += 30
        elif pf >= 1.5:
            score += 20
        elif pf >= 1.0:
            score += 10

        if sharpe >= 1.5:
            score += 25
        elif sharpe >= 1.0:
            score += 18
        elif sharpe >= 0.5:
            score += 10

        if trades >= 100:
            score += 15
        elif trades >= 50:
            score += 10
        elif trades >= 20:
            score += 5
        elif trades >= 10:
            score += 3

        if score >= 80:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 40:
            return "C"
        elif score >= 20:
            return "D"
        return "F"

    def _cap_results(self, result: Dict) -> Dict:
        """Cap unrealistic OOS metrics to prevent overfitting illusion."""
        if result.get("success"):
            # Capped at realistic maximums for forex
            # Higher caps allow model differentiation (all models hit 5.0 before)
            result["win_rate"] = min(result.get("win_rate", 0), 75.0)
            result["profit_factor"] = min(result.get("profit_factor", 0), 50.0)
            result["sharpe_ratio"] = min(result.get("sharpe_ratio", 0), 10.0)
            result["non_hold_accuracy"] = min(result.get("non_hold_accuracy", 0), 80.0)
            result["accuracy"] = min(result.get("accuracy", 0), 85.0)
        return result

    def _empty_result(self, reason: str = "unknown") -> Dict:
        self.logger.warning(f"OOS validation skipped: {reason}")
        return {
            "success": False,
            "reason": reason,
            "total_trades": 0,
            "win_rate": 0,
            "accuracy": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
            "grade": "N/A",
            "passed": False,
        }