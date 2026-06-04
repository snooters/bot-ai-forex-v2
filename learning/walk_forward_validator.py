from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from core.constants import LOOKAHEAD_5, RESULTS_DIR
from features.feature_pipeline import FeaturePipeline
from ml.ensemble import VotingEnsemble
from ml.trainer import ModelTrainer
from utils.helpers import (
    compute_sharpe_ratio, compute_profit_factor,
    compute_max_drawdown,
)
from utils.logger import get_logger


PF_THRESHOLD = 1.2
DD_THRESHOLD = 0.15


class WalkForwardValidator:
    def __init__(self):
        self.logger = get_logger("wf_validator")
        self.feature_pipeline = FeaturePipeline()

    def validate(
        self,
        df: pd.DataFrame,
        ensemble: VotingEnsemble,
        trainer: ModelTrainer,
        timeframe_label: str,
        window_size_months: int = 12,
        step_months: int = 6,
        min_train_years: int = 4,
    ) -> Dict:
        if df.empty or len(df) < 1000:
            return self._empty_result("insufficient data (<1000 rows)")

        df = df.sort_values("time").copy()
        if "time" not in df.columns:
            return self._empty_result("no 'time' column in data")

        df["_date"] = pd.to_datetime(df["time"], utc=True)

        min_date = df["_date"].min()
        max_date = df["_date"].max()
        total_span = (max_date - min_date).days / 365.25
        if total_span < 5:
            return self._empty_result(f"data span too short ({total_span:.1f} years, need >=5)")

        df_feat = self.feature_pipeline.compute_all(df.copy())

        available_cols = self.feature_pipeline.get_feature_columns()
        available_cols = [c for c in available_cols if c in df_feat.columns]

        windows = self._build_windows(df_feat, min_train_years)
        if not windows:
            return self._empty_result("no valid walk-forward windows")

        self.logger.info(f"Built {len(windows)} walk-forward windows for {timeframe_label}")

        all_results = []
        for i, (train_end, val_start, val_end, test_start, test_end) in enumerate(windows):
            self.logger.info(
                f"  Window {i+1}/{len(windows)}: "
                f"train<={train_end.date()} "
                f"val={val_start.date()}..{val_end.date()} "
                f"test={test_start.date()}..{test_end.date()}"
            )

            train_df = df_feat[df_feat["_date"] <= train_end].copy()
            val_df = df_feat[(df_feat["_date"] >= val_start) & (df_feat["_date"] <= val_end)].copy()
            test_df = df_feat[(df_feat["_date"] >= test_start) & (df_feat["_date"] <= test_end)].copy()

            if len(train_df) < 500 or len(val_df) < 100 or len(test_df) < 100:
                self.logger.warning(f"  Window {i+1}: insufficient data, skipping")
                continue

            try:
                X_train, y_train, _, _ = trainer.prepare_training_data(train_df, lookahead=LOOKAHEAD_5)
            except Exception as e:
                self.logger.warning(f"  Window {i+1}: train prep failed — {e}")
                continue

            test_clean = test_df.dropna(subset=available_cols).copy()
            if test_clean.empty or len(test_clean) < 20:
                continue

            y_true, oos_X = self._prepare_labels(test_clean, available_cols)
            if len(oos_X) < 10:
                continue

            try:
                preds = ensemble.predict(oos_X)
            except Exception as e:
                self.logger.warning(f"  Window {i+1}: prediction failed — {e}")
                continue

            val_result = self._validate_split(val_df, ensemble, available_cols)
            test_result = self._run_test(oos_X, y_true, preds, test_clean)

            window_result = {
                "window": i + 1,
                "train_samples": len(X_train),
                "val_accuracy": val_result.get("accuracy", 0),
                **test_result,
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
            }
            window_result["passed"] = self._check_passed(window_result)
            window_result["grade"] = self._compute_grade(window_result)
            all_results.append(window_result)

            self.logger.info(
                f"  Window {i+1}: Acc={test_result.get('accuracy',0):.1f}% "
                f"DirAcc={test_result.get('non_hold_accuracy',0):.1f}% "
                f"Trades={test_result.get('total_trades',0)} "
                f"PF={test_result.get('profit_factor',0):.2f} "
                f"Grade={window_result['grade']}"
            )

        if not all_results:
            return self._empty_result("no valid windows completed")

        return self._aggregate_results(all_results, timeframe_label)

    def _build_windows(
        self, df: pd.DataFrame, min_train_years: int = 4
    ) -> List[Tuple]:
        df_sorted = df.sort_values("_date")
        min_date = df_sorted["_date"].min()
        max_date = df_sorted["_date"].max()

        min_train_months = int(min_train_years * 12)
        total_months = int((max_date - min_date).days / 30.44)

        if total_months < min_train_months + 6:
            return []

        windows = []
        test_months = 3
        val_months = 3

        for train_end_offset in range(min_train_months, total_months - test_months, val_months + test_months):
            train_end = min_date + pd.DateOffset(months=train_end_offset)
            val_start = train_end + pd.Timedelta(hours=1)
            val_end = val_start + pd.DateOffset(months=val_months)
            test_start = val_end + pd.Timedelta(hours=1)
            test_end = test_start + pd.DateOffset(months=test_months)

            if test_end > max_date:
                break

            has_train = (df_sorted["_date"] <= train_end).sum() >= 500
            has_val = ((df_sorted["_date"] >= val_start) & (df_sorted["_date"] <= val_end)).sum() >= 50
            has_test = ((df_sorted["_date"] >= test_start) & (df_sorted["_date"] <= test_end)).sum() >= 50

            if has_train and has_val and has_test:
                windows.append((train_end, val_start, val_end, test_start, test_end))

        if not windows:
            train_end = min_date + pd.DateOffset(months=min_train_months)
            val_start = train_end + pd.Timedelta(hours=1)
            val_end = val_start + pd.DateOffset(months=6)
            test_start = val_end + pd.Timedelta(hours=1)
            test_end = test_start + pd.DateOffset(months=6)
            if test_end <= max_date:
                windows.append((train_end, val_start, val_end, test_start, test_end))

        return windows

    def _validate_split(
        self, val_df: pd.DataFrame, ensemble: VotingEnsemble, available_cols: List[str]
    ) -> Dict:
        val_clean = val_df.dropna(subset=available_cols).copy()
        if val_clean.empty or len(val_clean) < 10:
            return {"accuracy": 0}

        y_true_val, X_val = self._prepare_labels(val_clean, available_cols)
        if len(X_val) < 10:
            return {"accuracy": 0}

        try:
            preds = ensemble.predict(X_val)
            correct = (preds == y_true_val).sum()
            accuracy = float(correct / len(y_true_val)) if len(y_true_val) > 0 else 0
            return {"accuracy": accuracy, "samples": len(X_val)}
        except Exception:
            return {"accuracy": 0}

    def _prepare_labels(self, df: pd.DataFrame, available_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        future_close = df["close"].shift(-LOOKAHEAD_5)
        current_close = df["close"]
        future_return = (future_close - current_close) / current_close

        y_true = np.zeros(len(df), dtype=int)
        y_true[future_return > 0.001] = 0
        y_true[future_return < -0.001] = 1
        y_true[(future_return >= -0.001) & (future_return <= 0.001)] = 2

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

        for i in range(total_signals):
            pred = int(preds[i])
            actual = int(y_true[i])
            pred_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
            actual_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
            predicted_dir = pred_map.get(pred, "HOLD")
            actual_dir = actual_map.get(actual, "HOLD")
            correct_dir = predicted_dir == actual_dir

            if predicted_dir == "HOLD":
                continue

            directional_preds += 1
            row_idx = len(df) - len(y_true) + i
            if 0 <= row_idx < len(df) and row_idx + LOOKAHEAD_5 < len(df):
                entry_price = float(df.iloc[row_idx]["close"])
                future_price = float(df.iloc[row_idx + LOOKAHEAD_5]["close"])

                if predicted_dir == "BUY":
                    profit_pips = (future_price - entry_price) / 0.0001
                else:
                    profit_pips = (entry_price - future_price) / 0.0001

                profit = profit_pips * 0.10

                trade_signals.append({
                    "predicted": predicted_dir,
                    "actual": actual_dir,
                    "win": correct_dir,
                    "profit": profit,
                    "profit_pips": profit_pips,
                    "entry_price": entry_price,
                    "exit_price": future_price,
                })

        total_trades = len(trade_signals)
        non_hold_correct = sum(1 for t in trade_signals if t["win"])
        non_hold_accuracy = (non_hold_correct / total_trades * 100) if total_trades > 0 else 0

        if total_trades == 0:
            return {
                "success": False,
                "reason": "no directional predictions (all HOLD)",
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

    def _aggregate_results(self, results: List[Dict], timeframe_label: str) -> Dict:
        successful = [r for r in results if r.get("success", False)]
        total_windows = len(results)
        success_rate = len(successful) / total_windows * 100 if total_windows > 0 else 0

        agg = {
            "success": success_rate >= 50,
            "total_windows": total_windows,
            "successful_windows": len(successful),
            "success_rate": round(success_rate, 1),
            "timeframe": timeframe_label,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "windows": results,
        }

        if successful:
            for metric in [
                "accuracy", "non_hold_accuracy", "win_rate",
                "profit_factor", "sharpe_ratio", "max_drawdown_pct",
                "net_profit", "total_trades", "directional_predictions",
                "val_accuracy", "avg_win", "avg_loss", "avg_win_loss_ratio",
            ]:
                vals = [r.get(metric, 0) for r in successful]
                agg[f"avg_{metric}"] = round(np.mean(vals), 4) if vals else 0
                agg[f"std_{metric}"] = round(np.std(vals), 4) if len(vals) > 1 else 0

            passed_windows = sum(1 for r in successful if r.get("passed", False))
            agg["passed_windows"] = passed_windows
            agg["pass_rate"] = round(passed_windows / len(successful) * 100, 1) if successful else 0

        avg_val_acc = np.mean([r.get("val_accuracy", 0) for r in results]) if results else 0
        oos_acc = np.mean([r.get("accuracy", 0) for r in results if r.get("success", False)]) if successful else 0

        agg["grade"] = self._compute_aggregate_grade(agg)
        agg["passed"] = agg.get("pass_rate", 0) >= 60

        self.logger.info(
            f"WF {timeframe_label}: "
            f"windows={total_windows} "
            f"successful={len(successful)} "
            f"avg_val_acc={avg_val_acc:.1f}% "
            f"avg_oos_acc={oos_acc:.1f}% "
            f"pass_rate={agg.get('pass_rate',0):.1f}% "
            f"grade={agg['grade']} "
            f"passed={agg['passed']}"
        )
        return agg

    def _check_passed(self, result: Dict) -> bool:
        pf = result.get("profit_factor", 0)
        max_dd_pct = result.get("max_drawdown_pct", 100)
        total_trades = result.get("total_trades", 0)
        directional = result.get("directional_predictions", 0)
        accuracy = result.get("accuracy", 0)

        if not result.get("success", False):
            if result.get("reason") == "no directional predictions (all HOLD)" and accuracy >= 85:
                return True
            return False

        if total_trades < 3:
            return accuracy >= 85

        if directional < 10:
            return accuracy >= 75

        pf_ok = pf >= PF_THRESHOLD * 0.8
        dd_ok = max_dd_pct <= DD_THRESHOLD * 120
        return pf_ok and dd_ok

    def _compute_grade(self, result: Dict) -> str:
        wr = result.get("win_rate", 0)
        pf = result.get("profit_factor", 0)
        sharpe = result.get("sharpe_ratio", 0)
        trades = result.get("total_trades", 0)
        accuracy = result.get("accuracy", 0)

        if not result.get("success", False):
            if accuracy >= 95:
                return "B"
            elif accuracy >= 85:
                return "C"
            return "D"

        if trades < 3 and accuracy >= 90:
            return "C"
        if trades < 10:
            return "C" if accuracy >= 85 else "D"

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

    def _compute_aggregate_grade(self, agg: Dict) -> str:
        if not agg.get("success", False):
            return "D"
        total = agg.get("total_windows", 0)
        if total < 2:
            return "C"

        pass_rate = agg.get("pass_rate", 0)
        avg_pf = agg.get("avg_profit_factor", 0)
        avg_sharpe = agg.get("avg_sharpe_ratio", 0)
        avg_acc = agg.get("avg_accuracy", 0)

        score = 0
        if pass_rate >= 80:
            score += 30
        elif pass_rate >= 60:
            score += 20
        elif pass_rate >= 40:
            score += 10

        if avg_pf >= 1.5:
            score += 25
        elif avg_pf >= 1.2:
            score += 15
        elif avg_pf >= 1.0:
            score += 5

        if avg_sharpe >= 1.0:
            score += 20
        elif avg_sharpe >= 0.5:
            score += 10

        if avg_acc >= 65:
            score += 15
        elif avg_acc >= 55:
            score += 10
        elif avg_acc >= 45:
            score += 5

        if total >= 8:
            score += 10
        elif total >= 4:
            score += 5

        if score >= 80:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 40:
            return "C"
        elif score >= 20:
            return "D"
        return "F"

    def save_results(self, result: Dict, pair: str, timeframe: int, version: str) -> None:
        result_dir = Path(RESULTS_DIR) / pair / str(timeframe) / version / "walk_forward"
        result_dir.mkdir(parents=True, exist_ok=True)

        path = result_dir / "wf_results.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        self.logger.info(f"Walk-forward results saved to {path}")

    def _empty_result(self, reason: str = "unknown") -> Dict:
        self.logger.warning(f"Walk-forward validation skipped: {reason}")
        return {
            "success": False,
            "reason": reason,
            "total_windows": 0,
            "success_rate": 0,
            "grade": "N/A",
            "passed": False,
        }
