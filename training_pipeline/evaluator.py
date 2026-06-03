from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report
)

from .config import TrainingConfig
from .utils import setup_logger, safe_json_dump


class Evaluator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = setup_logger("evaluator", config.log_dir, config.log_level)
        self.labels = ["SELL", "HOLD", "BUY"]

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        report = {"num_samples": int(len(y_true))}

        report["accuracy"] = float(accuracy_score(y_true, y_pred))

        try:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            report["precision_weighted"] = float(prec)
            report["recall_weighted"] = float(rec)
            report["f1_weighted"] = float(f1)
        except Exception as e:
            self.logger.warning(f"Could not compute metrics: {e}")

        try:
            cm = confusion_matrix(y_true, y_pred)
            report["confusion_matrix"] = cm.tolist()
            report["classification_report"] = classification_report(
                y_true, y_pred,
                target_names=self.labels[:cm.shape[0]],
                output_dict=True,
                zero_division=0,
            )
        except Exception as e:
            self.logger.warning(f"Could not compute confusion matrix: {e}")

        try:
            class_acc = self._per_class_accuracy(y_true, y_pred)
            report["per_class_accuracy"] = class_acc
        except Exception as e:
            self.logger.warning(f"Could not compute per-class accuracy: {e}")

        self.logger.info(
            f"Accuracy: {report['accuracy']:.4f}, "
            f"F1: {report.get('f1_weighted', 'N/A')}"
        )
        return report

    def _per_class_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        classes = sorted(y_true.unique())
        result = {}
        for cls in classes:
            mask = y_true.values == cls
            if mask.sum() > 0:
                acc = (y_pred[mask] == cls).sum() / mask.sum()
                result[f"class_{int(cls)}"] = float(acc)
        return result

    def backtest(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray,
        label_col: str = "label_encoded",
    ) -> Dict:
        self.logger.info("Running backtest simulation...")
        df = df.copy()
        df["prediction"] = y_pred
        horizon = self.config.prediction_horizon

        hold_encoded = self.config.label_encoding["HOLD"]
        buy_encoded = self.config.label_encoding["BUY"]

        trade_mask = df["prediction"].values != hold_encoded
        exit_prices = df["close"].shift(-horizon).values
        valid_exit = ~pd.isna(exit_prices)
        mask = trade_mask & valid_exit

        if not mask.any():
            self.logger.warning("No trades generated in backtest")
            return {"total_trades": 0, "message": "no trades"}

        idx = np.where(mask)[0]
        entry = df["close"].values[idx]
        exit_ = exit_prices[idx]
        preds = df["prediction"].values[idx]

        side = np.where(preds == buy_encoded, 1, -1)
        rets = side * (exit_ - entry) / entry

        trades = []
        for i in range(len(idx)):
            ii = idx[i]
            trades.append({
                "entry_time": str(df.iloc[ii].get("timestamp", "")),
                "exit_time": str(df.iloc[min(ii + horizon, len(df) - 1)].get("timestamp", "")),
                "side": "BUY" if side[i] == 1 else "SELL",
                "entry_price": float(entry[i]),
                "exit_price": float(exit_[i]),
                "return": float(rets[i]),
                "profit": float(rets[i] * 10000),
            })

        returns = rets

        winning = (returns > 0).sum()
        losing = (returns <= 0).sum()
        total = len(trades)
        win_rate = winning / total if total > 0 else 0.0

        gross_profit = returns[returns > 0].sum() if returns[returns > 0].sum() > 0 else 0.0
        gross_loss = abs(returns[returns <= 0].sum()) if returns[returns <= 0].sum() < 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        total_return = float(returns.sum())
        avg_return = float(returns.mean())
        std_return = float(returns.std()) if len(returns) > 1 else 0.0
        sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0

        cumulative = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cumulative)
        dd = (cumulative - peak) / peak
        max_dd = float(dd.min())

        result = {
            "total_trades": total,
            "winning_trades": int(winning),
            "losing_trades": int(losing),
            "win_rate": float(round(win_rate, 4)),
            "profit_factor": float(round(profit_factor, 4)),
            "total_return": float(round(total_return, 6)),
            "total_return_pct": float(round(total_return * 100, 4)),
            "avg_return": float(round(avg_return, 6)),
            "sharpe_ratio": float(round(sharpe, 4)),
            "max_drawdown": float(round(max_dd, 6)),
            "max_drawdown_pct": float(round(max_dd * 100, 4)),
        }

        self.logger.info(
            f"Backtest: {total} trades, "
            f"WR={win_rate:.2%}, "
            f"PF={profit_factor:.2f}, "
            f"Sharpe={sharpe:.2f}"
        )
        return result

    def full_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        df: pd.DataFrame,
    ) -> Dict:
        eval_result = self.evaluate(y_true, y_pred, y_proba)
        bt_result = self.backtest(df, y_pred)
        report = {**eval_result, "backtest": bt_result}
        return report
