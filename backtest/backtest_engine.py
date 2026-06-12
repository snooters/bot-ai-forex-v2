import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from core.config import config
from core.constants import TradeDirection, Timeframe, PositionAction, TrendDirection, RESULTS_DIR
from data.market_data_engine import MarketDataEngine
from data.data_loader import DataLoader
from features.feature_pipeline import FeaturePipeline
from intelligence.trend_analysis import TrendAnalyzer
from intelligence.volatility_analysis import VolatilityAnalyzer
from intelligence.momentum_analysis import MomentumAnalyzer
from intelligence.market_regime import MarketRegimeDetector
from intelligence.market_scorer import MarketScorer
from llm.news_analyzer import NewsAnalyzer
from ml.predictor import MLPredictor
from learning.performance_analyzer import PerformanceAnalyzer
from decision.decision_engine import DecisionEngine
from utils.logger import get_logger


PRIMARY_TF = 5
CONTEXT_TFS = [15, 30, 60, 240]


class BacktestTrade:
    def __init__(self, direction: str, entry: float, volume: float, entry_time: datetime,
                 sl: float, tp: float, confidence: float, atr: float, is_runner: bool = False):
        self.direction = direction
        self.type = direction
        self.entry = entry
        self.volume = volume
        self.entry_time = entry_time
        self.sl = sl
        self.tp = tp
        self.confidence = confidence
        self.atr = atr
        self.is_runner = is_runner
        self.profit = 0.0
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.exit_reason: str = ""
        self.commission = 0.0
        self.net_profit = 0.0


class BacktestEngine:
    def __init__(
        self,
        data_engine: MarketDataEngine,
        ml_predictor: MLPredictor,
        feature_pipeline: FeaturePipeline,
        trend_analyzer: TrendAnalyzer,
        vol_analyzer: VolatilityAnalyzer,
        momentum_analyzer: MomentumAnalyzer,
        regime_detector: MarketRegimeDetector,
        market_scorer: MarketScorer,
        news_analyzer: NewsAnalyzer,
        decision_engine: Optional[DecisionEngine] = None,
        spread_pips: float = 1.0,
        commission_per_lot: float = 0.0,
        from_storage: bool = False,
    ):
        self.logger = get_logger("backtest_engine")
        self.data_engine = data_engine
        self.ml_predictor = ml_predictor
        self.feature_pipeline = feature_pipeline
        self.trend_analyzer = trend_analyzer
        self.vol_analyzer = vol_analyzer
        self.momentum_analyzer = momentum_analyzer
        self.regime_detector = regime_detector
        self.market_scorer = market_scorer
        self.news_analyzer = news_analyzer
        self.decision_engine = decision_engine
        self.performance = PerformanceAnalyzer()
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.from_storage = from_storage
        self.entry_check_interval = max(1, config.risk.get("backtest_entry_interval", 2))
        self.results_dir = Path(RESULTS_DIR) / "backtest"

    def _get_pip_value(self, symbol: str) -> float:
        base = symbol[:3]
        if base in ("JPY", "HUF", "TRY"):
            return 0.01
        return 0.0001

    def _load_multi_tf_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        self.logger.info("Loading multi-TF data...")

        tfs = [5, 15, 30, 60, 240]
        data: Dict[int, pd.DataFrame] = {}

        for tf in tfs:
            try:
                if self.from_storage:
                    df = self.data_engine.storage.load_data(symbol, tf, start_date, end_date)
                else:
                    df = self.data_engine.get_historical_data(symbol, tf)
                    if df.empty:
                        continue
                    df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
            except Exception as e:
                self.logger.warning(f"Failed to load {symbol} tf={tf}: {e}")
                continue

            if df.empty:
                self.logger.warning(f"No data for {symbol} tf={tf}")
                continue
            df = df.sort_values("time").reset_index(drop=True)
            df["_time"] = pd.to_datetime(df["time"])
            label = Timeframe.LABELS.get(tf, str(tf))
            self.logger.info(f"  {label}: {len(df)} candles ({df['time'].min()} -> {df['time'].max()})")
            data[tf] = df

        if 5 not in data:
            self.logger.error("No M5 data available")
            return None

        m5 = data[5].copy()
        for ctx_tf in [15, 30, 60, 240]:
            if ctx_tf not in data:
                self.logger.warning(f"No context data for tf={ctx_tf}, skipping")
                continue
            ctx = data[ctx_tf].copy()
            ctx_cols = {}
            for c in ["open", "high", "low", "close", "volume", "spread"]:
                if c in ctx.columns:
                    ctx_cols[f"{c}_tf{ctx_tf}"] = c
            if not ctx_cols:
                continue
            ctx_idx = ctx[["_time"] + list(ctx_cols.values())].copy()
            ctx_idx.columns = ["_time"] + list(ctx_cols.keys())
            m5 = pd.merge_asof(
                m5.sort_values("_time"),
                ctx_idx.sort_values("_time"),
                on="_time",
                direction="backward",
                suffixes=("", f"_{ctx_tf}_dup"),
            )

        m5.drop(columns=["_time"], inplace=True, errors="ignore")
        m5.ffill(inplace=True)
        return m5

    def _pip_profit(self, entry: float, exit_price: float, direction: str, volume: float, pip_value: float) -> float:
        if direction == TradeDirection.BUY.value:
            return (exit_price - entry) / pip_value * volume * 10
        return (entry - exit_price) / pip_value * volume * 10

    def _compute_commission(self, volume: float, price: float) -> float:
        return volume * price * self.commission_per_lot * 2

    def _check_sl_hit(self, pos: BacktestTrade, high: float, low: float) -> bool:
        if pos.sl == 0:
            return False
        if pos.direction == TradeDirection.BUY.value:
            return low <= pos.sl
        return high >= pos.sl

    def _check_tp_hit(self, pos: BacktestTrade, high: float, low: float) -> bool:
        if pos.tp == 0:
            return False
        if pos.direction == TradeDirection.BUY.value:
            return high >= pos.tp
        return low <= pos.tp

    def run_backtest(
        self,
        symbol: str,
        timeframe: int = Timeframe.M5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_balance: float = 10000.0,
        save_results: bool = True,
    ) -> Dict:
        self.logger.info(f"Running backtest: {symbol} "
                         f"{start_date.date() if start_date else '?'} to "
                         f"{end_date.date() if end_date else '?'}")

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=config.training.get("historical_years", 2) * 365)

        df = self._load_multi_tf_data(symbol, start_date, end_date)
        if df is None or df.empty:
            return {"error": "No data available"}

        self.logger.info("Computing features...")
        df = self.feature_pipeline.compute_all(df)
        if df.empty or len(df) < 50:
            return {"error": "Insufficient data after feature computation"}

        pip_value = self._get_pip_value(symbol)
        warmup = 250

        trades: List[Dict] = []
        positions: List[BacktestTrade] = []
        balance = initial_balance
        equity_curve = [balance]
        total_rows = len(df)
        self.logger.info(f"Simulating {total_rows} candles (entry check every {self.entry_check_interval} bars)...")

        for i in range(warmup, total_rows):
            row = df.iloc[i]

            balance = self._process_exits(positions, row, trades, balance, pip_value)
            self._process_trailing_stops(positions, row)
            balance = self._process_time_exit(positions, row, trades, balance, pip_value)

            if (i - warmup) % self.entry_check_interval == 0 and not positions and i < total_rows - 5:
                self._process_entry(
                    symbol, timeframe, row, positions, balance,
                    pip_value, df, i,
                )

            self._update_unrealized(positions, row, pip_value)
            equity = balance + sum(p.profit for p in positions)
            equity_curve.append(equity)

        for p in list(positions):
            last = df.iloc[-1]
            p.exit_price = last["close"]
            p.exit_time = last["time"]
            p.exit_reason = "end_of_backtest"
            p.profit = self._pip_profit(p.entry, last["close"], p.direction, p.volume, pip_value)
            p.commission = self._compute_commission(p.volume, p.entry) + self._compute_commission(p.volume, last["close"])
            p.net_profit = p.profit - p.commission
            trades.append(self._trade_to_dict(p))
            balance += p.net_profit
        positions.clear()

        results = self.performance.analyze_trades(trades)
        results["equity_curve"] = equity_curve
        results["final_balance"] = balance
        results["total_return"] = (balance - initial_balance) / initial_balance * 100
        results["trades"] = trades
        results["symbol"] = symbol
        results["timeframe"] = timeframe
        results["start_date"] = str(start_date)
        results["end_date"] = str(end_date)
        results["initial_balance"] = initial_balance
        results["spread_pips"] = self.spread_pips
        results["commission_per_lot"] = self.commission_per_lot

        self.logger.info(
            f"Backtest complete: return={results['total_return']:.1f}%, "
            f"trades={results['total_trades']}, win_rate={results['win_rate']:.1f}%"
        )

        if save_results:
            self._save_results(symbol, results)

        return results

    def _process_exits(
        self, positions: List[BacktestTrade], row: pd.Series,
        trades: List[Dict], balance: float, pip_value: float,
    ) -> float:
        for p in list(positions):
            if self._check_sl_hit(p, row["high"], row["low"]):
                p.exit_price = p.sl
                p.exit_time = row["time"]
                p.exit_reason = "stop_loss"
                p.profit = self._pip_profit(p.entry, p.sl, p.direction, p.volume, pip_value)
                p.commission = self._compute_commission(p.volume, p.entry) + self._compute_commission(p.volume, p.sl)
                p.net_profit = p.profit - p.commission
                trades.append(self._trade_to_dict(p))
                balance += p.net_profit
                positions.remove(p)
                continue

            if self._check_tp_hit(p, row["high"], row["low"]):
                p.exit_price = p.tp
                p.exit_time = row["time"]
                p.exit_reason = "take_profit"
                p.profit = self._pip_profit(p.entry, p.tp, p.direction, p.volume, pip_value)
                p.commission = self._compute_commission(p.volume, p.entry) + self._compute_commission(p.volume, p.tp)
                p.net_profit = p.profit - p.commission
                trades.append(self._trade_to_dict(p))
                balance += p.net_profit
                positions.remove(p)
        return balance

    def _process_trailing_stops(self, positions: List[BacktestTrade], row: pd.Series):
        for p in positions:
            if not p.is_runner or p.atr <= 0 or p.profit <= 0:
                continue
            is_buy = p.direction == TradeDirection.BUY.value
            price = row["high"] if is_buy else row["low"]
            trail_dist = p.atr * config.risk.get("trailing_atr_multiplier", 1.5)
            if is_buy:
                new_sl = price - trail_dist
                if new_sl > p.sl:
                    p.sl = new_sl
            else:
                new_sl = price + trail_dist
                if new_sl < p.sl or p.sl == 0:
                    p.sl = new_sl

    def _process_time_exit(
        self, positions: List[BacktestTrade], row: pd.Series,
        trades: List[Dict], balance: float, pip_value: float,
    ) -> float:
        max_hours = config.risk.get("max_hold_hours", 12)
        for p in list(positions):
            elapsed = (row["time"] - p.entry_time).total_seconds() / 3600
            if elapsed > max_hours:
                p.exit_price = row["close"]
                p.exit_time = row["time"]
                p.exit_reason = "time_exit"
                p.profit = self._pip_profit(p.entry, row["close"], p.direction, p.volume, pip_value)
                p.commission = self._compute_commission(p.volume, p.entry) + self._compute_commission(p.volume, row["close"])
                p.net_profit = p.profit - p.commission
                trades.append(self._trade_to_dict(p))
                balance += p.net_profit
                positions.remove(p)
        return balance

    def _process_entry(
        self, symbol: str, timeframe: int, row: pd.Series,
        positions: List[BacktestTrade], balance: float,
        pip_value: float, df: pd.DataFrame, idx: int,
    ):
        import time as _time
        t0 = _time.time()

        lookback = min(500, idx + 1)
        window = df.iloc[idx - lookback + 1:idx + 1] if lookback > 1 else df.iloc[[idx]]

        trend_result = self.trend_analyzer.analyze_trend(window)
        vol_result = self.vol_analyzer.analyze_volatility(window)
        momentum_result = self.momentum_analyzer.analyze_momentum(window)
        regime_result = self.regime_detector.detect_regime(
            trend_result, vol_result, momentum_result, window
        )
        sr_info = self.feature_pipeline.support_resistance.detect_levels(window)
        feature_summary = self.feature_pipeline.compute_features_summary(window)

        multi_tf_trends = {}
        last_feat = window.iloc[-1]
        for col in ["trend240", "trend60", "trend30", "trend15"]:
            if col in window.columns:
                val = last_feat.get(col)
                multi_tf_trends[col] = int(val) if pd.notna(val) else 0

        account_info = {"balance": balance, "equity": balance}

        ml_window = df.iloc[max(0, idx - 4):idx + 1]

        if self.decision_engine:
            decision = self.decision_engine.make_decision(
                symbol=symbol,
                df_entry={timeframe: ml_window},
                trend_result=trend_result,
                vol_result=vol_result,
                momentum_result=momentum_result,
                regime_result=regime_result,
                sr_info=sr_info,
                feature_summary=feature_summary,
                account_info=account_info,
                positions=[],
                spread=self.spread_pips,
                timeframe=timeframe,
                multi_tf_trends=multi_tf_trends,
            )
        else:
            if not self.ml_predictor.is_trained:
                return
            ml_signal = self.ml_predictor.get_buy_sell_hold(ml_window)
            confidence = ml_signal.get("confidence", 0)
            vol_level = vol_result.get("level", "low")
            if confidence >= config.ai_filter["min_confidence"] and vol_level in ("low", "medium"):
                decision = {
                    "action": ml_signal.get("signal", "HOLD"),
                    "no_trade": False,
                    "confidence": confidence,
                }
            else:
                return

        if decision.get("no_trade", True):
            return

        action = decision.get("action", TradeDirection.HOLD.value)
        if action not in (TradeDirection.BUY.value, TradeDirection.SELL.value):
            return

        atr = row.get("atr", 0)
        if atr == 0:
            atr = df.loc[idx, "atr"] if "atr" in df.columns else 0.001
        if atr <= 0:
            return

        entry = row["close"]
        sl_distance = atr * 1.5
        tp1_distance = atr * 1.0
        risk_amount = balance * config.risk["max_risk_pct"]
        sl_pips = sl_distance / pip_value
        volume = risk_amount / (sl_pips * pip_value * 10) if sl_pips > 0 else 0.01
        volume = max(min(round(volume, 2), 1.0), 0.01)
        volume_tp1 = round(volume * 0.5, 2)
        volume_run = round(volume - volume_tp1, 2)

        if action == TradeDirection.BUY.value:
            pos_tp1 = BacktestTrade(direction="BUY", entry=entry, volume=volume_tp1,
                entry_time=row["time"], sl=entry - sl_distance, tp=entry + tp1_distance,
                confidence=decision.get("confidence", 0), atr=atr)
            pos_run = BacktestTrade(direction="BUY", entry=entry, volume=volume_run,
                entry_time=row["time"], sl=entry - sl_distance, tp=0,
                confidence=decision.get("confidence", 0), atr=atr, is_runner=True)
        else:
            pos_tp1 = BacktestTrade(direction="SELL", entry=entry, volume=volume_tp1,
                entry_time=row["time"], sl=entry + sl_distance, tp=entry - tp1_distance,
                confidence=decision.get("confidence", 0), atr=atr)
            pos_run = BacktestTrade(direction="SELL", entry=entry, volume=volume_run,
                entry_time=row["time"], sl=entry + sl_distance, tp=0,
                confidence=decision.get("confidence", 0), atr=atr, is_runner=True)

        positions.append(pos_tp1)
        positions.append(pos_run)

        elapsed = _time.time() - t0
        self.logger.info(
            f"Entry: {action} {symbol} at {entry:.5f} vol={volume:.2f} "
            f"SL={sl_distance:.5f} TP1={tp1_distance:.5f} conf={decision.get('confidence', 0):.1%} "
            f"({elapsed:.2f}s)"
        )

    def _update_unrealized(self, positions: List[BacktestTrade], row: pd.Series, pip_value: float):
        for p in positions:
            p.profit = self._pip_profit(p.entry, row["close"], p.direction, p.volume, pip_value)

    def _trade_to_dict(self, t: BacktestTrade) -> Dict:
        return {
            "direction": t.direction,
            "entry": t.entry,
            "exit_price": t.exit_price,
            "volume": t.volume,
            "entry_time": str(t.entry_time) if t.entry_time else None,
            "exit_time": str(t.exit_time) if t.exit_time else None,
            "exit_reason": t.exit_reason,
            "profit": round(t.profit, 2),
            "net_profit": round(t.net_profit, 2),
            "commission": round(t.commission, 2),
            "confidence": t.confidence,
            "is_runner": t.is_runner,
        }

    def _save_results(self, symbol: str, results: Dict):
        try:
            out_dir = self.results_dir / symbol
            out_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save = {k: v for k, v in results.items() if k not in ("trades", "equity_curve")}
            save["num_trades"] = len(results.get("trades", []))
            save["trades"] = results.get("trades", [])

            filepath = out_dir / f"backtest_{timestamp}.json"
            with open(filepath, "w") as f:
                json.dump(save, f, indent=2, default=str)
            self.logger.info(f"Results saved to {filepath}")

            np.save(str(out_dir / f"equity_{timestamp}.npy"),
                    np.array(results.get("equity_curve", [])))
        except Exception as e:
            self.logger.warning(f"Failed to save results: {e}")
