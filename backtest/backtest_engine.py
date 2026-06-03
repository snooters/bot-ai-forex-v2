from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from core.constants import TradeDirection, Timeframe
from core.config import config
from data.market_data_engine import MarketDataEngine
from features.feature_pipeline import FeaturePipeline
from intelligence.trend_analysis import TrendAnalyzer
from intelligence.volatility_analysis import VolatilityAnalyzer
from intelligence.momentum_analysis import MomentumAnalyzer
from intelligence.market_regime import MarketRegimeDetector
from intelligence.market_scorer import MarketScorer
from llm.news_analyzer import NewsAnalyzer
from ml.trainer import ModelTrainer
from ml.predictor import MLPredictor
from learning.performance_analyzer import PerformanceAnalyzer
from utils.logger import get_logger


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
        self.performance = PerformanceAnalyzer()

    def run_backtest(
        self,
        symbol: str,
        timeframe: int,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 1000.0,
    ) -> Dict:
        self.logger.info(f"Running backtest: {symbol} tf={timeframe} "
                         f"{start_date.date()} to {end_date.date()}")

        df = self.data_engine.get_historical_data(symbol, timeframe)
        df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]

        if df.empty:
            self.logger.warning("No data for backtest period")
            return {"error": "No data"}

        df = self.feature_pipeline.compute_all(df)

        trades = []
        balance = initial_balance
        equity_curve = [balance]
        position = None

        for i in range(200, len(df)):
            current = df.iloc[:i + 1]
            row = df.iloc[i]

            if position is not None:
                if position["direction"] == TradeDirection.BUY.value:
                    profit = (row["close"] - position["entry"]) / 0.0001 * position["volume"] * 10
                else:
                    profit = (position["entry"] - row["close"]) / 0.0001 * position["volume"] * 10

                position["profit"] = profit
                position["exit_price"] = row["close"]
                position["exit_time"] = row["time"]

                if (position["direction"] == TradeDirection.BUY.value and
                    row["low"] <= position["sl"]):
                    profit = (position["sl"] - position["entry"]) / 0.0001 * position["volume"] * 10
                    position["profit"] = profit
                    position["exit_reason"] = "stop_loss"
                    trades.append(position)
                    balance += profit
                    position = None

                elif (position["direction"] == TradeDirection.SELL.value and
                      row["high"] >= position["sl"]):
                    profit = (position["entry"] - position["sl"]) / 0.0001 * position["volume"] * 10
                    position["profit"] = profit
                    position["exit_reason"] = "stop_loss"
                    trades.append(position)
                    balance += profit
                    position = None

                elif (position["direction"] == TradeDirection.BUY.value and
                      row["high"] >= position["tp"]):
                    profit = (position["tp"] - position["entry"]) / 0.0001 * position["volume"] * 10
                    position["profit"] = profit
                    position["exit_reason"] = "take_profit"
                    trades.append(position)
                    balance += profit
                    position = None

                elif (position["direction"] == TradeDirection.SELL.value and
                      row["low"] <= position["tp"]):
                    profit = (position["entry"] - position["tp"]) / 0.0001 * position["volume"] * 10
                    position["profit"] = profit
                    position["exit_reason"] = "take_profit"
                    trades.append(position)
                    balance += profit
                    position = None

            if position is None and i < len(df) - 5:
                trend = self.trend_analyzer.analyze_trend(current)
                vol = self.vol_analyzer.analyze_volatility(current)
                momentum = self.momentum_analyzer.analyze_momentum(current)

                ml_signal = self.ml_predictor.get_buy_sell_hold(current)
                confidence = ml_signal.get("confidence", 0)

                if confidence >= 70 and vol.get("level") in ["low", "medium"]:
                    atr = current["atr"].iloc[-1]
                    entry = row["close"]
                    sl_distance = atr * 1.5
                    tp_distance = atr * 3.0

                    risk_amount = balance * config.risk["max_risk_pct"]
                    pip_val = 0.0001
                    sl_pips = sl_distance / pip_val
                    volume = risk_amount / (sl_pips * pip_val * 10)
                    volume = max(min(round(volume, 2), 1.0), 0.01)

                    if ml_signal.get("signal") == "BUY":
                        position = {
                            "direction": TradeDirection.BUY.value,
                            "entry": entry,
                            "sl": entry - sl_distance,
                            "tp": entry + tp_distance,
                            "volume": volume,
                            "entry_time": row["time"],
                            "confidence": confidence,
                        }
                    elif ml_signal.get("signal") == "SELL":
                        position = {
                            "direction": TradeDirection.SELL.value,
                            "entry": entry,
                            "sl": entry + sl_distance,
                            "tp": entry - tp_distance,
                            "volume": volume,
                            "entry_time": row["time"],
                            "confidence": confidence,
                        }

            equity_curve.append(balance + (position.get("profit", 0) if position else 0))

        if position is not None:
            trades.append(position)

        results = self.performance.analyze_trades(trades)
        results["equity_curve"] = equity_curve
        results["final_balance"] = balance
        results["total_return"] = (balance - initial_balance) / initial_balance * 100
        results["trades"] = trades

        self.logger.info(f"Backtest complete: return={results['total_return']:.1f}%, "
                         f"trades={results['total_trades']}, win_rate={results['win_rate']:.1f}%")
        return results
