#!/usr/bin/env python3
"""
AI Forex Trading Bot v2 - Production-Ready System
==================================================
SURVIVAL > CONSISTENCY > PROFIT
"""

import asyncio
import argparse
import sys
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from core.config import config
from core.constants import (
    Timeframe, TradeDirection, TelegramEvent, EmergencyLevel,
    HEARTBEAT_INTERVAL_SECONDS
)
from data.market_data_engine import MarketDataEngine
from features.feature_pipeline import FeaturePipeline
from intelligence.trend_analysis import TrendAnalyzer
from intelligence.volatility_analysis import VolatilityAnalyzer
from intelligence.momentum_analysis import MomentumAnalyzer
from intelligence.market_regime import MarketRegimeDetector
from intelligence.timeframe_selector import TimeframeSelector
from intelligence.market_scorer import MarketScorer
from ml.trainer import ModelTrainer
from ml.predictor import MLPredictor
from ml.model_manager import ModelManager
from ml.ensemble import VotingEnsemble
from llm.llm_client import LLMClient
from llm.market_analyst import MarketAnalyst
from llm.news_analyzer import NewsAnalyzer
from decision.decision_engine import DecisionEngine
from risk.risk_manager import RiskManager
from trading.execution_engine import ExecutionEngine
from trading.entry_engine import EntryEngine
from trading.exit_engine import ExitEngine
from trading.position_manager import PositionManager
from learning.trade_logger import TradeLogger
from learning.trade_memory import TradeMemory
from learning.adaptive_memory import AdaptiveMemory
from learning.concept_drift import ConceptDriftDetector
from learning.auto_retrain import AutoRetrainEngine
from learning.performance_analyzer import PerformanceAnalyzer
from learning.oos_validator import OOSValidator
from learning.mistake_analyzer import MistakeAnalyzer
from learning.skill_scorer import SkillScorer
from learning.decision_logger import DecisionLogger
from learning.model_validator import ModelValidator
from learning.weekend_trainer import WeekendTrainer
from telegram.telegram_engine import TelegramEngine
from reports.report_engine import ReportEngine
from dashboard.dashboard import Dashboard
from utils.logger import get_logger
from utils.training_progress import TrainingProgress


class ForexBot:
    def __init__(self):
        self.logger = get_logger("forex_bot")
        self.running = False
        self.paused = False
        self._main_loop_task = None
        self._start_time = datetime.now()

        self.data_engine = MarketDataEngine()
        self.feature_pipeline = FeaturePipeline()
        self.trend_analyzer = TrendAnalyzer()
        self.vol_analyzer = VolatilityAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.timeframe_selector = TimeframeSelector()
        self.market_scorer = MarketScorer()

        self.trade_logger = TradeLogger()
        self.trade_memory = TradeMemory()

        self.model_trainer = ModelTrainer(trade_memory=self.trade_memory)
        self.model_manager = ModelManager()
        self.ml_predictor = None

        self.llm_client = LLMClient()
        self.market_analyst = MarketAnalyst(self.llm_client)
        self.news_analyzer = NewsAnalyzer()

        self.decision_engine = None
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine()
        self.entry_engine = None
        self.exit_engine = None
        self.position_manager = None
        self.adaptive_memory = AdaptiveMemory()
        self.drift_detector = ConceptDriftDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.oos_validator = OOSValidator()
        self.mistake_analyzer = MistakeAnalyzer()
        self.skill_scorer = SkillScorer()
        self.auto_retrain = None
        self.decision_logger = DecisionLogger()
        self.weekend_trainer = WeekendTrainer()
        self.model_validator = ModelValidator()

        self.telegram = TelegramEngine()
        self.report_engine = None

        self.dashboard = Dashboard()
        self._dashboard_refreshed = False
        self._last_dashboard_display = datetime.now()
        self._symbols = config.trading["pairs"]
        self._timeframes = [self._tf_to_minutes(tf) for tf in config.trading["timeframes"]]

        self._last_analysis: Dict = {}
        self._account_info: Dict = {}
        self._last_heartbeat_time = 0.0
        self._last_emergency_check: Dict = {}

    def _tf_to_minutes(self, tf: str) -> int:
        mapping = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
        return mapping.get(tf.upper(), 15)

    async def initialize(self):
        self.logger.info("=" * 60)
        self.logger.info("AI FOREX TRADING BOT v2 - INITIALIZING")
        self.logger.info("=" * 60)
        self.logger.info(f"Mode: {config.account['trading_mode']}")
        self.logger.info(f"Symbols: {self._symbols}")
        self.logger.info(f"Risk per trade: {config.risk['max_risk_pct']*100:.2f}%")
        self.logger.info(f"Emergency DD limits: caution={config.emergency['caution_dd']*100:.0f}% "
                         f"danger={config.emergency['danger_dd']*100:.0f}% "
                         f"critical={config.emergency['critical_dd']*100:.0f}%")
        self.logger.info("=" * 60)

        self.logger.info("Initializing Market Data Engine...")
        self.data_engine.initialize()

        self.logger.info("Initializing Account & Risk Manager...")
        account_info = self.data_engine.get_account_info()
        if account_info:
            self.risk_manager.initialize(account_info.get("balance", config.account["balance"]))
            self._account_info = account_info
        else:
            self.risk_manager.initialize(config.account["balance"])
            self._account_info = {
                "balance": config.account["balance"],
                "equity": config.account["balance"],
                "margin": 0,
                "margin_free": config.account["balance"],
                "margin_level": 0,
                "profit": 0,
                "leverage": config.account["leverage"],
            }

        try:
            synced = self.trade_logger.sync_from_mt5(self.data_engine.connector)
            if synced > 0:
                self.logger.info(f"Trade history synced: +{synced} trades from MT5")
        except Exception as e:
            self.logger.warning(f"MT5 trade history sync skipped: {e}")

        self.logger.info("Initializing ML Engine...")
        all_ensembles = {}
        model_version: Optional[str] = None
        trained_tfs = self.model_manager.get_trained_timeframes()
        if trained_tfs:
            model_version = self.model_manager.get_latest_version(trained_tfs[0])
            for tf in trained_tfs:
                try:
                    ensemble = self.model_manager.load_latest_for_timeframe(tf)
                    all_ensembles[tf] = ensemble
                    self.logger.info(f"Loaded model for {Timeframe.LABELS.get(tf, tf)}")
                except Exception as e:
                    self.logger.warning(f"Failed to load model for {Timeframe.LABELS.get(tf, tf)}: {e}")
        else:
            model_version = self.model_manager.get_latest_version()
            if model_version:
                self.logger.info(f"Loading legacy model: {model_version}")
                ensemble = self.model_manager.load_ensemble(model_version)
                all_ensembles[Timeframe.M15] = ensemble

        if all_ensembles:
            self.ml_predictor = MLPredictor(all_ensembles)
        else:
            self.logger.info("No existing model found. Will train on first run.")
            ensemble = self.model_trainer.get_ensemble()
            self.ml_predictor = MLPredictor(ensemble)

        config_tfs = set()
        for tf_str in config.trading["timeframes"]:
            tf_val = getattr(Timeframe, tf_str.upper(), None)
            if tf_val:
                config_tfs.add(tf_val)
        missing_tfs = sorted(config_tfs - set(self.ml_predictor.available_timeframes or []))
        if missing_tfs and self.data_engine.connector._mt5_available:
            self.logger.info(f"Training missing timeframes: {[Timeframe.LABELS.get(tf, str(tf)) for tf in missing_tfs]}")
            symbol = self._symbols[0]
            for tf in missing_tfs:
                try:
                    tf_label = Timeframe.LABELS.get(tf, str(tf))
                    self.logger.info(f"  {tf_label}: downloading data...")
                    self.model_trainer.ensemble = VotingEnsemble()
                    df = self.data_engine.get_historical_data(symbol, tf, years=config.training["historical_years"])
                    min_rows = 300 if tf <= Timeframe.M15 else 500
                    if df.empty or len(df) <= min_rows:
                        self.logger.warning(f"  {tf_label}: skipped — only {len(df)} rows, need {min_rows}")
                        continue
                    self.logger.info(f"  {tf_label}: {len(df)} rows — preparing features...")
                    try:
                        X, y, features, df_clean = self.model_trainer.prepare_training_data(df)
                    except Exception as e:
                        self.logger.warning(f"  {tf_label}: feature prep failed — {e}")
                        continue
                    recency = ModelTrainer.compute_recency_weights(df_clean["time"]) if "time" in df_clean.columns else None
                    self.logger.info(f"  {tf_label}: {len(X)} samples — training models...")
                    results = self.model_trainer.train_all_models(X, y, feature_cols=features, recency_weights=recency)
                    if self.model_trainer.get_ensemble().get_num_models() > 0:
                        version = self.model_manager.save_ensemble(
                            self.model_trainer.get_ensemble(), timeframe=tf
                        )
                        ensemble = self.model_manager.load_ensemble(version)
                        all_ensembles[tf] = ensemble
                        xgb_acc = results.get("models", {}).get("xgboost", {}).get("train_accuracy", 0)
                        xgb_val = results.get("models", {}).get("xgboost", {}).get("val_accuracy", 0)
                        rf_acc = results.get("models", {}).get("random_forest", {}).get("train_accuracy", 0)
                        rf_val = results.get("models", {}).get("random_forest", {}).get("val_accuracy", 0)
                        self.model_manager.save_performance(version, {
                            "accuracy": {
                                "xgboost": xgb_acc,
                                "xgboost_val": xgb_val,
                                "random_forest": rf_acc,
                                "random_forest_val": rf_val,
                            },
                            "samples": len(X),
                        })
                        self.logger.info(
                            f"  {tf_label}: DONE — XGB:{xgb_acc:.1%}(val:{xgb_val:.1%}) RF:{rf_acc:.1%}(val:{rf_val:.1%}) samples={len(X)} | v{version}"
                        )
                except Exception as e:
                    self.logger.warning(f"  {tf_label}: FAILED — {e}")
            if all_ensembles:
                self.ml_predictor = MLPredictor(all_ensembles)
                skill = self.model_manager.get_skill_level()
                retrains = self.model_manager.get_total_retrains()
                self.logger.info(f"All models ready. Retrains: {retrains} | Skill: {skill}")

        self.decision_engine = DecisionEngine(
            ml_predictor=self.ml_predictor,
            market_scorer=self.market_scorer,
            trade_memory=self.trade_memory,
        )
        self.exit_engine = ExitEngine(
            execution_engine=self.execution_engine,
            data_engine=self.data_engine,
        )
        self.position_manager = PositionManager(
            execution_engine=self.execution_engine,
            exit_engine=self.exit_engine,
            data_engine=self.data_engine,
        )
        self.entry_engine = EntryEngine(
            risk_manager=self.risk_manager,
            execution_engine=self.execution_engine,
            data_engine=self.data_engine,
        )
        self.auto_retrain = AutoRetrainEngine(
            trade_logger=self.trade_logger,
            model_trainer=self.model_trainer,
            model_manager=self.model_manager,
            trade_memory=self.trade_memory,
        )
        self.report_engine = ReportEngine(self.trade_logger)

        self.logger.info("Initializing Telegram Engine...")
        await self.telegram.initialize()

        if self.telegram._enabled:
            await self.telegram.send_event(TelegramEvent.BOT_STARTED, {
                "balance": self._account_info.get("balance", 0),
            })

        self.dashboard.update({
            "status": "initialized",
            "symbol": self._symbols[0] if self._symbols else "",
            "balance": self._account_info.get("balance", 0),
            "equity": self._account_info.get("equity", 0),
            "margin": self._account_info.get("margin", 0),
            "free_margin": self._account_info.get("margin_free", 0),
            "margin_level": self._account_info.get("margin_level", 0),
            "floating_profit": self._account_info.get("profit", 0),
            "model_version": model_version or "none",
            "retrain_count": self.model_manager.get_total_retrains(),
            "skill_level": self.model_manager.get_skill_level(),
            "models_summary": self.model_manager.get_models_summary(),
        })

        self.logger.info("System initialization complete")
        return True

    async def run(self):
        self.running = True
        self._main_loop_task = asyncio.create_task(self._main_loop())
        if config.account.get("learn_only"):
            self.logger.info("LEARN ONLY mode: AI will train without trading")
        self.logger.info("Bot started. Press Ctrl+C to stop.")

        try:
            await self._main_loop_task
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def _main_loop(self):
        if not self.ml_predictor.is_trained:
            self.logger.info("No trained model. Initiating initial training...")
            await self._initial_training()

        self.logger.info("Entering main trading loop...")
        last_heartbeat = 0.0
        last_report_check = datetime.now()
        last_data_refresh = datetime.now()

        while self.running:
            try:
                now = datetime.now()
                current_ts = now.timestamp()

                if self.paused:
                    await asyncio.sleep(1)
                    continue

                for symbol in self._symbols:
                    await self._process_symbol(symbol)

                self._check_emergency()

                if current_ts - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                    await self._heartbeat()
                    last_heartbeat = current_ts

                if (now - last_report_check).total_seconds() >= 3600:
                    await self._check_reports()
                    last_report_check = now

                self._update_dashboard()
                if not self._dashboard_refreshed and self._last_analysis:
                    self._dashboard_refreshed = True
                    self._last_dashboard_display = now
                    self.dashboard.display()
                elif (now - self._last_dashboard_display).total_seconds() >= 60:
                    self._last_dashboard_display = now
                    self.dashboard.display()

                if (now - last_data_refresh).total_seconds() >= 1800:
                    self.logger.info("Periodic data refresh: downloading latest candles...")
                    for symbol in self._symbols:
                        for tf in self._timeframes:
                            self.data_engine.refresh_stored_data(symbol, tf, count=2000)
                    last_data_refresh = now
                    self.logger.info("Periodic data refresh complete")

                retrain_needed, reason = self.auto_retrain.check_retrain_needed()
                if retrain_needed:
                    self.logger.info(f"Retrain triggered: {reason}")
                    await self._perform_retrain()

                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Main loop error: {e}", exc_info=True)
                await self.telegram.send_event(TelegramEvent.SYSTEM_ERROR, {
                    "error": str(e),
                    "module": "main_loop",
                    "action": "Auto-recovery - continuing in 30s",
                })
                await asyncio.sleep(30)

    def _align_multi_tf(self, symbol: str, count_m5: int = 500, count_context: int = 200):
        """Fetch M5 + context TFs and align to M5 timestamps."""
        import pandas as pd
        import numpy as np

        tfs = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
        data = {}
        for label, tf in tfs.items():
            cnt = count_m5 if tf == 5 else count_context
            df = self.data_engine.get_rates(symbol, tf, count=cnt)
            if df.empty:
                self.logger.warning(f"  MTF align: no data for {symbol} {label}")
                return None
            data[tf] = df.copy()

        m5 = data[5].sort_values("time").reset_index(drop=True)
        m5["_time"] = pd.to_datetime(m5["time"])

        for ctx_tf in [15, 30, 60, 240]:
            ctx = data[ctx_tf].sort_values("time").reset_index(drop=True)
            ctx["_time"] = pd.to_datetime(ctx["time"])
            # forward-fill higher TF values into M5
            ctx_cols = {}
            for c in ["open", "high", "low", "close", "volume", "spread"]:
                if c in ctx.columns:
                    ctx_cols[f"{c}_tf{ctx_tf}"] = c
            if not ctx_cols:
                continue
            ctx_idx = ctx[["_time"] + list(ctx_cols.values())].copy()
            ctx_idx.columns = ["_time"] + list(ctx_cols.keys())
            m5 = pd.merge_asof(m5.sort_values("_time"), ctx_idx.sort_values("_time"),
                               on="_time", direction="backward", suffixes=("", f"_{ctx_tf}_dup"))

        m5.drop(columns=["_time"], inplace=True)
        m5.ffill(inplace=True)
        return m5

    async def _process_symbol(self, symbol: str):
        try:
            perf_data = None
            try:
                trades = self.trade_logger.get_closed_trades()
                perf = self.performance_analyzer.analyze_trades(trades)
                if perf.get("by_timeframe"):
                    perf_data = perf["by_timeframe"]
            except Exception:
                pass

            selected_tfs = self.timeframe_selector.select_timeframes(
                {tf: self.data_engine.get_rates(symbol, tf, count=50) for tf in self._timeframes},
                performance_data=perf_data,
            )

            # ── Multi-TF: M5 entry + context from M15/M30/H1/H4 ──
            entry_tf = Timeframe.M5
            aligned = self._align_multi_tf(symbol, count_m5=500, count_context=200)
            if aligned is None or aligned.empty:
                return

            df_aligned_feat = self.feature_pipeline.compute_all(aligned)
            if df_aligned_feat.empty or len(df_aligned_feat) < 50:
                return

            trend_result = self.trend_analyzer.analyze_trend(df_aligned_feat)
            self.logger.debug(
                f"Trend [{symbol}] dir={trend_result.get('direction','?')} "
                f"score={trend_result.get('score',0):+.1f} "
                f"strength={trend_result.get('strength',0):.2f} "
                f"ema={trend_result.get('ema_alignment',{}).get('direction',0)} "
                f"slope={trend_result.get('slope_score',{}).get('direction',0)} "
                f"pos={trend_result.get('price_position',{}).get('direction',0):+.2f} "
                f"adx={trend_result.get('adx_score',{}).get('direction',0)} "
                f"div={trend_result.get('divergence',{}).get('direction',0)}"
            )

            vol_result = self.vol_analyzer.analyze_volatility(df_aligned_feat)
            momentum_result = self.momentum_analyzer.analyze_momentum(df_aligned_feat)
            regime_result = self.regime_detector.detect_regime(
                trend_result, vol_result, momentum_result, df_aligned_feat
            )

            sr = self.feature_pipeline.support_resistance.detect_levels(df_aligned_feat)
            feature_summary = self.feature_pipeline.compute_features_summary(df_aligned_feat)

            # Extract multi-TF trends from the last row of aligned data
            last_row = df_aligned_feat.iloc[-1]
            multi_tf_trends = {}
            for col in ["trend240", "trend60", "trend30", "trend15"]:
                if col in df_aligned_feat.columns:
                    val = last_row.get(col)
                    multi_tf_trends[col] = int(val) if pd.notna(val) else 0

            current_price = self.data_engine.get_current_price(symbol)
            price = current_price.get("bid", 0) if current_price else 0
            spread = self.data_engine.get_current_spread(symbol) or 0

            self._account_info = self.data_engine.get_account_info() or self._account_info
            positions = self.position_manager.get_open_positions(symbol)

            news = await self.news_analyzer.analyze_news(symbol)
            llm_analysis = await self.market_analyst.analyze_market(
                symbol, {
                    "trend": trend_result.get("direction", ""),
                    "regime": regime_result.get("regime", ""),
                    "volatility": vol_result.get("level", ""),
                    "price": price,
                },
                feature_summary.get("indicators", {})
            )

            multi_tf_trends_log = {k: multi_tf_trends.get(k, 0) for k in ["trend240", "trend60", "trend30", "trend15"]}
            self.logger.debug(f"MTF [{symbol}] trends={multi_tf_trends_log}")

            decision = self.decision_engine.make_decision(
                symbol=symbol,
                df_entry={entry_tf: df_aligned_feat},
                trend_result=trend_result,
                vol_result=vol_result,
                momentum_result=momentum_result,
                regime_result=regime_result,
                sr_info=sr,
                feature_summary=feature_summary,
                account_info=self._account_info,
                positions=positions,
                news_analysis=news,
                llm_analysis=llm_analysis,
                spread=spread,
                timeframe=entry_tf,
                multi_tf_trends=multi_tf_trends,
            )

            self.decision_logger.log_decision(symbol, decision)

            self._last_analysis[symbol] = {
                "trend": trend_result,
                "volatility": vol_result,
                "momentum": momentum_result,
                "regime": regime_result,
                "decision": decision,
                "timeframe": selected_tfs,
                "feature_summary": feature_summary,
                "sr": sr,
                "multi_tf_trends": multi_tf_trends,
            }

            ml_sig = decision.get("ml_signal", {})
            self.logger.debug(
                f"ML [{symbol}] sig={ml_sig.get('signal','?')} "
                f"buy={ml_sig.get('buy_prob',0):.0%} sell={ml_sig.get('sell_prob',0):.0%} "
                f"hold={ml_sig.get('hold_prob',0):.0%} conf={ml_sig.get('confidence',0):.0%} | "
                f"Final: {decision.get('action','?')} "
                f"(score={decision.get('market_score',0)} "
                f"conf={decision.get('confidence',0):.0%})"
            )

            is_trade_action = decision["action"] in (
                TradeDirection.BUY.value, TradeDirection.SELL.value,
                "WEAK_BUY", "WEAK_SELL"
            )

            if config.account.get("learn_only"):
                if not decision["no_trade"] and is_trade_action:
                    self.logger.info(f"LEARN ONLY: Would open {decision['action']} {symbol} (conf={decision.get('confidence',0):.0%}, score={decision.get('market_score',0)})")
                if positions:
                    pass
            else:
                if not decision["no_trade"] and is_trade_action:
                    if not positions:
                        atr = (df_entry_feat["atr"].iloc[-1] if "atr" in df_entry_feat.columns
                               and not df_entry_feat["atr"].empty else 0.001)

                        trade_result = self.entry_engine.open_trade(
                            symbol=symbol,
                            decision=decision,
                            account_info=self._account_info,
                            df_entry=df_entry_feat,
                            atr=atr,
                            current_price=price,
                            existing_positions=positions,
                        )

                        if trade_result:
                            self.trade_logger.log_trade_open(trade_result)
                            await self.telegram.send_event(TelegramEvent.OPEN_POSITION, {
                                **trade_result,
                                "confidence": decision.get("confidence", 0),
                                "balance": self._account_info.get("balance", 0),
                            })

            if not config.account.get("learn_only") and positions:
                actions = self.position_manager.manage_positions(
                    symbol=symbol,
                    trend_result=trend_result,
                    regime_result=regime_result,
                    confidence=decision.get("confidence", 0),
                    market_structure=self.feature_pipeline.market_structure.get_last_hh_ll(df_trend_feat),
                )

                for action in actions:
                    if action.get("action") in ["FULL_CLOSE", "PARTIAL_CLOSE"]:
                        closed_trade = self.trade_logger.log_trade_close(
                            ticket=action["ticket"],
                            exit_price=price,
                            exit_reason=action["action"],
                        )
                        if closed_trade:
                            self.trade_memory.record_from_trade_log(closed_trade)
                            await self.telegram.send_event(TelegramEvent.POSITION_CLOSED, {
                                "symbol": closed_trade.get("symbol", symbol),
                                "direction": closed_trade.get("direction", ""),
                                "entry_price": closed_trade.get("entry_price", 0),
                                "exit_price": price,
                                "profit": closed_trade.get("profit", 0),
                                "pips": closed_trade.get("profit_pips", 0),
                                "reason": action["action"],
                            })
                            if closed_trade.get("profit", 0) < 0:
                                await self.telegram.send_event(TelegramEvent.LOSS_ALERT, {
                                    "symbol": symbol,
                                    "loss": closed_trade.get("profit", 0),
                                    "balance": self._account_info.get("balance", 0),
                                    "drawdown": self._account_info.get("drawdown", 0),
                                })

        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def _check_emergency(self):
        emergency = self.risk_manager.check_emergency(self._account_info)
        if emergency:
            level = emergency["level"]
            prev_level = self._last_emergency_check.get("level", EmergencyLevel.NORMAL.value)

            if level != prev_level:
                self.logger.warning(
                    f"EMERGENCY LEVEL: {level} | "
                    f"DD: {emergency['drawdown']:.1f}% | "
                    f"Action: {emergency['action']}"
                )
                asyncio.create_task(
                    self.telegram.send_event(TelegramEvent.EMERGENCY_ALERT, emergency)
                )

                if level == EmergencyLevel.CRITICAL.value and config.emergency["auto_close_positions"]:
                    self.logger.warning("CRITICAL: Closing all positions due to emergency")
                    self.position_manager.close_all()

            self._last_emergency_check = emergency

    async def _check_reports(self):
        daily = self.report_engine.check_daily_report()
        if daily:
            await self.telegram.send_event(TelegramEvent.DAILY_REPORT, daily)

        weekly = self.report_engine.check_weekly_report()
        if weekly:
            await self.telegram.send_event(TelegramEvent.WEEKLY_REPORT, weekly)

        monthly = self.report_engine.check_monthly_report()
        if monthly:
            await self.telegram.send_event(TelegramEvent.MONTHLY_REPORT, monthly)

    async def _heartbeat(self):
        model_ver = self.model_manager.get_latest_version() or "none"
        positions = self.position_manager.get_open_positions()
        trades = self.trade_logger.get_trade_count()
        perf = self.performance_analyzer.analyze_trades(
            self.trade_logger.get_closed_trades(),
            start_balance=self._account_info.get("balance", 0),
        )
        acct_status = self.risk_manager.account_monitor.get_account_status(self._account_info)

        uptime = datetime.now() - self._start_time
        uptime_str = str(uptime).split(".")[0]

        mm = self.model_manager
        retrain_count = mm.get_total_retrains()
        skill = mm.get_skill_level()
        skill_score = mm.get_skill_score()
        summary = mm.get_models_summary()
        model_detail = " | ".join(
            f"{n}:{m['retrains']}r/{m['skill'][:4]}/{m.get('oos',{}).get('grade','?')}"
            for n, m in summary.items() if not n.startswith("_")
        )

        drift_summary = self.drift_detector.get_drift_summary()
        drift_detected = drift_summary.get("drift_detected", False)
        drift_score = drift_summary.get("last_drift", {}).get("score", 0) if drift_summary.get("last_drift") else 0

        pair_skills = self.skill_scorer.get_pair_skills()
        best_pair = self.skill_scorer.get_best_pair(pair_skills) or "N/A"
        worst_pair = self.skill_scorer.get_worst_pair(pair_skills) or "N/A"

        self.logger.info(
            f"HEARTBEAT | Retrains: {retrain_count} | "
            f"Skill: {skill}({skill_score}) | "
            f"Drift: {'YES' if drift_detected else 'no'} | "
            f"[{model_detail}] | "
            f"Best: {best_pair} | "
            f"Mode: {'LEARN' if config.account.get('learn_only') else 'LIVE'} | "
            f"Positions: {len(positions)} | "
            f"Trades: {trades} | "
            f"Balance: ${self._account_info.get('balance', 0):.2f} | "
            f"DD: {perf.get('max_drawdown', 0):.1f}%"
        )

        retrain_needed, retrain_reason = self.auto_retrain.check_retrain_needed() if self.auto_retrain else (False, "")

        regime = "N/A"
        for sym, analysis in self._last_analysis.items():
            regime = analysis.get("regime", {}).get("regime", "N/A")
            break

        await self.telegram.send_event(TelegramEvent.HEARTBEAT, {
            "status": "LEARN-ONLY" if config.account.get('learn_only') else "RUNNING" if not self.paused else "PAUSED",
            "uptime": uptime_str,
            "positions": len(positions),
            "balance": self._account_info.get("balance", 0),
            "equity": self._account_info.get("equity", 0),
            "model_version": model_ver,
            "total_trades": trades,
            "win_rate": perf.get("win_rate", 0),
            "drawdown": acct_status.get("current_drawdown", 0),
            "regime": regime,
            "retrain_count": self.model_manager.get_total_retrains(),
            "skill_level": self.model_manager.get_skill_level(),
            "skill_score": self.model_manager.get_skill_score(),
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "retrain_needed": retrain_needed,
            "best_pair": best_pair,
            "worst_pair": worst_pair,
        })

    async def _initial_training(self):
        self.logger.info("Starting initial model training...")
        symbol = self._symbols[0]

        ensembles = {}

        tf_list = [Timeframe.M5, Timeframe.M15, Timeframe.M30]
        synthetic = not self.data_engine.connector._mt5_available

        _train_start = time.monotonic()
        _train_times = []
        _train_results = {}
        for idx, tf in enumerate(tf_list):
            tf_label = Timeframe.LABELS.get(tf, tf)
            try:
                self.model_trainer.ensemble = VotingEnsemble()
                _step_start = time.monotonic()
                self.logger.info(f"TRAINING [{idx+1}/{len(tf_list)}] {tf_label} — downloading data...")
                df = (
                    self.data_engine.connector._simulate_rates(symbol, tf, 1000)
                    if synthetic
                    else self.data_engine.get_historical_data(symbol, tf, years=config.training["historical_years"])
                )
                min_rows = 300 if tf <= Timeframe.M15 else 500
                if df.empty or len(df) <= min_rows:
                    self.logger.warning(f"TRAINING [{idx+1}/{len(tf_list)}] {tf_label} — skipped (insufficient data)")
                    continue

                self.logger.info(f"TRAINING [{idx+1}/{len(tf_list)}] {tf_label} — preparing features ({len(df)} rows)...")
                X, y, features, df_clean = self.model_trainer.prepare_training_data(df)
                recency = ModelTrainer.compute_recency_weights(df_clean["time"]) if "time" in df_clean.columns else None
                self.logger.info(f"TRAINING [{idx+1}/{len(tf_list)}] {tf_label} — training models ({len(X)} samples)...")
                results = self.model_trainer.train_all_models(X, y, feature_cols=features, recency_weights=recency)
                if self.model_trainer.get_ensemble().get_num_models() > 0:
                    version = self.model_manager.save_ensemble(
                        self.model_trainer.get_ensemble(), timeframe=tf
                    )
                    ensembles[tf] = self.model_manager.load_ensemble(version)
                    _elapsed = time.monotonic() - _step_start
                    _train_times.append(_elapsed)
                    _avg = sum(_train_times) / len(_train_times)
                    _remaining = _avg * (len(tf_list) - idx - 1)

                    xgb_acc = results.get("models", {}).get("xgboost", {}).get("train_accuracy", 0)
                    xgb_val = results.get("models", {}).get("xgboost", {}).get("val_accuracy", 0)
                    rf_acc = results.get("models", {}).get("random_forest", {}).get("train_accuracy", 0)
                    rf_val = results.get("models", {}).get("random_forest", {}).get("val_accuracy", 0)
                    oos_data = {}
                    _train_results[tf_label] = {"xgb": xgb_acc, "xgb_val": xgb_val, "rf": rf_acc, "rf_val": rf_val, "samples": len(X), "oos": oos_data}
                    self.logger.info(
                        f"TRAINING [{idx+1}/{len(tf_list)}] {tf_label} — DONE in {_elapsed:.1f}s | "
                        f"XGB:{xgb_acc:.1%}(val:{xgb_val:.1%}) RF:{rf_acc:.1%}(val:{rf_val:.1%}) | "
                        f"ETA: {_remaining:.0f}s ({_remaining/60:.1f}m)"
                    )

                    if not synthetic:
                        perf_data = {
                            "initial_training": True,
                            "accuracy": {
                                "xgboost": xgb_acc,
                                "xgboost_val": xgb_val,
                                "random_forest": rf_acc,
                                "random_forest_val": rf_val,
                            },
                            "samples": len(X),
                        }
                        self.model_manager.save_performance(version, perf_data)

                        oos_result = self.oos_validator.validate(
                            df=df,
                            ensemble=self.model_trainer.get_ensemble(),
                            trainer=self.model_trainer,
                            timeframe_label=tf_label,
                            oos_split=0.2,
                        )
                        self.model_manager.save_oos_result(version, oos_result)
                        if oos_result.get("success"):
                            self.logger.info(
                                f"  {tf_label} OOS: WR={oos_result['win_rate']:.1f}% "
                                f"PF={oos_result['profit_factor']:.2f} "
                                f"Sharpe={oos_result['sharpe_ratio']:.2f} "
                                f"Grade={oos_result['grade']} "
                                f"Trades={oos_result['total_trades']}"
                            )
                        else:
                            self.logger.info(f"  {tf_label} OOS skipped: {oos_result.get('reason', 'unknown')}")

                        await self.telegram.send_event(TelegramEvent.MODEL_RETRAINED, {
                            "version": version,
                            "timeframe": tf_label,
                            "models": self.model_trainer.get_ensemble().get_num_models(),
                            "samples": len(X),
                            "accuracy": xgb_acc,
                            "oos_grade": oos_result.get("grade", "N/A"),
                            "oos_win_rate": oos_result.get("win_rate", 0),
                        })
            except Exception as e:
                self.logger.warning(f"TRAINING [{idx+1}/{len(tf_list)}] {tf_label} — FAILED: {e}")
                continue

        if ensembles:
            self.ml_predictor = MLPredictor(ensembles)
            self.decision_engine = DecisionEngine(
                ml_predictor=self.ml_predictor,
                market_scorer=self.market_scorer,
                trade_memory=self.trade_memory,
            )
            tfs_done = [Timeframe.LABELS.get(tf, str(tf)) for tf in ensembles]
            _total = time.monotonic() - _train_start
            skill = self.model_manager.get_skill_level()
            retrains = self.model_manager.get_total_retrains()
            summary = self.model_manager.get_models_summary()
            self.logger.info(
                f"TRAINING COMPLETE — {_total:.0f}s ({_total/60:.1f}m) | "
                f"Timeframes: {', '.join(tfs_done)} | "
                f"Retrains: {retrains} | Skill: {skill}"
            )
            for tf_name, acc in _train_results.items():
                oos = _train_results[tf_name].get("oos", {})
                oos_str = f" OOS: WR={oos.get('win_rate',0):.1f}% PF={oos.get('profit_factor',0):.2f} Grade={oos.get('grade','N/A')}" if oos.get("win_rate", 0) > 0 else ""
                self.logger.info(
                    f"  {tf_name}: XGB={acc['xgb']:.1%}(val:{acc['xgb_val']:.1%}) RF={acc['rf']:.1%}(val:{acc['rf_val']:.1%}) samples={acc['samples']}{oos_str}"
                )

            self._update_dashboard()

            skill = self.model_manager.get_skill_level()
            summary = self.model_manager.get_models_summary()
            model_lines = "\n".join(
                f"  {n}: v{m['version']} | {m['skill']}"
                for n, m in sorted(summary.items()) if not n.startswith("_")
            )
            await self.telegram.send_event(TelegramEvent.SKILL_UP, {
                "old_skill": "Newborn",
                "new_skill": skill,
                "total_retrains": 0,
                "active_models": len(ensembles),
                "models_detail": model_lines,
            })
        else:
            self.logger.warning("All training attempts failed. Running with untrained model.")

    def _retry_model_params(self, attempt: int) -> tuple:
        if attempt == 1:
            return 2.0, {
                "xgboost": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03},
                "random_forest": {"n_estimators": 300, "max_depth": 10},
                "lightgbm": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03},
            }
        elif attempt == 2:
            return 4.0, {
                "xgboost": {"n_estimators": 400, "max_depth": 10, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.7},
                "random_forest": {"n_estimators": 400, "max_depth": 12, "min_samples_split": 5},
                "lightgbm": {"n_estimators": 400, "max_depth": 10, "learning_rate": 0.02, "subsample": 0.7, "colsample_bytree": 0.7},
            }
        else:
            return 6.0, {
                "xgboost": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "gamma": 0.5, "reg_alpha": 0.5, "reg_lambda": 2.0},
                "random_forest": {"n_estimators": 200, "max_depth": 6, "min_samples_split": 20},
                "lightgbm": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "reg_alpha": 0.5, "reg_lambda": 2.0},
            }

    async def _perform_retrain(self, data_override: Optional[Dict] = None):
        self.logger.info("Starting auto retrain...")
        symbol = self._symbols[0]
        trained_timeframes = self.model_manager.get_trained_timeframes()
        max_retries = 3

        if data_override:
            trained_timeframes = sorted(data_override.keys())
        elif not trained_timeframes:
            trained_timeframes = [self._tf_to_minutes(tf) for tf in config.trading["timeframes"]]

        old_skill = self.model_manager.get_skill_level()
        old_retrains = self.model_manager.get_total_retrains()
        ensembles = {}
        rejected_versions = []
        accepted_list = []

        _train_times = []
        with TrainingProgress() as progress:
            for idx, tf in enumerate(trained_timeframes):
                tf_label = Timeframe.LABELS.get(tf, tf)
                progress.begin_tf(str(tf_label), attempt=1, max_attempts=max_retries)
                _step_start = time.monotonic()
                try:
                    if data_override and tf in data_override:
                        df = data_override[tf]
                        self.logger.info(f"  {tf_label}: using pre-loaded data ({len(df)} candles)")
                    else:
                        df = self.data_engine.get_historical_data(symbol, tf, years=1)
                    if df.empty:
                        self.logger.warning(f"  {tf_label}: no data, skipping")
                        continue
                    self.logger.info(f"  {tf_label}: {len(df)} candles loaded — preparing features...")

                    old_version = self.model_manager.get_latest_version(tf)
                    accepted = False
                    best_oos = None
                    best_version = None
                    attempt_versions = []

                    for attempt in range(1, max_retries + 1):
                        weight_mult, model_params = self._retry_model_params(attempt)
                        self.logger.info(f"RETRAIN [{idx+1}/{len(trained_timeframes)}] {tf_label} — attempt {attempt}/{max_retries} (w={weight_mult})...")

                        self.auto_retrain.model_trainer.ensemble = VotingEnsemble()
                        result = self.auto_retrain.retrain(
                            df, timeframe=tf,
                            sample_weight_multiplier=weight_mult,
                            model_params=model_params,
                            progress=progress,
                            tf_label=tf_label,
                        )
                        if not result.get("success"):
                            self.logger.warning(f"  {tf_label} attempt {attempt}: training failed — {result.get('error', 'unknown')}")
                            continue

                        attempt_version = result["version"]
                        attempt_versions.append(attempt_version)

                        models_data = result.get("models", {}) or {}
                        perf_data = {"accuracy": {}}
                        for m_name in ["xgboost", "random_forest", "lightgbm"]:
                            m_data = models_data.get(m_name, {}) or {}
                            perf_data["accuracy"][m_name] = m_data.get("train_accuracy", 0) or 0
                            perf_data["accuracy"][f"{m_name}_val"] = m_data.get("val_accuracy", 0) or 0
                        self.model_manager.save_performance(attempt_version, perf_data)

                        ensemble = self.model_manager.load_ensemble(attempt_version)

                        progress.begin_oos()
                        oos_result = self.oos_validator.validate(
                            df=df, ensemble=ensemble, trainer=self.model_trainer,
                            timeframe_label=tf_label, oos_split=0.2,
                        )
                        progress.end_oos()
                        self.model_manager.save_oos_result(attempt_version, oos_result)

                        if oos_result.get("success"):
                            self.logger.info(
                                f"  {tf_label} attempt {attempt}/{max_retries} OOS: WR={oos_result['win_rate']:.1f}% "
                                f"PF={oos_result['profit_factor']:.2f} Sharpe={oos_result.get('sharpe_ratio',0):.2f} "
                                f"Grade={oos_result['grade']} Trades={oos_result['total_trades']} "
                                f"Passed={oos_result.get('passed',False)}"
                            )
                        else:
                            self.logger.info(
                                f"  {tf_label} attempt {attempt}/{max_retries} OOS: FAILED "
                                f"({oos_result.get('reason', 'unknown')}) "
                                f"acc={oos_result.get('accuracy',0):.1f}%"
                            )

                        if best_oos is None or (
                            oos_result.get("success", False) and
                            oos_result.get("profit_factor", 0) > best_oos.get("profit_factor", 0)
                        ):
                            best_oos = oos_result
                            best_version = attempt_version
                            self.logger.info(f"  {tf_label} attempt {attempt}: best so far (PF={oos_result.get('profit_factor',0):.2f})")

                        oos_success = oos_result.get("success", False)
                        oos_acc = oos_result.get("accuracy", 0)
                        fallback_eligible = not oos_success and oos_acc >= 50
                        self.logger.info(
                            f"  {tf_label} attempt {attempt}: oos_success={oos_success} "
                            f"acc={oos_acc:.1f}% → "
                            f"{'PASS' if oos_success else 'FALLBACK_ELIGIBLE' if fallback_eligible else 'REJECT'}"
                        )

                        if oos_success or fallback_eligible:
                            effective_pass = oos_result.get("passed", False) if oos_result.get("success") else True
                            if effective_pass:
                                if old_version:
                                    old_oos = self.model_manager.get_oos_result(old_version)
                                    if old_oos.get("success"):
                                        new_score = oos_result.get("win_rate", 0) * oos_result.get("profit_factor", 0)
                                        old_score = old_oos.get("win_rate", 0) * old_oos.get("profit_factor", 0)
                                        if new_score >= old_score * 0.5:
                                            accepted = True
                                            self.logger.info(f"  {tf_label} attempt {attempt}: ACCEPTED (score={new_score:.2f} >= old*0.5={old_score*0.5:.2f})")
                                    else:
                                        accepted = True
                                        self.logger.info(f"  {tf_label} attempt {attempt}: ACCEPTED (no old OOS)")
                                else:
                                    accepted = True
                                    self.logger.info(f"  {tf_label} attempt {attempt}: ACCEPTED (first ever)")
                            else:
                                self.logger.info(f"  {tf_label} attempt {attempt}: not passed (PF={oos_result.get('profit_factor',0):.2f})")

                        if accepted:
                            progress.end_tf()
                            ensembles[tf] = self.model_manager.load_ensemble(attempt_version)
                            self.model_manager.increment_retrain_count(tf)
                            accepted_list.append(tf_label)
                            rejected_versions.extend(v for v in attempt_versions if v != attempt_version)
                            await self.telegram.send_event(TelegramEvent.MODEL_RETRAINED, {
                                "version": attempt_version,
                                "timeframe": tf_label,
                                "models": ensemble.get_num_models(),
                                "samples": len(result.get("X", [])),
                                "accuracy": result.get("models", {}).get("xgboost", {}).get("train_accuracy", 0),
                                "oos_grade": oos_result.get("grade", "N/A"),
                                "oos_win_rate": oos_result.get("win_rate", 0),
                            })
                            break
                        else:
                            self.logger.info(f"  {tf_label} attempt {attempt}: not accepted, retrying...")

                    _elapsed = time.monotonic() - _step_start
                    _train_times.append(_elapsed)
                    _avg = sum(_train_times) / len(_train_times)
                    _remaining = _avg * (len(trained_timeframes) - idx - 1)

                    if not accepted:
                        use_version = old_version or best_version
                        if use_version:
                            try:
                                ensembles[tf] = self.model_manager.load_ensemble(use_version)
                            except Exception:
                                pass
                        rejected_versions.extend(attempt_versions)
                        self.logger.warning(
                            f"  {tf_label}: all {max_retries} attempts failed, keeping {use_version} "
                            f"(best OOS: Grade={best_oos.get('grade','N/A') if best_oos else 'N/A'} "
                            f"WR={best_oos.get('win_rate',0) if best_oos else 0:.1f}% "
                            f"PF={best_oos.get('profit_factor',0) if best_oos else 0:.2f})"
                        )

                    self.logger.info(f"RETRAIN [{idx+1}/{len(trained_timeframes)}] {tf_label} — {'ACCEPTED' if accepted else 'KEPT OLD'} in {_elapsed:.1f}s | ETA: {_remaining:.0f}s ({_remaining/60:.1f}m)")

                except Exception as e:
                    self.logger.warning(f"Retrain failed for {Timeframe.LABELS.get(tf, tf)}: {e}")
                    try:
                        ensembles[tf] = self.model_manager.load_latest_for_timeframe(tf)
                    except Exception:
                        pass
                    continue

            keep_versions = set()
            for tf_ in ensembles:
                v = self.model_manager.get_latest_version(tf_)
                if v:
                    keep_versions.add(v)
            for v in set(rejected_versions):
                if v and v not in keep_versions and v != "none":
                    try:
                        self.model_manager.delete_version(v)
                    except Exception:
                        pass

            if ensembles:
                self.ml_predictor = MLPredictor(ensembles)
                self.decision_engine = DecisionEngine(
                    ml_predictor=self.ml_predictor,
                    market_scorer=self.market_scorer,
                    trade_memory=self.trade_memory,
                )
                self._update_dashboard()

                if accepted_list:
                    self.logger.info(f"Models accepted: {', '.join(accepted_list)}")

                new_skill = self.model_manager.get_skill_level()
                new_retrains = self.model_manager.get_total_retrains()
                if new_skill != old_skill:
                    summary = self.model_manager.get_models_summary()
                    model_lines = "\n".join(
                        f"  {n}: v{m['version']} | {m['retrains']}x | {m['skill']} (score:{m.get('skill_score',0)})"
                        for n, m in sorted(summary.items()) if not n.startswith("_")
                    )
                    await self.telegram.send_event(TelegramEvent.SKILL_UP, {
                        "old_skill": old_skill,
                        "new_skill": new_skill,
                        "total_retrains": new_retrains,
                        "active_models": len(ensembles),
                        "models_detail": model_lines,
                    })

    def _update_dashboard(self):
        trades = self.trade_logger.get_closed_trades()
        perf = self.performance_analyzer.analyze_trades(
            trades,
            start_balance=self._account_info.get("balance", 0),
        )
        acct_status = self.risk_manager.account_monitor.get_account_status(self._account_info)

        drift_summary = self.drift_detector.get_drift_summary()
        retrain_needed, retrain_reason = self.auto_retrain.check_retrain_needed() if self.auto_retrain else (False, "")
        closed_trades = self.trade_logger.get_closed_trades()
        if closed_trades:
            trades_by_pair: Dict[str, List] = {}
            for t in closed_trades:
                sym = t.get("symbol", "UNKNOWN")
                trades_by_pair.setdefault(sym, []).append(t)
            self.skill_scorer.compute_per_pair(trades_by_pair)
        pair_skills = self.skill_scorer.get_pair_skills()
        best_pair = self.skill_scorer.get_best_pair(pair_skills) or "N/A"
        worst_pair = self.skill_scorer.get_worst_pair(pair_skills) or "N/A"

        mistake_report = self.mistake_analyzer.analyze_losses(self.trade_logger.get_closed_trades())

        has_analysis = len(self._last_analysis) > 0
        state = {
            "analysis_ready": has_analysis,
            "balance": self._account_info.get("balance", 0),
            "equity": self._account_info.get("equity", 0),
            "margin": self._account_info.get("margin", 0),
            "free_margin": self._account_info.get("margin_free", 0),
            "margin_level": self._account_info.get("margin_level", 0),
            "floating_profit": self._account_info.get("profit", 0),
            "open_positions": len(self.position_manager.get_open_positions()),
            "total_trades": perf.get("total_trades", 0),
            "win_rate": perf.get("win_rate", 0),
            "profit_factor": perf.get("profit_factor", 0),
            "drawdown": acct_status.get("current_drawdown", 0),
            "learning_status": "active" if config.learning["enabled"] else "disabled",
            "model_version": self.model_manager.get_latest_version() or "none",
            "retrain_count": self.model_manager.get_total_retrains(),
            "last_retrain": self.model_manager.get_last_retrain_time() or "never",
            "skill_level": self.model_manager.get_skill_level(),
            "skill_score": self.model_manager.get_skill_score(),
            "models_summary": self.model_manager.get_models_summary(),
            "emergency_level": acct_status.get("emergency_level", "NORMAL"),
            "drift_detected": drift_summary.get("drift_detected", False),
            "drift_score": drift_summary.get("last_drift", {}).get("score", 0) if drift_summary.get("last_drift") else 0,
            "retrain_needed": retrain_needed,
            "retrain_reason": retrain_reason,
            "best_pair": best_pair,
            "worst_pair": worst_pair,
            "pair_skills": pair_skills,
            "mistake_summary": mistake_report.get("summary", ""),
        }

        for symbol, analysis in self._last_analysis.items():
            state["symbol"] = symbol
            trend = analysis.get("trend", {})
            state["trend"] = trend.get("direction", "N/A")
            state["trend_strength"] = trend.get("strength", 0)
            state["trend_score"] = trend.get("score", 0)
            diverg = trend.get("divergence", {})
            state["divergence_type"] = diverg.get("type") if diverg.get("direction") else None

            vol = analysis.get("volatility", {})
            state["vol_level"] = vol.get("level", "N/A")
            state["vol_score"] = vol.get("score", 0)
            state["atr"] = vol.get("atr", 0)
            state["vol_expanding"] = vol.get("expanding", False)

            mom = analysis.get("momentum", {})
            state["momentum_score"] = mom.get("score", 0)
            state["momentum_strength"] = mom.get("strength", "N/A")

            regime_result = analysis.get("regime", {})
            state["regime"] = regime_result.get("regime", "N/A")
            state["regime_confidence"] = regime_result.get("confidence", 0)

            dec = analysis.get("decision", {})
            state["market_score"] = dec.get("market_score", 0)
            state["confidence"] = dec.get("confidence", 0)
            state["current_action"] = dec.get("action", "HOLD")
            state["ml_signal"] = dec.get("ml_signal")
            state["decision_reasons"] = dec.get("reasons", [])
            state["no_trade_reasons"] = dec.get("no_trade_reasons", [])
            state["no_trade"] = dec.get("no_trade", False)
            state["entry_price"] = dec.get("entry_price") or 0
            state["stop_loss"] = dec.get("stop_loss") or 0
            state["take_profit"] = dec.get("take_profit") or 0

            tf_info = analysis.get("timeframe", {})
            dec_tf = analysis.get("decision", {}).get("timeframe")
            if dec_tf:
                state["selected_timeframe"] = Timeframe.LABELS.get(dec_tf, "N/A")
            else:
                state["selected_timeframe"] = Timeframe.LABELS.get(tf_info.get("entry", 0), "N/A")
            raw_tf_scores = tf_info.get("scores", {})
            state["tf_scores"] = {
                Timeframe.LABELS.get(k, str(k)): v
                for k, v in raw_tf_scores.items()
            }

            strategy_full = self.regime_detector.get_strategy_for_regime(
                regime_result.get("regime", "SIDEWAYS")
            )
            state["strategy"] = strategy_full.get("action", "HOLD")
            state["aggressiveness"] = strategy_full.get("aggressiveness", "N/A")
            state["trailing_stop"] = strategy_full.get("trailing_stop", False)

            fs = analysis.get("feature_summary", {})
            ind = fs.get("indicators", {})
            state["rsi"] = ind.get("rsi", 50)
            state["macd"] = ind.get("macd", 0)
            state["macd_signal"] = ind.get("macd_signal", 0)
            state["adx"] = ind.get("adx", 0)
            state["ema_20"] = ind.get("ema_20", 0)
            state["ema_50"] = ind.get("ema_50", 0)
            state["ema_200"] = ind.get("ema_200", 0)

            ms = fs.get("market_structure", {})
            state["market_structure"] = ms.get("current", "N/A")
            state["has_bos"] = ms.get("has_bos", False)
            state["has_choch"] = ms.get("has_choch", False)

            state["price_action"] = fs.get("price_action", {}).get("current", "N/A")
            state["candle_pattern"] = fs.get("candle_pattern", {}).get("current", "N/A")
            state["candle_signal"] = str(fs.get("candle_pattern", {}).get("signal", "N/A"))

            tick = self.data_engine.get_current_price(symbol)
            if tick:
                state["current_price"] = tick.get("bid", 0)

        self.dashboard.update(state)

    async def shutdown(self, reason: str = "User request"):
        self.logger.info(f"Initiating graceful shutdown: {reason}")
        self.running = False

        trades = self.trade_logger.get_closed_trades()
        perf = self.performance_analyzer.analyze_trades(
            trades,
            start_balance=self._account_info.get("balance", 0),
        )
        positions = self.position_manager.get_open_positions()

        tf_lines = []
        for tf_name, tf_perf in perf.get("by_timeframe", {}).items():
            tf_lines.append(
                f"  {tf_name}: {tf_perf.get('total_trades', 0)}t "
                f"WR:{tf_perf.get('win_rate', 0):.1f}% "
                f"PF:{tf_perf.get('profit_factor', 0):.2f}"
            )

        active_tfs = self.ml_predictor.available_timeframes if hasattr(self, 'ml_predictor') else []
        tf_labels = [Timeframe.LABELS.get(tf, str(tf)) for tf in active_tfs]

        msg = (
            f"🛑 *BOT SHUTDOWN*\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Reason: {reason}\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📊 *Last State*\n"
            f"Balance: ${self._account_info.get('balance', 0):.2f}\n"
            f"Equity: ${self._account_info.get('equity', 0):.2f}\n"
            f"Open Positions: {len(positions)}\n"
            f"Total Trades: {perf.get('total_trades', 0)}\n"
            f"Win Rate: {perf.get('win_rate', 0):.1f}%\n"
            f"Profit Factor: {perf.get('profit_factor', 0):.2f}\n"
            f"Max DD: {perf.get('max_drawdown', 0):.1f}%\n"
            f"Active TF: {', '.join(tf_labels) or 'none'}\n"
            f"Retrains: {self.model_manager.get_total_retrains()}\n"
            f"Skill: {self.model_manager.get_skill_level()}\n"
            f"Model: {self.model_manager.get_latest_version() or 'none'}\n"
        )
        if tf_lines:
            msg += "📈 *Per Timeframe*\n" + "\n".join(tf_lines) + "\n"

        model_info = self.model_manager.get_models_summary()
        msg += "🧠 *Models*\n"
        for name, m in sorted(model_info.items()):
            if name.startswith("_"):
                continue
            msg += f"  {name}: v{m['version']} | {m['retrains']}x retrain | {m['skill']}\n"
        msg += f"  Total: {model_info.get('_total', {}).get('retrains', 0)} retrains | Skill: {model_info.get('_total', {}).get('skill', 'N/A')}\n"

        if positions:
            self.logger.info(f"Closing {len(positions)} open positions...")
            self.position_manager.close_all()

        self.data_engine.shutdown()
        await self.telegram.send_message(msg)
        await self.telegram.shutdown()
        await self.llm_client.close()
        self.logger.info("Shutdown complete")

    def pause(self):
        self.paused = True
        self.logger.info("Bot paused")
        asyncio.create_task(
            self.telegram.send_message("⏸ *BOT PAUSED*\nTrading paused by user")
        )

    def resume(self):
        self.paused = False
        self.logger.info("Bot resumed")
        asyncio.ensure_future(
            self.telegram.send_message("▶️ *BOT RESUMED*\nTrading resumed")
        )


async def cmd_train(bot: ForexBot, args: argparse.Namespace):
    bot.logger.info("=" * 50)
    bot.logger.info("TRAINING COMMAND")
    bot.logger.info("=" * 50)

    if args.from_storage:
        bot.logger.info("Loading data from storage (no MT5 download)...")
        symbol = bot._symbols[0]
        timeframes = [int(t.strip()) for t in args.timeframes.split(",")] if args.timeframes else bot._timeframes
        days = args.days or config.training.get("rolling_window_days", 180)
        data_override = {}
        for tf in timeframes:
            tf_label = Timeframe.LABELS.get(tf, str(tf))
            bot.logger.info(f"  {tf_label}: loading {days} days from storage...")
            df = bot.data_engine.load_training_data(symbol, tf, days=days)
            if not df.empty:
                data_override[tf] = df
                bot.logger.info(f"  {tf_label}: {len(df)} candles loaded ({days} days)")
            else:
                bot.logger.warning(f"  {tf_label}: no data in storage")
        if not data_override:
            bot.logger.warning("No data found in storage. Run 'download' first.")
            await bot.shutdown(reason="No data")
            return
        bot.logger.info("Training from stored data...")
        await bot._perform_retrain(data_override=data_override)
    elif args.force:
        bot.logger.info("Force mode: training will proceed regardless of market status")
        await bot._perform_retrain()
    elif not bot.model_manager.is_market_open():
        bot.logger.info("Market is closed. Proceeding with training...")
        await bot._perform_retrain()
    else:
        bot.logger.warning("Market is currently open. Use --force to train anyway.")
        bot.logger.info("Downloading fresh data only...")
        symbol = bot._symbols[0]
        timeframes = [int(t.strip()) for t in args.timeframes.split(",")] if args.timeframes else bot._timeframes
        for tf in timeframes:
            tf_label = Timeframe.LABELS.get(tf, str(tf))
            bot.logger.info(f"  {tf_label}: downloading & appending...")
            df = bot.data_engine.get_historical_data(symbol, tf, years=config.training["historical_years"])
            if not df.empty:
                bot.logger.info(f"  {tf_label}: {len(df)} candles available")
            else:
                bot.logger.warning(f"  {tf_label}: no data")
        bot.logger.info("Run with --force to retrain models.")

    mm = bot.model_manager
    summary = mm.get_models_summary()
    print()
    print("=" * 80)
    print("TRAINING RESULT SUMMARY")
    print("=" * 80)
    print(f"{'TF':<8} {'Version':<16} {'Retr':<6} {'Skill':<16} {'Score':<7} {'OOS WR':<8} {'OOS PF':<8} {'Grade':<6}")
    print("-" * 80)
    total_retrains = 0
    for name, m in sorted(summary.items()):
        if name.startswith("_"):
            continue
        oos = m.get("oos", {})
        oos_wr = f"{oos.get('win_rate', 0):.0f}%" if oos.get("win_rate", 0) else "-"
        oos_pf = f"{oos.get('profit_factor', 0):.2f}" if oos.get("profit_factor", 0) else "-"
        oos_grade = oos.get("grade", "-")
        print(f"{name:<8} {m.get('version', '-'):<16} {m.get('retrains', 0):<6} {m.get('skill', '-'):<16} {m.get('skill_score', 0):<7} {oos_wr:<8} {oos_pf:<8} {oos_grade:<6}")
        total_retrains += m.get('retrains', 0)
    print("-" * 80)
    print(f"Total retrains: {total_retrains} | Skill: {mm.get_skill_level()} ({mm.get_skill_score()}/100)")
    print("=" * 80)
    print()

    try:
        import winsound
        winsound.Beep(880, 200)
        winsound.Beep(1100, 200)
        winsound.Beep(1320, 400)
    except Exception:
        print("\a")

    try:
        msg = "*TRAINING COMPLETE*\n\n"
        for name, m in sorted(summary.items()):
            if name.startswith("_"):
                continue
            oos = m.get("oos", {})
            oos_wr = f"{oos.get('win_rate', 0):.0f}%" if oos.get("win_rate", 0) else "-"
            oos_pf = f"{oos.get('profit_factor', 0):.2f}" if oos.get("profit_factor", 0) else "-"
            oos_grade = oos.get("grade", "-")
            acc = m.get("accuracy", {})
            xgb_val = f"{acc.get('xgboost_val', 0)*100:.0f}%" if acc.get('xgboost_val', 0) else "-"
            rf_val = f"{acc.get('random_forest_val', 0)*100:.0f}%" if acc.get('random_forest_val', 0) else "-"
            msg += f"*{name}*: v{m.get('version', '-')} | {m.get('retrains', 0)}x | {m.get('skill', '-')} ({m.get('skill_score', 0)})\n"
            msg += f"  Val: XGB={xgb_val} RF={rf_val} | OOS: WR={oos_wr} PF={oos_pf} Grade={oos_grade}\n"
        msg += f"\nSkill: {mm.get_skill_level()} ({mm.get_skill_score()}/100)"
        await bot.telegram.send_message(msg, parse_mode="Markdown")
    except Exception as e:
        bot.logger.warning(f"Failed to send training summary to Telegram: {e}")

    bot.logger.info("Training command complete.")
    await bot.shutdown(reason="Training complete")


async def cmd_validate(bot: ForexBot, args: argparse.Namespace):
    bot.logger.info("=" * 50)
    bot.logger.info("VALIDATE: Comparing production vs candidate models")
    bot.logger.info("=" * 50)

    timeframes = bot._timeframes if args.all else ([args.timeframe] if args.timeframe else bot._timeframes)
    results = {}
    for tf in timeframes:
        tf_label = Timeframe.LABELS.get(tf, str(tf))
        prod = None
        cand = None
        try:
            prod = bot.model_manager.load_production(tf)
        except Exception:
            bot.logger.info(f"  {tf_label}: no production model")
        try:
            cand = bot.model_manager.load_candidate(tf)
        except Exception:
            bot.logger.info(f"  {tf_label}: no candidate model")

        if prod is None or cand is None:
            bot.logger.info(f"  {tf_label}: SKIPPED — need both production and candidate")
            continue

        result = bot.model_validator.validate(prod, cand, tf)
        results[tf_label] = result
        status = "PROMOTE" if result.get("promote") else "REJECT"
        score = result.get("score", 0)
        bot.logger.info(
            f"  {tf_label}: {status} (score={score:.4f})\n"
            f"    Production: WR={result['production_metrics'].get('win_rate',0):.1f}% PF={result['production_metrics'].get('profit_factor',0):.2f} Sharpe={result['production_metrics'].get('sharpe_ratio',0):.2f}\n"
            f"    Candidate:  WR={result['candidate_metrics'].get('win_rate',0):.1f}% PF={result['candidate_metrics'].get('profit_factor',0):.2f} Sharpe={result['candidate_metrics'].get('sharpe_ratio',0):.2f}\n"
            f"    Reason: {result.get('reject_reason', 'Candidate passes validation')}"
        )

    print(f"\nValidated {len(results)} timeframe(s).")
    await bot.shutdown(reason="Validation complete")


async def cmd_promote(bot: ForexBot, args: argparse.Namespace):
    bot.logger.info("=" * 50)
    bot.logger.info("PROMOTE: Promoting candidate to production")
    bot.logger.info("=" * 50)

    timeframes = bot._timeframes if args.all else ([args.timeframe] if args.timeframe else bot._timeframes)
    promoted = 0
    for tf in timeframes:
        tf_label = Timeframe.LABELS.get(tf, str(tf))
        success, msg = bot.model_manager.promote_candidate(tf)
        if success:
            bot.logger.info(f"  {tf_label}: {msg}")
            promoted += 1
        else:
            bot.logger.info(f"  {tf_label}: FAILED — {msg}")

    print(f"\nPromoted {promoted}/{len(timeframes)} timeframe(s).")
    await bot.shutdown(reason="Promotion complete")


async def cmd_rollback(bot: ForexBot, args: argparse.Namespace):
    bot.logger.info("=" * 50)
    bot.logger.info("ROLLBACK: Restoring previous production model")
    bot.logger.info("=" * 50)

    timeframes = bot._timeframes if args.all else ([args.timeframe] if args.timeframe else bot._timeframes)
    rolled_back = 0
    for tf in timeframes:
        tf_label = Timeframe.LABELS.get(tf, str(tf))
        success, msg = bot.model_manager.rollback(tf)
        if success:
            bot.logger.info(f"  {tf_label}: {msg}")
            rolled_back += 1
        else:
            bot.logger.info(f"  {tf_label}: FAILED — {msg}")

    print(f"\nRolled back {rolled_back}/{len(timeframes)} timeframe(s).")
    await bot.shutdown(reason="Rollback complete")


async def cmd_status(bot: ForexBot, args: argparse.Namespace):
    bot.logger.info("=" * 50)
    bot.logger.info("SYSTEM STATUS")
    bot.logger.info("=" * 50)

    mm = bot.model_manager
    summary = mm.get_models_summary()
    retrain_counts = mm.get_retrain_counts()
    trained_tfs = mm.get_trained_timeframes()
    retrain_needed, retrain_reason = bot.auto_retrain.check_retrain_needed() if bot.auto_retrain else (False, "")
    drift = bot.drift_detector.get_drift_summary()
    pair_skills = bot.skill_scorer.get_pair_skills()
    wknd = bot.weekend_trainer.get_status()
    trades = bot.trade_logger.get_closed_trades()
    perf = bot.performance_analyzer.analyze_trades(trades) if trades else {}
    positions = bot.position_manager.get_open_positions()
    acct = bot.data_engine.get_account_info() or {}

    print(f"System:              {'RUNNING' if bot.running else 'STOPPED'}")
    print(f"Mode:                {'LEARN-ONLY' if config.account.get('learn_only') else 'LIVE'}")
    print(f"Symbol:              {', '.join(bot._symbols)}")
    print(f"Balance:             ${acct.get('balance', 0):.2f}")
    print(f"Equity:              ${acct.get('equity', 0):.2f}")
    print(f"Open Positions:      {len(positions)}")
    print(f"Total Trades:        {perf.get('total_trades', 0)}")
    print(f"Win Rate:            {perf.get('win_rate', 0):.1f}%")
    print(f"Profit Factor:       {perf.get('profit_factor', 0):.2f}")
    print(f"Max Drawdown:        {perf.get('max_drawdown', 0):.1f}%")
    print(f"Sharpe Ratio:        {perf.get('sharpe_ratio', 0):.2f}")
    print(f"Retrain Count:       {mm.get_total_retrains()}")
    print(f"Skill Level:         {mm.get_skill_level()} ({mm.get_skill_score()}/100)")
    print(f"Last Retrain:        {retrain_counts.get('last_retrain', 'N/A')}")
    print(f"Retrain Needed:      {'YES' if retrain_needed else 'no'} ({retrain_reason if retrain_needed else ''})")
    print(f"Drift Detected:      {'YES' if drift.get('drift_detected') else 'no'}")
    print(f"Weekend:             {'yes' if wknd.get('is_weekend') else 'no'}, market {'open' if wknd.get('market_open') else 'closed'}")
    print(f"Weekend Train:       {'ready' if wknd.get('should_train') else 'not due'} (last: {wknd.get('last_training', 'never')})")
    print()

    print(f"{'Timeframe':<12} {'Version':<20} {'Retrains':<10} {'Skill':<20} {'Score':<8} {'OOS':<30}")
    print("-" * 100)
    for name, m in sorted(summary.items()):
        if name.startswith("_"):
            continue
        oos = m.get("oos", {})
        oos_str = f"WR={oos.get('win_rate',0):.1f}% PF={oos.get('profit_factor',0):.2f} G={oos.get('grade','N/A')}"
        version = m.get("version", "none")[:18]
        print(f"{name:<12} {version:<20} {m.get('retrains',0):<10} {str(m.get('skill','N/A'))[:18]:<20} {m.get('skill_score',0):<8} {oos_str}")
    print()

    if trained_tfs:
        print(f"{'Version History':-^60}")
        for tf in trained_tfs:
            tf_label = Timeframe.LABELS.get(tf, str(tf))
            prod_ver = mm.get_production_version(tf) or "none"
            cand = "yes" if mm.load_candidate(tf) is not None else "no"
            archives = len(mm.get_archive_versions(tf))
            print(f"  {tf_label:<6} | production: {prod_ver[:18]:<18} | candidate: {cand:<3} | archives: {archives}")

    if pair_skills:
        best = bot.skill_scorer.get_best_pair(pair_skills)
        worst = bot.skill_scorer.get_worst_pair(pair_skills)
        print(f"\nBest Pair:  {best}")
        print(f"Worst Pair: {worst}")

    await bot.shutdown(reason="Status command")


async def cmd_download(bot: ForexBot, args: argparse.Namespace):
    bot.logger.info("=" * 50)
    bot.logger.info("DOWNLOAD: Downloading latest market data")
    bot.logger.info("=" * 50)

    symbols = [args.pair] if args.pair else bot._symbols

    if args.timeframes:
        timeframes = [int(t.strip()) for t in args.timeframes.split(",")]
    elif args.timeframe:
        timeframes = [args.timeframe]
    else:
        timeframes = bot._timeframes

    total = 0
    for symbol in symbols:
        for tf in timeframes:
            tf_label = Timeframe.LABELS.get(tf, str(tf))
            count = args.days * (1440 // tf) if args.days else 5000
            bot.logger.info(f"  {symbol} {tf_label}: downloading {count} candles ({args.days or 'default'} days)...")
            n = bot.data_engine.refresh_stored_data(symbol, tf, count=count)
            if n > 0:
                bot.logger.info(f"  {symbol} {tf_label}: {n} candles saved")
                total += n
            else:
                bot.logger.warning(f"  {symbol} {tf_label}: no data")

    bot.logger.info(f"Download complete. Total candles saved: {total}")
    print(bot.dashboard.get_display_text())
    await bot.shutdown(reason="Download complete")


async def cmd_backtest(bot: ForexBot, args: argparse.Namespace):
    bot.logger.info("=" * 50)
    bot.logger.info("BACKTEST MODE")
    bot.logger.info("=" * 50)

    symbol = args.pair or bot._symbols[0]
    days = args.days or config.training.get("historical_years", 2) * 365

    bot.logger.info(f"Running backtest on {symbol} (days={days})")

    from backtest.backtest_engine import BacktestEngine

    all_trades = []
    for tf in bot._timeframes:
        tf_label = Timeframe.LABELS.get(tf, str(tf))
        bot.logger.info(f"  {tf_label}: loading data & running backtest...")
        try:
            bt = BacktestEngine(
                data_engine=bot.data_engine,
                ml_predictor=bot.ml_predictor,
                feature_pipeline=bot.feature_pipeline,
                trend_analyzer=bot.trend_analyzer,
                vol_analyzer=bot.vol_analyzer,
                momentum_analyzer=bot.momentum_analyzer,
                regime_detector=bot.regime_detector,
                market_scorer=bot.market_scorer,
                news_analyzer=bot.news_analyzer,
            )
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            result = bt.run_backtest(symbol, tf, start_date, end_date)
            trades = result.get("trades", [])
            all_trades.extend(trades)
            bot.logger.info(f"  {tf_label}: {len(trades)} trades, return={result.get('total_return', 0):.1f}%")
        except Exception as e:
            bot.logger.warning(f"  {tf_label}: error — {e}")

    if all_trades:
        perf_result = bot.performance_analyzer.analyze_trades(all_trades)
        print(f"\nBacktest Results ({len(all_trades)} total trades):")
        print(f"  Win Rate:      {perf_result.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor: {perf_result.get('profit_factor', 0):.2f}")
        print(f"  Sharpe Ratio:  {perf_result.get('sharpe_ratio', 0):.2f}")
        print(f"  Max DD:        {perf_result.get('max_drawdown', 0):.1f}%")
        print(f"  Expectancy:    {perf_result.get('expectancy', 0):.4f}")
        print(f"  Total P&L:     ${perf_result.get('total_profit', 0):.2f}")
    else:
        print("\nNo trades generated in backtest.")

    await bot.shutdown(reason="Backtest complete")


async def main():
    parser = argparse.ArgumentParser(description="AI Forex Trading Bot v2")
    parser.add_argument("command", nargs="?", default="live",
                        choices=["live", "train", "backtest", "validate", "promote", "rollback", "status", "download"],
                        help="Command to run (default: live)")
    parser.add_argument("--force", "-f", action="store_true", help="Force action (bypass safety checks)")
    parser.add_argument("--pair", type=str, default=None, help="Symbol/pair to operate on")
    parser.add_argument("--all-pairs", action="store_true", help="Operate on all configured pairs")
    parser.add_argument("--days", type=int, default=None, help="Number of days of data")
    parser.add_argument("--model", type=str, default=None, choices=["xgboost", "random_forest", "lightgbm"],
                        help="Model type to train")
    parser.add_argument("--timeframe", type=int, default=None,
                        help="Timeframe in minutes (e.g. 5, 15, 30)")
    parser.add_argument("--timeframes", type=str, default=None,
                        help="Comma-separated timeframes (e.g. 5,15,30)")
    parser.add_argument("--all", action="store_true", help="Apply to all timeframes")
    parser.add_argument("--from-storage", action="store_true", help="Train from stored data only (no MT5 download)")

    args = parser.parse_args()

    bot = ForexBot()
    try:
        await bot.initialize()
        bot.dashboard.display()

        cmd = args.command
        if cmd == "train":
            await cmd_train(bot, args)
        elif cmd == "validate":
            await cmd_validate(bot, args)
        elif cmd == "promote":
            await cmd_promote(bot, args)
        elif cmd == "rollback":
            await cmd_rollback(bot, args)
        elif cmd == "status":
            await cmd_status(bot, args)
        elif cmd == "download":
            await cmd_download(bot, args)
        elif cmd == "backtest":
            await cmd_backtest(bot, args)
        else:
            await bot.run()
    except asyncio.CancelledError:
        print("\nShutdown signal received...")
        await bot.shutdown(reason="Ctrl+C / Shutdown signal")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
        await bot.shutdown(reason="Keyboard interrupt")
    except Exception as e:
        logger = get_logger("main")
        logger.error(f"Fatal error: {e}", exc_info=True)
        try:
            await bot.shutdown(reason=f"Fatal error: {type(e).__name__}")
        except Exception:
            pass
    else:
        await bot.shutdown(reason="Normal exit")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot terminated by user")
