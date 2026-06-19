#!/usr/bin/env python3
"""
AI Forex Trading Bot v2 - Production-Ready System
==================================================
SURVIVAL > CONSISTENCY > PROFIT
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
np.seterr(divide="ignore", invalid="ignore")

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
from dashboard.tiktok_dashboard import TikTokDashboard
from monitor.health_server import update_state as health_update
from monitor.health_server import update_full_state, update_candles
from utils.logger import get_logger
from utils.training_progress import TrainingProgress
from utils import sounds as sound_utils
from utils.readiness_estimator import estimate_real_readiness

# Ensemble v2 integration
from ensemble.integration import EnsembleIntegration


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
        self.ensemble_integration = None
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
        if config.dashboard.get("hidden"):
            self.dashboard.hide()
        self._dashboard_refreshed = False
        self._last_dashboard_display = datetime.now()
        self._tiktok_mode = False
        self.tiktok_dashboard = TikTokDashboard()
        self._last_tiktok_display = datetime.now()
        self._symbols = config.trading["pairs"]
        self._timeframes = [self._tf_to_minutes(tf) for tf in config.trading["timeframes"]]

        self._last_analysis: Dict = {}
        self._account_info: Dict = {}
        self._last_heartbeat_time = 0.0
        self._last_emergency_check: Dict = {}
        self._equity_history: List[float] = []  # for web dashboard equity sparkline


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

        try:
            from monitor.health_server import start_server
            import threading
            t = threading.Thread(target=start_server, daemon=True)
            t.start()
            self.logger.info("Health check server started on 127.0.0.1:9090")
        except Exception as e:
            self.logger.debug(f"Health check server not started: {e}")

        self.logger.info("Initializing Market Data Engine...")
        self.data_engine.initialize()

        account_info = self.data_engine.get_account_info()
        if account_info and account_info.get("login"):
            self.trade_memory.set_account_id(str(account_info["login"]))

        self.logger.info("Initializing Account & Risk Manager...")
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

        # Seed equity history for web dashboard sparkline
        initial_equity = self._account_info.get("equity", 0) or self._account_info.get("balance", 0) or config.account["balance"]
        self._equity_history = [initial_equity]

        try:
            if self.data_engine.connector._mt5_available:
                self.data_engine.connector.ensure_connected()
            synced = self.trade_logger.sync_from_mt5(self.data_engine.connector)
            if synced > 0:
                self.logger.info(f"Trade history synced: +{synced} trades from MT5")
        except Exception as e:
            self.logger.warning(f"MT5 trade history sync skipped: {e}")

        self.logger.info("Initializing ML Engine...")
        all_ensembles = {}
        model_version: Optional[str] = None
        trained_tfs = self.model_manager.get_trained_timeframes()
        
        # ── Model inventory health check ──
        version_count = len(self.model_manager.list_versions())
        retrain_count = self.model_manager.get_total_retrains()
        if retrain_count > 10 and version_count < retrain_count * 0.3:
            self.logger.warning(
                f"Model inventory anomaly: {version_count} versions on disk "
                f"vs {retrain_count} total retrains. Some models may have been lost!"
            )
        
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
                        X, y, features, df_clean = self.model_trainer.prepare_training_data(
                            df, pair=symbol, timeframe=tf,
                        )
                    except Exception as e:
                        self.logger.warning(f"  {tf_label}: feature prep failed — {e}")
                        continue
                    recency = ModelTrainer.compute_recency_weights(df_clean["time"]) if "time" in df_clean.columns else None
                    X, y, trade_weights = self.model_trainer.incorporate_trade_outcomes(
                        X, y, features, pair=symbol, timeframe=tf, min_trades=5,
                    )
                    # Extend recency weights to match merged length
                    if recency is not None and len(recency) < len(X):
                        pad_len = len(X) - len(recency)
                        recency = np.hstack([recency, np.ones(pad_len, dtype=recency.dtype)])
                    self.logger.info(f"  {tf_label}: {len(X)} samples — training models...")
                    results = self.model_trainer.train_all_models(
                        X, y, feature_cols=features, recency_weights=recency,
                        trade_outcome_weights=trade_weights, tf_label=tf_label,
                    )
                    if self.model_trainer.get_ensemble().get_num_models() > 0:
                        version = self.model_manager.save_ensemble(
                            self.model_trainer.get_ensemble(), timeframe=tf
                        )

                        # Save feature importance for this TF (used for pruning next retrain)
                        self.model_trainer.save_feature_importance(
                            self.model_trainer.get_ensemble(), features, symbol, tf, version
                        )

                        # Save performance data BEFORE loading so ensemble weights
                        # can be computed from val_accuracy
                        perf_acc = {}
                        for m_name in ["xgboost", "random_forest", "lightgbm", "lstm"]:
                            m_result = results.get("models", {}).get(m_name, {}) or {}
                            if isinstance(m_result, dict):
                                perf_acc[m_name] = m_result.get("train_accuracy", 0) or 0
                                perf_acc[f"{m_name}_val"] = m_result.get("val_accuracy", 0) or 0
                        self.model_manager.save_performance(version, {
                            "accuracy": perf_acc,
                            "samples": len(X),
                        })

                        ensemble = self.model_manager.load_ensemble(version)
                        all_ensembles[tf] = ensemble
                        xgb_val = perf_acc.get("xgboost_val", 0)
                        rf_val = perf_acc.get("random_forest_val", 0)
                        lgb_val = perf_acc.get("lightgbm_val", 0)
                        self.logger.info(
                            f"  {tf_label}: DONE — XGB(val={xgb_val:.1%}) RF(val={rf_val:.1%}) "
                            f"LGB(val={lgb_val:.1%}) samples={len(X)} | v{version} "
                            f"weights={ {k: f'{v:.2f}' for k, v in ensemble.weights.items()} }"
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
        # Ensemble v2 integration (activated via ENSEMBLE_MODE=true)
        if config.ensemble.get("enabled", False):
            self.ensemble_integration = EnsembleIntegration()
            if self.ensemble_integration.initialize():
                self.logger.info("Ensemble v2 initialized: H4+H1+M5 models loaded")
            else:
                self.logger.warning("Ensemble v2: failed to load models, falling back to legacy")
                self.ensemble_integration = None
        else:
            self.logger.debug("Ensemble v2 disabled (ENSEMBLE_MODE=false)")
        
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
        last_mt5_sync = datetime.now()

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

                if self._tiktok_mode:
                    if not self._dashboard_refreshed and self._last_analysis:
                        self._last_tiktok_display = now
                        self.tiktok_dashboard.display()
                    elif (now - self._last_tiktok_display).total_seconds() >= 3:
                        self._last_tiktok_display = now
                        self.tiktok_dashboard.display()

                if (now - last_data_refresh).total_seconds() >= 1800:
                    self.logger.info("Periodic data refresh: downloading latest candles...")
                    for symbol in self._symbols:
                        for tf in self._timeframes:
                            self.data_engine.refresh_stored_data(symbol, tf, count=2000)
                    last_data_refresh = now
                    self.logger.info("Periodic data refresh complete")

                if (now - last_mt5_sync).total_seconds() >= 3600:
                    try:
                        if self.data_engine.connector._mt5_available:
                            self.data_engine.connector.ensure_connected()
                        synced = self.trade_logger.sync_from_mt5(self.data_engine.connector)
                        if synced > 0:
                            self.logger.info(f"MT5 trade sync: +{synced} new trades")
                    except Exception as e:
                        self.logger.debug(f"MT5 periodic sync skipped: {e}")
                    last_mt5_sync = now

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

    async def _align_multi_tf(self, symbol: str, count_m5: int = 500, count_context: int = 200):
        """Fetch M5 + context TFs and align to M5 timestamps."""
        import pandas as pd
        import numpy as np

        tfs = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}
        data = {}
        for label, tf in tfs.items():
            cnt = count_m5 if tf == 5 else count_context
            df = await self.data_engine.get_rates_async(symbol, tf, count=cnt, force_refresh=True)
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
            except Exception as e:
                self.logger.debug(f"Performance analysis failed for {symbol}: {e}")

            selected_tfs = self.timeframe_selector.select_timeframes(
                {tf: self.data_engine.get_rates(symbol, tf, count=50, force_refresh=True) for tf in self._timeframes},
                performance_data=perf_data,
            )

            # ── Multi-TF: M5 entry + context from M15/M30/H1/H4 ──
            entry_tf = Timeframe.M5
            aligned = await self._align_multi_tf(symbol, count_m5=500, count_context=200)
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

            pair_skill_score = self.skill_scorer.get_pair_skills().get(symbol, 50)

            # ── Ensemble v2 mode ──
            use_ensemble = (self.ensemble_integration is not None and 
                           self.ensemble_integration.is_loaded and
                           config.ensemble.get("enabled", False))
            
            if use_ensemble:
                self.ensemble_integration.compute_all_features()
                decision = self.ensemble_integration.get_decision()
                decision["symbol"] = symbol
                # Log ensemble decision
                self.logger.info(
                    f"[Ensemble] {symbol}: {decision['action']} "
                    f"(conf={decision['confidence']:.0%}, "
                    f"score={decision.get('raw_score', 0):.2f})"
                )
                if decision["action"] in ("BUY", "SELL"):
                    decision["no_trade"] = False
                    decision["market_score"] = int(decision["confidence"] * 100)
                    # Add basic SL/TP
                    atr_val = df_aligned_feat.get("atr", pd.Series([0.001])).iloc[-1]
                    decision["stop_loss"] = price - atr_val * 1.5 if decision["action"] == "BUY" else price + atr_val * 1.5
                    decision["take_profit"] = price + atr_val * 2.5 if decision["action"] == "BUY" else price - atr_val * 2.5
                    decision["entry_price"] = price
            else:
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
                pair_skill_score=pair_skill_score,
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
            if use_ensemble:
                details = decision.get("details", {})
                self.logger.info(
                    f"[Ensemble] H4={details.get('h4',{}).get('direction','?')} "
                    f"({details.get('h4',{}).get('confidence',0):.0%}) | "
                    f"H1={details.get('h1',{}).get('signal','?')} "
                    f"({details.get('h1',{}).get('confidence',0):.0%}) | "
                    f"M5_pullback={details.get('m5',{}).get('pullback_predicted','?')} | "
                    f"Final: {decision.get('action','?')} "
                    f"(conf={decision.get('confidence',0):.0%})"
                )
            else:
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
                        atr = (df_aligned_feat["atr"].iloc[-1] if "atr" in df_aligned_feat.columns
                               and not df_aligned_feat["atr"].empty else 0.001)

                        trade_result = self.entry_engine.open_trade(
                            symbol=symbol,
                            decision=decision,
                            account_info=self._account_info,
                            df_entry=df_aligned_feat,
                            atr=atr,
                            current_price=price,
                            existing_positions=positions,
                        )

                        if trade_result:
                            entry_indicators = self._extract_indicators_from_df(df_aligned_feat)
                            self.trade_logger.log_trade_open(trade_result, indicators=entry_indicators)
                            sound_utils.entry()
                            await self.telegram.send_event(TelegramEvent.OPEN_POSITION, {
                                **trade_result,
                                "confidence": decision.get("confidence", 0),
                                "balance": self._account_info.get("balance", 0),
                            })

            if not config.account.get("learn_only") and positions:
                atr_val = float(df_aligned_feat["atr"].iloc[-1]) if "atr" in df_aligned_feat.columns and not df_aligned_feat["atr"].empty else 0.001
                rsi_val = float(df_aligned_feat["rsi"].iloc[-1]) if "rsi" in df_aligned_feat.columns and not df_aligned_feat["rsi"].empty else 50.0
                actions = self.position_manager.manage_positions(
                    symbol=symbol,
                    trend_result=trend_result,
                    regime_result=regime_result,
                    confidence=decision.get("confidence", 0),
                    market_structure=self.feature_pipeline.market_structure.get_last_hh_ll(df_aligned_feat),
                    multi_tf_trends=multi_tf_trends,
                    momentum_result=momentum_result,
                    vol_result=vol_result,
                    rsi=rsi_val,
                    atr=atr_val,
                    balance=self._account_info.get("balance", 0),
                )

                for action in actions:
                    if action.get("action") in ["FULL_CLOSE", "PARTIAL_CLOSE"]:
                        closed_trade = self.trade_logger.log_trade_close(
                            ticket=action["ticket"],
                            exit_price=price,
                            exit_reason=action["action"],
                        )
                        if closed_trade:
                            # Pass entry-time indicators snapshot to trade memory
                            entry_indicators = closed_trade.get("entry_indicators", {})
                            self.trade_memory.record_from_trade_log(closed_trade, indicators=entry_indicators)
                            await self.telegram.send_event(TelegramEvent.POSITION_CLOSED, {
                                "symbol": closed_trade.get("symbol", symbol),
                                "direction": closed_trade.get("direction", ""),
                                "entry_price": closed_trade.get("entry_price", 0),
                                "exit_price": price,
                                "profit": closed_trade.get("profit", 0),
                                "pips": closed_trade.get("profit_pips", 0),
                                "reason": action["action"],
                            })
                            if closed_trade.get("profit", 0) > 0:
                                sound_utils.win()
                            elif closed_trade.get("profit", 0) < 0:
                                sound_utils.loss()
                            if closed_trade.get("profit", 0) < 0:
                                await self.telegram.send_event(TelegramEvent.LOSS_ALERT, {
                                    "symbol": symbol,
                                    "loss": closed_trade.get("profit", 0),
                                    "balance": self._account_info.get("balance", 0),
                                    "drawdown": self._account_info.get("drawdown", 0),
                                })

        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def _extract_indicators_from_df(self, df) -> Dict:
        """Extract key indicator values from dataframe for trade memory snapshot."""
        from learning.trade_memory import INDICATOR_FIELDS
        indicators = {}
        if df is None or df.empty:
            return indicators
        last = df.iloc[-1]
        for field in INDICATOR_FIELDS:
            if field in df.columns:
                val = last.get(field)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    try:
                        indicators[field] = float(val)
                    except (ValueError, TypeError):
                        indicators[field] = str(val)
        return indicators

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

        health_update(
            is_running=not self.paused,
            is_connected=self.data_engine.connector.is_connected if hasattr(self.data_engine, 'connector') else False,
            account_balance=self._account_info.get("balance", 0),
            account_equity=self._account_info.get("equity", 0),
            open_positions=len(positions),
            emergency_level=acct_status.get("emergency_level", "NORMAL"),
            trading_paused=acct_status.get("trading_paused", False),
            total_retrains=retrain_count,
            skill_level=skill,
            uptime_seconds=uptime.total_seconds(),
            mode=config.account["trading_mode"],
        )

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
                X, y, features, df_clean = self.model_trainer.prepare_training_data(
                    df, pair=symbol, timeframe=tf,
                )
                recency = ModelTrainer.compute_recency_weights(df_clean["time"]) if "time" in df_clean.columns else None
                X, y, trade_weights = self.model_trainer.incorporate_trade_outcomes(
                    X, y, features, pair=symbol, timeframe=tf, min_trades=5,
                )
                self.logger.info(f"TRAINING [{idx+1}/{len(tf_list)}] {tf_label} — training models ({len(X)} samples)...")
                results = self.model_trainer.train_all_models(
                    X, y, feature_cols=features, recency_weights=recency,
                    trade_outcome_weights=trade_weights, tf_label=tf_label,
                )
                if self.model_trainer.get_ensemble().get_num_models() > 0:
                    version = self.model_manager.save_ensemble(
                        self.model_trainer.get_ensemble(), timeframe=tf
                    )
                    # Save feature importance for this TF (used for pruning next retrain)
                    self.model_trainer.save_feature_importance(
                        self.model_trainer.get_ensemble(), features, symbol, tf, version
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

                        _bt = config.training["buy_threshold"]
                        _st = config.training["sell_threshold"]
                        oos_result = self.oos_validator.validate(
                            df=df,
                            ensemble=self.model_trainer.get_ensemble(),
                            trainer=self.model_trainer,
                            timeframe_label=tf_label,
                            oos_split=0.2,
                            buy_threshold=_bt,
                            sell_threshold=_st,
                            timeframe=tf,
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

    def _retry_model_params(self, attempt: int, warm_start: bool = False) -> tuple:
        """Get model params for retry attempts.
        
        Args:
            attempt: 1, 2, or 3
            warm_start: If True, use fine-tuning params (smaller LR, fewer estimators)
                        karena model sudah punya dasar dari versi sebelumnya.
        """
        if warm_start:
            # Fine-tuning mode: model sudah punya dasar, tinggal adaptasi ke data baru
            if attempt == 1:
                return 1.5, {
                    "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.01, "subsample": 0.8},
                    "random_forest": {"n_estimators": 100, "max_depth": 8, "min_samples_split": 10},
                    "lightgbm": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.01, "subsample": 0.8},
                }
            elif attempt == 2:
                return 2.0, {
                    "xgboost": {"n_estimators": 150, "max_depth": 8, "learning_rate": 0.015, "subsample": 0.7},
                    "random_forest": {"n_estimators": 150, "max_depth": 10, "min_samples_split": 5},
                    "lightgbm": {"n_estimators": 150, "max_depth": 8, "learning_rate": 0.015, "subsample": 0.7},
                }
            else:
                return 3.0, {
                    "xgboost": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02, "gamma": 0.3, "reg_alpha": 0.3},
                    "random_forest": {"n_estimators": 200, "max_depth": 6, "min_samples_split": 20},
                    "lightgbm": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.02, "reg_alpha": 0.3},
                }
        else:
            # From-scratch mode: model belum punya dasar, perlu training penuh
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

                    old_version = self.model_manager.get_best_version(tf) or self.model_manager.get_latest_version(tf)
                    accepted = False
                    best_oos = None
                    best_version = None
                    attempt_versions = []

                    for attempt in range(1, max_retries + 1):
                        is_warm_start = old_version is not None
                        weight_mult, model_params = self._retry_model_params(attempt, warm_start=is_warm_start)
                        self.logger.info(f"RETRAIN [{idx+1}/{len(trained_timeframes)}] {tf_label} — attempt {attempt}/{max_retries} (w={weight_mult}, warm_start={is_warm_start})...")

                        self.auto_retrain.model_trainer.ensemble = VotingEnsemble()
                        result = self.auto_retrain.retrain(
                            df, timeframe=tf,
                            sample_weight_multiplier=weight_mult,
                            model_params=model_params,
                            progress=progress,
                            tf_label=tf_label,
                            pair=symbol,
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

                        try:
                            ensemble = self.model_manager.load_ensemble(attempt_version)
                        except Exception as e:
                            self.logger.warning(f"  {tf_label} attempt {attempt}: failed to load ensemble: {e}")
                            continue

                        progress.begin_oos()
                        try:
                            _bt = config.training["buy_threshold"]
                            _st = config.training["sell_threshold"]
                            oos_result = self.oos_validator.validate(
                                df=df, ensemble=ensemble, trainer=self.model_trainer,
                                timeframe_label=tf_label, oos_split=0.2,
                                buy_threshold=_bt, sell_threshold=_st,
                                timeframe=tf,
                            )
                        except Exception as e:
                            oos_result = self.oos_validator._empty_result(f"OOS validation crashed: {e}")
                        progress.end_oos()
                        self.model_manager.save_oos_result(attempt_version, oos_result)
                        self.logger.info(f"  {tf_label} attempt {attempt}: OOS saved for {attempt_version}")

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
                            f"acc={oos_acc:.1f}% -> "
                            f"{'PASS' if oos_success else 'FALLBACK_ELIGIBLE' if fallback_eligible else 'REJECT'}"
                        )

                        if oos_success or fallback_eligible:
                            effective_pass = oos_result.get("passed", False) if oos_result.get("success") else True
                            if effective_pass:
                                new_score = self.model_manager._compute_oos_numeric_score(oos_result)
                                if old_version:
                                    # Re-evaluasi old model pada data yang SAMA untuk fair comparison
                                    try:
                                        old_ensemble = self.model_manager.load_ensemble(old_version)
                                        if old_ensemble.get_num_models() > 0:
                                            _bt = config.training["buy_threshold"]
                                            _st = config.training["sell_threshold"]
                                            old_oos_result = self.oos_validator.validate(
                                                df=df, ensemble=old_ensemble,
                                                trainer=self.model_trainer,
                                                timeframe_label=tf_label, oos_split=0.2,
                                                buy_threshold=_bt, sell_threshold=_st,
                                                timeframe=tf,
                                            )
                                            old_oos = old_oos_result
                                            old_score = self.model_manager._compute_oos_numeric_score(old_oos_result)
                                            self.logger.info(
                                                f"  {tf_label}: re-evaluated {old_version} on same data: "
                                                f"WR={old_oos.get('win_rate',0):.1f}% "
                                                f"PF={old_oos.get('profit_factor',0):.2f} "
                                                f"Score={old_score:.1f}"
                                            )
                                    except Exception as e:
                                        self.logger.warning(f"  {tf_label}: could not re-evaluate {old_version}: {e}")
                                        old_oos = self.model_manager.get_oos_result(old_version)
                                        old_score = self.model_manager._compute_oos_numeric_score(old_oos)

                                    if old_score > 0:
                                        better, reason = ModelManager.is_model_better(oos_result, old_oos)
                                        if not better:
                                            self.logger.info(
                                                f"  {tf_label} attempt {attempt}: REJECTED ({reason})"
                                            )
                                        else:
                                            accepted = True
                                            self.logger.info(
                                                f"  {tf_label} attempt {attempt}: ACCEPTED ({reason})"
                                            )
                                    else:
                                        accepted = True
                                        self.logger.info(f"  {tf_label} attempt {attempt}: ACCEPTED (first ever, score={new_score:.1f})")
                                else:
                                    accepted = True
                                    self.logger.info(f"  {tf_label} attempt {attempt}: ACCEPTED (first ever, score={new_score:.1f})")
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
                    except Exception as load_e:
                        self.logger.warning(f"Failed to load fallback ensemble for {Timeframe.LABELS.get(tf, tf)}: {load_e}")
                    continue

            # ── Safe cleanup: only delete rejected versions per-TF, protect best ones ──
            # Build a map of TF -> the ACTUAL version loaded in ensembles
            active_versions = {}  # TF label -> version string
            for tf_, ensemble in ensembles.items():
                v = getattr(ensemble, 'version', None)
                if v:
                    active_versions[Timeframe.LABELS.get(tf_, str(tf_))] = v
                else:
                    # Fallback: get latest but log warning
                    v = self.model_manager.get_latest_version(tf_)
                    if v:
                        active_versions[Timeframe.LABELS.get(tf_, str(tf_))] = v
                        self.logger.warning(f"  Cleanup: ensemble has no version attr for {Timeframe.LABELS.get(tf_, str(tf_))}, using latest={v}")
            
            # Filter rejected versions per-TF and skip protected ones
            deleted_count = 0
            for v in set(rejected_versions):
                if not v or v == "none":
                    continue
                # Skip if this version is the active one for its TF
                if v in active_versions.values():
                    self.logger.debug(f"  Cleanup: skipping {v} (currently active)")
                    continue
                # Skip protected versions (production, golden, top-3 by OOS, recent self-learn)
                if self.model_manager.is_version_protected(v):
                    self.logger.debug(f"  Cleanup: skipping {v} (protected)")
                    continue
                try:
                    self.model_manager.delete_version(v)
                    deleted_count += 1
                except Exception as del_e:
                    self.logger.warning(f"Failed to delete rejected version {v}: {del_e}")
            if deleted_count > 0:
                self.logger.info(f"Cleanup: archived {deleted_count} rejected model versions")

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

    def _get_economic_calendar(self) -> Optional[Dict]:
        """Return upcoming high-impact economic events for EURUSD dashboard display.
        
        Returns a dict with nearest high-impact news event, or None if none found.
        This is purely for dashboard display and does not affect trading logic.
        """
        try:
            now = datetime.now()
            today = now.date()
            weekday = now.weekday()  # 0=Mon, 6=Sun
            hour = now.hour
            minute = now.minute
            
            # ── Known high-impact USD events (3rd week of month typical pattern) ──
            # In a real deployment, replace this with an economic calendar API
            events = []
            
            # Determine which day of month we're in
            day = today.day
            
            # FOMC / Fed events — typically 2nd-3rd Wed of month
            if 15 <= day <= 25:
                events.append({
                    "time": f"{today}T14:00",
                    "event": "FOMC Meeting Minutes",
                    "impact": "HIGH",
                    "currency": "USD",
                    "forecast": "--",
                    "previous": "--",
                })
            
            # NFP — typically 1st Friday
            if today.weekday() == 4 and day <= 7:
                events.append({
                    "time": f"{today}T08:30",
                    "event": "Non-Farm Payrolls (NFP)",
                    "impact": "HIGH",
                    "currency": "USD",
                    "forecast": "--",
                    "previous": "--",
                })
            
            # CPI — typically 2nd-3rd week
            if 10 <= day <= 20:
                events.append({
                    "time": f"{today}T08:30",
                    "event": "Consumer Price Index (CPI) MoM",
                    "impact": "HIGH",
                    "currency": "USD",
                    "forecast": "--",
                    "previous": "--",
                })
            
            # ECB / EUR events
            if 10 <= day <= 25:
                events.append({
                    "time": f"{today}T07:45",
                    "event": "ECB Interest Rate Decision",
                    "impact": "HIGH",
                    "currency": "EUR",
                    "forecast": "--",
                    "previous": "--",
                })
            
            # ADP Non-Farm — 1st-2nd Wed
            if day <= 10 and today.weekday() == 2:
                events.append({
                    "time": f"{today}T08:15",
                    "event": "ADP Non-Farm Employment Change",
                    "impact": "HIGH",
                    "currency": "USD",
                    "forecast": "--",
                    "previous": "--",
                })
            
            # GDP — quarterly: Jan, Apr, Jul, Oct
            if today.month in [1, 4, 7, 10] and day >= 20:
                events.append({
                    "time": f"{today}T08:30",
                    "event": "GDP Growth Rate QoQ",
                    "impact": "HIGH",
                    "currency": "USD",
                    "forecast": "--",
                    "previous": "--",
                })
            
            # ISM Manufacturing PMI — 1st business day
            if day <= 5:
                events.append({
                    "time": f"{today}T10:00",
                    "event": "ISM Manufacturing PMI",
                    "impact": "HIGH",
                    "currency": "USD",
                    "forecast": "--",
                    "previous": "--",
                })
            
            # Find nearest upcoming event
            upcoming = None
            for evt in events:
                try:
                    evt_dt = datetime.fromisoformat(evt["time"])
                    if evt_dt > now:
                        time_diff = (evt_dt - now).total_seconds()
                        if upcoming is None or time_diff < (datetime.fromisoformat(upcoming["time"]) - now).total_seconds():
                            upcoming = evt
                except (ValueError, TypeError):
                    continue
            
            if upcoming:
                try:
                    evt_dt = datetime.fromisoformat(upcoming["time"])
                    upcoming["time_display"] = evt_dt.strftime("%H:%M")
                    upcoming["countdown_min"] = int((evt_dt - now).total_seconds() / 60)
                except (ValueError, TypeError):
                    upcoming["time_display"] = "--:--"
                    upcoming["countdown_min"] = 0
                return upcoming
            
            # Fallback: return a generic next-session event
            sessions = [
                ("Sydney Open", 22, 0, "AUD"),
                ("Tokyo Open", 23, 0, "JPY"),
                ("London Open", 7, 0, "GBP"),
                ("New York Open", 12, 30, "USD"),
            ]
            for name, evt_hour, evt_min, _ in sessions:
                evt_dt = now.replace(hour=evt_hour, minute=evt_min, second=0, microsecond=0)
                # If already past, move to next day
                if evt_dt <= now:
                    evt_dt = evt_dt + timedelta(days=1)
                if upcoming is None or evt_dt < datetime.fromisoformat(upcoming["time"]):
                    upcoming = {
                        "time": evt_dt.isoformat(),
                        "time_display": evt_dt.strftime("%H:%M"),
                        "countdown_min": int((evt_dt - now).total_seconds() / 60),
                        "event": f"{name} Market Opens",
                        "impact": "MEDIUM",
                        "currency": "--",
                        "forecast": "--",
                        "previous": "--",
                    }
            
            return upcoming
        except Exception as exc:
            self.logger.debug(f"Economic calendar error: {exc}")
            return None

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
        uptime_sec = (datetime.now() - self._start_time).total_seconds()
        uptime_str = f"{int(uptime_sec//3600):02d}:{int((uptime_sec%3600)//60):02d}:{int(uptime_sec%60):02d}"
        # ── Equity history for web dashboard sparkline ──
        current_equity_val = self._account_info.get("equity", 0) or self._account_info.get("balance", 0)
        self._equity_history.append(current_equity_val)
        if len(self._equity_history) > 200:
            self._equity_history = self._equity_history[-200:]

        # ── Track month-start balance (persisted across restarts) ──
        try:
            _ms_path = Path("learning/trade_history/month_start.json")
            _now = datetime.now()
            _current_month = _now.month
            _current_year = _now.year
            _current_bal = self._account_info.get("balance", 0)
            if _ms_path.is_file():
                ms_data = json.loads(_ms_path.read_text())
                if ms_data.get("month") == _current_month and ms_data.get("year") == _current_year:
                    _month_start_balance = ms_data.get("balance", _current_bal)
                else:
                    # New month — save current balance as start
                    _month_start_balance = _current_bal
                    _ms_path.write_text(json.dumps({"month": _current_month, "year": _current_year, "balance": _current_bal}))
            else:
                _month_start_balance = _current_bal
                _ms_path.parent.mkdir(parents=True, exist_ok=True)
                _ms_path.write_text(json.dumps({"month": _current_month, "year": _current_year, "balance": _current_bal}))
        except Exception:
            _month_start_balance = self._account_info.get("balance", 0)

        state = {
            "analysis_ready": has_analysis,
            "uptime": uptime_str,
            "balance": self._account_info.get("balance", 0),
            "equity": current_equity_val,
            "month_start_balance": _month_start_balance,
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

        readiness = estimate_real_readiness(
            trades=self.trade_logger.get_closed_trades(),
            current_dd=acct_status.get("current_drawdown", 0),
        )
        state["real_readiness_score"] = readiness["score"]
        state["real_readiness_eta"] = readiness["eta_str"]
        state["real_readiness_bar"] = readiness["bar"]
        state["real_readiness_detail"] = readiness["detail"]

        # ── Economic calendar: upcoming high-impact events ──
        state["_news"] = self._get_economic_calendar()

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

            # ── Per-TF predictions for dashboard ──
            multi_tf = analysis.get("multi_tf_trends", {})
            state["tf_predictions"] = {}
            tf_label_map = {15: "M15", 30: "M30", 60: "H1", 240: "H4"}
            for trend_col, tf_label in tf_label_map.items():
                trend_dir = multi_tf.get(f"trend{trend_col}", 0)
                tf_score = state["tf_scores"].get(tf_label, 50)
                # Derive prediction from trend direction + market score
                if trend_dir > 0:
                    pred_dir = "UP"
                    pred_conf = min(95, max(55, tf_score / 100 * 70 + 30))
                elif trend_dir < 0:
                    pred_dir = "DOWN"
                    pred_conf = min(95, max(55, abs(tf_score) / 100 * 70 + 30))
                else:
                    pred_dir = "SIDEWAYS"
                    pred_conf = 50
                state["tf_predictions"][tf_label] = {
                    "direction": pred_dir,
                    "confidence": round(pred_conf, 1),
                    "score": round(tf_score, 1),
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

        state["last_signals"] = [
            {
                "direction": t.get("direction", "N/A"),
                "symbol": t.get("symbol", "N/A"),
                "profit": t.get("profit", 0),
                "exit_time": str(t.get("exit_time", ""))[:16],
            }
            for t in self.trade_logger.get_closed_trades()[-5:]
        ]

        # Equity curve for web dashboard sparkline (send as compact list)
        state["equity_curve"] = self._equity_history[-100:]  # max 100 points

        # ── Web dashboard: broadcast full state + cache candles ──
        state["mode"] = config.account.get("trading_mode", "simulation")
        open_positions = self.position_manager.get_open_positions()
        state["open_positions_detail"] = [
            {
                "ticket": p.get("ticket", 0),
                "type": p.get("type", ""),
                "volume": p.get("volume", 0),
                "price_open": p.get("price_open", 0),
                "profit": p.get("profit", 0),
                "sl": p.get("sl", 0),
                "tp": p.get("tp", 0),
                "symbol": p.get("symbol", ""),
            }
            for p in open_positions
        ]
        
        # Ensemble v2 stats for streaming overlay
        if self.ensemble_integration is not None and config.ensemble.get("enabled", False):
            ens_stats = self.ensemble_integration.get_stats()
            state["ensemble_mode"] = True
            state["ensemble_h4_acc"] = ens_stats.get("h4_acc", 0)
            state["ensemble_h1_acc"] = ens_stats.get("h1_acc", 0)
            state["ensemble_m5_acc"] = ens_stats.get("m5_acc", 0)
            state["ensemble_daily_trades"] = ens_stats.get("daily_trades", 0)
        else:
            state["ensemble_mode"] = False
        
        # ── Dashboard streaming data (win rate, profit factor, monthly target) ──
        closed_trades = self.trade_logger.get_closed_trades()
        bot_trades = [t for t in closed_trades if t.get("exit_reason") != "MT5"]
        if bot_trades:
            bot_wins = sum(1 for t in bot_trades if t.get("profit", 0) > 0)
            bot_wr = min(100, max(0, bot_wins / len(bot_trades) * 100))
            state["win_rate"] = bot_wr
            state["total_trades"] = len(bot_trades)
            bot_gross_profit = sum(t.get("profit", 0) for t in bot_trades if t.get("profit", 0) > 0)
            bot_gross_loss = abs(sum(t.get("profit", 0) for t in bot_trades if t.get("profit", 0) < 0))
            state["profit_factor"] = round(bot_gross_profit / bot_gross_loss, 2) if bot_gross_loss > 0 else 0.0
        else:
            state["win_rate"] = 0
            state["total_trades"] = 0
            state["profit_factor"] = 0.0
        
        state["monthly_target_pct"] = config.ensemble.get("monthly_target_pct", 10)
        
        # Win streak
        streak = 0
        for t in reversed(bot_trades if bot_trades else []):
            if t.get("profit", 0) > 0:
                streak += 1
            elif t.get("profit", 0) < 0:
                break
        state["win_streak"] = streak
        
        update_full_state(state)

        # Cache M5 candles for chart (every 5th call to reduce MT5 load)
        try:
            if not hasattr(self, '_candle_cache_counter'):
                self._candle_cache_counter = 0
            self._candle_cache_counter += 1
            if self._candle_cache_counter >= 5:
                self._candle_cache_counter = 0
                for sym in config.trading.get("pairs", ["EURUSD"]):
                    m5_df = self.data_engine.get_rates(sym, 5, count=100, use_cache=False)
                    if m5_df is not None and not m5_df.empty:
                        candles_list = []
                        for _, row in m5_df.iterrows():
                            if "time" in m5_df.columns:
                                ts = int(row["time"].timestamp())
                            else:
                                continue
                            candles_list.append({
                                "time": ts,
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                            })
                        if candles_list:
                            update_candles(sym, candles_list)
        except Exception as exc:
            self.logger.debug(f"Web dashboard candle cache: {exc}")

        if self._tiktok_mode:
            self.tiktok_dashboard.set_win_streak(state.get("win_streak", 0))
            self.tiktok_dashboard.update(state)

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


async def cmd_simulate(bot: ForexBot, args: argparse.Namespace):
    from simulation.simulator import Simulator
    from learning.trade_outcome_trainer import TradeOutcomeTrainer
    from simulation.dashboard_panel import render_simulation_panel

    symbol = args.pair or bot._symbols[0]
    days = args.days or 180
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    bot.logger.info("=" * 50)
    bot.logger.info(f"SELF-LEARN SIMULATION: {symbol} ({days} days)")
    bot.logger.info("=" * 50)

    # Enable simulation mode for relaxed filters
    if bot.decision_engine is not None:
        bot.decision_engine.set_simulation_mode(True)
        bot.logger.info("DecisionEngine set to SIMULATION MODE")

    sim = Simulator(
        symbol=symbol,
        from_date=start_date,
        to_date=end_date,
        initial_balance=10000,
        volume_fixed=0.01,
        pair_suffix="",
        decision_engine=bot.decision_engine if hasattr(bot, 'decision_engine') else None,
    )
    result = await sim.run()

    trades = result.get("trades", [])
    print(f"\nSimulation: {symbol} | {start_date.date()} to {end_date.date()}")
    print(f"  Candles processed: {result.get('total_candles_processed', 0)}")
    print(f"  Trades: {result.get('total_trades', 0)} ({result.get('trades_opened', 0)} opened)")
    print(f"  Win Rate: {result.get('stats', {}).get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor: {result.get('stats', {}).get('profit_factor', 0):.2f}")
    print(f"  Sharpe: {result.get('stats', {}).get('sharpe_ratio', 0):.2f}")
    print(f"  Max DD: {result.get('stats', {}).get('max_drawdown_pct', 0)*100:.1f}%")
    print(f"  Net Profit: ${result.get('stats', {}).get('net_profit', 0):.2f}")
    print(f"  Skill: {result.get('learning', {}).get('skill', {}).get('level', 'N/A')} "
          f"({result.get('learning', {}).get('skill', {}).get('score', 0)}/100)")
    print(f"  Readiness: {result.get('learning', {}).get('real_account_readiness', 'N/A')}")

    if trades:
        by_side = {}
        for t in trades:
            s = t.get("side", "")
            by_side.setdefault(s, {"total": 0, "wins": 0})
            by_side[s]["total"] += 1
            if t.get("net_pnl", 0) > 0:
                by_side[s]["wins"] += 1
        for s, d in by_side.items():
            wr = d["wins"] / d["total"] * 100 if d["total"] > 0 else 0
            print(f"  {s}: {d['total']} trades, WR={wr:.1f}%")

    sim_trades = result.get("trades", [])
    sim_trades_with_feat = sum(1 for t in sim_trades if t.get("feature_vector"))
    print(f"\n  Trades with feature vector: {sim_trades_with_feat}/{len(sim_trades)}")

    # Export trade details for analysis
    export_path = f"data/sim_trades_{symbol}_{days}d.json"
    try:
        import json
        class DateEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                return str(obj)
        export_trades = []
        for t in sim_trades:
            et = {k: v for k, v in t.items() if k != "feature_vector"}
            export_trades.append(et)
        with open(export_path, "w") as f:
            json.dump(export_trades, f, cls=DateEncoder, indent=2)
        print(f"\n  Trade details exported to {export_path}")
    except Exception as e:
        print(f"  Export warning: {e}")

    print("\n" + render_simulation_panel(result))
    bot._simulation_result = result
    await bot.shutdown(reason="Simulation complete")


async def cmd_self_learn(bot: ForexBot, args: argparse.Namespace):
    from simulation.simulator import Simulator
    from learning.trade_outcome_trainer import TradeOutcomeTrainer
    from ml.trainer import ModelTrainer
    from ml.ensemble import VotingEnsemble

    symbol = args.pair or bot._symbols[0]
    days = args.days or 180
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    bot.logger.info("=" * 50)
    bot.logger.info(f"SELF-LEARN FULL PIPELINE: {symbol} ({days} days)")
    bot.logger.info("=" * 50)

    if args.fast_mode:
        bot.logger.info("FAST MODE enabled: skipping LSTM, using single TF ensemble")
        if hasattr(bot, 'decision_engine') and bot.decision_engine is not None:
            from ml.predictor import MLPredictor
            old_ensembles = bot.decision_engine.ml_predictor._ensembles
            m5_ensemble = old_ensembles.get(5) or list(old_ensembles.values())[0]
            m5_models = {k: v for k, v in m5_ensemble.models.items() if k != "lstm"}
            if len(m5_models) < len(m5_ensemble.models):
                bot.logger.info(f"  Removed LSTM from ensemble, keeping: {list(m5_models.keys())}")
            m5_ensemble.models = m5_models
            m5_ensemble.weights = {k: m5_ensemble.weights.get(k, 1.0) for k in m5_models}
            fast_ensembles = {5: m5_ensemble}
            bot.decision_engine.ml_predictor = MLPredictor(fast_ensembles)
            bot.logger.info(f"  Fast MLPredictor: 1 ensemble (M5 only)")

    sim_cache = Path(f"data/sim_cache_{symbol}.json")
    if sim_cache.exists() and getattr(args, 'resume', False):
        bot.logger.info(f"Resuming from cached simulation data ({sim_cache})")
        with open(sim_cache) as f:
            sim_result = json.loads(f.read())
        sim_trades = sim_result.get("trades", [])
    else:
        bot.logger.info("Phase 1/3: Running simulation...")
        # Enable simulation mode for relaxed filters
        if bot.decision_engine is not None:
            bot.decision_engine.set_simulation_mode(True)
            bot.logger.info("DecisionEngine set to SIMULATION MODE")
        sim = Simulator(
            symbol=symbol,
            from_date=start_date,
            to_date=end_date,
            initial_balance=10000,
            volume_fixed=0.01,
            pair_suffix="",
            decision_engine=bot.decision_engine if hasattr(bot, 'decision_engine') else None,
        )
        sim_result = await sim.run()
        sim_trades = sim_result.get("trades", [])
        feat_count = sum(1 for t in sim_trades if t.get("feature_vector"))
        bot.logger.info(f"Simulation: {len(sim_trades)} trades ({feat_count} with feature vectors)")

        if not sim_trades or feat_count == 0:
            bot.logger.warning("No trades or feature vectors from simulation, aborting")
            await bot.shutdown(reason="Self-learn: no sim trades")
            return

        # Cache sim result so Phase 3 can be retried without re-running simulation
        try:
            sim_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(sim_cache, 'w') as f:
                f.write(json.dumps(sim_result, default=str))
            bot.logger.info(f"Simulation data cached to {sim_cache}")
        except Exception as e:
            bot.logger.warning(f"Could not cache simulation data: {e}")

    bot.logger.info("Phase 2/3: Converting simulation trades to training data...")
    tot = TradeOutcomeTrainer()

    bot.logger.info("Phase 3/3: Retraining models with sim data...")
    trained_timeframes = bot.model_manager.get_trained_timeframes() or [
        bot._tf_to_minutes(tf) for tf in config.trading["timeframes"]
    ]
    for tf in trained_timeframes:
        tf_label = Timeframe.LABELS.get(tf, str(tf))
        bot.logger.info(f"  {tf_label}: loading data...")
        df = bot.data_engine.get_historical_data(symbol, tf, years=1)
        if df.empty:
            bot.logger.warning(f"  {tf_label}: no data, skipping")
            continue
        try:
            bot.model_trainer.ensemble = VotingEnsemble()
            X, y, features, df_clean = bot.model_trainer.prepare_training_data(
                df, pair=symbol, timeframe=tf,
            )
        except Exception as e:
            bot.logger.warning(f"  {tf_label}: prepare failed: {e}")
            continue

        # Convert simulation trades using the SAME feature set as the training data
        X_sim, y_sim = tot.convert_from_simulation(sim_result, features, pair=symbol)
        bot.logger.info(f"  {tf_label}: {len(y_sim)} sim samples "
                        f"(BUY={(y_sim==0).sum()} SELL={(y_sim==1).sum()} HOLD={(y_sim==2).sum()})")

        recency = ModelTrainer.compute_recency_weights(df_clean["time"]) if "time" in df_clean.columns else None
        X, y, sw = tot.merge_with_ohlc(X, y, X_sim, y_sim, upsample_wins=True, win_weight=2.0)

        # Extend recency weights to match merged length (sim trades get weight 1.0)
        if recency is not None and len(recency) < len(X):
            pad_len = len(X) - len(recency)
            recency = np.hstack([recency, np.ones(pad_len, dtype=recency.dtype)])

        bot.logger.info(f"  {tf_label}: {len(X)} samples — training...")
        results = bot.model_trainer.train_all_models(
            X, y, feature_cols=features, recency_weights=recency,
            trade_outcome_weights=sw, tf_label=tf_label,
        )
        if bot.model_trainer.get_ensemble().get_num_models() > 0:
            version = bot.model_manager.save_ensemble(
                bot.model_trainer.get_ensemble(), timeframe=tf,
                source="self_learn",
            )
            # Save feature importance for this TF (used for pruning next retrain)
            bot.model_trainer.save_feature_importance(
                bot.model_trainer.get_ensemble(), features, symbol, tf, version
            )
            bot.logger.info(f"  {tf_label}: model saved as v{version}")
            
            # ── Run OOS validation immediately so the model has a score ──
            try:
                oos_result = bot.oos_validator.validate(
                    df=df, ensemble=bot.model_trainer.get_ensemble(),
                    trainer=bot.model_trainer,
                    timeframe_label=tf_label, oos_split=0.2,
                    timeframe=tf,
                )
                bot.model_manager.save_oos_result(version, oos_result)
                wr = oos_result.get("win_rate", 0)
                pf = oos_result.get("profit_factor", 0)
                grade = oos_result.get("grade", "N/A")
                bot.logger.info(f"  {tf_label}: OOS validation — WR={wr:.1f}% PF={pf:.2f} Grade={grade}")
            except Exception as e:
                bot.logger.warning(f"  {tf_label}: OOS validation failed — {e}")
            
            # Save performance data
            perf_acc = {}
            for m_name in ["xgboost", "random_forest", "lightgbm", "lstm"]:
                m_result = results.get("models", {}).get(m_name, {}) or {}
                if isinstance(m_result, dict):
                    perf_acc[m_name] = m_result.get("train_accuracy", 0) or 0
                    perf_acc[f"{m_name}_val"] = m_result.get("val_accuracy", 0) or 0
            bot.model_manager.save_performance(version, {"accuracy": perf_acc})

    bot.logger.info("Self-learn complete!")
    await bot.shutdown(reason="Self-learn complete")


async def cmd_backtest(bot: ForexBot, args: argparse.Namespace):
    bot.logger.info("=" * 50)
    bot.logger.info("BACKTEST MODE (Multi-TF)")
    bot.logger.info("=" * 50)

    symbol = args.pair or bot._symbols[0]
    days = args.days or config.training.get("historical_years", 2) * 365
    from_storage = getattr(args, "from_storage", False)

    bot.logger.info(f"Running backtest on {symbol} (days={days}, from_storage={from_storage})")

    from backtest.backtest_engine import BacktestEngine

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
        decision_engine=bot.decision_engine if hasattr(bot, "decision_engine") else None,
        from_storage=from_storage,
    )

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    result = bt.run_backtest(symbol, Timeframe.M5, start_date, end_date)

    trades = result.get("trades", [])
    if trades:
        print(f"\n=== Backtest Results ({symbol}) ===")
        print(f"  Period:        {start_date.date()} to {end_date.date()}")
        print(f"  Trades:        {len(trades)}")
        print(f"  Win Rate:      {result.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor: {result.get('profit_factor', 0):.2f}")
        print(f"  Sharpe Ratio:  {result.get('sharpe_ratio', 0):.2f}")
        print(f"  Max DD:        {result.get('max_drawdown', 0):.1f}%")
        print(f"  Expectancy:    {result.get('expectancy', 0):.4f}")
        print(f"  Net Profit:    ${result.get('net_profit', 0):.2f}")
        print(f"  Total Return:  {result.get('total_return', 0):.1f}%")
        print(f"  Final Balance: ${result.get('final_balance', 0):.2f}")

        tp_trades = [t for t in trades if t.get("exit_reason") == "take_profit"]
        sl_trades = [t for t in trades if t.get("exit_reason") == "stop_loss"]
        time_trades = [t for t in trades if t.get("exit_reason") == "time_exit"]
        print(f"\n  Exit breakdown:")
        print(f"    Take Profit: {len(tp_trades)}")
        print(f"    Stop Loss:   {len(sl_trades)}")
        print(f"    Time Exit:   {len(time_trades)}")
        print(f"    End of BT:   {len([t for t in trades if t.get('exit_reason') == 'end_of_backtest'])}")
    else:
        print(f"\nNo trades generated in backtest for {symbol}.")

    await bot.shutdown(reason="Backtest complete")


async def main():
    parser = argparse.ArgumentParser(description="AI Forex Trading Bot v2")
    parser.add_argument("command", nargs="?", default="live",
                        choices=["live", "train", "backtest", "validate", "promote", "rollback", "status", "download", "simulate", "self-learn"],
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
    parser.add_argument("--tiktok", action="store_true", help="TikTok mode: simplified dashboard + sound effects")
    parser.add_argument("--fast-mode", action="store_true", help="Fast mode: skip LSTM, use single TF ensemble")
    parser.add_argument("--resume", action="store_true", help="Resume self-learn from cached simulation data (skip Phase 1)")

    args = parser.parse_args()

    bot = ForexBot()
    try:
        if args.tiktok:
            bot._tiktok_mode = True
            sound_utils.set_enabled(True)
            bot.dashboard.hide()
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
        elif cmd == "simulate":
            await cmd_simulate(bot, args)
        elif cmd == "self-learn":
            await cmd_self_learn(bot, args)
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
        except Exception as shutdown_e:
            logger.error(f"Shutdown after fatal error also failed: {shutdown_e}")
    else:
        await bot.shutdown(reason="Normal exit")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot terminated by user")
