from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.config import config
from core.constants import ML_WEIGHT, INTELLIGENCE_WEIGHT, TrendDirection, TradeDirection, Timeframe
from core.exceptions import DecisionError
from decision.confidence_calculator import ConfidenceCalculator
from decision.no_trade_engine import NoTradeEngine
from decision.trend_reversal_detector import TrendReversalDetector
from intelligence.market_scorer import MarketScorer
from learning.trade_memory import TradeMemory
from ml.predictor import MLPredictor
from utils.logger import get_logger


class DecisionEngine:
    def __init__(
        self,
        ml_predictor: MLPredictor,
        market_scorer: MarketScorer,
        trade_memory: Optional[TradeMemory] = None,
        simulation_mode: bool = False,
    ):
        self.logger = get_logger("decision_engine")
        self.ml_predictor = ml_predictor
        self.market_scorer = market_scorer
        self.trade_memory = trade_memory
        self.confidence_calculator = ConfidenceCalculator()
        self.no_trade_engine = NoTradeEngine()
        self._last_decision: Optional[Dict] = None
        self._regime_classifier = None
        self._last_trade_direction: Optional[str] = None
        self._last_trade_time: Optional[datetime] = None
        self.simulation_mode = simulation_mode
        if simulation_mode:
            self.logger.info("DecisionEngine in SIMULATION MODE (relaxed filters)")

    def set_simulation_mode(self, enabled: bool = True):
        """Enable or disable simulation mode (relaxed trading filters)."""
        self.simulation_mode = enabled
        self.logger.info(f"DecisionEngine simulation_mode={enabled}")

    def _init_regime_classifier(self):
        if self._regime_classifier is not None:
            return
        try:
            from analysis.regime_classifier import RegimeClassifier
            self._regime_classifier = RegimeClassifier()
        except Exception as e:
            self.logger.debug(f"RegimeClassifier not available: {e}")

    def _classify_regime(self, df) -> Optional[Dict]:
        self._init_regime_classifier()
        if self._regime_classifier is None:
            return None
        try:
            return self._regime_classifier.classify(df)
        except Exception as e:
            self.logger.warning(f"Regime classification failed: {e}")
            return None

    def make_decision(
        self,
        symbol: str,
        df_entry: Dict,
        trend_result: Dict,
        vol_result: Dict,
        momentum_result: Dict,
        regime_result: Dict,
        sr_info: Dict,
        feature_summary: Dict,
        account_info: Optional[Dict] = None,
        positions: Optional[list] = None,
        news_analysis: Optional[Dict] = None,
        llm_analysis: Optional[Dict] = None,
        spread: float = 0.0,
        timeframe: Optional[int] = None,
        consensus: Optional[Dict] = None,
        reversal_info: Optional[Dict] = None,
        multi_tf_trends: Optional[Dict] = None,
        pair_skill_score: Optional[float] = None,
    ) -> Dict:
        entry_tfs = list(df_entry.keys()) if isinstance(df_entry, dict) else []
        entry_tf = timeframe or (entry_tfs[0] if entry_tfs else Timeframe.M15)
        df = df_entry[entry_tf] if isinstance(df_entry, dict) else df_entry

        decision = {
            "symbol": symbol,
            "action": TradeDirection.HOLD.value,
            "confidence": 0.0,
            "market_score": 0,
            "reasons": [],
            "no_trade": True,
            "no_trade_reasons": [],
            "ml_signal": None,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0,
            "timeframe": entry_tf,
            "reversal_force_close": False,
            "model_version": self.ml_predictor.get_model_version(timeframe=entry_tf) if hasattr(self, 'ml_predictor') and self.ml_predictor else "unknown",
        }

        # ── Check trend reversal first ──
        if reversal_info and reversal_info.get("severity", 0) >= TrendReversalDetector.CRITICAL:
            decision["action"] = TradeDirection.HOLD.value
            decision["no_trade"] = True
            decision["no_trade_reasons"].append(f"Trend reversal: {reversal_info.get('reason', '')}")
            decision["reasons"].append(f"Blocked by reversal: {reversal_info.get('to_trend', '')}")
            decision["reversal_force_close"] = True
            self._last_decision = decision
            return decision

        try:
            # ── Use multi-TF consensus ML signal if available ──
            if consensus:
                ml_signal = {
                    "signal": consensus.get("signal", "HOLD"),
                    "confidence": consensus.get("confidence", 0),
                    "buy_prob": consensus.get("buy_prob", 0),
                    "sell_prob": consensus.get("sell_prob", 0),
                    "hold_prob": consensus.get("hold_prob", 0),
                }
            else:
                ml_signal = self.ml_predictor.get_buy_sell_hold(df, timeframe=entry_tf)
            decision["ml_signal"] = ml_signal

            new_regime = self._classify_regime(df)
            if new_regime is not None and new_regime.get("confidence", 0) > 0.5:
                # Only override caller's regime if classifier has meaningful confidence
                # (>0.5 avoids replacing a pre-classified regime with fallback RANGING)
                regime_result = new_regime
                decision["regime_classifier"] = new_regime.get("regime")

            market_score = self.market_scorer.compute_market_score(
                trend_result, vol_result, momentum_result,
                regime_result, sr_info, feature_summary
            )
            decision["market_score"] = market_score

            confidence = self.confidence_calculator.calculate_confidence(
                ml_signal=ml_signal,
                market_score=market_score,
                trend_result=trend_result,
                regime_result=regime_result,
                sr_info=sr_info,
                news_analysis=news_analysis,
                llm_analysis=llm_analysis,
                pair_skill_score=pair_skill_score,
            )
            decision["confidence"] = confidence

            balance = (account_info or {}).get("balance", 0)
            trend_dir = trend_result.get("direction", "SIDEWAYS")
            ml_signal_dir = ml_signal.get("signal", "HOLD")

            no_trade_severity = self.no_trade_engine.should_no_trade(
                confidence=confidence,
                market_score=market_score,
                spread=spread,
                news_analysis=news_analysis,
                regime_result=regime_result,
                existing_positions=positions,
                balance=balance,
                trend_result=trend_result,
                signal=ml_signal_dir,
            )
            buy_prob = ml_signal.get("buy_prob", 0)
            sell_prob = ml_signal.get("sell_prob", 0)
            hold_prob = ml_signal.get("hold_prob", 0)

            if no_trade_severity >= NoTradeEngine.CRITICAL and not self.simulation_mode:
                decision["no_trade"] = True
                decision["no_trade_reasons"] = self.no_trade_engine.reasons
                decision["action"] = TradeDirection.HOLD.value
                critical_reason = "; ".join(self.no_trade_engine.reasons)
                decision["reasons"].append(f"Blocked: {critical_reason}")
                self._last_decision = decision
                return decision

            memory_check = self._check_trade_memory(
                direction=ml_signal_dir,
                regime=regime_result.get("regime", ""),
                timeframe=Timeframe.LABELS.get(entry_tf, "M15"),
            )
            if memory_check.get("block"):
                if not self.simulation_mode:
                    decision["no_trade"] = True
                    decision["no_trade_reasons"] = [memory_check["reason"]]
                    decision["action"] = TradeDirection.HOLD.value
                    decision["reasons"].append(f"Memory block: {memory_check['reason']}")
                    self._last_decision = decision
                    return decision
                else:
                    decision["reasons"].append(f"Memory block (simulation override): {memory_check['reason']}")

            reduce_size = memory_check.get("reduce_size", False)

            dynamic_min = config.get_dynamic_min_confidence(balance)
            min_conf = max(0.50, dynamic_min)
            # In simulation mode, use much lower threshold to generate more trades
            if self.simulation_mode:
                min_conf = 0.50

            if confidence >= min_conf and confidence < 0.70:
                trade_type = "WEAK_SIGNAL"
            elif confidence >= 0.70:
                trade_type = "STRONG_SIGNAL"
            else:
                trade_type = "NO_SIGNAL"

            if trade_type == "NO_SIGNAL":
                direction_bias = self._get_direction_bias(
                    ml_signal_dir, buy_prob, sell_prob, trend_dir,
                    momentum_result=momentum_result,
                )
                if direction_bias:
                    decision["action"] = f"WEAK_{direction_bias}"
                    decision["no_trade"] = False
                    decision["reasons"].append(
                        f"Weak {direction_bias} bias (confidence={confidence:.1%}, score={market_score})"
                    )
                else:
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append(
                        f"Below threshold: conf={confidence:.1%} < {min_conf:.0%})"
                    )
            else:
                intel_base = market_score / 100.0
                # Dampened boost: reduce bias when market_score is far from 50
                center = 0.5
                dampening = 0.3  # 0 = no market influence, 1 = full market influence
                buy_boost = center + (intel_base - center) * dampening
                sell_boost = center - (intel_base - center) * dampening
                combined_buy = (
                    ml_signal.get("buy_prob", 0) * ML_WEIGHT +
                    buy_boost * INTELLIGENCE_WEIGHT
                )
                combined_sell = (
                    ml_signal.get("sell_prob", 0) * ML_WEIGHT +
                    sell_boost * INTELLIGENCE_WEIGHT
                )

                counter_trend_min_conf = config.ai_filter.get("counter_trade_min_confidence", 0.60)

                # Compensate for model's SELL prediction bias (80% SELL vs 20% BUY).
                # 1.2x means "prefer BUY when buy_prob is within 20% of sell_prob".
                buy_bias = config.ai_filter.get("buy_bias_correction", 1.2)

                if combined_buy * buy_bias > combined_sell:
                    if self._validate_entry(direction="BUY", confidence=confidence,
                                            trend_result=trend_result, sr_info=sr_info, df=df,
                                            multi_tf_trends=multi_tf_trends, reasons=decision["reasons"]):
                        decision["action"] = TradeDirection.BUY.value
                        decision["no_trade"] = False
                        label = "STRONG" if confidence >= 0.70 else "WEAK"
                        decision["reasons"].append(f"{label} BUY (conf={confidence:.0%}, score={market_score})")
                    else:
                        direction_bias = self._get_direction_bias(
                            ml_signal_dir, buy_prob, sell_prob, trend_dir,
                            momentum_result=momentum_result,
                        )
                        decision["action"] = f"WEAK_{direction_bias}" if direction_bias else TradeDirection.HOLD.value
                        decision["no_trade"] = False if direction_bias else True
                        if direction_bias:
                            decision["reasons"].append(f"BUY validation failed, weak {direction_bias} bias")

                elif combined_sell > combined_buy * buy_bias:
                    if self._validate_entry(direction="SELL", confidence=confidence,
                                            trend_result=trend_result, sr_info=sr_info, df=df,
                                            multi_tf_trends=multi_tf_trends, reasons=decision["reasons"]):
                        decision["action"] = TradeDirection.SELL.value
                        decision["no_trade"] = False
                        label = "STRONG" if confidence >= 0.70 else "WEAK"
                        decision["reasons"].append(f"{label} SELL (conf={confidence:.0%}, score={market_score})")
                    else:
                        direction_bias = self._get_direction_bias(
                            ml_signal_dir, buy_prob, sell_prob, trend_dir,
                            momentum_result=momentum_result,
                        )
                        decision["action"] = f"WEAK_{direction_bias}" if direction_bias else TradeDirection.HOLD.value
                        decision["no_trade"] = False if direction_bias else True
                        if direction_bias:
                            decision["reasons"].append(f"SELL validation failed, weak {direction_bias} bias")
                else:
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append("Combined signal equal - HOLD")

            # ── Trend-based direction override (ML-gated) ──
            # In simulation mode, skip trend override to test raw ML performance
            if not decision["no_trade"] and not self.simulation_mode:
                trend_dir = trend_result.get("direction", "")
                current_action = decision.get("action", TradeDirection.HOLD.value)
                ml_buy_prob = ml_signal.get("buy_prob", 0)
                ml_sell_prob = ml_signal.get("sell_prob", 0)

                if trend_dir == TrendDirection.STRONG_BULLISH.value:
                    if current_action in (TradeDirection.HOLD.value, TradeDirection.SELL.value,
                                          TradeDirection.WEAK_SELL.value):
                        # Don't override if ML confidently disagrees
                        if ml_sell_prob < 0.50:
                            decision["action"] = TradeDirection.BUY.value
                            decision["reasons"].append("Trend override: STRONG_BULLISH -> BUY")
                        else:
                            decision["reasons"].append(
                                f"Trend override suppressed: ML sell_prob={ml_sell_prob:.2f}"
                            )
                elif trend_dir == TrendDirection.STRONG_BEARISH.value:
                    if current_action in (TradeDirection.HOLD.value, TradeDirection.BUY.value,
                                          TradeDirection.WEAK_BUY.value):
                        # Don't override if ML confidently disagrees
                        if ml_buy_prob < 0.50:
                            decision["action"] = TradeDirection.SELL.value
                            decision["reasons"].append("Trend override: STRONG_BEARISH -> SELL")
                        else:
                            decision["reasons"].append(
                                f"Trend override suppressed: ML buy_prob={ml_buy_prob:.2f}"
                            )

            # ── RSI + MACD safety override (dynamic thresholds by trend) ──
            # In simulation mode, skip RSI/MACD safety overrides
            if not decision["no_trade"] and not self.simulation_mode:
                ind = feature_summary.get("indicators", {})
                rsi_val = ind.get("rsi", 50)
                macd_val = ind.get("macd", 0)
                macd_sig = ind.get("macd_signal", 0)
                current_action = decision.get("action", TradeDirection.HOLD.value)

                # Dynamic RSI thresholds based on trend direction
                # In strong trends, RSI can stay extended for long periods
                if trend_dir in ("STRONG_BULLISH", "BULLISH"):
                    rsi_overbought = 85  # Tolerate higher RSI in uptrend
                    rsi_oversold = 20
                elif trend_dir in ("STRONG_BEARISH", "BEARISH"):
                    rsi_overbought = 65
                    rsi_oversold = 15  # Tolerate lower RSI in downtrend
                else:
                    rsi_overbought = 70  # Normal thresholds for sideways
                    rsi_oversold = 30

                # Overbought: RSI > threshold → no BUY
                if rsi_val > rsi_overbought and current_action in (TradeDirection.BUY.value, "WEAK_BUY"):
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append(
                        f"Safety: RSI={rsi_val:.1f} > {rsi_overbought} overbought (trend={trend_dir}), no BUY"
                    )
                    decision["reasons"].append(f"RSI overbought safety override (threshold={rsi_overbought})")
                # Oversold: RSI < threshold → no SELL
                elif rsi_val < rsi_oversold and current_action in (TradeDirection.SELL.value, "WEAK_SELL"):
                    decision["action"] = TradeDirection.HOLD.value
                    decision["no_trade"] = True
                    decision["no_trade_reasons"].append(
                        f"Safety: RSI={rsi_val:.1f} < {rsi_oversold} oversold (trend={trend_dir}), no SELL"
                    )
                    decision["reasons"].append(f"RSI oversold safety override (threshold={rsi_oversold})")
                # MACD-confirmed extreme: use dynamic thresholds too
                elif rsi_val < rsi_oversold and macd_val < macd_sig:
                    if current_action in (TradeDirection.BUY.value, "WEAK_BUY"):
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        decision["no_trade_reasons"].append(
                            f"Safety: RSI={rsi_val:.1f} < {rsi_oversold} + MACD bearish, no BUY"
                        )
                        decision["reasons"].append(f"RSI+MACD safety override (oversold threshold={rsi_oversold})")
                elif rsi_val > rsi_overbought and macd_val > macd_sig:
                    if current_action in (TradeDirection.SELL.value, "WEAK_SELL"):
                        decision["action"] = TradeDirection.HOLD.value
                        decision["no_trade"] = True
                        decision["no_trade_reasons"].append(
                            f"Safety: RSI={rsi_val:.1f} > {rsi_overbought} + MACD bullish, no SELL"
                        )
                        decision["reasons"].append(f"RSI+MACD safety override (overbought threshold={rsi_overbought})")

            # ── Multi-TF trend alignment BONUS ──
            # H4, H1, M30, M15 = context TFs
            # This is a BONUS signal, not a blocker.
            # If TFs agree → boost confidence. If they disagree → just note it.
            if multi_tf_trends and not decision["no_trade"]:
                h4 = multi_tf_trends.get("trend240", 0)
                h1 = multi_tf_trends.get("trend60", 0)
                m30 = multi_tf_trends.get("trend30", 0)
                m15 = multi_tf_trends.get("trend15", 0)
                current_action = decision.get("action", TradeDirection.HOLD.value)

                tf_values = [("H4", h4), ("H1", h1), ("M30", m30), ("M15", m15)]

                if current_action in (TradeDirection.BUY.value, "WEAK_BUY"):
                    agree = sum(1 for _, v in tf_values if v > 0)
                    disagree = sum(1 for _, v in tf_values if v < 0)
                    if agree >= 3:
                        decision["reasons"].append(f"MTF bonus: {agree}/4 TFs bullish")
                        decision["confidence"] = min(decision["confidence"] + 0.05, 1.0)
                    elif disagree >= 3:
                        decision["reasons"].append(f"MTF note: {disagree}/4 TFs bearish (reducing confidence)")
                        decision["confidence"] = max(decision["confidence"] - 0.05, 0.0)
                    else:
                        decision["reasons"].append(f"MTF note: {agree} bullish / {disagree} bearish")

                elif current_action in (TradeDirection.SELL.value, "WEAK_SELL"):
                    agree = sum(1 for _, v in tf_values if v < 0)
                    disagree = sum(1 for _, v in tf_values if v > 0)
                    if agree >= 3:
                        decision["reasons"].append(f"MTF bonus: {agree}/4 TFs bearish")
                        decision["confidence"] = min(decision["confidence"] + 0.05, 1.0)
                    elif disagree >= 3:
                        decision["reasons"].append(f"MTF note: {disagree}/4 TFs bullish (reducing confidence)")
                        decision["confidence"] = max(decision["confidence"] - 0.05, 0.0)
                    else:
                        decision["reasons"].append(f"MTF note: {agree} bearish / {disagree} bullish")

            # ── Anti-whipsaw: cegah flip-flop ──
            # Skip in simulation mode to allow more trades for learning
            if not decision["no_trade"] and decision.get("action") in (TradeDirection.BUY.value, TradeDirection.SELL.value) and not self.simulation_mode:
                new_dir = decision["action"]
                opposite_dir = TradeDirection.SELL.value if new_dir == TradeDirection.BUY.value else TradeDirection.BUY.value

                # Cek posisi terbuka
                if positions:
                    for pos in positions:
                        pos_dir = pos.get("type", "")
                        if pos_dir == opposite_dir:
                            decision["action"] = TradeDirection.HOLD.value
                            decision["no_trade"] = True
                            decision["no_trade_reasons"].append(
                                f"Anti-whipsaw: existing {pos_dir} position open, cannot open {new_dir}"
                            )
                            decision["reasons"].append(f"Anti-whipsaw blocked: {new_dir} vs open {pos_dir}")

                # Cek cooldown dari posisi sebelumnya (in-memory)
                if not decision["no_trade"] and self._last_trade_direction == opposite_dir:
                    cooldown_min = 60  # 60 menit cooldown
                    if self._last_trade_time is not None:
                        elapsed = (datetime.now() - self._last_trade_time).total_seconds() / 60.0
                        if elapsed < cooldown_min:
                            decision["action"] = TradeDirection.HOLD.value
                            decision["no_trade"] = True
                            decision["no_trade_reasons"].append(
                                f"Anti-whipsaw: {self._last_trade_direction} was {elapsed:.0f}min ago "
                                f"(cooldown={cooldown_min}min), cannot open {new_dir}"
                            )
                            decision["reasons"].append(
                                f"Anti-whipsaw cooldown: {self._last_trade_direction}→{new_dir} "
                                f"({elapsed:.0f}/{cooldown_min}min)"
                            )

            # ── Reversal WARNING: reduce confidence ──
            if reversal_info and reversal_info.get("severity") == TrendReversalDetector.WARNING and not decision["no_trade"]:
                decision["confidence"] *= 0.7
                decision["reasons"].append(f"Trend warning: {reversal_info.get('reason', '')}")

            if no_trade_severity == NoTradeEngine.WARNING and not decision["no_trade"]:
                decision["reasons"].append(f"Warnings: {'; '.join(self.no_trade_engine.reasons)}")
                decision["no_trade_reasons"] = self.no_trade_engine.reasons

        except Exception as e:
            self.logger.error(f"Decision error: {e}")
            decision["action"] = TradeDirection.HOLD.value
            decision["no_trade"] = True
            decision["reasons"].append(f"Error: {e}")

        # ── Track last trade direction for whipsaw prevention ──
        if not decision["no_trade"] and decision.get("action") in (TradeDirection.BUY.value, TradeDirection.SELL.value):
            self._last_trade_direction = decision["action"]
            # Note: time updated when trade actually opens, not just decision
            # Set here for approximate tracking
            if self._last_trade_time is None:
                self._last_trade_time = datetime.now()

        self._last_decision = decision
        return decision

    def _check_trade_memory(
        self,
        direction: str,
        regime: str,
        timeframe: str,
    ) -> Dict:
        if not self.trade_memory or direction in ("HOLD", ""):
            return {"block": False, "reduce_size": False}

        result = self.trade_memory.find_by_pattern(
            direction=direction,
            regime=regime,
            timeframe=timeframe,
            min_trades=3,
        )

        if result.get("closed", 0) < 3:
            return {"block": False, "reduce_size": False}

        win_rate = result.get("win_rate", 0)
        losses = result.get("losses", 0)

        if win_rate < 0.25 and losses >= 3:
            return {
                "block": True,
                "reduce_size": False,
                "reason": (
                    f"Pattern {direction}/{regime} win rate {win_rate:.0%} "
                    f"({result['closed']} trades, {losses} losses)"
                ),
            }

        if win_rate < 0.40 and losses >= 3:
            return {
                "block": False,
                "reduce_size": True,
                "reason": f"Pattern {direction}/{regime} win rate {win_rate:.0%} — reducing size",
            }

        if result.get("profit_factor", 0) < 0.5 and result.get("closed", 0) >= 5:
            return {
                "block": False,
                "reduce_size": True,
                "reason": f"PF={result['profit_factor']:.2f} — reducing size",
            }

        return {"block": False, "reduce_size": False}

    def _get_direction_bias(
        self,
        ml_signal_dir: str,
        buy_prob: float,
        sell_prob: float,
        trend_dir: str,
        momentum_result: Optional[Dict] = None,
    ) -> Optional[str]:
        if ml_signal_dir in ("BUY", "SELL"):
            return ml_signal_dir
        margin = abs(buy_prob - sell_prob)
        if margin < 0.05:
            # If sell_prob >= 20% and bearish momentum, allow SELL
            if sell_prob >= 0.20 and momentum_result:
                mom_dir = momentum_result.get("direction", 0)
                if mom_dir < 0:
                    return "SELL"
            if buy_prob >= 0.20 and momentum_result:
                mom_dir = momentum_result.get("direction", 0)
                if mom_dir > 0:
                    return "BUY"
            return None
        dominant = "BUY" if buy_prob > sell_prob else "SELL"
        if "BULLISH" in trend_dir and dominant == "BUY":
            return dominant
        if "BEARISH" in trend_dir and dominant == "SELL":
            return dominant
        if margin > 0.10:
            return dominant
        return None

    def _validate_entry(
        self,
        direction: str,
        confidence: float,
        trend_result: Dict,
        sr_info: Dict,
        df,
        multi_tf_trends: Optional[Dict] = None,
        reasons: Optional[List] = None,
    ) -> bool:
        # ── Multi-TF guard: cegah counter-trend entry tanpa confidence tinggi ──
        # Hanya aktif di non-simulation mode
        if multi_tf_trends and not self.simulation_mode:
            h4 = multi_tf_trends.get("trend240", 0)
            h1 = multi_tf_trends.get("trend60", 0)
            threshold = 0.65  # Counter-trend butuh confidence > normal (0.55)

            if direction == "BUY":
                if h4 < 0 and h1 < 0 and confidence < threshold:
                    if reasons is not None:
                        reasons.append(f"Counter-trend BUY blocked: H4({h4}) H1({h1}) bearish, conf={confidence:.2f}")
                    return False
            elif direction == "SELL":
                if h4 > 0 and h1 > 0 and confidence < threshold:
                    if reasons is not None:
                        reasons.append(f"Counter-trend SELL blocked: H4({h4}) H1({h1}) bullish, conf={confidence:.2f}")
                    return False

        if "atr" in df.columns and not df["atr"].empty:
            atr = df["atr"].iloc[-1]
            if atr > 0:
                atr_pct = atr / df["close"].iloc[-1]
                if atr_pct > config.risk.get("max_atr_pct", 0.02):
                    return False

        # ── Candle confirmation: prevent fake breakouts ──
        # Require the last 2 COMPLETED candles to show momentum in signal direction.
        # NOTE: df.iloc[-1] is the CURRENT (possibly incomplete) candle in LIVE mode
        # because MT5 copy_rates_from_pos(start_pos=0) includes the forming candle.
        # We use df.iloc[-3] (signal) and df.iloc[-2] (confirm) which are always completed.
        if len(df) >= 4:
            signal_candle = df.iloc[-3]
            confirm_candle = df.iloc[-2]
            sig_close = float(signal_candle.get("close", 0))
            con_close = float(confirm_candle.get("close", 0))
            con_open = float(confirm_candle.get("open", 0))

            if direction == "BUY":
                # Confirm candle must be bullish AND close higher than signal close
                if con_close <= con_open:
                    return False  # Bearish candle — does not confirm
                if con_close <= sig_close:
                    return False  # No upward momentum
            elif direction == "SELL":
                # Confirm candle must be bearish AND close lower than signal close
                if con_close >= con_open:
                    return False  # Bullish candle — does not confirm
                if con_close >= sig_close:
                    return False  # No downward momentum

        return True

    @property
    def last_decision(self) -> Optional[Dict]:
        return self._last_decision
