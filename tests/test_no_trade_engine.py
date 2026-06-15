from decision.no_trade_engine import NoTradeEngine


class TestNoTradeEngine:
    def setup_method(self):
        self.engine = NoTradeEngine()

    def test_okay_with_good_confidence_and_score(self):
        sev = self.engine.should_no_trade(confidence=0.75, market_score=70, spread=1)
        assert sev == NoTradeEngine.OK

    def test_quality_gate_low_confidence(self):
        sev = self.engine.should_no_trade(confidence=0.30, market_score=70, spread=1)
        assert sev >= NoTradeEngine.CRITICAL
        assert any("Quality gate" in r for r in self.engine.reasons)

    def test_quality_gate_low_score(self):
        sev = self.engine.should_no_trade(confidence=0.75, market_score=20, spread=1)
        assert sev >= NoTradeEngine.CRITICAL
        assert any("Quality gate" in r for r in self.engine.reasons)

    def test_high_spread_warning(self):
        sev = self.engine.should_no_trade(confidence=0.75, market_score=70, spread=10)
        assert sev >= NoTradeEngine.WARNING
        assert any("Spread" in r for r in self.engine.reasons)

    def test_news_driven_critical(self, sample_regime_news_driven):
        sev = self.engine.should_no_trade(
            confidence=0.75, market_score=70, spread=1,
            regime_result=sample_regime_news_driven,
        )
        assert sev == NoTradeEngine.CRITICAL
        assert any("News-driven" in r for r in self.engine.reasons)

    def test_sideways_regime_warning(self):
        sev = self.engine.should_no_trade(
            confidence=0.75, market_score=70, spread=1,
            regime_result={"regime": "SIDEWAYS", "confidence": 0.6, "volatility_score": 25},
            signal="BUY",
        )
        assert sev >= NoTradeEngine.WARNING
        assert any("SIDEWAYS" in r for r in self.engine.reasons)

    def test_regime_signal_mismatch_critical(self):
        sev = self.engine.should_no_trade(
            confidence=0.75, market_score=70, spread=1,
            regime_result={"regime": "STRONG_TRENDING_BULLISH", "confidence": 0.8, "volatility_score": 50},
            signal="SELL",
        )
        assert sev == NoTradeEngine.CRITICAL
        assert any("SELL signal in STRONG_TRENDING_BULLISH" in r for r in self.engine.reasons)

    def test_ranging_regime_critical(self):
        sev = self.engine.should_no_trade(
            confidence=0.75, market_score=70, spread=1,
            regime_result={"regime": "RANGING", "confidence": 0.35, "volatility_score": 20},
            signal="BUY",
        )
        assert sev == NoTradeEngine.CRITICAL
        assert any("RANGING" in r for r in self.engine.reasons)

    def test_no_warning_without_signal(self):
        sev = self.engine.should_no_trade(
            confidence=0.75, market_score=70, spread=1,
            regime_result={"regime": "SIDEWAYS", "confidence": 0.6, "volatility_score": 25},
        )
        assert sev == NoTradeEngine.OK

    def test_high_volatility_critical(self):
        sev = self.engine.should_no_trade(
            confidence=0.65, market_score=70, spread=1,
            regime_result={"regime": "HIGH_VOLATILITY", "confidence": 0.7, "volatility_score": 85},
        )
        assert sev == NoTradeEngine.CRITICAL
        assert any("volatility" in r for r in self.engine.reasons)

    def test_max_positions_critical(self):
        sev = self.engine.should_no_trade(
            confidence=0.75, market_score=70, spread=1,
            existing_positions=[{"ticket": 1}, {"ticket": 2}, {"ticket": 3}],
            balance=10000,
        )
        assert sev == NoTradeEngine.CRITICAL
        assert any("Max positions" in r for r in self.engine.reasons)
