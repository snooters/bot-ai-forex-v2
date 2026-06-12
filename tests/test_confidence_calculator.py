from decision.confidence_calculator import ConfidenceCalculator


class TestConfidenceCalculator:
    def setup_method(self):
        self.calc = ConfidenceCalculator()

    def test_confidence_zero_without_signal(self):
        assert self.calc.calculate_confidence(ml_signal=None, market_score=0, trend_result={}, regime_result={}, sr_info={}) == 0.0

    def test_buy_confidence_basic(self, sample_ml_signal, sample_trend_bullish, sample_regime_trending_bullish):
        conf = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal,
            market_score=70,
            trend_result=sample_trend_bullish,
            regime_result=sample_regime_trending_bullish,
            sr_info={},
        )
        assert 0.3 < conf <= 1.0

    def test_skill_score_boost_high(self, sample_ml_signal, sample_trend_bullish, sample_regime_trending_bullish):
        conf_no_skill = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result=sample_trend_bullish, regime_result=sample_regime_trending_bullish,
            sr_info={}, pair_skill_score=50,
        )
        conf_high_skill = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result=sample_trend_bullish, regime_result=sample_regime_trending_bullish,
            sr_info={}, pair_skill_score=100,
        )
        assert conf_high_skill > conf_no_skill

    def test_skill_score_penalty_low(self, sample_ml_signal, sample_trend_bullish, sample_regime_trending_bullish):
        conf_mid = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result=sample_trend_bullish, regime_result=sample_regime_trending_bullish,
            sr_info={}, pair_skill_score=50,
        )
        conf_low = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result=sample_trend_bullish, regime_result=sample_regime_trending_bullish,
            sr_info={}, pair_skill_score=0,
        )
        assert conf_low < conf_mid

    def test_skill_score_multiplier_range(self, sample_ml_signal, sample_trend_bullish, sample_regime_trending_bullish):
        conf_0 = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result=sample_trend_bullish, regime_result=sample_regime_trending_bullish,
            sr_info={}, pair_skill_score=0,
        )
        conf_100 = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result=sample_trend_bullish, regime_result=sample_regime_trending_bullish,
            sr_info={}, pair_skill_score=100,
        )
        ratio = conf_100 / conf_0 if conf_0 > 0 else 0
        assert 1.25 <= ratio <= 1.40

    def test_confidence_clamped(self, sample_ml_signal, sample_trend_bullish, sample_regime_trending_bullish):
        conf = self.calc.calculate_confidence(
            ml_signal={**sample_ml_signal, "confidence": 0.99, "buy_prob": 0.95, "sell_prob": 0.02, "hold_prob": 0.03},
            market_score=100,
            trend_result={"direction": "STRONG_BULLISH", "strength": 0.9, "score": 0.85},
            regime_result=sample_regime_trending_bullish,
            sr_info={},
        )
        assert conf <= 1.0

    def test_confidence_floor(self):
        conf = self.calc.calculate_confidence(
            ml_signal={"signal": "HOLD", "confidence": 0.0, "buy_prob": 0.0, "sell_prob": 0.0, "hold_prob": 1.0},
            market_score=0,
            trend_result={"direction": "SIDEWAYS"},
            regime_result={"regime": "SIDEWAYS", "confidence": 0.5},
            sr_info={},
        )
        assert conf >= 0.0

    def test_regime_penalty_news_driven(self, sample_ml_signal, sample_trend_bullish):
        conf_sideways = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result=sample_trend_bullish,
            regime_result={"regime": "SIDEWAYS", "confidence": 0.5},
            sr_info={},
        )
        conf_news = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result=sample_trend_bullish,
            regime_result={"regime": "NEWS_DRIVEN", "confidence": 0.8},
            sr_info={},
        )
        assert conf_news < conf_sideways

    def test_trend_alignment_boost(self, sample_ml_signal):
        conf_aligned = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result={"direction": "BULLISH", "strength": 0.6, "score": 0.5},
            regime_result={"regime": "WEAK_TRENDING_BULLISH", "confidence": 0.6},
            sr_info={},
        )
        conf_against = self.calc.calculate_confidence(
            ml_signal=sample_ml_signal, market_score=70,
            trend_result={"direction": "BEARISH", "strength": 0.6, "score": 0.5},
            regime_result={"regime": "WEAK_TRENDING_BEARISH", "confidence": 0.6},
            sr_info={},
        )
        assert conf_aligned > conf_against
