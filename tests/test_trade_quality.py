from learning.trade_quality import TradeQualityScorer


class TestTradeQualityScorer:
    def setup_method(self):
        self.scorer = TradeQualityScorer()

    def test_high_quality_trade(self, sample_trade_buy_win):
        score = self.scorer.score_trade(sample_trade_buy_win)
        assert 60 <= score <= 100

    def test_low_quality_trade(self, sample_trade_buy_loss):
        score = self.scorer.score_trade(sample_trade_buy_loss)
        assert score <= 50

    def test_win_trade_scores_higher_than_loss(self, sample_trade_buy_win, sample_trade_buy_loss):
        win_score = self.scorer.score_trade(sample_trade_buy_win)
        loss_score = self.scorer.score_trade(sample_trade_buy_loss)
        assert win_score > loss_score

    def test_entry_quality_high_confidence(self, sample_trade_buy_win):
        score = self.scorer._score_entry_quality(sample_trade_buy_win)
        assert score >= 30

    def test_exit_quality_tp_hit(self, sample_trade_buy_win):
        score = self.scorer._score_exit_quality(sample_trade_buy_win)
        assert score >= 25

    def test_exit_quality_sl_hit(self, sample_trade_buy_loss):
        score = self.scorer._score_exit_quality(sample_trade_buy_loss)
        assert score <= 10

    def test_score_range(self, sample_trade_buy_win):
        for _ in range(20):
            score = self.scorer.score_trade(sample_trade_buy_win)
            assert 0 <= score <= 100

    def test_empty_trade_scores_minimal(self):
        score = self.scorer.score_trade({})
        assert score <= 10
