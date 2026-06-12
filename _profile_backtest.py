"""Profile what takes time in backtest per-bar analysis."""
import sys, time
sys.path.insert(0, ".")
from datetime import datetime, timedelta
from data.market_data_engine import MarketDataEngine
from features.feature_pipeline import FeaturePipeline
from intelligence.trend_analysis import TrendAnalyzer
from intelligence.volatility_analysis import VolatilityAnalyzer
from intelligence.momentum_analysis import MomentumAnalyzer
from intelligence.market_regime import MarketRegimeDetector
from intelligence.market_scorer import MarketScorer
from ml.model_manager import ModelManager
from ml.predictor import MLPredictor

print("Loading data...")
data_engine = MarketDataEngine()
df = data_engine.storage.load_data("EURUSD", 5)
df = df.tail(500).reset_index(drop=True)
print(f"Data: {len(df)} rows")

feat_pipe = FeaturePipeline()
trend = TrendAnalyzer()
vol = VolatilityAnalyzer()
mom = MomentumAnalyzer()
regime = MarketRegimeDetector()
scorer = MarketScorer()

model_manager = ModelManager()
try:
    ensemble = model_manager.load_production(5)
    ml = MLPredictor(ensemble)
except Exception:
    ml = MLPredictor({})

# Time feature computation
print("\nTiming feature computation...")
t0 = time.time()
df_feat = feat_pipe.compute_all(df)
t_feat = time.time() - t0
print(f"  Feature compute_all: {t_feat:.2f}s")

# Time individual analyses
n = len(df_feat)
for _ in range(3):
    idx = n - 1
    window = df_feat.iloc[max(0, idx-499):idx+1]

    t0 = time.time()
    r1 = trend.analyze_trend(window)
    t1 = time.time()

    r2 = vol.analyze_volatility(window)
    t2 = time.time()

    r3 = mom.analyze_momentum(window)
    t3 = time.time()

    r4 = regime.detect_regime(r1, r2, r3, window)
    t4 = time.time()

    r5 = feat_pipe.support_resistance.detect_levels(window)
    t5 = time.time()

    r6 = feat_pipe.compute_features_summary(window)
    t6 = time.time()

    r7 = ml.get_buy_sell_hold(window)
    t7 = time.time()

    print(f"\nPer-bar breakdown:")
    print(f"  analyze_trend:           {t1-t0:.3f}s")
    print(f"  analyze_volatility:      {t2-t1:.3f}s")
    print(f"  analyze_momentum:        {t3-t2:.3f}s")
    print(f"  detect_regime:           {t4-t3:.3f}s")
    print(f"  detect_levels (S/R):     {t5-t4:.3f}s")
    print(f"  compute_features_summary:{t6-t5:.3f}s")
    print(f"  ml_predictor:            {t7-t6:.3f}s")
    print(f"  TOTAL:                   {t7-t0:.3f}s")

t_total = t7 - t0
bars_per_second = 1 / t_total if t_total > 0 else 0
print(f"\nEstimated: {bars_per_second:.0f} bars/sec => {(3133/bars_per_second)/60:.1f} min for 3133 bars")
