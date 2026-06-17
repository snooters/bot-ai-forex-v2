"""OOS validation for v2 self-learn models (50 features for all TFs)"""
import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from core.config import config
from data.data_loader import DataLoader
from data.data_storage import ParquetStorage
from ml.model_manager import ModelManager
from ml.trainer import ModelTrainer
from learning.oos_validator import OOSValidator

models = [
    (5, "v55_M5"), (15, "v30_M15"), (30, "v30_M30"),
    (60, "v24_H1"), (240, "v19_H4"),
]

mm = ModelManager()
for tf_min, ver in models:
    label = {5:"M5",15:"M15",30:"M30",60:"H1",240:"H4"}[tf_min]
    print(f"{ver} ({label})...", end=" ", flush=True)
    if tf_min == 5:
        df = DataLoader("EURUSD").load_aligned()
    else:
        df = ParquetStorage().load_data("EURUSD", tf_min)
    if df is None or df.empty:
        print("NO DATA")
        continue
    ensemble = mm.load_ensemble(ver)
    if ensemble is None or ensemble.get_num_models() == 0:
        print("NO MODEL")
        continue
    oos = OOSValidator().validate(df, ensemble, ModelTrainer(), label,
        oos_split=0.2,
        buy_threshold=config.training["buy_threshold"],
        sell_threshold=config.training["sell_threshold"],
        timeframe=tf_min)
    mm.save_oos_result(ver, oos)
    score = mm._compute_oos_numeric_score(oos)
    print(f"Score={score:.1f} WR={oos.get('win_rate',0):.1f}% PF={oos.get('profit_factor',0):.2f} Grade={oos.get('grade','N/A')}")

print("\n=== BEST VERSIONS ===")
for tf, label in [(5,"M5"),(15,"M15"),(30,"M30"),(60,"H1"),(240,"H4")]:
    best = mm.get_best_version(tf)
    oos = mm.get_oos_result(best) if best else {}
    score = mm._compute_oos_numeric_score(oos) if oos else 0
    print(f"  {label}: {best} (score={score:.1f})")
