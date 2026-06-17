"""Final OOS validation of all new models"""
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
    (5, "v54_M5"), (5, "v53_M5"), (5, "v52_M5"), (5, "v51_M5"),
    (15, "v29_M15"), (30, "v29_M30"), (60, "v23_H1"), (240, "v18_H4"),
]

mm = ModelManager()
for tf_min, ver in models:
    label = {5:"M5",15:"M15",30:"M30",60:"H1",240:"H4"}[tf_min]
    print(f"Validating {ver} ({label})...", end=" ", flush=True)
    
    if tf_min == 5:
        df = DataLoader("EURUSD").load_aligned()
    else:
        df = ParquetStorage().load_data("EURUSD", tf_min)
    if df is None or df.empty: print("NO DATA"); continue
    
    ensemble = mm.load_ensemble(ver)
    if ensemble is None or ensemble.get_num_models() == 0: print("NO MODEL"); continue
    
    oos = OOSValidator().validate(df, ensemble, ModelTrainer(), label, oos_split=0.2,
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
