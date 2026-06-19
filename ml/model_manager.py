import json
import os
import glob
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import numpy as np

from core.constants import MODEL_DIR, Timeframe
from core.exceptions import ModelNotFoundError, ModelError
from ml.xgboost_model import XGBoostModel
from ml.random_forest_model import RandomForestModel
from ml.lightgbm_model import LightGBMModel
from ml.ensemble import VotingEnsemble
from ml.lstm_model import LSTMModel
from learning.skill_scorer import SkillScorer
from utils.logger import get_logger


PRODUCTION_DIR = "production"
CANDIDATE_DIR = "candidate"
ARCHIVE_DIR = "archive"
PROTECTED_VERSIONS_COUNT = 3  # Top N versions per TF are protected from deletion


class ModelManager:
    def __init__(self):
        self.logger = get_logger("model_manager")
        self._model_dir = Path(MODEL_DIR)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._production_dir = self._model_dir / PRODUCTION_DIR
        self._candidate_dir = self._model_dir / CANDIDATE_DIR
        self._archive_dir = self._model_dir / ARCHIVE_DIR
        for d in [self._production_dir, self._candidate_dir, self._archive_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self._current_version: Optional[str] = None
        self._version_metadata: Dict[str, Dict] = {}

    def _tf_label(self, timeframe: int) -> str:
        return Timeframe.LABELS.get(timeframe, f"tf{timeframe}")

    @staticmethod
    def _version_sort_key(version: str) -> tuple:
        import re
        m = re.match(r"(?:v|prod_)(\d+)", version)
        num = int(m.group(1)) if m else 0
        return (num, version)

    def _tf_timeframe_dir(self, timeframe: int) -> Path:
        label = self._tf_label(timeframe)
        return self._model_dir / label

    def _version_dir(self, version: str) -> Path:
        return self._model_dir / f"model_{version}"

    def save_to_production(self, ensemble: VotingEnsemble, timeframe: int, source: str = "auto_retrain") -> str:
        tf_label = self._tf_label(timeframe)
        tf_dir = self._production_dir / tf_label
        tf_dir.mkdir(parents=True, exist_ok=True)

        version = f"prod_{tf_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metadata = {
            "version": version,
            "type": "production",
            "timeframe": tf_label,
            "timeframe_minutes": timeframe,
            "created_at": datetime.now().isoformat(),
            "source": source,
            "models": {},
        }
        if ensemble.feature_cols:
            metadata["feature_cols"] = ensemble.feature_cols

        for name, model in ensemble.models.items():
            model_path = str(tf_dir / f"{name}.ubj")
            try:
                model.save(model_path)
                metadata["models"][name] = {"path": model_path, "trained": model.is_trained}
            except Exception as e:
                self.logger.error(f"Failed to save {name}: {e}")

        current_path = tf_dir / "current.txt"
        try:
            current_path.write_text(version)
        except Exception as e:
            self.logger.warning(f"Failed to write current.txt: {e}")

        self.logger.info(f"Saved production model {version} for {tf_label}")
        return version

    def load_production(self, timeframe: int) -> VotingEnsemble:
        tf_label = self._tf_label(timeframe)
        tf_dir = self._production_dir / tf_label
        if not tf_dir.exists():
            raise ModelNotFoundError(f"No production model for {tf_label}")

        version = self._get_current_production_version(timeframe)
        if version:
            model_files = list(tf_dir.glob("*.ubj")) + list(tf_dir.glob("*.model")) + list(tf_dir.glob("*.pkl"))
            if model_files:
                return self._load_models_from_dir(tf_dir)

        raise ModelNotFoundError(f"No production model files for {tf_label}")

    def _get_current_production_version(self, timeframe: int) -> Optional[str]:
        tf_label = self._tf_label(timeframe)
        current_path = self._production_dir / tf_label / "current.txt"
        if current_path.exists():
            try:
                return current_path.read_text().strip()
            except Exception:
                pass
        return None

    def save_to_candidate(self, ensemble: VotingEnsemble, timeframe: int, source_version: str = "", source: str = "auto_retrain") -> str:
        tf_label = self._tf_label(timeframe)
        tf_dir = self._candidate_dir / tf_label
        tf_dir.mkdir(parents=True, exist_ok=True)

        version = f"cand_{tf_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metadata = {
            "version": version,
            "type": "candidate",
            "timeframe": tf_label,
            "timeframe_minutes": timeframe,
            "source_version": source_version,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "models": {},
        }
        if ensemble.feature_cols:
            metadata["feature_cols"] = ensemble.feature_cols

        existing = list(tf_dir.glob("*"))
        for f in existing:
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            except Exception:
                pass

        for name, model in ensemble.models.items():
            model_path = str(tf_dir / f"{name}.ubj")
            try:
                model.save(model_path)
                metadata["models"][name] = {"path": model_path, "trained": model.is_trained}
            except Exception as e:
                self.logger.error(f"Failed to save {name}: {e}")

        meta_path = tf_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved candidate model {version} for {tf_label}")
        return version

    def load_candidate(self, timeframe: int) -> Optional[VotingEnsemble]:
        tf_label = self._tf_label(timeframe)
        tf_dir = self._candidate_dir / tf_label
        if not tf_dir.exists():
            return None
        model_files = list(tf_dir.glob("*.ubj")) + list(tf_dir.glob("*.model")) + list(tf_dir.glob("*.pkl"))
        if not model_files:
            return None
        try:
            return self._load_models_from_dir(tf_dir)
        except Exception as e:
            self.logger.warning(f"Failed to load candidate for {tf_label}: {e}")
            return None

    def promote_candidate(self, timeframe: int) -> Tuple[bool, str]:
        tf_label = self._tf_label(timeframe)
        candidate = self.load_candidate(timeframe)
        if candidate is None:
            return False, f"No candidate model for {tf_label}"

        try:
            production = self.load_production(timeframe)
        except ModelNotFoundError:
            production = None

        if production:
            validation = self._compare_ensembles(production, candidate, timeframe)
            if not validation["promote"]:
                reason = validation.get("reject_reason", "validation failed")
                return False, f"Candidate rejected: {reason}"

        tf_dir = self._production_dir / tf_label
        if production and tf_dir.exists():
            archive_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_label_dir = self._archive_dir / tf_label
            archive_label_dir.mkdir(parents=True, exist_ok=True)
            archive_version = f"archive_{tf_label}_{archive_ts}"
            archive_version_dir = archive_label_dir / archive_version
            archive_version_dir.mkdir(parents=True, exist_ok=True)
            for f in tf_dir.glob("*"):
                if f.is_file():
                    shutil.copy2(f, archive_version_dir / f.name)

        candidate_tf_dir = self._candidate_dir / tf_label
        for f in candidate_tf_dir.glob("*"):
            if f.is_file() and f.name != "metadata.json":
                dest = tf_dir / f.name
                shutil.copy2(f, dest)

        version = f"prod_{tf_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        current_path = tf_dir / "current.txt"
        current_path.write_text(version)

        meta_path = tf_dir / "metadata.json"
        metadata = {
            "version": version,
            "type": "production",
            "timeframe": tf_label,
            "timeframe_minutes": timeframe,
            "promoted_at": datetime.now().isoformat(),
            "models": {name: {"path": str(tf_dir / f"{name}.ubj"), "trained": True} for name in candidate.models},
        }
        if candidate.feature_cols:
            metadata["feature_cols"] = candidate.feature_cols
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Promoted candidate to production for {tf_label} (version: {version})")
        return True, f"Promoted candidate to production for {tf_label}: {version}"

    def rollback(self, timeframe: int) -> Tuple[bool, str]:
        tf_label = self._tf_label(timeframe)
        archive_label_dir = self._archive_dir / tf_label
        if not archive_label_dir.exists():
            return False, f"No archived models for {tf_label}"

        archives = sorted([d for d in archive_label_dir.iterdir() if d.is_dir()], reverse=True)
        if not archives:
            return False, f"No archived models for {tf_label}"

        latest_archive = archives[0]
        tf_dir = self._production_dir / tf_label
        tf_dir.mkdir(parents=True, exist_ok=True)

        for f in latest_archive.glob("*"):
            if f.is_file():
                shutil.copy2(f, tf_dir / f.name)

        version = f"rollback_{tf_label}_{latest_archive.name}"
        current_path = tf_dir / "current.txt"
        current_path.write_text(version)

        self.logger.info(f"Rolled back production for {tf_label} to {latest_archive.name}")
        return True, f"Rolled back {tf_label} to {latest_archive.name}"

    def get_archive_versions(self, timeframe: int) -> List[str]:
        tf_label = self._tf_label(timeframe)
        archive_label_dir = self._archive_dir / tf_label
        if not archive_label_dir.exists():
            return []
        return sorted([d.name for d in archive_label_dir.iterdir() if d.is_dir()], reverse=True)

    def get_production_version(self, timeframe: int) -> Optional[str]:
        return self._get_current_production_version(timeframe)

    def has_production_model(self, timeframe: int) -> bool:
        try:
            self.load_production(timeframe)
            return True
        except ModelNotFoundError:
            return False

    def _compare_ensembles(self, production: VotingEnsemble, candidate: VotingEnsemble, timeframe: int) -> Dict:
        from learning.model_validator import ModelValidator
        validator = ModelValidator()
        return validator.validate(production, candidate, timeframe)

    def _load_models_from_dir(self, directory: Path) -> VotingEnsemble:
        ensemble = VotingEnsemble()
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                ensemble.feature_cols = meta.get("feature_cols")
            except Exception:
                pass
        model_class_map = {
            "xgboost": XGBoostModel,
            "random_forest": RandomForestModel,
            "lightgbm": LightGBMModel,
            "lstm": LSTMModel,
        }
        for name, model_class in model_class_map.items():
            model_path = directory / f"{name}.ubj"
            alt_path = directory / f"{name}.model"
            pkl_path = directory / f"{name}.pkl"
            if model_path.exists():
                try:
                    model = model_class()
                    model.load(str(model_path))
                    ensemble.register_model(name, model)
                except Exception as e:
                    self.logger.warning(f"Failed to load {name}: {e}")
            elif alt_path.exists():
                try:
                    model = model_class()
                    model.load(str(alt_path))
                    ensemble.register_model(name, model)
                except Exception as e:
                    self.logger.warning(f"Failed to load {name}: {e}")
            elif pkl_path.exists():
                try:
                    model = model_class()
                    model.load(str(pkl_path))
                    ensemble.register_model(name, model)
                except Exception as e:
                    self.logger.warning(f"Failed to load {name}: {e}")

        # ── Weight ensemble by val_accuracy ──
        self._apply_ensemble_weights(ensemble, directory)
        return ensemble

    def _apply_ensemble_weights(self, ensemble: VotingEnsemble, directory: Path):
        """Set ensemble model weights based on validation accuracy from performance.json."""
        perf_path = directory / "performance.json"
        if perf_path.exists():
            try:
                with open(perf_path) as f:
                    perf = json.load(f)
                acc = perf.get("accuracy", {})
                val_accs = {}
                for name in ensemble.models:
                    val_key = f"{name}_val"
                    if val_key in acc:
                        val_accs[name] = float(acc[val_key])
                    elif name in acc:
                        val_accs[name] = float(acc[name])
                if val_accs:
                    ensemble.set_weights_from_val_accuracy(val_accs)
                    return
            except Exception as e:
                self.logger.debug(f"Cannot read performance for weighting: {e}")

        # Fallback: look up aggregated performance from version history
        # (e.g., for production dirs without their own performance.json)
        try:
            meta_path = directory / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                version = meta.get("version", "")
                if version:
                    perf = self._load_performance(version)
                    acc = perf.get("accuracy", {})
                    val_accs = {}
                    for name in ensemble.models:
                        val_key = f"{name}_val"
                        if val_key in acc:
                            val_accs[name] = float(acc[val_key])
                        elif name in acc:
                            val_accs[name] = float(acc[name])
                    if val_accs:
                        ensemble.set_weights_from_val_accuracy(val_accs)
                        return
        except Exception:
            pass

        self.logger.info("No performance data for ensemble weighting — using uniform weights")

    def save_ensemble(self, ensemble: VotingEnsemble, version: Optional[str] = None, timeframe: Optional[int] = None, source: str = "auto_retrain") -> str:
        if version is None:
            version = self._get_next_version(timeframe)
        version_dir = self._model_dir / f"model_{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "timeframe": self._tf_label(timeframe) if timeframe else None,
            "timeframe_minutes": timeframe,
            "source": source,
            "models": {},
        }
        if ensemble.feature_cols:
            metadata["feature_cols"] = ensemble.feature_cols
        for name, model in ensemble.models.items():
            ext = "ubj"
            model_path = str(version_dir / f"{name}.{ext}")
            try:
                model.save(model_path)
                metadata["models"][name] = {"path": model_path, "trained": model.is_trained}
                self.logger.info(f"Saved {name} model to {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to save {name}: {e}")
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        calibrator = getattr(ensemble, "calibrator", None)
        if calibrator is not None and calibrator.is_fitted:
            try:
                calib_path = self._model_dir / "calibration" / (self._tf_label(timeframe) if timeframe else "default")
                calibrator.save(str(calib_path))
                metadata["calibration_path"] = str(calib_path)
                self.logger.info(f"Saved calibrator to {calib_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save calibrator: {e}")

        self._current_version = version
        self._version_metadata[version] = metadata
        self.logger.info(f"Saved model version {version} (timeframe={metadata['timeframe']})")
        return version

    def load_ensemble(self, version: str) -> VotingEnsemble:
        version_dir = self._model_dir / f"model_{version}"
        if not version_dir.exists():
            raise ModelNotFoundError(f"Model version {version} not found")
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotFoundError(f"Metadata not found for version {version}")
        with open(metadata_path) as f:
            metadata = json.load(f)
        ensemble = VotingEnsemble()
        ensemble.feature_cols = metadata.get("feature_cols")
        model_class_map = {
            "xgboost": XGBoostModel,
            "random_forest": RandomForestModel,
            "lightgbm": LightGBMModel,
            "lstm": LSTMModel,
        }
        for name, info in metadata.get("models", {}).items():
            model_path = info.get("path")
            if not model_path or not os.path.exists(model_path):
                alt_exts = [".ubj", ".model", ".pkl"]
                model_path = None
                for ext in alt_exts:
                    candidate = str(Path(info.get("path", "")).parent / f"{name}{ext}") if info.get("path") else None
                    if candidate and os.path.exists(candidate):
                        model_path = candidate
                        break
                if not model_path:
                    self.logger.warning(f"Model file not found for {name}")
                    continue
            try:
                model_class = model_class_map.get(name)
                if model_class:
                    model = model_class()
                    model.load(model_path)
                    ensemble.register_model(name, model)
                    self.logger.info(f"Loaded {name} from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load {name}: {e}")
        # ── Weight ensemble by val_accuracy ──
        self._apply_ensemble_weights(ensemble, version_dir)

        ensemble.version = version
        self._current_version = version
        self._version_metadata[version] = metadata
        self.logger.info(f"Loaded model version {version} with {ensemble.get_num_models()} models")
        return ensemble

    def get_latest_version(self, timeframe: Optional[int] = None) -> Optional[str]:
        versions = self.list_versions(timeframe)
        return versions[-1] if versions else None

    def list_versions(self, timeframe: Optional[int] = None) -> List[str]:
        versions = []
        for pattern in ["model_v*"]:
            for d in self._model_dir.glob(pattern):
                if d.is_dir():
                    version = d.name.replace("model_", "")
                    if timeframe is not None:
                        suffix = self._tf_label(timeframe)
                        if not version.endswith(f"_{suffix}"):
                            continue
                    versions.append(version)
        if self._production_dir.exists():
            for tf_dir in self._production_dir.iterdir():
                if tf_dir.is_dir():
                    cur_path = tf_dir / "current.txt"
                    if cur_path.exists():
                        prod_version = cur_path.read_text().strip()
                        if prod_version and prod_version not in versions:
                            if timeframe is not None:
                                suffix = self._tf_label(timeframe)
                                if prod_version.endswith(f"_{suffix}") or tf_dir.name == suffix:
                                    versions.append(prod_version)
                            else:
                                versions.append(prod_version)
        versions = list(set(versions))
        return sorted(versions, key=self._version_sort_key)

    def get_trained_timeframes(self) -> List[int]:
        tfs = set()
        for v in self.list_versions():
            meta = self._load_metadata(v)
            tf_minutes = meta.get("timeframe_minutes") if meta else None
            if tf_minutes:
                tfs.add(tf_minutes)
        for prod_dir in self._production_dir.iterdir():
            if prod_dir.is_dir():
                for tf_name in Timeframe.LABELS.values():
                    if prod_dir.name == tf_name:
                        for tf_val, label in Timeframe.LABELS.items():
                            if label == tf_name:
                                tfs.add(tf_val)
        return sorted(tfs)

    def has_model_for_timeframe(self, timeframe: int) -> bool:
        return self.get_latest_version(timeframe) is not None

    def load_latest_for_timeframe(self, timeframe: int) -> VotingEnsemble:
        try:
            return self.load_production(timeframe)
        except ModelNotFoundError:
            pass

        best_version = self.get_best_version(timeframe)
        if best_version:
            try:
                return self.load_ensemble(best_version)
            except Exception:
                pass

        version = self.get_latest_version(timeframe)
        if not version:
            raise ModelNotFoundError(f"No model for timeframe {self._tf_label(timeframe)}")
        return self.load_ensemble(version)

    def _load_metadata(self, version: str) -> Dict:
        version_dir = self._model_dir / f"model_{version}"
        meta_path = version_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def compare_versions(self, version_a: str, version_b: str) -> Dict:
        perf_a = self._load_performance(version_a)
        perf_b = self._load_performance(version_b)
        comparison = {"version_a": version_a, "version_b": version_b, "details": {}}
        for metric in ["win_rate", "profit_factor", "sharpe", "avg_return"]:
            va = perf_a.get(metric, 0) if perf_a else 0
            vb = perf_b.get(metric, 0) if perf_b else 0
            comparison["details"][metric] = {"a": va, "b": vb, "better": "a" if va >= vb else "b"}
        comparison["a_better"] = sum(1 for v in comparison["details"].values() if v["better"] == "a")
        comparison["b_better"] = sum(1 for v in comparison["details"].values() if v["better"] == "b")
        return comparison

    def _get_persistent_next_version(self, timeframe: Optional[int] = None) -> int:
        """Get the next version number from a persistent counter file.
        
        This prevents counter reset when model directories are deleted.
        Falls back to directory scanning if counter file is missing.
        """
        counter_path = self._model_dir / "version_counter.json"
        key = self._tf_label(timeframe) if timeframe else "default"
        counter = {}
        if counter_path.exists():
            try:
                with open(counter_path) as f:
                    counter = json.load(f)
            except Exception:
                pass
        next_num = counter.get(key, 0) + 1
        counter[key] = next_num
        try:
            with open(counter_path, "w") as f:
                json.dump(counter, f, indent=2)
        except Exception:
            pass
        return next_num

    def _get_next_version(self, timeframe: Optional[int] = None) -> str:
        # Try persistent counter first
        try:
            next_num = self._get_persistent_next_version(timeframe)
        except Exception:
            next_num = 0
        
        if next_num > 0:
            base = f"v{next_num}"
            if timeframe is not None:
                base += f"_{self._tf_label(timeframe)}"
            return base
        
        # Fallback to directory scanning
        versions = self.list_versions(timeframe)
        nums = []
        for v in versions:
            stem = v.split("_")[0] if "_" in v else v
            stem = stem.replace("v", "")
            if stem.isdigit():
                nums.append(int(stem))
        next_num = max(nums) + 1 if nums else 1
        base = f"v{next_num}"
        if timeframe is not None:
            base += f"_{self._tf_label(timeframe)}"
        return base

    def _load_performance(self, version: str) -> Dict:
        version_dir = self._model_dir / f"model_{version}"
        perf_path = version_dir / "performance.json"
        if perf_path.exists():
            try:
                with open(perf_path) as f:
                    perf = json.load(f)
                # Cap legacy OOS values agar tidak tampil 98% WR / 68571 PF
                oos = perf.get("oos")
                if oos:
                    perf["oos"] = self._cap_legacy_oos(oos)
                return perf
            except Exception:
                pass
        return {}

    @staticmethod
    def _cap_legacy_oos(oos: Dict) -> Dict:
        """Cap unrealistic OOS metrics dari legacy/old model versions.
        Mencegah tampilan WR=98% atau PF=68571 di dashboard, tapi tetap
        membiarkan nilai realistis seperti v28 (WR=36%, PF=6.74) lewat.
        """
        if oos.get("success"):
            wr_raw = float(oos.get("win_rate", 0))
            pf_raw = float(oos.get("profit_factor", 0))
            sh_raw = float(oos.get("sharpe_ratio", 0))

            # Hanya cap kalau jelas-jelas tidak realistis (>99% pasti legacy bug)
            # v10_M15 punya WR=98% PF=68571 — itu PALSU
            # v28_M5 punya WR=36% PF=6.74 — itu NYATA
            oos["win_rate"] = min(wr_raw, 75.0) if wr_raw > 80 else wr_raw
            if pf_raw > 50:
                if wr_raw > 80:
                    # Legacy bug: PF>50 AND WR>80 jelas tidak realistis
                    # (contoh: v10_M15 PF=68571 WR=98%)
                    oos["profit_factor"] = min(pf_raw, 5.0)
                else:
                    # Model modern dengan PF realistis > 50 (sangat jarang)
                    oos["profit_factor"] = min(pf_raw, 50.0)
            else:
                oos["profit_factor"] = pf_raw
            oos["sharpe_ratio"] = min(sh_raw, 3.0) if sh_raw > 4 else sh_raw
            # Accuracy capping
            acc_raw = float(oos.get("accuracy", 0))
            oos["accuracy"] = min(acc_raw, 85.0) if acc_raw > 90 else acc_raw
            nha_raw = float(oos.get("non_hold_accuracy", 0))
            oos["non_hold_accuracy"] = min(nha_raw, 80.0) if nha_raw > 85 else nha_raw
            # Re-komputasi OOS numeric score dengan capped values
            wr = oos["win_rate"]
            pf = oos["profit_factor"]
            sharpe = oos["sharpe_ratio"]
            trades = oos.get("total_trades", 0)
            score = 0
            if wr >= 65: score += 30
            elif wr >= 55: score += 20
            elif wr >= 50: score += 10
            else: score += min(wr / 10, 5)
            if pf >= 2.0: score += 25
            elif pf >= 1.5: score += 18
            elif pf >= 1.0: score += 8
            if sharpe >= 1.5: score += 20
            elif sharpe >= 1.0: score += 12
            elif sharpe >= 0.5: score += 6
            if trades >= 100: score += 15
            elif trades >= 50: score += 10
            elif trades >= 20: score += 5
            oos["oos_score"] = min(score, 100)
        return oos

    def save_performance(self, version: str, performance: Dict):
        version_dir = self._model_dir / f"model_{version}"
        if not version_dir.exists():
            self.logger.warning(f"Cannot save performance for {version}: directory not found")
            return
        perf_path = version_dir / "performance.json"
        try:
            # Merge dengan data existing (misal OOS result) agar tidak overwrite
            existing = {}
            if perf_path.exists():
                try:
                    with open(perf_path) as f:
                        existing = json.load(f)
                except Exception:
                    pass
            existing.update(performance)
            with open(perf_path, "w") as f:
                json.dump(existing, f, indent=2)
            self.logger.info(f"Saved performance data for {version}")
        except Exception as e:
            self.logger.error(f"Failed to save performance for {version}: {e}")

    def _counter_path(self) -> Path:
        return self._model_dir / "retrain_counter.json"

    def increment_retrain_count(self, timeframe: Optional[int] = None):
        counts = self._load_retrain_counts()
        key = self._tf_label(timeframe) if timeframe else "total"
        if key not in counts:
            counts[key] = 0
        counts[key] += 1
        counts["total"] = counts.get("total", 0) + 1
        counts["last_retrain"] = datetime.now().isoformat()
        try:
            with open(self._counter_path(), "w") as f:
                json.dump(counts, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save retrain counter: {e}")

    def save_oos_result(self, version: str, oos_result: Dict):
        oos_score = self._compute_oos_numeric_score(oos_result)
        if version.startswith("cand_"):
            tf_label = version.split("_")[1]
            cand_dir = self._candidate_dir / tf_label
            if cand_dir.exists():
                perf_path = cand_dir / "performance.json"
                perf = {}
                if perf_path.exists():
                    try:
                        with open(perf_path) as f:
                            perf = json.load(f)
                    except Exception:
                        pass
                perf["oos"] = oos_result
                perf["oos_score"] = oos_score
                with open(perf_path, "w") as f:
                    json.dump(perf, f, indent=2)
                return
            cand_version_dir = self._model_dir / f"model_{version}"
            if cand_version_dir.exists():
                perf_path = cand_version_dir / "performance.json"
                perf = {}
                if perf_path.exists():
                    try:
                        with open(perf_path) as f:
                            perf = json.load(f)
                    except Exception:
                        pass
                perf["oos"] = oos_result
                perf["oos_score"] = oos_score
                with open(perf_path, "w") as f:
                    json.dump(perf, f, indent=2)
                return
        version_dir = self._model_dir / f"model_{version}"
        if not version_dir.exists():
            return
        perf = self._load_performance(version)
        perf["oos"] = oos_result
        perf["oos_score"] = oos_score
        perf_path = version_dir / "performance.json"
        try:
            with open(perf_path, "w") as f:
                json.dump(perf, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save OOS for {version}: {e}")

    def get_oos_result(self, version: str) -> Dict:
        perf = self._load_performance(version)
        return perf.get("oos", {})

    def _compute_oos_numeric_score(self, oos: Dict) -> float:
        if not oos or not oos.get("success"):
            return 0
        wr = oos.get("win_rate", 0)
        pf = oos.get("profit_factor", 0)
        sharpe = oos.get("sharpe_ratio", 0)
        trades = oos.get("total_trades", 0)
        score = 0
        # WR component — graded, not truncated
        if wr >= 65:
            score += 30
        elif wr >= 55:
            score += 20
        elif wr >= 50:
            score += 10
        else:
            score += min(wr / 10, 5)  # partial credit: up to 5 pts for WR below 50%
        # PF component — kontinu agar PF=6.22 > PF=5.0
        if pf >= 2.0:
            # 20 base + up to 20 extra untuk PF di atas 2.0 (PF=10 → max 40)
            score += 20 + min(pf * 2.0, 20)
        elif pf >= 1.5:
            score += 14
        elif pf >= 1.0:
            score += 8
        # Sharpe component — granular tiers
        if sharpe >= 2.0:
            score += 23
        elif sharpe >= 1.5:
            score += 20
        elif sharpe >= 1.0:
            score += 12
        elif sharpe >= 0.5:
            score += 6
        # Trade count component — statistik signifikan
        if trades >= 100:
            score += 15
        elif trades >= 50:
            score += 10
        elif trades >= 20:
            score += 5
        return min(score, 100)

    @staticmethod
    def is_model_better(new_oos: Dict, old_oos: Dict) -> Tuple[bool, str]:
        """Multi-dimensional comparison: tidak cuma total score.
        Menerima model baru jika:
          1. Total score lebih tinggi (default), ATAU
          2. PF naik >= 10% DAN WR tidak turun >= 10%
        """
        if not old_oos or not old_oos.get("success"):
            return True, "no existing model"

        new_wr = new_oos.get("win_rate", 0)
        new_pf = new_oos.get("profit_factor", 0)
        new_sharpe = new_oos.get("sharpe_ratio", 0)
        old_wr = old_oos.get("win_rate", 0)
        old_pf = old_oos.get("profit_factor", 0)
        old_sharpe = old_oos.get("sharpe_ratio", 0)

        # 1. Score comparison
        new_score = ModelManager._compute_oos_numeric_score_static(new_oos)
        old_score = ModelManager._compute_oos_numeric_score_static(old_oos)

        if new_score >= old_score:
            return True, f"score {new_score:.1f} >= {old_score:.1f}"

        # 2. PF improvement with acceptable WR degradation
        pf_improved = old_pf > 0 and (new_pf - old_pf) / old_pf >= 0.10
        wr_not_degraded = old_wr == 0 or (old_wr > 0 and (new_wr - old_wr) / old_wr >= -0.10)
        if pf_improved and wr_not_degraded:
            return True, f"PF {new_pf:.2f} vs {old_pf:.2f} (improved), WR {new_wr:.1f}% vs {old_wr:.1f}% (stable)"

        # 3. Sharpe improvement (if WR didn't degrade much)
        sharpe_improved = old_sharpe > 0 and (new_sharpe - old_sharpe) / old_sharpe >= 0.15
        wr_slight_degraded = old_wr == 0 or (old_wr > 0 and (new_wr - old_wr) / old_wr >= -0.20)
        if sharpe_improved and wr_slight_degraded:
            return True, f"Sharpe {new_sharpe:.2f} vs {old_sharpe:.2f} (improved), WR stable"

        return False, f"score {new_score:.1f} < {old_score:.1f}"

    def _compute_oos_numeric_score(self, oos: Dict) -> float:
        return self._compute_oos_numeric_score_static(oos)

    @staticmethod
    def _compute_oos_numeric_score_static(oos: Dict) -> float:
        """Static version so is_model_better() can use it without instance."""
        if not oos or not oos.get("success"):
            return 0
        wr = oos.get("win_rate", 0)
        pf = oos.get("profit_factor", 0)
        sharpe = oos.get("sharpe_ratio", 0)
        trades = oos.get("total_trades", 0)
        score = 0
        if wr >= 65:
            score += 30
        elif wr >= 55:
            score += 20
        elif wr >= 50:
            score += 10
        else:
            score += min(wr / 10, 5)
        if pf >= 2.0:
            score += 20 + min(pf * 2.0, 20)
        elif pf >= 1.5:
            score += 14
        elif pf >= 1.0:
            score += 8
        if sharpe >= 2.0:
            score += 23
        elif sharpe >= 1.5:
            score += 20
        elif sharpe >= 1.0:
            score += 12
        elif sharpe >= 0.5:
            score += 6
        if trades >= 100:
            score += 15
        elif trades >= 50:
            score += 10
        elif trades >= 20:
            score += 5
        return min(score, 100)

    def _extract_val_accuracy(self, perf: Dict, oos: Dict) -> float:
        legacy_val = max(
            perf.get("accuracy", {}).get("xgboost_val", 0),
            perf.get("accuracy", {}).get("random_forest_val", 0),
        )
        if legacy_val > 0:
            return legacy_val
        oos_val = oos.get("val_accuracy", 0)
        if oos_val > 0:
            return oos_val if oos_val <= 1.0 else oos_val / 100.0
        oos_acc = oos.get("accuracy", 0)
        if oos_acc > 0:
            return oos_acc if oos_acc <= 1.0 else oos_acc / 100.0
        return 0

    def _get_aggregate_history(self) -> List[Dict]:
        tfs = self.get_trained_timeframes()
        if not tfs:
            return []
        all_history = []
        for tf in tfs:
            all_history.extend(self.get_version_history(tf))
        all_history.sort(key=lambda h: h.get("created_at", ""))
        return all_history

    def get_best_version(self, timeframe: int) -> Optional[str]:
        versions = self.list_versions(timeframe)
        if not versions:
            return None
        if len(versions) == 1:
            return versions[0]
        best_ver = None
        best_score = -1
        for ver in versions:
            oos = self.get_oos_result(ver)
            score = self._compute_oos_numeric_score(oos)
            if score > 0:
                if score > best_score:
                    best_score = score
                    best_ver = ver
                elif score == best_score and best_ver is not None:
                    # Tie-break: prefer higher WR, then higher PF, then newer version
                    cur_oos = self.get_oos_result(best_ver)
                    cur_wr = cur_oos.get("win_rate", 0)
                    cur_pf = cur_oos.get("profit_factor", 0)
                    new_wr = oos.get("win_rate", 0)
                    new_pf = oos.get("profit_factor", 0)
                    if new_wr > cur_wr:
                        best_ver = ver
                    elif new_wr == cur_wr and new_pf > cur_pf:
                        best_ver = ver
        if best_ver is None and versions:
            best_ver = versions[-1]
        return best_ver

    def get_version_history(self, timeframe: int) -> List[Dict]:
        versions = self.list_versions(timeframe)
        history = []
        for ver in versions:
            perf = self._load_performance(ver)
            oos = perf.get("oos", {})
            meta = self._load_metadata(ver)
            history.append({
                "version": ver,
                "created_at": meta.get("created_at", ""),
                "oos_score": perf.get("oos_score", 0),
                "oos_win_rate": oos.get("win_rate", 0),
                "oos_profit_factor": oos.get("profit_factor", 0),
                "oos_sharpe": oos.get("sharpe_ratio", 0),
                "oos_passed": oos.get("passed", False),
                "oos_grade": oos.get("grade", "N/A"),
                "oos_trades": oos.get("total_trades", 0),
                "val_accuracy": max(
                    perf.get("accuracy", {}).get("xgboost_val", 0),
                    perf.get("accuracy", {}).get("random_forest_val", 0),
                ),
            })
        return history

    def _get_best_version_for_skill(self, timeframe: int) -> Optional[str]:
        """Find the version with the best OOS data for skill computation.
        Falls back to latest version if no OOS data exists."""
        best = self.get_best_version(timeframe)
        if best:
            oos = self.get_oos_result(best)
            if oos.get("success"):
                return best
        # Fall back: walk backwards through versions to find one with OOS data
        versions = self.list_versions(timeframe)
        for ver in reversed(versions):
            oos = self.get_oos_result(ver)
            if oos.get("success"):
                return ver
        # Last resort: return latest
        return versions[-1] if versions else None

    def get_skill_level(self, timeframe: Optional[int] = None) -> str:
        scorer = SkillScorer()
        if timeframe is not None:
            cnt = self.get_retrain_count(timeframe)
            version = self._get_best_version_for_skill(timeframe)
            oos = self.get_oos_result(version) if version else {}
            perf = self._load_performance(version) if version else {}
            val_acc = self._extract_val_accuracy(perf, oos)
            history = self.get_version_history(timeframe)
            _, score = scorer.compute_global(
                retrain_count=cnt, oos_results=oos,
                val_accuracy=val_acc, version_history=history,
            )
        else:
            # Aggregate skill across all timeframes
            tfs = self.get_trained_timeframes()
            if not tfs:
                return "Newborn"
            total_score = 0
            for tf in tfs:
                total_score += self.get_skill_score(tf)
            avg_score = total_score // len(tfs)
            skill = scorer._map_score_to_skill(avg_score, self.get_total_retrains())
            return skill
        return scorer._map_score_to_skill(score, cnt)

    def get_skill_score(self, timeframe: Optional[int] = None) -> int:
        scorer = SkillScorer()
        if timeframe is not None:
            cnt = self.get_retrain_count(timeframe)
            version = self._get_best_version_for_skill(timeframe)
            oos = self.get_oos_result(version) if version else {}
            perf = self._load_performance(version) if version else {}
            val_acc = self._extract_val_accuracy(perf, oos)
            history = self.get_version_history(timeframe)
            _, score = scorer.compute_global(
                retrain_count=cnt, oos_results=oos,
                val_accuracy=val_acc, version_history=history,
            )
        else:
            # Aggregate across all timeframes
            tfs = self.get_trained_timeframes()
            if not tfs:
                return 0
            total_score = 0
            for tf in tfs:
                total_score += self.get_skill_score(tf)
            score = total_score // len(tfs)
        return score

    def get_models_summary(self) -> Dict[str, Dict]:
        summary = {}
        for tf in self.get_trained_timeframes():
            label = self._tf_label(tf)
            version = self._get_best_version_for_skill(tf) or self.get_production_version(tf) or "none"
            perf = self._load_performance(version) if version != "none" else {}
            oos = perf.get("oos", {})
            summary[label] = {
                "version": version,
                "retrains": self.get_retrain_count(tf),
                "skill": self.get_skill_level(tf),
                "skill_score": self.get_skill_score(tf),
                "accuracy": perf.get("accuracy", {}),
                "oos": {
                    "win_rate": oos.get("win_rate", 0),
                    "profit_factor": oos.get("profit_factor", 0),
                    "sharpe_ratio": oos.get("sharpe_ratio", 0),
                    "grade": oos.get("grade", "N/A"),
                    "passed": oos.get("passed", False),
                    "trades": oos.get("total_trades", 0),
                },
            }
        summary["_total"] = {
            "retrains": self.get_total_retrains(),
            "skill": self.get_skill_level(),
            "skill_score": self.get_skill_score(),
            "models": len(summary),
        }
        return summary

    @staticmethod
    def is_protected_version(version: str) -> bool:
        """Check if a version string is a protected/production version."""
        return version.startswith("prod_") or version.startswith("golden_") or version.startswith("archive_")

    def _extract_timeframe_from_version(self, version: str) -> Optional[int]:
        """Extract timeframe minutes from version string (e.g., 'v3_M5' -> 5)."""
        parts = version.split("_")
        if len(parts) >= 2:
            tf_label = parts[-1]
            for val, label in Timeframe.LABELS.items():
                if label == tf_label:
                    return val
        return None

    def is_version_protected(self, version: str, timeframe: Optional[int] = None) -> bool:
        """Check if a version is protected from deletion.
        
        Protected versions include:
        - Production versions (prod_*)
        - Golden/fallback versions (golden_*)
        - Top N best versions per TF by OOS score
        - Self-learn models created in the last 24 hours
        """
        # Static protection by name pattern
        if self.is_protected_version(version):
            return True
        
        # Auto-detect timeframe from version string if not provided
        if timeframe is None:
            timeframe = self._extract_timeframe_from_version(version)
        
        # Check metadata for source=self_learn — protect for 24h
        meta = self._load_metadata(version)
        if meta:
            source = meta.get("source", "")
            if source == "self_learn":
                created_str = meta.get("created_at", "")
                if created_str:
                    try:
                        created = datetime.fromisoformat(created_str)
                        age_hours = (datetime.now() - created).total_seconds() / 3600
                        if age_hours < 24:
                            return True
                    except Exception:
                        pass
        
        # Check if version is in top N protected versions for this TF
        if timeframe is not None:
            # Cache protected versions lookup to avoid repeated scanning
            cache_key = f"protected_{timeframe}"
            if not hasattr(self, '_protected_cache'):
                self._protected_cache = {}
            if cache_key not in self._protected_cache:
                self._protected_cache[cache_key] = set(self.get_protected_versions(timeframe))
            if version in self._protected_cache[cache_key]:
                return True
        
        return False

    def get_protected_versions(self, timeframe: int) -> List[str]:
        """Get the top N protected versions for a timeframe based on OOS score."""
        versions = self.list_versions(timeframe)
        scored = []
        for v in versions:
            oos = self.get_oos_result(v)
            score = self._compute_oos_numeric_score(oos)
            if score > 0:
                scored.append((v, score))
        scored.sort(key=lambda x: -x[1])  # descending by score
        return [v for v, _ in scored[:PROTECTED_VERSIONS_COUNT]]

    def delete_version(self, version: str, archive: bool = True, force: bool = False):
        """Delete or archive a model version.
        
        Args:
            version: The version string (e.g. 'v3_M5')
            archive: If True, move to archive/ instead of permanent deletion.
            force: If True, bypass protection check.
        """
        version_dir = self._model_dir / f"model_{version}"
        if not version_dir.exists():
            return
        if not force and self.is_version_protected(version):
            self.logger.warning(f"Version {version} is protected — skipping deletion")
            return
        if archive:
            archive_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_dest = self._archive_dir / "deleted" / f"{version}_{archive_ts}"
            archive_dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(version_dir), str(archive_dest))
                self.logger.info(f"Archived model version {version} -> {archive_dest}")
                return
            except Exception as e:
                self.logger.warning(f"Archive failed for {version}, falling back to delete: {e}")
        shutil.rmtree(version_dir)
        self.logger.info(f"Deleted model version {version}")

    def get_retrain_counts(self) -> Dict:
        return self._load_retrain_counts()

    def get_total_retrains(self) -> int:
        return self._load_retrain_counts().get("total", 0)

    def get_last_retrain_time(self) -> Optional[str]:
        return self._load_retrain_counts().get("last_retrain")

    def get_retrain_count(self, timeframe: int) -> int:
        key = self._tf_label(timeframe)
        return self._load_retrain_counts().get(key, 0)

    def _load_retrain_counts(self) -> Dict:
        p = self._counter_path()
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"total": 0}

    def load_best_ensemble(self, timeframe: int) -> Optional[VotingEnsemble]:
        """Load the best known ensemble for warm-start/continued training.
        Returns None if no model exists (first-time training).
        """
        try:
            # Priority 1: production model
            return self.load_production(timeframe)
        except ModelNotFoundError:
            pass

        try:
            # Priority 2: best version by OOS score
            best_ver = self.get_best_version(timeframe)
            if best_ver:
                return self.load_ensemble(best_ver)
        except Exception:
            pass

        try:
            # Priority 3: latest version
            latest = self.get_latest_version(timeframe)
            if latest:
                return self.load_ensemble(latest)
        except Exception:
            pass

        return None

    def is_market_open(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        return True

    def to_dict(self) -> Dict:
        return {
            "total_retrains": self.get_total_retrains(),
            "skill_level": self.get_skill_level(),
            "skill_score": self.get_skill_score(),
            "current_production": {
                tf: self.get_production_version(tf)
                for tf in self.get_trained_timeframes()
            },
            "archive_count": {
                tf: len(self.get_archive_versions(tf))
                for tf in self.get_trained_timeframes()
            },
        }
