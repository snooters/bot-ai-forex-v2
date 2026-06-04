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

    def _tf_timeframe_dir(self, timeframe: int) -> Path:
        label = self._tf_label(timeframe)
        return self._model_dir / label

    def _version_dir(self, version: str) -> Path:
        return self._model_dir / f"model_{version}"

    def save_to_production(self, ensemble: VotingEnsemble, timeframe: int) -> str:
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
            "models": {},
        }

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

    def save_to_candidate(self, ensemble: VotingEnsemble, timeframe: int, source_version: str = "") -> str:
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
            "created_at": datetime.now().isoformat(),
            "models": {},
        }

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
        return ensemble

    def save_ensemble(self, ensemble: VotingEnsemble, version: Optional[str] = None, timeframe: Optional[int] = None) -> str:
        if version is None:
            version = self._get_next_version(timeframe)
        version_dir = self._model_dir / f"model_{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "timeframe": self._tf_label(timeframe) if timeframe else None,
            "timeframe_minutes": timeframe,
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
        self._current_version = version
        self._version_metadata[version] = metadata
        self.logger.info(f"Loaded model version {version} with {ensemble.get_num_models()} models")
        return ensemble

    def get_latest_version(self, timeframe: Optional[int] = None) -> Optional[str]:
        versions = self.list_versions(timeframe)
        return versions[-1] if versions else None

    def list_versions(self, timeframe: Optional[int] = None) -> List[str]:
        versions = []
        for d in self._model_dir.glob("model_v*"):
            if d.is_dir():
                version = d.name.replace("model_", "")
                if timeframe is not None:
                    suffix = self._tf_label(timeframe)
                    if not version.endswith(f"_{suffix}"):
                        continue
                versions.append(version)
        return sorted(versions)

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

    def _get_next_version(self, timeframe: Optional[int] = None) -> str:
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
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_performance(self, version: str, performance: Dict):
        version_dir = self._model_dir / f"model_{version}"
        if not version_dir.exists():
            return
        perf_path = version_dir / "performance.json"
        with open(perf_path, "w") as f:
            json.dump(performance, f, indent=2)

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
        version_dir = self._model_dir / f"model_{version}"
        if not version_dir.exists():
            return
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
                perf["oos_score"] = self._compute_oos_numeric_score(oos_result)
                with open(perf_path, "w") as f:
                    json.dump(perf, f, indent=2)
                return
        perf = self._load_performance(version)
        perf["oos"] = oos_result
        perf["oos_score"] = self._compute_oos_numeric_score(oos_result)
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
        if wr >= 65:
            score += 30
        elif wr >= 55:
            score += 20
        elif wr >= 50:
            score += 10
        if pf >= 2.0:
            score += 25
        elif pf >= 1.5:
            score += 18
        elif pf >= 1.0:
            score += 8
        if sharpe >= 1.5:
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
            if score > best_score and score > 0:
                best_score = score
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

    def get_skill_level(self, timeframe: Optional[int] = None) -> str:
        if timeframe is not None:
            cnt = self.get_retrain_count(timeframe)
            has = self.has_model_for_timeframe(timeframe)
            version = self.get_latest_version(timeframe)
            oos = self.get_oos_result(version) if version else {}
            perf = self._load_performance(version) if version else {}
            val_acc = max(
                perf.get("accuracy", {}).get("xgboost_val", 0),
                perf.get("accuracy", {}).get("random_forest_val", 0),
            )
            history = self.get_version_history(timeframe)
        else:
            cnt = self.get_total_retrains()
            has = bool(self.list_versions())
            version = self.get_latest_version()
            oos = self.get_oos_result(version) if version else {}
            perf = self._load_performance(version) if version else {}
            val_acc = max(
                perf.get("accuracy", {}).get("xgboost_val", 0),
                perf.get("accuracy", {}).get("random_forest_val", 0),
            )
            history = self.get_version_history(self.get_trained_timeframes()[0]) if self.get_trained_timeframes() else []
        scorer = SkillScorer()
        skill, _ = scorer.compute_global(
            retrain_count=cnt,
            oos_results=oos,
            val_accuracy=val_acc,
            version_history=history,
        )
        return skill

    def get_skill_score(self, timeframe: Optional[int] = None) -> int:
        if timeframe is not None:
            cnt = self.get_retrain_count(timeframe)
            version = self.get_latest_version(timeframe)
            oos = self.get_oos_result(version) if version else {}
            perf = self._load_performance(version) if version else {}
            val_acc = max(
                perf.get("accuracy", {}).get("xgboost_val", 0),
                perf.get("accuracy", {}).get("random_forest_val", 0),
            )
            history = self.get_version_history(timeframe)
        else:
            cnt = self.get_total_retrains()
            version = self.get_latest_version()
            oos = self.get_oos_result(version) if version else {}
            perf = self._load_performance(version) if version else {}
            val_acc = max(
                perf.get("accuracy", {}).get("xgboost_val", 0),
                perf.get("accuracy", {}).get("random_forest_val", 0),
            )
            history = self.get_version_history(self.get_trained_timeframes()[0]) if self.get_trained_timeframes() else []
        scorer = SkillScorer()
        _, score = scorer.compute_global(
            retrain_count=cnt,
            oos_results=oos,
            val_accuracy=val_acc,
            version_history=history,
        )
        return score

    def get_models_summary(self) -> Dict[str, Dict]:
        summary = {}
        for tf in self.get_trained_timeframes():
            label = self._tf_label(tf)
            version = self.get_latest_version(tf) or self.get_production_version(tf) or "none"
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

    def delete_version(self, version: str):
        version_dir = self._model_dir / f"model_{version}"
        if version_dir.exists():
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
