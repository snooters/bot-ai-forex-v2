from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from core.constants import HISTORICAL_DIR
from utils.logger import get_logger


PRIMARY_TF = 5
CONTEXT_TFS = [15, 30, 60, 240]
ALL_TFS = [PRIMARY_TF] + CONTEXT_TFS


class MultiTimeframeData:
    def __init__(self, symbol: str, data: Dict[int, pd.DataFrame]):
        self.symbol = symbol
        self.data = data
        self.primary_tf = PRIMARY_TF
        self.context_tfs = CONTEXT_TFS
        self.primary = data.get(PRIMARY_TF)
        self._validate()

    def _validate(self):
        if self.primary is None or self.primary.empty:
            raise ValueError(f"No primary TF data ({PRIMARY_TF}) for {self.symbol}")
        for tf, df in self.data.items():
            if not df.empty and "time" not in df.columns:
                raise ValueError(f"TF {tf} missing 'time' column")

    def get_aligned(self) -> pd.DataFrame:
        return self.data.get(PRIMARY_TF, pd.DataFrame())

    def get_context(self, tf: int) -> Optional[pd.DataFrame]:
        return self.data.get(tf)

    @property
    def available_timeframes(self) -> List[int]:
        return [tf for tf, df in self.data.items() if not df.empty]

    @property
    def start_date(self) -> Optional[datetime]:
        if self.primary is not None and not self.primary.empty:
            return self.primary["time"].min()
        return None

    @property
    def end_date(self) -> Optional[datetime]:
        if self.primary is not None and not self.primary.empty:
            return self.primary["time"].max()
        return None

    @property
    def rows(self) -> int:
        return len(self.primary) if self.primary is not None else 0

    def summary(self) -> Dict:
        return {
            "symbol": self.symbol,
            "primary_tf": PRIMARY_TF,
            "context_tfs": CONTEXT_TFS,
            "rows": {tf: len(df) for tf, df in self.data.items() if not df.empty},
            "date_range": {
                "start": str(self.start_date),
                "end": str(self.end_date),
            },
        }


class DataLoader:
    def __init__(self, symbol: str = ""):
        if not symbol:
            symbol = config.trading["pairs"][0] if config.trading["pairs"] else "EURUSD"
        self.logger = get_logger("data_loader")
        self.symbol = symbol
        self._base_dir = Path(HISTORICAL_DIR) / symbol
        self._storage: Dict[int, Optional[pd.DataFrame]] = {}

    def load_all(self) -> MultiTimeframeData:
        result: Dict[int, pd.DataFrame] = {}
        for tf in ALL_TFS:
            df = self._load_parquet(tf)
            if df is not None and not df.empty:
                result[tf] = self._clean(df)
            else:
                self.logger.warning(f"TF {tf}: no data for {self.symbol}")
                result[tf] = pd.DataFrame()
        return MultiTimeframeData(self.symbol, result)

    def load_aligned(self) -> pd.DataFrame:
        mtf = self.load_all()
        primary = mtf.primary
        if primary is None or primary.empty:
            raise ValueError(f"No primary data for {self.symbol}")

        primary = primary.copy()
        primary.sort_values("time", inplace=True)
        primary.set_index("time", inplace=True)

        for tf in CONTEXT_TFS:
            ctx = mtf.get_context(tf)
            if ctx is None or ctx.empty:
                continue
            ctx = ctx.copy()
            ctx.sort_values("time", inplace=True)
            ctx.set_index("time", inplace=True)
            cols = {c: f"{c}_tf{tf}" for c in ctx.columns
                    if c not in ("time", "symbol", "timeframe")}
            ctx_renamed = ctx[list(cols.keys())].rename(columns=cols)
            primary = primary.join(ctx_renamed, how="left")

        filled_cols = [c for c in primary.columns if c.startswith("open_tf")
                       or c.startswith("high_tf") or c.startswith("low_tf")
                       or c.startswith("close_tf") or c.startswith("volume_tf")
                       or c.startswith("spread_tf")]
        for c in filled_cols:
            primary[c] = primary[c].ffill()

        primary.reset_index(inplace=True)
        primary.dropna(subset=["open", "high", "low", "close"], inplace=True)
        return primary

    def _load_parquet(self, timeframe: int) -> Optional[pd.DataFrame]:
        filepath = self._base_dir / f"tf_{timeframe}.parquet"
        if not filepath.exists():
            self.logger.debug(f"File not found: {filepath}")
            return None
        try:
            df = pd.read_parquet(str(filepath))
            if df.empty:
                return None
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
            return df
        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {e}")
            return None

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "time" in df.columns:
            df.sort_values("time", inplace=True)
            df.drop_duplicates(subset=["time"], keep="last", inplace=True)
            df.reset_index(drop=True, inplace=True)
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def get_context_features(
        self, df_aligned: pd.DataFrame
    ) -> pd.DataFrame:
        result = df_aligned.copy()
        for tf in CONTEXT_TFS:
            ctx = self._load_parquet(tf)
            if ctx is None or ctx.empty:
                continue
            ctx_orig = ctx.sort_values("time").set_index("time")

            ema20 = ctx_orig["close"].ewm(span=20, adjust=False).mean()
            trend = np.where(ctx_orig["close"] > ema20, 1,
                             np.where(ctx_orig["close"] < ema20, -1, 0))
            vol = (ctx_orig["high"] - ctx_orig["low"]).rolling(14).mean() / ctx_orig["close"].rolling(14).mean()
            mom = ctx_orig["close"].pct_change(periods=3, fill_method=None)
            feat = pd.DataFrame({
                f"trend{tf}": trend,
                f"volatility{tf}": vol,
                f"momentum{tf}": mom,
            }, index=ctx_orig.index)
            result["_time_key"] = result["time"]
            result = result.merge(feat, left_on="time", right_index=True, how="left")

        ctx_cols = [c for c in result.columns
                    if any(c.startswith(p) for p in ("trend", "volatility", "momentum"))]
        for c in ctx_cols:
            result[c] = result[c].ffill()

        if "_time_key" in result.columns:
            result.drop(columns=["_time_key"], inplace=True)

        return result


def load_multi_tf(
    symbol: str = "",
) -> pd.DataFrame:
    if not symbol:
        symbol = config.trading["pairs"][0] if config.trading["pairs"] else "EURUSD"
    loader = DataLoader(symbol)
    aligned = loader.load_aligned()
    return loader.get_context_features(aligned)


def get_available_symbols() -> List[str]:
    base = Path(HISTORICAL_DIR)
    if not base.exists():
        return []
    return [d.name for d in base.iterdir() if d.is_dir()]


def get_data_stats(symbol: str) -> Dict:
    stats = {}
    base = Path(HISTORICAL_DIR) / symbol
    if not base.exists():
        return stats
    for tf in ALL_TFS:
        fp = base / f"tf_{tf}.parquet"
        if fp.exists():
            df = pd.read_parquet(str(fp))
            stats[tf] = {
                "rows": len(df),
                "start": str(df["time"].min()) if "time" in df.columns else "?",
                "end": str(df["time"].max()) if "time" in df.columns else "?",
                "size_mb": round(fp.stat().st_size / 1024 / 1024, 1),
            }
    return stats
