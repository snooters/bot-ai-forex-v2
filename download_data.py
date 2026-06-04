"""Download historical data dari MT5, resample M5 -> M15/M30/H1/H4.

M5  = entry TF (source dari MT5)
M15 = context (resample dari M5)
M30 = context (resample dari M5)
H1  = context (resample dari M5)
H4  = context (resample dari M5)

Catatan:
  - Hanya M5 yang di-download dari MT5.
  - M15/M30/H1/H4 di-resample dari M5 (OHLCV aggregation).
  - Data historis tergantung broker (demo FTMO ~1.5 tahun).
  - Jika perlu data lebih tua, gunakan broker dengan history lebih panjang.

Usage:
    python download_data.py
    python download_data.py --pair EURUSD --year 2025
    python download_data.py --all
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from core.config import config
from core.constants import HISTORICAL_DIR
from data.mt5_connector import MT5Connector
from data.data_storage import ParquetStorage
from utils.logger import get_logger

logger = get_logger("download_data")

PRIMARY_TF = 5
CONTEXT_TFS = [15, 30, 60, 240]
ALL_TFS = [PRIMARY_TF] + CONTEXT_TFS
TF_LABELS = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MT5 historical data since 2019")
    parser.add_argument("--pair", default=config.trading["pairs"][0],
                        help=f"Currency pair (default: {config.trading['pairs'][0]})")
    parser.add_argument("--year", type=int, default=2025,
                        help="Start year (default: 2025, tergantung broker)")
    parser.add_argument("--all", action="store_true",
                        help="Download all pairs from config")
    return parser.parse_args()


def _resample_ohlc(m5: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """Resample M5 OHLCV ke timeframe yang lebih besar."""
    rule = f"{tf_minutes}min"
    resampled = m5.resample(rule, on="time").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "spread": "mean",
    })
    resampled.dropna(subset=["open", "close"], inplace=True)
    resampled.reset_index(inplace=True)
    return resampled


def download_m5_range(
    connector: MT5Connector,
    symbol: str,
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    """Download M5 data via copy_rates_from_pos (max yg broker sediakan)."""
    mt5 = connector._mt5
    if mt5 is None:
        return pd.DataFrame()

    connector.ensure_connected()
    mt5.symbol_select(symbol, True)

    # Binary search max candle
    lo, hi = 50000, 200000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        r = mt5.copy_rates_from_pos(symbol, 5, 0, mid)
        if r is not None:
            lo = mid
        else:
            hi = mid - 1
    max_count = lo
    rates = mt5.copy_rates_from_pos(symbol, 5, 0, max_count)
    if rates is None or len(rates) == 0:
        logger.error(f"  No data: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.sort_values("time", inplace=True)
    df.drop_duplicates(subset=["time"], keep="last", inplace=True)

    tmin = df["time"].min()
    tmax = df["time"].max()
    logger.info(f"  [M5] {len(df):,} candles, {tmin.date()} -> {tmax.date()}")

    # Filter dari from_date
    df = df[df["time"] >= pd.Timestamp(from_date)].copy()
    logger.info(f"  [M5] after filter: {len(df):,} candles since {from_date.date()}")

    df_out = pd.DataFrame()
    df_out["time"] = df["time"]
    df_out["open"] = df["open"].astype(float)
    df_out["high"] = df["high"].astype(float)
    df_out["low"] = df["low"].astype(float)
    df_out["close"] = df["close"].astype(float)
    df_out["volume"] = df["tick_volume"].astype(float) if "tick_volume" in df.columns else df["volume"].astype(float)
    df_out["spread"] = df["spread"].astype(float) if "spread" in df.columns else 0.0
    df_out["symbol"] = symbol
    df_out["timeframe"] = 5
    return df_out


def _resolve_symbol(mt5, symbol: str) -> str:
    """Auto-detect full symbol name (add .fl if needed)."""
    if mt5.symbol_select(symbol, True):
        return symbol
    alt = f"{symbol}.fl"
    if not symbol.endswith(".fl") and mt5.symbol_select(alt, True):
        logger.info(f"  Symbol '{symbol}' not found, using '{alt}'")
        return alt
    return symbol


def download_and_save(
    connector: MT5Connector,
    storage: ParquetStorage,
    symbol: str,
    from_year: int,
) -> None:
    """Download M5 via MT5, resample ke M15/M30/H1/H4, simpan semua ke parquet."""
    from_year = max(from_year, 2010)
    from_date = datetime(from_year, 1, 1)
    to_date = datetime.now()

    # Auto-detect symbol (.fl suffix)
    symbol = _resolve_symbol(connector._mt5, symbol)

    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading {symbol} M5 since {from_year}")
    logger.info(f"{'='*60}")

    # ── Step 1: Download M5 raw dari MT5 ──
    logger.info(f"  [M5] fetching {from_date.date()} -> {to_date.date()}...")
    m5 = download_m5_range(connector, symbol, from_date, to_date)

    if m5.empty:
        logger.error(f"  [M5] No data received from MT5!")
        return

    n = len(m5)
    tmin = m5["time"].min()
    tmax = m5["time"].max()
    is_sim = (m5["time"].dt.microsecond > 0).any()
    logger.info(f"  [M5] {n:,} candles, {tmin} -> {tmax}")
    if is_sim:
        logger.warning(f"  [M5] WARNING: Fractional seconds detected — "
                       f"data mungkin simulasi! Gunakan MT5 real.")

    # ── Step 2: Simpan M5 ──
    storage.save_data(symbol, 5, m5)

    # ── Step 3: Resample ke M15, M30, H1, H4 ──
    m5_sorted = m5.sort_values("time").reset_index(drop=True)
    logger.info(f"\nResampling M5 -> higher timeframes...")
    for tf in [15, 30, 60, 240]:
        label = TF_LABELS.get(tf, f"TF{tf}")
        df_tf = _resample_ohlc(m5_sorted, tf)
        df_tf["symbol"] = symbol
        df_tf["timeframe"] = tf
        storage.save_data(symbol, tf, df_tf)
        logger.info(f"  [{label}] {len(df_tf):,} candles "
                    f"({df_tf['time'].min().date()} -> {df_tf['time'].max().date()})")

    # ── Summary ──
    logger.info(f"\nDownload complete for {symbol}:")
    for tf in ALL_TFS:
        label = TF_LABELS.get(tf, f"TF{tf}")
        saved = storage.load_data(symbol, tf)
        if not saved.empty:
            logger.info(f"  {label}: {len(saved):,} candles "
                        f"({saved['time'].min().date()} -> {saved['time'].max().date()})")
        else:
            logger.warning(f"  {label}: NO DATA")


def main() -> None:
    args = parse_args()

    # Inisialisasi MT5
    connector = MT5Connector()
    logger.info("Connecting to MT5...")
    try:
        connector.connect()
    except Exception as e:
        logger.error(f"Failed to connect to MT5: {e}")
        logger.error("Pastikan MetaTrader 5 sudah running dan login.")
        sys.exit(1)

    if not connector._mt5_available:
        logger.error("MetaTrader5 Python package tidak terinstall.")
        sys.exit(1)

    storage = ParquetStorage()

    if args.all:
        symbols = config.trading["pairs"]
        logger.info(f"Downloading ALL pairs: {symbols}")
    else:
        symbols = [args.pair]

    for symbol in symbols:
        download_and_save(connector, storage, symbol, args.year)

    connector.disconnect()
    logger.info("\nAll downloads finished.")


if __name__ == "__main__":
    main()
