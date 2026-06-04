"""Download historical data dari MT5 — semua TF langsung dari MT5.

M5  = entry TF (source dari MT5)
M15 = context (langsung dari MT5 atau resample fallback)
M30 = context (langsung dari MT5 atau resample fallback)
H1  = context (langsung dari MT5 atau resample fallback)
H4  = context (langsung dari MT5 atau resample fallback)

Usage:
    python download_data.py
    python download_data.py --pair EURUSD --year 2019
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
    parser = argparse.ArgumentParser(description="Download MT5 historical data")
    parser.add_argument("--pair", default=config.trading["pairs"][0],
                        help=f"Currency pair (default: {config.trading['pairs'][0]})")
    parser.add_argument("--year", type=int, default=2019,
                        help="Start year (default: 2019, tergantung broker)")
    parser.add_argument("--all", action="store_true",
                        help="Download all pairs from config")
    return parser.parse_args()


def _resolve_symbol(mt5, symbol: str) -> str:
    if mt5.symbol_select(symbol, True):
        return symbol
    alt = f"{symbol}.fl"
    if not symbol.endswith(".fl") and mt5.symbol_select(alt, True):
        logger.info(f"  Symbol '{symbol}' not found, using '{alt}'")
        return alt
    return symbol


def _download_tf(
    connector: MT5Connector,
    symbol: str,
    tf: int,
    from_date: datetime,
) -> pd.DataFrame:
    """Download timeframe tertentu dari MT5 via copy_rates_from_pos (binary search max candles)."""
    mt5 = connector._mt5
    if mt5 is None:
        return pd.DataFrame()

    connector.ensure_connected()
    mt5.symbol_select(symbol, True)

    tf_max = {5: 200000, 15: 70000, 30: 35000, 60: 18000, 240: 5000}
    hi = tf_max.get(tf, 50000)
    lo = max(hi // 4, 5000)

    while lo < hi:
        mid = (lo + hi + 1) // 2
        r = mt5.copy_rates_from_pos(symbol, tf, 0, mid)
        if r is not None:
            lo = mid
        else:
            hi = mid - 1

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, lo)
    if rates is None or len(rates) == 0:
        logger.warning(f"  [{TF_LABELS.get(tf, str(tf))}] No data from MT5: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.sort_values("time", inplace=True)
    df.drop_duplicates(subset=["time"], keep="last", inplace=True)
    df = df[df["time"] >= pd.Timestamp(from_date)].copy()

    logger.info(f"  [{TF_LABELS.get(tf, str(tf))}] {len(df):,} candles "
                f"({df['time'].min().date()} -> {df['time'].max().date()})")

    df_out = pd.DataFrame()
    df_out["time"] = df["time"]
    df_out["open"] = df["open"].astype(float)
    df_out["high"] = df["high"].astype(float)
    df_out["low"] = df["low"].astype(float)
    df_out["close"] = df["close"].astype(float)
    df_out["volume"] = df["tick_volume"].astype(float) if "tick_volume" in df.columns else df["volume"].astype(float)
    df_out["spread"] = df["spread"].astype(float) if "spread" in df.columns else 0.0
    df_out["symbol"] = symbol
    df_out["timeframe"] = tf
    return df_out


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


def download_and_save(
    connector: MT5Connector,
    storage: ParquetStorage,
    symbol: str,
    from_year: int,
) -> None:
    """Download M5 + M15/M30/H1/H4 langsung dari MT5, resample sebagai fallback."""
    from_year = max(from_year, 2010)
    from_date = datetime(from_year, 1, 1)
    to_date = datetime.now()

    symbol = _resolve_symbol(connector._mt5, symbol)

    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading {symbol} since {from_year}")
    logger.info(f"{'='*60}")
    logger.info(f"  Period: {from_date.date()} -> {to_date.date()}")

    # ── Step 1: Download M5 dari MT5 ──
    logger.info(f"\n>>> M5 (primary, from MT5)")
    m5 = _download_tf(connector, symbol, 5, from_date)
    if m5.empty:
        logger.error("  [M5] No data! Aborting.")
        return
    storage.save_data(symbol, 5, m5)

    # ── Step 2: Download context TFs langsung dari MT5 ──
    for tf in CONTEXT_TFS:
        label = TF_LABELS.get(tf, str(tf))
        logger.info(f"\n>>> {label} (from MT5)")

        df = _download_tf(connector, symbol, tf, from_date)
        m5_resampled = _resample_ohlc(m5, tf)

        # Pilih mana yang lebih banyak datanya
        if not df.empty and not m5_resampled.empty:
            winner = "MT5" if len(df) >= len(m5_resampled) else "resample"
            if winner == "resample":
                logger.info(f"  -> Using resample ({len(m5_resampled):,} rows vs MT5 {len(df):,})")
                df = m5_resampled
            else:
                logger.info(f"  -> Using MT5 direct ({len(df):,} rows vs resample {len(m5_resampled):,})")
                # Gabung: isi gap yg tidak ada di MT5 dari resample
                overlap = df["time"].isin(m5_resampled["time"])
                missing = m5_resampled[~m5_resampled["time"].isin(df["time"])]
                if len(missing) > 0:
                    logger.info(f"  -> Adding {len(missing):,} missing candles from resample")
                    df = pd.concat([df, missing], ignore_index=True)
                    df.sort_values("time", inplace=True)
                    df.drop_duplicates(subset=["time"], keep="last", inplace=True)
        elif df.empty and not m5_resampled.empty:
            logger.info(f"  -> MT5 empty, using resample ({len(m5_resampled):,} rows)")
            df = m5_resampled
        elif not df.empty and m5_resampled.empty:
            logger.info(f"  -> Using MT5 direct ({len(df):,} rows)")
        else:
            logger.warning(f"  -> NO DATA for {label}")
            continue

        df["symbol"] = symbol
        df["timeframe"] = tf
        storage.save_data(symbol, tf, df)

    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"Download complete for {symbol}:")
    logger.info(f"{'='*60}")
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
