"""
Download EURUSD.fl data from MT5 local cache.
First: scroll the chart in MT5 to load old data into cache,
then run this script to extract it.
"""
import pandas as pd
import MetaTrader5 as mt5
from pathlib import Path
from datetime import datetime

SYMBOL = "EURUSD.fl"
OUTPUT_DIR = Path("data/historical/EURUSD.fl")
TIMEFRAMES = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4"}

mt5.initialize()
mt5.symbol_select(SYMBOL, True)

print(f"Downloading data from MT5 local cache for {SYMBOL}...")
print(f"(make sure you've scrolled MT5 chart to load old data first)\n")

for tf, label in sorted(TIMEFRAMES.items()):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, 50000)
    if rates is None:
        print(f"{label}: no data (try with smaller count)")
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, 10000)
    if rates is None:
        print(f"{label}: FAILED - {mt5.last_error()}")
        continue
    
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    span = (df["time"].max() - df["time"].min()).days
    earliest = df["time"].min()
    latest = df["time"].max()
    print(f"{label}: {len(df):>7} rows, {earliest.date()} -> {latest.date()}, span={span:>4}d")

    filepath = OUTPUT_DIR / f"tf_{tf}.parquet"
    
    try:
        existing = pd.read_parquet(filepath)
        existing["time"] = pd.to_datetime(existing["time"])
        earliest_existing = existing["time"].min().date()
        
        if earliest.date() < earliest_existing:
            combined = pd.concat([df, existing], ignore_index=True)
            combined.sort_values("time", inplace=True)
            combined.drop_duplicates(subset=["time"], keep="last", inplace=True)
            combined.reset_index(drop=True, inplace=True)
            combined.to_parquet(filepath, index=False)
            print(f"  -> Extended! {len(combined)} rows total ({earliest.date()} to {latest.date()})")
        else:
            print(f"  -> No older data found (earliest: {earliest.date()} vs existing: {earliest_existing})")
    except Exception as e:
        print(f"  -> Could not merge: {e}")
        print(f"  -> Saved directly")
        df.to_parquet(filepath, index=False)

mt5.shutdown()
print(f"\nDone. Files in {OUTPUT_DIR}:")
for f in sorted(OUTPUT_DIR.glob("*.parquet")):
    df = pd.read_parquet(f)
    ts = pd.to_datetime(df["time"])
    print(f"  {f.name}: {len(df):>7} rows, {ts.min().date()} -> {ts.max().date()}, span={(ts.max()-ts.min()).days}d")
