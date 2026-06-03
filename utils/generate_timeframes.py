import pandas as pd
import numpy as np
from pathlib import Path


def ohlc_resample(df, rule):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "spread": "mean",
    }
    if vol_col in df.columns:
        agg_dict[vol_col] = "sum"
    if "real_volume" in df.columns:
        agg_dict["real_volume"] = "sum"

    resampled = df.resample(rule).agg(agg_dict).dropna(subset=["open"])
    resampled = resampled.reset_index()
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    return resampled


def main():
    base = Path("data/historical/EURUSD.fl")
    tf_5 = base / "tf_5.parquet"

    if not tf_5.exists():
        print("No M5 data found at", tf_5)
        return

    m5 = pd.read_parquet(tf_5)
    print(f"M5: {len(m5):,} rows | {m5['time'].min()} -> {m5['time'].max()}")

    # Generate H1 (60 min)
    h1 = ohlc_resample(m5, "60min")
    h1.to_parquet(base / "tf_60.parquet", index=False)
    print(f"H1: {len(h1):,} rows -> saved tf_60.parquet")

    # Generate H4 (240 min)
    h4 = ohlc_resample(m5, "240min")
    h4.to_parquet(base / "tf_240.parquet", index=False)
    print(f"H4: {len(h4):,} rows -> saved tf_240.parquet")

    # Generate M1 not possible from M5 (need tick data)
    print("\nNOTE: M1 (tf_1.parquet) cannot be derived from M5.")
    print("You need actual tick/M1 data for that timeframe.")

    print("\nFinal file list:")
    for f in sorted(base.glob("*.parquet")):
        df = pd.read_parquet(f)
        print(f"  {f.name:20s} {len(df):>8,} rows | {df['time'].min()} -> {df['time'].max()}")


if __name__ == "__main__":
    main()
