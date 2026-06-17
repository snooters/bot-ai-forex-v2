"""Data preparation for 3-model ensemble system.

Loads parquet data, computes features, and generates targets
for H4, H1, and M5 models.
"""
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from ensemble.config import CONFIG


DATA_DIR = Path(r"H:\source\pyton\BOT AI FOREX V2\data\historical\EURUSD")

# TF to parquet file mapping
TF_FILES = {
    5: "tf_5.parquet",
    15: "tf_15.parquet",
    30: "tf_30.parquet",
    60: "tf_60.parquet",
    240: "tf_240.parquet",
}


def load_tf_data(timeframe: int) -> Optional[pd.DataFrame]:
    """Load a timeframe's parquet data."""
    fname = TF_FILES.get(timeframe)
    if not fname:
        return None
    fp = DATA_DIR / fname
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    # Convert time column to datetime and set as index
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


def add_technical_features(df: pd.DataFrame, closes: np.ndarray, 
                           highs: np.ndarray, lows: np.ndarray,
                           volumes: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute common technical features."""
    n = len(closes)
    features = {}
    
    # EMAs
    ema20 = pd.Series(closes).ewm(span=20).mean().values if n > 20 else np.full(n, np.nan)
    ema50 = pd.Series(closes).ewm(span=50).mean().values if n > 50 else np.full(n, np.nan)
    ema200 = pd.Series(closes).ewm(span=200).mean().values if n > 200 else np.full(n, np.nan)
    
    features["ema20"] = ema20
    features["ema50"] = ema50
    features["ema200"] = ema200
    
    # EMA slopes (rate of change over 3 bars)
    if n > 23:
        features["ema20_slope"] = np.concatenate([[np.nan]*3, np.diff(ema20, 3)]) if n > 23 else np.full(n, np.nan)
    else:
        features["ema20_slope"] = np.full(n, np.nan)
    
    if n > 53:
        features["ema50_slope"] = np.concatenate([[np.nan]*3, np.diff(ema50, 3)])
    else:
        features["ema50_slope"] = np.full(n, np.nan)
    
    if n > 203:
        features["ema200_slope"] = np.concatenate([[np.nan]*3, np.diff(ema200, 3)])
    else:
        features["ema200_slope"] = np.full(n, np.nan)
    
    # RSI 14
    if n > 14:
        delta = np.diff(closes)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + np.where(avg_loss != 0, rs, 100)))
        # Pad with NaN to match length
        rsi_padded = np.concatenate([[np.nan], rsi])
        if len(rsi_padded) == n - 14:  # Handle edge
            rsi_padded = np.concatenate([np.full(14, np.nan), rsi])
        features["rsi14"] = rsi_padded[:n]
    else:
        features["rsi14"] = np.full(n, np.nan)
    
    # ATR 14
    if n > 15:
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(14).mean().values
        features["atr"] = atr[:n]
    else:
        features["atr"] = np.full(n, np.nan)
    
    # Volume ratio (current vs 20-bar average)
    if n > 20 and volumes is not None:
        vol_avg = pd.Series(volumes).rolling(20).mean().values
        features["volume_ratio"] = np.where(vol_avg > 0, volumes / vol_avg, 1.0)
    else:
        features["volume_ratio"] = np.ones(n)
    
    # ADX 14
    if n > 28:
        # Simplified ADX
        up_move = np.diff(highs)
        down_move = np.diff(lows)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        # Smoothed with 14-period
        tr_series = pd.Series(np.concatenate([[np.nan], tr]))
        plus_dm_series = pd.Series(np.concatenate([[0], plus_dm]))
        minus_dm_series = pd.Series(np.concatenate([[0], minus_dm]))
        
        atr_14 = tr_series.rolling(14).mean().values
        plus_di_14 = 100 * plus_dm_series.rolling(14).mean().values / np.where(atr_14 > 0, atr_14, 1)
        minus_di_14 = 100 * minus_dm_series.rolling(14).mean().values / np.where(atr_14 > 0, atr_14, 1)
        dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) > 0, (plus_di_14 + minus_di_14), 1)
        adx = pd.Series(dx).rolling(14).mean().values
        features["adx"] = adx[:n]
    else:
        features["adx"] = np.full(n, np.nan)
    
    # MACD
    if n > 26:
        ema12 = pd.Series(closes).ewm(span=12).mean().values
        ema26 = pd.Series(closes).ewm(span=26).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9).mean().values
        features["macd_line"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_line_minus_signal"] = macd_line - signal_line
    else:
        features["macd_line"] = np.full(n, np.nan)
        features["macd_signal"] = np.full(n, np.nan)
        features["macd_line_minus_signal"] = np.full(n, np.nan)
    
    # Price position vs EMA20 (%)
    features["price_pos_vs_ema20"] = np.where(ema20 > 0, (closes - ema20) / ema20 * 100, 0)
    
    # Range % (high-low)/close
    features["range_pct"] = np.where(closes > 0, (highs - lows) / closes * 100, 0)
    
    # Previous 3 bars direction
    if n > 3:
        dir_3 = np.sign(closes[1:] - closes[:-1])
        dir_3_sum = pd.Series(np.concatenate([[0], dir_3])).rolling(3).sum().values
        features["prev_3_dir"] = dir_3_sum
    else:
        features["prev_3_dir"] = np.zeros(n)
    
    # Consecutive same-direction bars
    if n > 1:
        dirs = np.sign(np.diff(closes))
        consec = np.zeros(n)
        for i in range(1, n):
            if i > 0 and np.sign(closes[i] - closes[i-1]) == np.sign(closes[i-1] - closes[i-2]) if i > 1 else False:
                consec[i] = consec[i-1] + 1
            else:
                consec[i] = 0
        features["consecutive_bars"] = consec
    else:
        features["consecutive_bars"] = np.zeros(n)
    
    # Candle body vs wick ratio
    body = np.abs(closes - df["open"].values)
    wick = np.maximum(highs - np.maximum(closes, df["open"].values), 
                      np.minimum(closes, df["open"].values) - lows)
    features["body_wick_ratio"] = np.where(wick > 0, body / (body + wick), 0.5)
    
    return features


def prepare_h4_data() -> Optional[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """Prepare H4 data: features + target for trend prediction.
    
    Target: 1 if close 4 bars ahead > current close, else 0
    Horizon: 4 H4 bars = ~16 hours
    """
    df = load_tf_data(240)  # H4
    if df is None or len(df) < 500:
        return None
    
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    volumes = df.get("volume", df.get("tick_volume", pd.Series(np.ones(len(df))))).values.astype(np.float64)
    times = df.index.values
    
    # Compute features
    feat_dict = add_technical_features(df, closes, highs, lows, volumes)
    
    # H4-specific features
    n = len(closes)
    
    # H1 trend slope
    h1_df = load_tf_data(60)
    if h1_df is not None:
        h1_closes = h1_df["close"].values.astype(np.float64)
        h1_times = h1_df.index
        # Resample: align H1 slope to H4 bars
        h1_returns = pd.Series(np.diff(h1_closes) / h1_closes[:-1], index=h1_times[1:])
        h1_slope = h1_returns.rolling(4).mean().resample("4h").mean()
        # Align to H4 index
        h4_idx = df.index
        aligned_slope = h1_slope.reindex(h4_idx, method="ffill").values
        feat_dict["h1_trend_slope"] = aligned_slope[:n]
    else:
        feat_dict["h1_trend_slope"] = np.zeros(n)
    
    # M30 momentum
    m30_df = load_tf_data(30)
    if m30_df is not None:
        m30_closes = m30_df["close"].values.astype(np.float64)
        m30_returns = pd.Series(np.diff(m30_closes) / m30_closes[:-1], index=m30_df.index[1:])
        m30_mom = m30_returns.rolling(2).mean().resample("30min").mean()
        # Resample from 30min to 4H
        m30_mom_4h = m30_mom.resample("4h").mean()
        h4_idx = df.index
        aligned_mom = m30_mom_4h.reindex(h4_idx, method="ffill").values
        feat_dict["m30_momentum"] = aligned_mom[:n]
    else:
        feat_dict["m30_momentum"] = np.zeros(n)
    
    # M30 close vs H4 EMA20
    if m30_df is not None:
        m30_close = m30_df["close"].resample("4h").last()
        h4_ema20 = feat_dict["ema20"]
        h4_ema20_series = pd.Series(h4_ema20, index=df.index)
        aligned_m30_close = m30_close.reindex(df.index, method="ffill").values[:n]
        feat_dict["m30_close_vs_h4_ema20"] = np.where(
            h4_ema20_series.values[:n] > 0,
            (aligned_m30_close - h4_ema20_series.values[:n]) / h4_ema20_series.values[:n] * 100,
            0
        )
    else:
        feat_dict["m30_close_vs_h4_ema20"] = np.zeros(n)
    
    # H1 RSI
    if h1_df is not None:
        h1_feat = add_technical_features(h1_df, 
                                          h1_df["close"].values.astype(np.float64),
                                          h1_df["high"].values.astype(np.float64),
                                          h1_df["low"].values.astype(np.float64),
                                          h1_df.get("volume", h1_df.get("tick_volume", pd.Series(np.ones(len(h1_df))))).values.astype(np.float64))
        h1_rsi = pd.Series(h1_feat["rsi14"], index=h1_df.index).resample("4h").last()
        feat_dict["h1_rsi"] = h1_rsi.reindex(df.index, method="ffill").values[:n]
    else:
        feat_dict["h1_rsi"] = np.full(n, 50)
    
    # Build feature matrix for H4
    feature_cols = CONFIG.H4_FEATURES
    X = np.column_stack([feat_dict.get(col, np.zeros(n)) for col in feature_cols])
    
    # Target: 1 if close 4 bars ahead > current close
    target = np.zeros(n, dtype=np.int32)
    for i in range(n - 4):
        target[i] = 1 if closes[i + 4] > closes[i] else 0
    # Last 4 bars have no target
    target[-4:] = -1
    
    # Remove rows with NaN
    valid = ~np.isnan(X).any(axis=1) & (target != -1)
    X, target, valid_times = X[valid], target[valid], times[valid]
    
    print(f"[H4] Total samples: {len(X)}, Bull: {target.sum()} ({target.mean()*100:.1f}%)")
    return X, target, pd.DatetimeIndex(valid_times)


def prepare_h1_data() -> Optional[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """Prepare H1 data: features + target for entry signal.
    
    Target: 1 if cumulative return over 2 bars > 10 pips (0.0010), else 0
    Horizon: 2 H1 bars = ~2 hours
    """
    df = load_tf_data(60)  # H1
    if df is None or len(df) < 1000:
        return None
    
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    volumes = df.get("volume", df.get("tick_volume", pd.Series(np.ones(len(df))))).values.astype(np.float64)
    times = df.index.values
    n = len(closes)
    
    # Compute features
    feat_dict = add_technical_features(df, closes, highs, lows, volumes)
    
    # H4 trend (from H4 model output or simple EMA)
    h4_df = load_tf_data(240)
    if h4_df is not None:
        h4_ema20 = pd.Series(h4_df["close"].values.astype(np.float64)).ewm(span=20).mean().values
        h4_current = h4_df["close"].values.astype(np.float64)
        h4_trend = (h4_current > h4_ema20).astype(np.int32)
        h4_trend_series = pd.Series(h4_trend, index=h4_df.index).resample("1h").ffill()
        feat_dict["h4_trend"] = h4_trend_series.reindex(df.index, method="ffill").values[:n]
    else:
        feat_dict["h4_trend"] = np.zeros(n)
    
    # EMA5/20 cross
    ema5 = pd.Series(closes).ewm(span=5).mean().values
    ema20 = feat_dict["ema20"]
    # Above = 1, below = -1
    ema5_above = np.sign(ema5 - ema20)
    if n > 1:
        ema5_prev = np.concatenate([[0], ema5_above[:-1]])
        cross = np.where((ema5_above == 1) & (ema5_prev == -1), 1,  # Bull cross
                        np.where((ema5_above == -1) & (ema5_prev == 1), -1, 0))  # Bear cross
        feat_dict["ema5_20_cross"] = cross
    else:
        feat_dict["ema5_20_cross"] = np.zeros(n)
    
    # Stochastics K
    if n > 14:
        low_14 = pd.Series(lows).rolling(14).min().values
        high_14 = pd.Series(highs).rolling(14).max().values
        stoch_k = np.where(high_14 != low_14, 
                          (closes - low_14) / (high_14 - low_14) * 100,
                          50)
        feat_dict["stoch_k"] = stoch_k
    else:
        feat_dict["stoch_k"] = np.full(n, 50)
    
    # Volume spike
    vol_avg = pd.Series(volumes).rolling(20).mean().values
    feat_dict["volume_spike"] = np.where(vol_avg > 0, volumes / vol_avg, 1.0)
    
    # M5 price position (last 12 M5 bars)
    m5_df = load_tf_data(5)
    if m5_df is not None and len(m5_df) > 0:
        m5_high = m5_df["high"].resample("1h").max()
        m5_low = m5_df["low"].resample("1h").min()
        m5_close = m5_df["close"].resample("1h").last()
        # Price position within hour: 0=low, 1=high
        price_pos = pd.Series(
            np.where(m5_high.values != m5_low.values, 
                    (m5_close.values - m5_low.values) / (m5_high.values - m5_low.values + 1e-10), 0.5),
            index=m5_high.index
        )
        feat_dict["m5_price_pos_12bar"] = price_pos.reindex(df.index, method="ffill").values[:n]
    else:
        feat_dict["m5_price_pos_12bar"] = np.full(n, 0.5)
    
    # Previous bar direction
    if n > 1:
        prev_dir = np.sign(closes[1:] - closes[:-1])
        feat_dict["prev_bar_dir"] = np.concatenate([[0], prev_dir])
    else:
        feat_dict["prev_bar_dir"] = np.zeros(n)
    
    # Range vs ATR
    feat_dict["range_vs_atr"] = np.where(feat_dict["atr"] > 0, 
                                         (highs - lows) / feat_dict["atr"], 1.0)
    
    # Session features
    hour_of_day = pd.Series(df.index.hour, index=df.index)
    feat_dict["session_asia"] = ((hour_of_day >= 0) & (hour_of_day < 8)).astype(np.int32).values
    feat_dict["session_london"] = ((hour_of_day >= 8) & (hour_of_day < 16)).astype(np.int32).values
    feat_dict["session_overlap"] = ((hour_of_day >= 12) & (hour_of_day < 16)).astype(np.int32).values
    
    # Build feature matrix for H1
    feature_cols = CONFIG.H1_FEATURES
    X = np.column_stack([feat_dict.get(col, np.zeros(n)) for col in feature_cols])
    
    # Target: 1 if 2-bar forward price goes UP (any positive)
    # Direction prediction is more learnable than magnitude
    target = np.zeros(n, dtype=np.int32)
    for i in range(n - 2):
        cum_return = (closes[i + 2] - closes[i]) / closes[i]
        target[i] = 1 if cum_return > 0 else 0
    target[-2:] = -1
    
    # Remove rows with NaN
    valid = ~np.isnan(X).any(axis=1) & (target != -1)
    X, target, valid_times = X[valid], target[valid], times[valid]
    
    print(f"[H1] Total samples: {len(X)}, Win: {target.sum()} ({target.mean()*100:.1f}%)")
    return X, target, pd.DatetimeIndex(valid_times)


def prepare_m5_data() -> Optional[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """Prepare M5 data: features + target for pullback prediction.
    
    Target: 1 if price drops > 3 pips within next 6 bars (30 min), else 0
    """
    df = load_tf_data(5)  # M5
    if df is None or len(df) < 5000:
        return None
    
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    volumes = df.get("volume", df.get("tick_volume", pd.Series(np.ones(len(df))))).values.astype(np.float64)
    times = df.index.values
    n = len(closes)
    
    # Compute features
    feat_dict = add_technical_features(df, closes, highs, lows, volumes)
    
    # RSI direction (change from previous)
    rsi = feat_dict["rsi14"]
    if n > 1:
        rsi_dir = np.concatenate([[0], np.diff(rsi)])
        feat_dict["rsi_direction"] = rsi_dir
    else:
        feat_dict["rsi_direction"] = np.zeros(n)
    
    # Price vs MA20
    feat_dict["price_vs_ma20"] = feat_dict["price_pos_vs_ema20"]
    
    # Volume spike
    vol_avg = pd.Series(volumes).rolling(20).mean().values
    feat_dict["volume_spike"] = np.where(vol_avg > 0, volumes / vol_avg, 1.0)
    
    # H1 trend direction
    h1_df = load_tf_data(60)
    if h1_df is not None:
        h1_closes = h1_df["close"].values.astype(np.float64)
        h1_ema20 = pd.Series(h1_closes).ewm(span=20).mean().values
        h1_trend = (h1_closes[-1] > h1_ema20[-1]).astype(np.int32) if len(h1_closes) > 0 else 0
        feat_dict["h1_trend"] = np.full(n, h1_trend)
    else:
        feat_dict["h1_trend"] = np.zeros(n)
    
    # Spread
    feat_dict["spread"] = df.get("spread", pd.Series(np.full(n, 16.6))).values.astype(np.float64)
    
    # Build feature matrix for M5
    feature_cols = CONFIG.M5_FEATURES
    X = np.column_stack([feat_dict.get(col, np.zeros(n)) for col in feature_cols])
    
    # Target: 1 if min price in next 6 bars drops > 3 pips (0.0003)
    THRESHOLD = 0.0003  # 3 pips
    target = np.zeros(n, dtype=np.int32)
    for i in range(n - 6):
        future_min = np.min(closes[i:i+6])
        target[i] = 1 if (future_min < closes[i] - THRESHOLD) else 0
    target[-6:] = -1
    
    # Remove rows with NaN
    valid = ~np.isnan(X).any(axis=1) & (target != -1)
    X, target, valid_times = X[valid], target[valid], times[valid]
    
    print(f"[M5] Total samples: {len(X)}, Pullback: {target.sum()} ({target.mean()*100:.1f}%)")
    return X, target, pd.DatetimeIndex(valid_times)
