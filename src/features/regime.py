"""
Market regime detection features.

Includes:
- Hurst exponent (trending vs mean-reverting)
- Volatility regime (rolling percentile)
- ADX-based trend strength
- Realized volatility ratio (short vs long)
- Day-of-week / hour-of-day cyclical encodings
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ── Hurst Exponent ────────────────────────────────────────────────────────────

def _hurst_rs(series: np.ndarray) -> float:
    """
    Estimate Hurst exponent via R/S analysis on a log-return series.
    H > 0.5 → trending, H < 0.5 → mean-reverting, H ≈ 0.5 → random walk.
    Returns NaN if insufficient data.
    """
    n = len(series)
    if n < 20:
        return np.nan

    lags = [2, 4, 8, 16, 32]
    lags = [l for l in lags if l < n // 2]
    if len(lags) < 3:
        return np.nan

    rs_vals = []
    for lag in lags:
        sub_rs = []
        for start in range(0, n - lag, lag):
            chunk = series[start : start + lag]
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            dev_cs = np.cumsum(chunk - mean)
            r = dev_cs.max() - dev_cs.min()
            s = chunk.std(ddof=1)
            if s > 0:
                sub_rs.append(r / s)
        if sub_rs:
            rs_vals.append(np.mean(sub_rs))

    if len(rs_vals) < 3:
        return np.nan

    log_lags = np.log(lags[: len(rs_vals)])
    log_rs = np.log(rs_vals)
    slope, _, r, _, _ = linregress(log_lags, log_rs)
    return float(np.clip(slope, 0.01, 0.99))


def rolling_hurst(series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling Hurst exponent on log returns."""
    log_ret = np.log(series / series.shift(1)).fillna(0)
    # raw=True passes numpy array directly; _hurst_rs expects numpy array
    return log_ret.rolling(window).apply(lambda x: _hurst_rs(x), raw=True)


# ── Volatility Regime ─────────────────────────────────────────────────────────

def volatility_regime(
    log_ret: pd.Series,
    short_window: int = 24,
    long_window: int = 168,
    pct_window: int = 720,
) -> pd.DataFrame:
    """
    Classify volatility into regime:
    - realized_vol_short: 24h realized vol
    - realized_vol_long: 168h realized vol
    - vol_ratio: short/long (>1 = vol expansion)
    - vol_regime_pct: percentile of current short vol over 30d rolling
    """
    rv_short = log_ret.rolling(short_window).std()
    rv_long = log_ret.rolling(long_window).std()
    vol_ratio = rv_short / rv_long.replace(0, np.nan)
    vol_pct = rv_short.rolling(pct_window).rank(pct=True)

    return pd.DataFrame(
        {
            "vol_short": rv_short,
            "vol_long": rv_long,
            "vol_ratio": vol_ratio,
            "vol_regime_pct": vol_pct,
        }
    )


# ── Calendar Features ─────────────────────────────────────────────────────────

def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Cyclically encoded time features — important for intraday crypto patterns.
    Sine/cosine encoding avoids artificial discontinuities.
    """
    hour = index.hour
    dow = index.dayofweek
    month = index.month
    woy = index.isocalendar().week.astype(int)

    feats = {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "woy_sin": np.sin(2 * np.pi * woy / 52),
        "woy_cos": np.cos(2 * np.pi * woy / 52),
        # Raw (useful for GBDT which can split on exact hour)
        "hour": hour.astype(float),
        "dow": dow.astype(float),
        "is_weekend": (dow >= 5).astype(float),
        "is_asian_session": ((hour >= 0) & (hour < 8)).astype(float),
        "is_london_session": ((hour >= 7) & (hour < 16)).astype(float),
        "is_ny_session": ((hour >= 13) & (hour < 22)).astype(float),
    }
    return pd.DataFrame(feats, index=index)


# ── Main regime feature builder ───────────────────────────────────────────────

def compute_regime_features(
    df: pd.DataFrame,
    hurst_window: int = 100,
) -> pd.DataFrame:
    """
    Compute regime features for 1h OHLCV DataFrame.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))

    parts = []

    # Hurst exponent
    h = rolling_hurst(df["close"], hurst_window)
    h_df = pd.DataFrame({"hurst": h}, index=df.index)
    h_df["hurst_trending"] = (h > 0.55).astype(float)
    h_df["hurst_reverting"] = (h < 0.45).astype(float)
    parts.append(h_df)

    # Volatility regime
    vr = volatility_regime(log_ret)
    parts.append(vr)

    # Calendar features
    cal = calendar_features(df.index)
    parts.append(cal)

    # Autocorrelation features (for regime detection)
    for lag in [1, 2, 3, 12, 24]:
        auto = log_ret.rolling(48).apply(
            lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else np.nan,
            raw=False,
        )
        # rolling autocorr via manual formula
        r = log_ret
        r_lag = log_ret.shift(lag)
        # rolling correlation = rolling covariance / (std * std)
        window = 48
        cov = (r * r_lag).rolling(window).mean() - r.rolling(window).mean() * r_lag.rolling(window).mean()
        std1 = r.rolling(window).std().replace(0, np.nan)
        std2 = r_lag.rolling(window).std().replace(0, np.nan)
        autocorr_feat = pd.DataFrame(
            {f"autocorr_{lag}": cov / (std1 * std2)}, index=df.index
        )
        parts.append(autocorr_feat)

    result = pd.concat(parts, axis=1)
    return result
