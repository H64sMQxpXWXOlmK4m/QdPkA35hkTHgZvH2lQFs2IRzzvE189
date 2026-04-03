"""
Technical indicators computed on a single OHLCV DataFrame.

All functions accept a DataFrame with columns:
  open, high, low, close, volume
and return a DataFrame of features (same index).

No lookahead: every indicator uses only past and current bar data.
The caller is responsible for shifting by 1 period when needed to ensure
that prediction at time t uses only information known at time t.
"""

import numpy as np
import pandas as pd


# ── Helper utilities ──────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    h = df["high"]
    l = df["low"]
    pc = df["close"].shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


# ── Individual indicators ─────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    line = ema_fast - ema_slow
    sig = _ema(line, signal)
    hist = line - sig
    return pd.DataFrame(
        {"macd_line": line, "macd_signal": sig, "macd_hist": hist},
        index=close.index,
    )


def bollinger_bands(
    close: pd.Series, period: int = 20, n_std: float = 2.0
) -> pd.DataFrame:
    mid = _sma(close, period)
    std = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
            "bb_pctb": pct_b,
        },
        index=close.index,
    )


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _true_range(df)
    return tr.ewm(span=period, adjust=False).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["close"] - low_min) / denom
    d = _sma(k, d_period)
    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=df.index)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = _sma(tp, period)
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    h = df["high"]
    l = df["low"]
    pc = df["close"].shift(1)

    plus_dm = (h - h.shift(1)).clip(lower=0)
    minus_dm = (l.shift(1) - l).clip(lower=0)
    # Only keep the dominant DM
    mask = plus_dm >= minus_dm
    plus_dm = plus_dm.where(mask, 0)
    minus_dm = minus_dm.where(~mask, 0)

    tr = _true_range(df)
    atr_val = tr.ewm(span=period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    highest_h = df["high"].rolling(period).max()
    lowest_l = df["low"].rolling(period).min()
    denom = (highest_h - lowest_l).replace(0, np.nan)
    return -100 * (highest_h - df["close"]) / denom


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff())
    obv = (direction * df["volume"]).fillna(0).cumsum()
    return obv


def donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    mid = (upper + lower) / 2
    pct = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"donch_upper": upper, "donch_lower": lower, "donch_mid": mid, "donch_pct": pct},
        index=df.index,
    )


def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, mult: float = 2.0) -> pd.DataFrame:
    mid = _ema(df["close"], ema_period)
    atr_val = atr(df, atr_period)
    upper = mid + mult * atr_val
    lower = mid - mult * atr_val
    pct = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"kelt_upper": upper, "kelt_lower": lower, "kelt_mid": mid, "kelt_pct": pct},
        index=df.index,
    )


def vwap_rolling(df: pd.DataFrame, period: int = 24) -> pd.Series:
    """Rolling VWAP over `period` bars (approximation using close * volume)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    numerator = (tp * df["volume"]).rolling(period).sum()
    denominator = df["volume"].rolling(period).sum().replace(0, np.nan)
    return numerator / denominator


def rate_of_change(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period)


def momentum(close: pd.Series, period: int) -> pd.Series:
    return close - close.shift(period)


# ── Main feature builder ──────────────────────────────────────────────────────

def compute_technical_features(
    df: pd.DataFrame,
    tf_prefix: str = "",
    rsi_periods: list = (7, 14, 21),
    atr_periods: list = (7, 14),
    return_lags: list = (1, 2, 3, 4, 6, 8, 12, 24, 48, 168),
    vol_windows: list = (6, 12, 24, 48, 168),
) -> pd.DataFrame:
    """
    Compute full set of technical features for one timeframe.
    All features are price-agnostic (normalized/ratio/percentage).

    Parameters
    ----------
    df : OHLCV DataFrame
    tf_prefix : string prefix added to all column names, e.g. "4h_"
    """
    p = tf_prefix + "_" if tf_prefix else ""
    feats = {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    open_ = df["open"]

    # ── Returns ───────────────────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    for lag in return_lags:
        feats[f"{p}ret_{lag}"] = close.pct_change(lag)
        feats[f"{p}logret_{lag}"] = np.log(close / close.shift(lag))

    # ── Candle body features ──────────────────────────────────────────────────
    feats[f"{p}body"] = (close - open_) / open_.replace(0, np.nan)
    feats[f"{p}upper_shadow"] = (high - close.clip(lower=open_)) / open_.replace(0, np.nan)
    feats[f"{p}lower_shadow"] = (open_.clip(upper=close) - low) / open_.replace(0, np.nan)
    feats[f"{p}hl_range"] = (high - low) / open_.replace(0, np.nan)
    feats[f"{p}gap"] = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)

    # ── Volatility ────────────────────────────────────────────────────────────
    for w in vol_windows:
        rv = log_ret.rolling(w).std()
        feats[f"{p}vol_{w}"] = rv
        # Volatility regime: current vol percentile over 30-day rolling window
        # (computed as quantile transform — bounded 0-1)
        rank_window = max(w * 4, 168)
        feats[f"{p}vol_rank_{w}"] = rv.rolling(rank_window).rank(pct=True)

    # ── RSI ───────────────────────────────────────────────────────────────────
    for period in rsi_periods:
        r = rsi(close, period)
        feats[f"{p}rsi_{period}"] = r / 100.0  # normalize to [0,1]
        # RSI deviation from neutral
        feats[f"{p}rsi_{period}_dev"] = (r - 50) / 50

    # ── MACD ─────────────────────────────────────────────────────────────────
    m = macd(close)
    feats[f"{p}macd_line"] = m["macd_line"] / close.replace(0, np.nan)
    feats[f"{p}macd_hist"] = m["macd_hist"] / close.replace(0, np.nan)
    feats[f"{p}macd_hist_chg"] = m["macd_hist"].diff() / close.replace(0, np.nan)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = bollinger_bands(close)
    feats[f"{p}bb_pctb"] = bb["bb_pctb"]
    feats[f"{p}bb_width"] = bb["bb_width"]

    # ── ATR ───────────────────────────────────────────────────────────────────
    for period in atr_periods:
        a = atr(df, period)
        feats[f"{p}atr_{period}"] = a / close.replace(0, np.nan)  # normalized

    # ── Stochastic ────────────────────────────────────────────────────────────
    st = stochastic(df)
    feats[f"{p}stoch_k"] = st["stoch_k"] / 100.0
    feats[f"{p}stoch_d"] = st["stoch_d"] / 100.0
    feats[f"{p}stoch_kd_diff"] = (st["stoch_k"] - st["stoch_d"]) / 100.0

    # ── CCI ───────────────────────────────────────────────────────────────────
    c = cci(df)
    feats[f"{p}cci"] = c / 200.0  # scale to roughly [-1, 1]

    # ── ADX ───────────────────────────────────────────────────────────────────
    a_df = adx(df)
    feats[f"{p}adx"] = a_df["adx"] / 100.0
    feats[f"{p}di_diff"] = (a_df["plus_di"] - a_df["minus_di"]) / 100.0

    # ── Williams %R ───────────────────────────────────────────────────────────
    feats[f"{p}willr"] = williams_r(df) / 100.0 + 0.5  # center at 0.5

    # ── OBV ───────────────────────────────────────────────────────────────────
    obv = on_balance_volume(df)
    obv_sma = _sma(obv, 20)
    feats[f"{p}obv_slope"] = (obv - obv.shift(5)) / (obv.abs().rolling(20).mean() + 1e-9)

    # ── Donchian ──────────────────────────────────────────────────────────────
    don = donchian_channels(df)
    feats[f"{p}donch_pct"] = don["donch_pct"]

    # ── Keltner ───────────────────────────────────────────────────────────────
    kelt = keltner_channels(df)
    feats[f"{p}kelt_pct"] = kelt["kelt_pct"]

    # ── VWAP ─────────────────────────────────────────────────────────────────
    vwap = vwap_rolling(df, 24)
    feats[f"{p}vwap_dev"] = (close - vwap) / vwap.replace(0, np.nan)

    # ── Moving average features ───────────────────────────────────────────────
    for w in [7, 20, 50, 200]:
        ma = _sma(close, w)
        feats[f"{p}sma_{w}_dev"] = (close - ma) / ma.replace(0, np.nan)

    for w in [9, 21, 55, 200]:
        ema_val = _ema(close, w)
        feats[f"{p}ema_{w}_dev"] = (close - ema_val) / ema_val.replace(0, np.nan)

    # MA crosses (sign of difference)
    feats[f"{p}sma_7_20_cross"] = _sma(close, 7) / _sma(close, 20).replace(0, np.nan) - 1
    feats[f"{p}sma_20_50_cross"] = _sma(close, 20) / _sma(close, 50).replace(0, np.nan) - 1
    feats[f"{p}ema_9_21_cross"] = _ema(close, 9) / _ema(close, 21).replace(0, np.nan) - 1

    # ── Price position in rolling range ───────────────────────────────────────
    for w in [12, 24, 48, 168]:
        rolling_low = low.rolling(w).min()
        rolling_high = high.rolling(w).max()
        denom = (rolling_high - rolling_low).replace(0, np.nan)
        feats[f"{p}price_pos_{w}"] = (close - rolling_low) / denom

    # ── Rate of change ────────────────────────────────────────────────────────
    for p_lag in [5, 10, 20]:
        feats[f"{p}roc_{p_lag}"] = rate_of_change(close, p_lag)

    # ── Volume features ───────────────────────────────────────────────────────
    for w in [6, 12, 24]:
        vol_ma = _sma(volume, w)
        feats[f"{p}vol_ratio_{w}"] = volume / vol_ma.replace(0, np.nan)
    feats[f"{p}vol_chg"] = volume.pct_change()
    # Volume-weighted return
    feats[f"{p}vol_price_trend"] = log_ret * (volume / volume.rolling(24).mean().replace(0, np.nan))

    result = pd.DataFrame(feats, index=df.index)
    return result
