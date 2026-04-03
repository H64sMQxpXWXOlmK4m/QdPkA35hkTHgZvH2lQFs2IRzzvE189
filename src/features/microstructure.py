"""
Microstructure features derived from Binance kline data.

Binance provides per-candle: taker_buy_base, taker_buy_quote, num_trades,
quote_volume. These reveal order-flow imbalance — a powerful short-term signal.
"""

import numpy as np
import pandas as pd


def compute_microstructure_features(
    df: pd.DataFrame,
    tf_prefix: str = "",
    windows: tuple = (3, 6, 12, 24),
) -> pd.DataFrame:
    """
    Compute order-flow microstructure features.

    Required columns in df: volume, quote_volume, num_trades,
    taker_buy_base, taker_buy_quote  (standard Binance kline columns).
    Falls back gracefully if these are missing.
    """
    p = tf_prefix + "_" if tf_prefix else ""
    feats = {}

    has_micro = all(c in df.columns for c in ["taker_buy_base", "quote_volume", "num_trades"])

    if not has_micro:
        # No extended data; return empty
        return pd.DataFrame(index=df.index)

    vol = df["volume"].replace(0, np.nan)
    qvol = df["quote_volume"].replace(0, np.nan)
    buy_base = df["taker_buy_base"]
    buy_quote = df["taker_buy_quote"]
    sell_base = vol - buy_base
    sell_quote = qvol - buy_quote
    trades = df["num_trades"].replace(0, np.nan)

    # ── Taker buy ratio (base volume) ─────────────────────────────────────────
    tbr = buy_base / vol  # 0-1, >0.5 means more buying
    feats[f"{p}taker_buy_ratio"] = tbr
    feats[f"{p}taker_buy_ratio_dev"] = tbr - 0.5  # deviation from neutral

    # ── Quote taker buy ratio ─────────────────────────────────────────────────
    qtbr = buy_quote / qvol
    feats[f"{p}quote_taker_ratio"] = qtbr

    # ── Average trade size (buy vs sell) ─────────────────────────────────────
    # Proxy: separate buy trades aren't available, but avg size = vol/trades
    feats[f"{p}avg_trade_size"] = vol / trades
    feats[f"{p}avg_trade_size_norm"] = (vol / trades) / (vol / trades).rolling(24).mean().replace(0, np.nan)

    # ── Volume imbalance (buy pressure) ──────────────────────────────────────
    denom = (buy_base + sell_base).replace(0, np.nan)
    feats[f"{p}vol_imbalance"] = (buy_base - sell_base) / denom

    quote_denom = (buy_quote + sell_quote).replace(0, np.nan)
    feats[f"{p}quote_imbalance"] = (buy_quote - sell_quote) / quote_denom

    # ── Rolling aggregates of buy ratio ──────────────────────────────────────
    for w in windows:
        roll_tbr = tbr.rolling(w).mean()
        feats[f"{p}tbr_ma_{w}"] = roll_tbr
        feats[f"{p}tbr_dev_{w}"] = tbr - roll_tbr          # deviation from short MA
        feats[f"{p}tbr_std_{w}"] = tbr.rolling(w).std()

    # ── Cumulative order-flow indicator (like OBV but with buy ratio) ─────────
    # Signed volume: positive for buy-dominated, negative for sell-dominated
    signed_vol = (2 * tbr - 1) * vol
    feats[f"{p}signed_vol"] = signed_vol / vol  # normalized per bar
    roll_sv_24 = signed_vol.rolling(24).sum()
    roll_vol_24 = vol.rolling(24).sum().replace(0, np.nan)
    feats[f"{p}cum_flow_24"] = roll_sv_24 / roll_vol_24

    # ── Trade intensity ───────────────────────────────────────────────────────
    feats[f"{p}trade_intensity"] = trades / trades.rolling(24).mean().replace(0, np.nan)
    feats[f"{p}trade_intensity_chg"] = trades.pct_change()

    # ── Price impact proxy ────────────────────────────────────────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))
    feats[f"{p}price_impact"] = log_ret / signed_vol.replace(0, np.nan).abs()
    feats[f"{p}ret_per_trade"] = log_ret / trades

    # ── Amihud illiquidity ratio ──────────────────────────────────────────────
    feats[f"{p}amihud"] = log_ret.abs() / qvol

    # ── Rolling z-score of imbalance ─────────────────────────────────────────
    for w in [12, 24]:
        imb = feats[f"{p}vol_imbalance"]
        imb_ma = imb.rolling(w).mean()
        imb_std = imb.rolling(w).std().replace(0, np.nan)
        feats[f"{p}imb_zscore_{w}"] = (imb - imb_ma) / imb_std

    return pd.DataFrame(feats, index=df.index)
