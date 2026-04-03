"""
Cross-asset features: correlations, leadership, relative strength, betas.

At any time t, these features capture how each asset relates to others —
e.g. BTC's 1h return as a leading indicator for alts.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_cross_asset_features(
    hourly_data: Dict[str, pd.DataFrame],
    target_symbol: str,
    corr_windows: tuple = (24, 168),
    btc_symbol: str = "BTC/USDT",
) -> pd.DataFrame:
    """
    Compute cross-asset features for `target_symbol`.

    Parameters
    ----------
    hourly_data : dict mapping symbol → 1h OHLCV DataFrame (all must share same index range)
    target_symbol : the asset we are predicting
    corr_windows : rolling window sizes for correlation
    btc_symbol : reference asset (usually BTC)
    """
    # Build close-price panel aligned to a common index
    closes = {}
    for sym, df in hourly_data.items():
        if not df.empty and "close" in df.columns:
            closes[sym] = df["close"]

    if len(closes) < 2:
        return pd.DataFrame()

    panel = pd.DataFrame(closes).sort_index()
    log_rets = np.log(panel / panel.shift(1))

    target_ret = log_rets[target_symbol] if target_symbol in log_rets.columns else None
    btc_ret = log_rets[btc_symbol] if btc_symbol in log_rets.columns else None
    others = [s for s in log_rets.columns if s != target_symbol]

    feats = {}
    ref_index = log_rets.index

    # ── BTC leadership features ───────────────────────────────────────────────
    if btc_ret is not None:
        # BTC lagged returns (already-shifted = known at bar close)
        for lag in [1, 2, 3]:
            feats[f"btc_ret_lag{lag}"] = btc_ret.shift(lag)

        # BTC direction (up/down) as categorical signal
        feats["btc_up_lag1"] = (btc_ret.shift(1) > 0).astype(float)

        # BTC short-term momentum
        for w in [4, 12, 24]:
            feats[f"btc_mom_{w}"] = btc_ret.shift(1).rolling(w).sum()

    # ── Relative strength vs BTC ───────────────────────────────────────────────
    if target_ret is not None and btc_ret is not None and target_symbol != btc_symbol:
        for w in [12, 24, 48, 168]:
            target_cum = target_ret.shift(1).rolling(w).sum()
            btc_cum = btc_ret.shift(1).rolling(w).sum()
            feats[f"rel_strength_btc_{w}"] = target_cum - btc_cum
            feats[f"rel_strength_btc_{w}_sign"] = np.sign(feats[f"rel_strength_btc_{w}"])

    # ── Rolling correlations ──────────────────────────────────────────────────
    if target_ret is not None:
        for other_sym in others[:5]:  # limit to avoid explosion
            if other_sym not in log_rets.columns:
                continue
            other_ret = log_rets[other_sym]
            safe_name = other_sym.split("/")[0].lower()
            for w in corr_windows:
                corr = target_ret.rolling(w).corr(other_ret)
                feats[f"corr_{safe_name}_{w}"] = corr.shift(1)

    # ── Rolling beta to BTC ───────────────────────────────────────────────────
    if target_ret is not None and btc_ret is not None and target_symbol != btc_symbol:
        for w in [24, 168]:
            def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
                cov = y.rolling(window).cov(x)
                var = x.rolling(window).var().replace(0, np.nan)
                return cov / var
            feats[f"beta_btc_{w}"] = rolling_beta(target_ret, btc_ret, w).shift(1)

    # ── Market breadth: fraction of assets that are up this bar ──────────────
    n_up = (log_rets.shift(1) > 0).sum(axis=1)
    feats["market_breadth"] = n_up / len(log_rets.columns)

    # ── Cross-sectional momentum rank ─────────────────────────────────────────
    if target_ret is not None:
        for w in [12, 24]:
            rolling_sum = log_rets.shift(1).rolling(w).sum()
            # rank of target among all assets (0=weakest, 1=strongest)
            feats[f"cs_rank_{w}"] = rolling_sum.rank(axis=1, pct=True)[target_symbol]

    # ── Average correlation (connectedness) ───────────────────────────────────
    if len(others) >= 2 and target_ret is not None:
        corrs_24 = []
        for s in others[:5]:
            if s in log_rets.columns:
                corrs_24.append(target_ret.rolling(24).corr(log_rets[s]))
        if corrs_24:
            feats["avg_corr_24"] = pd.concat(corrs_24, axis=1).mean(axis=1).shift(1)

    df_out = pd.DataFrame(feats, index=ref_index)
    return df_out
