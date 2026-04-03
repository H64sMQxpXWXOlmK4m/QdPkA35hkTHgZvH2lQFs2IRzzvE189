"""
Master feature pipeline.

Orchestrates technical, microstructure, cross-asset, and regime features
across multiple timeframes, aligns everything to the 1h index, and enforces
strict no-lookahead alignment.

Convention:
  - For the 1h bar ending at time T (opened at T-1h, closed at T):
    * target  = is close_T > open_{T-1h}?  (i.e. is bar [T-1h, T] an up-bar?)
    * features = computed from data strictly BEFORE T, i.e. bar T-1h and earlier
  - In practice: compute indicators on the full series, then shift each
    feature column by +1 so that bar index i uses only information through bar i-1.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import FEATURE_CONFIG, SYMBOLS
from src.features.technical import compute_technical_features
from src.features.microstructure import compute_microstructure_features
from src.features.cross_asset import compute_cross_asset_features
from src.features.regime import compute_regime_features

logger = logging.getLogger(__name__)


def _resample_to_1h(df: pd.DataFrame, source_tf: str) -> pd.DataFrame:
    """
    For timeframes coarser than 1h (e.g. 4h, 1d), forward-fill each bar
    value for all constituent 1h bars.  This makes the multi-timeframe
    merge straightforward.
    """
    if source_tf in ("1h",):
        return df
    # For 4h: each bar covers 4 1h slots; forward-fill
    # We assume the 4h bar CLOSES at its timestamp, so it is known at that
    # 1h index and all subsequent 1h bars until the next 4h close.
    return df.resample("1h").last().ffill()


def _align_and_shift(feat_df: pd.DataFrame, ref_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Reindex to ref_index (1h), shift by +1 to eliminate lookahead,
    then forward-fill NAs arising from the reindex.
    """
    aligned = feat_df.reindex(ref_index).ffill()
    # Shift by 1: feature at bar i reflects information through bar i-1
    return aligned.shift(1)


def build_feature_matrix(
    symbol: str,
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    ref_symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the complete feature matrix for `symbol`.

    Parameters
    ----------
    symbol : e.g. "BTC/USDT"
    all_data : nested dict: all_data[symbol][timeframe] = OHLCV DataFrame
    ref_symbol : passed through for cross-asset (defaults to symbol)

    Returns
    -------
    DataFrame indexed by hourly timestamps.
    All features are pre-shifted: feature at row t uses data known at t-1h.
    (After this, build_labels() creates target for bar starting at that row.)
    """
    sym_data = all_data.get(symbol, {})
    df_1h = sym_data.get("1h", pd.DataFrame())
    if df_1h.empty:
        raise ValueError(f"No 1h data for {symbol}")

    ref_index = df_1h.index  # canonical index
    parts = []

    # ── 1h technical features ─────────────────────────────────────────────────
    logger.info(f"  [{symbol}] Computing 1h technical features...")
    feats_1h = compute_technical_features(
        df_1h,
        tf_prefix="1h",
        rsi_periods=FEATURE_CONFIG.rsi_periods,
        atr_periods=FEATURE_CONFIG.atr_periods,
        return_lags=FEATURE_CONFIG.return_lags,
        vol_windows=FEATURE_CONFIG.vol_windows,
    )
    # shift by 1 to eliminate lookahead
    feats_1h_shifted = feats_1h.shift(1)
    parts.append(feats_1h_shifted)

    # ── 1h microstructure ─────────────────────────────────────────────────────
    logger.info(f"  [{symbol}] Computing microstructure features...")
    micro_1h = compute_microstructure_features(df_1h, tf_prefix="1h")
    if not micro_1h.empty:
        parts.append(micro_1h.shift(1))

    # ── 5m / 15m features aggregated to 1h ───────────────────────────────────
    for tf in ["5m", "15m"]:
        df_tf = sym_data.get(tf, pd.DataFrame())
        if df_tf.empty:
            continue
        logger.info(f"  [{symbol}] Computing {tf} technical features...")
        feats_tf = compute_technical_features(
            df_tf,
            tf_prefix=tf,
            rsi_periods=[7, 14],
            atr_periods=[7],
            return_lags=[1, 3, 6, 12],
            vol_windows=[6, 12, 24],
        )
        # Resample to 1h (take last bar value of each 1h window)
        feats_tf_1h = feats_tf.resample("1h").last()
        feats_tf_aligned = _align_and_shift(feats_tf_1h, ref_index)
        parts.append(feats_tf_aligned)

        # Microstructure at finer resolution
        micro_tf = compute_microstructure_features(df_tf, tf_prefix=tf)
        if not micro_tf.empty:
            micro_tf_1h = micro_tf.resample("1h").last()
            micro_tf_aligned = _align_and_shift(micro_tf_1h, ref_index)
            parts.append(micro_tf_aligned)

    # ── 4h features ───────────────────────────────────────────────────────────
    df_4h = sym_data.get("4h", pd.DataFrame())
    if not df_4h.empty:
        logger.info(f"  [{symbol}] Computing 4h technical features...")
        feats_4h = compute_technical_features(
            df_4h,
            tf_prefix="4h",
            rsi_periods=[7, 14],
            atr_periods=[7, 14],
            return_lags=[1, 2, 4, 7, 14],
            vol_windows=[4, 7, 14, 28],
        )
        micro_4h = compute_microstructure_features(df_4h, tf_prefix="4h")
        # each 4h bar is known from its close time; forward-fill to 1h
        feats_4h_1h = feats_4h.resample("1h").last().reindex(ref_index).ffill()
        parts.append(feats_4h_1h.shift(1))
        if not micro_4h.empty:
            micro_4h_1h = micro_4h.resample("1h").last().reindex(ref_index).ffill()
            parts.append(micro_4h_1h.shift(1))

    # ── 1d features ───────────────────────────────────────────────────────────
    df_1d = sym_data.get("1d", pd.DataFrame())
    if not df_1d.empty:
        logger.info(f"  [{symbol}] Computing 1d technical features...")
        feats_1d = compute_technical_features(
            df_1d,
            tf_prefix="1d",
            rsi_periods=[7, 14],
            atr_periods=[7],
            return_lags=[1, 3, 7, 14, 30],
            vol_windows=[7, 14, 30],
        )
        feats_1d_1h = feats_1d.resample("1h").last().reindex(ref_index).ffill()
        parts.append(feats_1d_1h.shift(1))

    # ── Regime features ───────────────────────────────────────────────────────
    logger.info(f"  [{symbol}] Computing regime features...")
    regime = compute_regime_features(df_1h, hurst_window=FEATURE_CONFIG.hurst_window)
    parts.append(regime.shift(1))

    # ── Cross-asset features ──────────────────────────────────────────────────
    # We only compute cross-asset for 1h timeframe
    hourly_data_all = {s: all_data[s].get("1h", pd.DataFrame()) for s in all_data}
    valid_hourly = {s: d for s, d in hourly_data_all.items() if not d.empty}
    if len(valid_hourly) >= 2 and symbol in valid_hourly:
        logger.info(f"  [{symbol}] Computing cross-asset features...")
        cross = compute_cross_asset_features(
            valid_hourly,
            target_symbol=symbol,
            corr_windows=FEATURE_CONFIG.corr_windows,
        )
        # cross-asset features are already shifted inside (we shift by 1 in that module)
        cross_aligned = cross.reindex(ref_index).ffill()
        parts.append(cross_aligned)

    # ── Combine all features ──────────────────────────────────────────────────
    logger.info(f"  [{symbol}] Concatenating {len(parts)} feature blocks...")
    features = pd.concat(parts, axis=1)
    features = features.reindex(ref_index)

    # Remove columns that are mostly NaN (>50%)
    thresh = 0.5 * len(features)
    features = features.dropna(axis=1, thresh=int(thresh))

    # Replace infinities with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info(f"  [{symbol}] Feature matrix: {features.shape}")
    return features


def build_labels(df_1h: pd.DataFrame) -> pd.Series:
    """
    Binary label: 1 if bar closes higher than it opens, 0 otherwise.
    Label at index t = is_up for the bar starting at t.
    """
    return (df_1h["close"] > df_1h["open"]).astype(int)


def build_dataset(
    symbol: str,
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    min_bars: int = 500,
) -> tuple:
    """
    Build (X, y) for model training/evaluation.

    Returns
    -------
    X : pd.DataFrame of features
    y : pd.Series of labels (same index as X)
    """
    df_1h = all_data[symbol].get("1h", pd.DataFrame())
    if df_1h.empty:
        raise ValueError(f"No 1h data for {symbol}")

    X = build_feature_matrix(symbol, all_data)
    y = build_labels(df_1h)

    # Align X and y on the same index
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    # Drop rows where ALL features are NaN (warm-up period)
    valid = X.notna().any(axis=1)
    X = X[valid]
    y = y[valid]

    # Require minimum number of complete rows
    complete = X.dropna()
    logger.info(
        f"[{symbol}] Dataset: {len(X)} total rows, "
        f"{len(complete)} fully complete rows"
    )

    return X, y
