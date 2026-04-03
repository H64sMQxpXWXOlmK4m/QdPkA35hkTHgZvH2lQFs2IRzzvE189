"""
Purged walk-forward cross-validation for time series.

Implements:
1. Expanding-window walk-forward CV with a gap (purge) between train and validation
   to prevent leakage from correlated observations.
2. Walk-forward backtesting: train on all data up to t, test on next window.

Gap convention: after the training window ends, skip `gap_bars` bars before
starting validation. This prevents any autocorrelation / label leakage.
"""

from typing import Generator, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


def expanding_wf_splits(
    n: int,
    n_splits: int = 6,
    min_train: int = 1000,
    gap: int = 24,
    val_size: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate (train_idx, val_idx) pairs for expanding-window walk-forward CV.

    Parameters
    ----------
    n : total number of samples
    n_splits : number of folds
    min_train : minimum training set size (in bars)
    gap : number of bars to skip between train end and val start (purge)
    val_size : size of each validation window (default: equal splits of remaining)

    Yields
    ------
    (train_indices, val_indices)
    """
    available = n - min_train - gap
    if available <= 0:
        raise ValueError(f"n={n} too small for min_train={min_train}+gap={gap}")

    if val_size is None:
        val_size = max(1, available // n_splits)

    for i in range(n_splits):
        val_start = min_train + gap + i * val_size
        val_end = min(val_start + val_size, n)
        if val_start >= n:
            break

        # Expanding window: training always starts at 0
        train_end = val_start - gap
        if train_end < min_train:
            continue

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)

        if len(val_idx) == 0:
            continue

        yield train_idx, val_idx


def rolling_wf_splits(
    n: int,
    train_size: int = 2000,
    val_size: int = 500,
    step: int = 250,
    gap: int = 24,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Sliding-window walk-forward CV (fixed training window size).

    Parameters
    ----------
    n : total number of samples
    train_size : fixed size of training window
    val_size : size of each validation window
    step : how many bars to advance per fold
    gap : purge gap between train and val
    """
    start = 0
    while True:
        train_end = start + train_size
        val_start = train_end + gap
        val_end = val_start + val_size

        if val_end > n:
            break

        yield np.arange(start, train_end), np.arange(val_start, val_end)
        start += step


# Re-export for convenience
__all__ = ["expanding_wf_splits", "rolling_wf_splits"]
