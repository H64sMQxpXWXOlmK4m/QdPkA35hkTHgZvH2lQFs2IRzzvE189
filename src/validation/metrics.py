"""
Calibration and performance metrics for probabilistic binary classification.

Metrics implemented:
- Brier score (and decomposition: reliability + resolution + uncertainty)
- Log loss
- ROC-AUC
- Expected Calibration Error (ECE)
- Reliability diagram data
- Empirical hit-rate table per probability bucket
- Maximum Calibration Error (MCE)
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE = Σ (n_bin/n) |avg_prob - avg_label| over all bins.
    Uses equal-width bins in [0, 1].
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """MCE = max over all bins of |avg_prob - avg_label|."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    errors = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() < 3:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        errors.append(abs(avg_conf - avg_acc))
    return max(errors) if errors else 0.0


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      bin_mid, mean_prob, fraction_pos, count
    for plotting a reliability diagram.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if lo == bins[-2]:  # last bin inclusive on right
            mask = (y_prob >= lo) & (y_prob <= hi)
        cnt = mask.sum()
        if cnt == 0:
            rows.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "bin_mid": (lo + hi) / 2,
                    "mean_prob": (lo + hi) / 2,
                    "fraction_pos": np.nan,
                    "count": 0,
                }
            )
        else:
            rows.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "bin_mid": (lo + hi) / 2,
                    "mean_prob": y_prob[mask].mean(),
                    "fraction_pos": y_true[mask].mean(),
                    "count": cnt,
                }
            )
    return pd.DataFrame(rows)


def hit_rate_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Empirical hit-rate table per probability bucket.
    Useful for verifying calibration in a human-readable way.
    """
    rd = reliability_diagram_data(y_true, y_prob, n_bins)
    rd["calibration_error"] = (rd["mean_prob"] - rd["fraction_pos"]).abs()
    rd["label"] = rd.apply(
        lambda r: f"{r['bin_lo']:.0%}–{r['bin_hi']:.0%}", axis=1
    )
    return rd[["label", "mean_prob", "fraction_pos", "calibration_error", "count"]]


def brier_decomposition(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Murphy (1973) decomposition of Brier score into:
      Brier = Reliability - Resolution + Uncertainty
    Lower reliability is better (0 = perfect calibration).
    Higher resolution is better.
    """
    y_mean = y_true.mean()
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    n = len(y_true)

    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        cnt = mask.sum()
        if cnt == 0:
            continue
        o_k = y_true[mask].mean()
        p_k = y_prob[mask].mean()
        reliability += cnt * (p_k - o_k) ** 2
        resolution += cnt * (o_k - y_mean) ** 2

    reliability /= n
    resolution /= n
    uncertainty = y_mean * (1 - y_mean)
    brier = reliability - resolution + uncertainty

    return {
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    n_bins: int = 15,
) -> Dict:
    """
    Compute the full suite of metrics.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    acc = (y_pred == y_true).mean()

    metrics = {
        "label": label,
        "n_samples": len(y_true),
        "accuracy": acc,
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins),
        "fraction_up": y_true.mean(),
        "mean_prob": y_prob.mean(),
        "prob_std": y_prob.std(),
    }

    decomp = brier_decomposition(y_true, y_prob, n_bins)
    metrics.update(
        {
            "brier_reliability": decomp["reliability"],
            "brier_resolution": decomp["resolution"],
            "brier_uncertainty": decomp["uncertainty"],
        }
    )

    return metrics


def format_metrics_table(metrics_list: list) -> str:
    """Pretty-print a list of metric dicts as a table."""
    df = pd.DataFrame(metrics_list)
    float_cols = df.select_dtypes("float").columns
    for col in float_cols:
        df[col] = df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    return df.to_string(index=False)
