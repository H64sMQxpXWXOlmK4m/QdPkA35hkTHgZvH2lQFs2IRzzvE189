"""
train.py — Master training pipeline.

Usage:
  python train.py                      # train all assets
  python train.py --symbol BTC/USDT   # train one asset
  python train.py --fetch              # fetch/update data first, then train

Walk-forward backtest is run automatically after training to validate
calibration and performance across multiple out-of-sample windows.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import LOGS_DIR, MODELS_DIR, SYMBOLS, TRAIN_CONFIG
from src.data.fetcher import fetch_all, load_all
from src.features.pipeline import build_dataset
from src.models.ensemble import StackedEnsemble
from src.validation.metrics import (
    compute_all_metrics,
    format_metrics_table,
    hit_rate_table,
    reliability_diagram_data,
)
from src.validation.walk_forward import expanding_wf_splits

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "train.log"),
    ],
)
logger = logging.getLogger("train")


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    symbol: str,
    n_splits: int = TRAIN_CONFIG.n_wf_splits,
    min_train: int = TRAIN_CONFIG.min_train_bars,
    gap: int = TRAIN_CONFIG.gap_bars,
) -> pd.DataFrame:
    """
    Run purged expanding-window walk-forward backtest.
    Trains a model on each expanding window and evaluates on OOS window.
    Returns a DataFrame of per-fold metrics.
    """
    logger.info(f"[{symbol}] Starting walk-forward backtest ({n_splits} folds)...")
    n = len(X)
    splits = list(expanding_wf_splits(n, n_splits=n_splits, min_train=min_train, gap=gap))

    if not splits:
        logger.warning(f"[{symbol}] No valid WF splits (n={n}, min_train={min_train})")
        return pd.DataFrame()

    fold_results = []
    all_probs = np.full(n, np.nan)
    all_labels = y.values.copy()

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(
            f"  Fold {fold_idx+1}/{len(splits)}: "
            f"train={len(train_idx)} bars, val={len(val_idx)} bars"
        )
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        # Train ensemble (inner CV splits for Optuna)
        model = StackedEnsemble(
            n_optuna_trials=max(20, TRAIN_CONFIG.n_optuna_trials // 3),
            optuna_timeout=180,
        )
        try:
            model.fit(X_tr, y_tr)
        except Exception as e:
            logger.error(f"  Fold {fold_idx+1} training failed: {e}")
            continue

        # Predict on validation set
        y_prob = model.predict_proba(X_val)
        all_probs[val_idx] = y_prob

        # Compute fold metrics
        metrics = compute_all_metrics(
            y_val.values, y_prob, label=f"fold_{fold_idx+1}"
        )
        metrics["train_bars"] = len(train_idx)
        metrics["val_start"] = str(X.index[val_idx[0]])
        metrics["val_end"] = str(X.index[val_idx[-1]])
        fold_results.append(metrics)

        logger.info(
            f"    AUC={metrics['roc_auc']:.4f} "
            f"Brier={metrics['brier']:.4f} "
            f"ECE={metrics['ece']:.4f} "
            f"Acc={metrics['accuracy']:.4f}"
        )

    if not fold_results:
        return pd.DataFrame()

    results_df = pd.DataFrame(fold_results)
    logger.info(f"\n[{symbol}] WF Summary:")
    logger.info(format_metrics_table(fold_results))

    # Aggregate OOS predictions
    oos_mask = ~np.isnan(all_probs)
    if oos_mask.sum() > 100:
        oos_metrics = compute_all_metrics(
            all_labels[oos_mask], all_probs[oos_mask], label="oos_aggregate"
        )
        logger.info(f"\n[{symbol}] Aggregate OOS: {oos_metrics}")

        # Save reliability diagram data
        rd = reliability_diagram_data(all_labels[oos_mask], all_probs[oos_mask])
        rd.to_csv(MODELS_DIR / f"{symbol.replace('/', '_')}_reliability.csv", index=False)

        # Save hit rate table
        ht = hit_rate_table(all_labels[oos_mask], all_probs[oos_mask])
        ht.to_csv(MODELS_DIR / f"{symbol.replace('/', '_')}_hitrate.csv", index=False)

        # Plot reliability diagram
        _plot_reliability(rd, symbol, all_probs[oos_mask], all_labels[oos_mask])

    return results_df


def _plot_reliability(
    rd: pd.DataFrame,
    symbol: str,
    y_prob: np.ndarray,
    y_true: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    ax = axes[0]
    valid = rd[rd["count"] > 0]
    ax.plot(valid["mean_prob"], valid["fraction_pos"], "o-", label="Model", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.fill_between(
        valid["mean_prob"],
        valid["fraction_pos"] - 0.05,
        valid["fraction_pos"] + 0.05,
        alpha=0.15,
        color="blue",
    )
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title(f"{symbol} — Reliability Diagram (OOS)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Histogram of probabilities
    ax2 = axes[1]
    ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Up bars", color="green", density=True)
    ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Down bars", color="red", density=True)
    ax2.axvline(0.5, color="black", linestyle="--")
    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(f"{symbol} — Probability Distribution", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / f"{symbol.replace('/', '_')}_calibration.png", dpi=120)
    plt.close()


# ── Final model training ──────────────────────────────────────────────────────

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    symbol: str,
    test_fraction: float = 0.15,
) -> Dict:
    """
    Train the final production model on (1-test_fraction) of the data,
    evaluate on holdout test set, then retrain on ALL data for production.
    """
    logger.info(f"[{symbol}] Training final production model...")

    # Hold out last 15% for final evaluation
    n = len(X)
    test_start = int(n * (1 - test_fraction))
    X_train, X_test = X.iloc[:test_start], X.iloc[test_start:]
    y_train, y_test = y.iloc[:test_start], y.iloc[test_start:]

    logger.info(
        f"  Train: {len(X_train)} bars ({X_train.index[0]} → {X_train.index[-1]})"
    )
    logger.info(
        f"  Test:  {len(X_test)} bars ({X_test.index[0]} → {X_test.index[-1]})"
    )

    # Train on all-but-test
    model = StackedEnsemble()
    model.fit(X_train, y_train)

    # Evaluate on holdout test
    y_prob_test = model.predict_proba(X_test)
    test_metrics = compute_all_metrics(y_test.values, y_prob_test, label="holdout_test")
    logger.info(
        f"[{symbol}] Holdout test — "
        f"AUC={test_metrics['roc_auc']:.4f} "
        f"Brier={test_metrics['brier']:.4f} "
        f"ECE={test_metrics['ece']:.4f} "
        f"Acc={test_metrics['accuracy']:.4f}"
    )

    # Retrain on ALL data for production
    logger.info(f"[{symbol}] Retraining on FULL dataset for production...")
    prod_model = StackedEnsemble()
    prod_model.fit(X, y)
    prod_model.save(symbol)

    return {
        "symbol": symbol,
        "test_metrics": test_metrics,
        "model": prod_model,
        "n_train": len(X),
        "feature_importance": prod_model.feature_importance,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train crypto direction forecaster")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol to train (e.g. BTC/USDT)")
    parser.add_argument("--fetch", action="store_true", help="Fetch/update data before training")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest only")
    parser.add_argument("--no-wf", action="store_true", help="Skip walk-forward backtest")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.fetch:
        logger.info("Fetching / updating data from Binance...")
        all_data = fetch_all(symbols=symbols)
    else:
        logger.info("Loading cached data...")
        all_data = load_all(symbols=symbols)

    # ── Train per asset ───────────────────────────────────────────────────────
    summary_rows = []

    for symbol in symbols:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*70}")

        try:
            X, y = build_dataset(symbol, all_data)
        except Exception as e:
            logger.error(f"[{symbol}] Feature build failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        if len(X) < 500:
            logger.warning(f"[{symbol}] Too few samples ({len(X)}), skipping.")
            continue

        # Walk-forward backtest
        wf_results = pd.DataFrame()
        if not args.no_wf:
            try:
                wf_results = walk_forward_backtest(X, y, symbol)
            except Exception as e:
                logger.error(f"[{symbol}] WF backtest failed: {e}")
                import traceback
                traceback.print_exc()

        # Train final production model
        if not args.backtest:
            try:
                result = train_final_model(X, y, symbol)
                m = result["test_metrics"]
                row = {
                    "symbol": symbol,
                    "holdout_acc": m["accuracy"],
                    "holdout_auc": m["roc_auc"],
                    "holdout_brier": m["brier"],
                    "holdout_ece": m["ece"],
                    "n_bars": result["n_train"],
                }

                if not wf_results.empty:
                    row["wf_auc_mean"] = wf_results["roc_auc"].astype(float).mean()
                    row["wf_brier_mean"] = wf_results["brier"].astype(float).mean()
                    row["wf_ece_mean"] = wf_results["ece"].astype(float).mean()

                summary_rows.append(row)

                # Save feature importances
                fi = result["feature_importance"]
                if fi is not None and len(fi) > 0:
                    fi.head(40).to_csv(
                        MODELS_DIR / f"{symbol.replace('/', '_')}_feature_importance.csv"
                    )

            except Exception as e:
                logger.error(f"[{symbol}] Training failed: {e}")
                import traceback
                traceback.print_exc()

    # ── Summary table ─────────────────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = MODELS_DIR / "training_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)
        logger.info("\n" + summary_df.to_string(index=False))
        logger.info(f"\nSummary saved to {summary_path}")

    logger.info("\nTraining complete.")


if __name__ == "__main__":
    main()
