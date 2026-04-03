"""
predict.py — Production prediction script.

Fetches the latest market data, computes features, loads the trained ensemble
model, and outputs calibrated up-direction probabilities for all assets.

Usage:
  python predict.py                         # predict all assets
  python predict.py --symbol BTC/USDT       # predict one asset
  python predict.py --json                  # output as JSON
  python predict.py --no-fetch              # use cached data only

Output (per asset):
  Symbol        : BTC/USDT
  Prediction at : 2026-04-03 17:00:00 UTC  (for candle 17:00→18:00)
  P(Up)         : 0.623
  P(Down)       : 0.377
  Direction     : UP
  Confidence    : 0.246  (= |P-0.5| * 2, range 0–1)
  Components    : LGBM=0.61 XGB=0.63 CatBoost=0.62
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.config import LOGS_DIR, MODELS_DIR, PREDICTION_LOG, SYMBOLS
from src.data.fetcher import update_data, load_parquet, _make_exchange
from src.features.pipeline import build_feature_matrix
from src.models.ensemble import StackedEnsemble

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "predict.log"),
    ],
)
logger = logging.getLogger("predict")


# ── Data refresh ──────────────────────────────────────────────────────────────

def refresh_data(symbols: List[str], timeframes=("1h", "4h", "1d", "5m", "15m")) -> Dict:
    """Update data for all symbols/timeframes incrementally."""
    exchange = _make_exchange()
    all_data = {}
    for sym in symbols:
        all_data[sym] = {}
        for tf in timeframes:
            try:
                df = update_data(sym, tf, exchange)
                all_data[sym][tf] = df
            except Exception as e:
                logger.warning(f"Failed to update {sym} {tf}: {e}")
                all_data[sym][tf] = load_parquet(sym, tf)
        time.sleep(0.1)
    return all_data


def load_cached_data(symbols: List[str], timeframes=("1h", "4h", "1d", "5m", "15m")) -> Dict:
    all_data = {}
    for sym in symbols:
        all_data[sym] = {}
        for tf in timeframes:
            all_data[sym][tf] = load_parquet(sym, tf)
    return all_data


# ── Single asset prediction ───────────────────────────────────────────────────

def predict_symbol(symbol: str, all_data: Dict) -> Optional[Dict]:
    """Load model and predict for the latest bar."""
    # Load model
    try:
        model = StackedEnsemble.load(symbol)
    except FileNotFoundError:
        logger.warning(f"[{symbol}] No trained model found. Run train.py first.")
        return None
    except Exception as e:
        logger.error(f"[{symbol}] Model load error: {e}")
        return None

    # Compute features
    try:
        X = build_feature_matrix(symbol, all_data)
    except Exception as e:
        logger.error(f"[{symbol}] Feature computation error: {e}")
        return None

    if X is None or X.empty:
        logger.warning(f"[{symbol}] No features available.")
        return None

    # Use the LAST row: features at the most recently closed bar
    X_latest = X.iloc[[-1]]
    pred_ts = X.index[-1]       # timestamp of most recently closed 1h bar
    candle_start = pred_ts
    candle_end = pred_ts + pd.Timedelta(hours=1)

    # Open price for upcoming candle = close of most recently completed bar
    df_1h = all_data[symbol].get("1h", pd.DataFrame())
    open_price = float(df_1h["close"].iloc[-1]) if not df_1h.empty else np.nan

    # Predict
    try:
        components = model.predict_proba_components(X_latest)
        p_up = float(components["calibrated"][0])
    except Exception as e:
        logger.error(f"[{symbol}] Prediction error: {e}")
        return None

    p_down = 1.0 - p_up
    direction = "UP" if p_up >= 0.5 else "DOWN"
    confidence = abs(p_up - 0.5) * 2  # 0=no edge, 1=max certainty

    return {
        "symbol": symbol,
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_closed_bar": str(pred_ts),
        "candle_start": str(candle_start),
        "candle_end": str(candle_end),
        "open_price": open_price,
        "p_up": round(p_up, 4),
        "p_down": round(p_down, 4),
        "direction": direction,
        "confidence": round(confidence, 4),
        "components": {
            "lgbm": round(float(components["lgbm"][0]), 4),
            "xgb": round(float(components["xgb"][0]), 4),
            "catboost": round(float(components["catboost"][0]), 4),
            "meta": round(float(components["meta"][0]), 4),
        },
    }


def predict_all(symbols: List[str], all_data: Dict) -> List[Dict]:
    results = []
    for sym in symbols:
        logger.info(f"Predicting {sym}...")
        r = predict_symbol(sym, all_data)
        if r:
            results.append(r)
    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_human(results: List[Dict]) -> None:
    print()
    print("=" * 65)
    print(f"  HOURLY DIRECTION FORECASTS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)
    for r in results:
        arrow = "▲" if r["direction"] == "UP" else "▼"
        conf_bar = "█" * int(r["confidence"] * 10)
        print(f"\n  {r['symbol']:<12}  Candle: {r['candle_start'][:16]} → {r['candle_end'][:16]} UTC")
        print(f"  {'─'*60}")
        print(f"  P(Up)   = {r['p_up']:.1%}   P(Down) = {r['p_down']:.1%}")
        print(f"  Signal  = {arrow} {r['direction']:<5}   Confidence = {r['confidence']:.1%}  [{conf_bar:<10}]")
        print(f"  Open    ≈ {r['open_price']:,.4f}")
        c = r["components"]
        print(
            f"  Models  : LGBM={c['lgbm']:.3f}  XGB={c['xgb']:.3f}  "
            f"CatBoost={c['catboost']:.3f}  Meta={c['meta']:.3f}"
        )
    print()
    print("=" * 65)
    print("  NOTE: Calibrated probabilities. NOT financial advice.")
    print("=" * 65)
    print()


def log_prediction(results: List[Dict]) -> None:
    """Append predictions to JSONL log."""
    with open(PREDICTION_LOG, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict hourly direction probabilities")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS

    if args.no_fetch:
        logger.info("Loading cached data (no fetch)...")
        all_data = load_cached_data(symbols)
    else:
        logger.info("Fetching latest data from Binance...")
        all_data = refresh_data(symbols)

    results = predict_all(symbols, all_data)

    if not results:
        logger.error("No predictions generated. Run train.py first.")
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_human(results)

    log_prediction(results)


if __name__ == "__main__":
    main()
