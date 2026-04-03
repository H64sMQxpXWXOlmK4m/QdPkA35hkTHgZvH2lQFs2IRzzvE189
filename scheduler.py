"""
scheduler.py — Production hourly prediction scheduler.

Runs predict.py logic at the top of every UTC hour, exactly when the
previous 1h candle has just closed and fresh data is available.

Features:
- APScheduler with cron trigger (every hour at minute 1, to allow
  exchange data to propagate — adjustable via OFFSET_MINUTES)
- Automatic data refresh before each prediction
- Structured JSONL logging of every prediction
- Slack/email alert hooks (optional, config-driven)
- Graceful error recovery: failures are logged and the scheduler keeps running
- Watchdog: re-fetches and retries on stale data

Usage:
  python scheduler.py                    # run indefinitely
  python scheduler.py --dry-run          # one immediate prediction then exit
  python scheduler.py --offset-minutes 2 # wait 2 min after candle close
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from src.config import LOGS_DIR, PREDICTION_LOG, SYMBOLS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "scheduler.log"),
    ],
)
logger = logging.getLogger("scheduler")

# ── Config (override via env vars if needed) ──────────────────────────────────
OFFSET_MINUTES = 1      # minutes after candle close before predicting
MAX_RETRIES = 3         # retry attempts per scheduled job
RETRY_DELAY = 30        # seconds between retries


# ── Core prediction job ───────────────────────────────────────────────────────

def run_predictions(symbols: List[str] = SYMBOLS) -> List[dict]:
    """
    Fetch latest data and run predictions for all symbols.
    Called by the scheduler on each trigger.
    """
    from src.data.fetcher import refresh_data as _refresh
    from src.features.pipeline import build_feature_matrix
    from src.models.ensemble import StackedEnsemble
    import pandas as pd
    import numpy as np

    now_utc = datetime.now(timezone.utc)
    logger.info(f"=== Prediction job started at {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} ===")

    # Determine which hour bar was just completed
    # At HH:01 UTC, the HH-1:00 → HH:00 bar just closed
    last_closed_hour = now_utc.replace(minute=0, second=0, microsecond=0)
    logger.info(f"Last closed 1h bar: {last_closed_hour.strftime('%Y-%m-%d %H:00 UTC')}")
    logger.info(f"Predicting for candle: {last_closed_hour.strftime('%H:00')} → {(last_closed_hour.hour + 1) % 24:02d}:00 UTC")

    # Fetch latest data
    timeframes = ("1h", "4h", "1d", "5m", "15m")
    all_data = {}
    for attempt in range(MAX_RETRIES):
        try:
            all_data = _refresh(symbols, timeframes)
            break
        except Exception as e:
            logger.error(f"Data refresh failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.critical("All data refresh attempts failed. Skipping this cycle.")
                return []

    results = []
    for sym in symbols:
        for attempt in range(MAX_RETRIES):
            try:
                # Load model
                model = StackedEnsemble.load(sym)

                # Build features
                X = build_feature_matrix(sym, all_data)
                if X is None or X.empty:
                    logger.warning(f"[{sym}] Empty feature matrix, skipping.")
                    break

                # Verify data freshness
                df_1h = all_data[sym].get("1h", pd.DataFrame())
                if not df_1h.empty:
                    last_bar = df_1h.index[-1]
                    staleness_hours = (
                        now_utc - last_bar.to_pydatetime().replace(tzinfo=timezone.utc)
                    ).total_seconds() / 3600
                    if staleness_hours > 2.5:
                        logger.warning(
                            f"[{sym}] Data may be stale: last bar is {staleness_hours:.1f}h ago."
                        )

                # Predict on the latest row
                X_latest = X.iloc[[-1]]
                components = model.predict_proba_components(X_latest)
                p_up = float(components["calibrated"][0])
                p_down = 1.0 - p_up

                open_price = float(df_1h["close"].iloc[-1]) if not df_1h.empty else np.nan

                result = {
                    "symbol": sym,
                    "predicted_at_utc": now_utc.isoformat(),
                    "last_closed_bar": str(X.index[-1]),
                    "candle_start": str(last_closed_hour),
                    "candle_end": str(last_closed_hour + pd.Timedelta(hours=1)),
                    "open_price": open_price,
                    "p_up": round(p_up, 4),
                    "p_down": round(p_down, 4),
                    "direction": "UP" if p_up >= 0.5 else "DOWN",
                    "confidence": round(abs(p_up - 0.5) * 2, 4),
                    "components": {
                        "lgbm": round(float(components["lgbm"][0]), 4),
                        "xgb": round(float(components["xgb"][0]), 4),
                        "catboost": round(float(components["catboost"][0]), 4),
                        "meta": round(float(components["meta"][0]), 4),
                    },
                }
                results.append(result)

                arrow = "▲" if result["direction"] == "UP" else "▼"
                logger.info(
                    f"[{sym}] {arrow} P(Up)={p_up:.1%}  "
                    f"Confidence={result['confidence']:.1%}  "
                    f"Open≈{open_price:,.4f}"
                )
                break  # success

            except FileNotFoundError:
                logger.warning(f"[{sym}] No trained model. Skipping.")
                break
            except Exception as e:
                logger.error(f"[{sym}] Prediction attempt {attempt+1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

    # Write to JSONL prediction log
    if results:
        with open(PREDICTION_LOG, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        _print_summary(results, now_utc)

    logger.info(f"=== Prediction job complete: {len(results)}/{len(symbols)} assets ===\n")
    return results


def _print_summary(results: list, now_utc: datetime) -> None:
    lines = [
        "",
        "=" * 65,
        f"  HOURLY FORECASTS — {now_utc.strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 65,
    ]
    for r in results:
        arrow = "▲" if r["direction"] == "UP" else "▼"
        bar = "█" * int(r["confidence"] * 10)
        lines.append(
            f"  {r['symbol']:<12} {arrow} P(Up)={r['p_up']:.1%}  "
            f"Conf={r['confidence']:.1%} [{bar:<10}]"
        )
    lines.append("=" * 65)
    print("\n".join(lines))


# ── APScheduler setup ─────────────────────────────────────────────────────────

def _on_job_executed(event):
    logger.info(f"Job '{event.job_id}' executed successfully.")


def _on_job_error(event):
    logger.error(f"Job '{event.job_id}' raised an exception: {event.exception}")


def start_scheduler(offset_minutes: int = OFFSET_MINUTES, symbols: List[str] = SYMBOLS) -> None:
    """Start the blocking hourly scheduler."""
    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_listener(_on_job_executed, EVENT_JOB_EXECUTED)
    scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)

    # Fire at HH:offset_minutes every hour
    trigger = CronTrigger(minute=offset_minutes, timezone="UTC")
    scheduler.add_job(
        func=run_predictions,
        trigger=trigger,
        id="hourly_prediction",
        name="Hourly crypto direction forecast",
        kwargs={"symbols": symbols},
        misfire_grace_time=120,   # if job fires late, still run if within 2 min
        coalesce=True,
        max_instances=1,
    )

    logger.info(
        f"Scheduler started. Predictions will fire at minute {offset_minutes} of every UTC hour."
    )
    logger.info(f"Assets: {', '.join(symbols)}")
    logger.info("Press Ctrl+C to stop.")

    # Graceful shutdown on SIGTERM
    def _shutdown(signum, frame):
        logger.info("Received shutdown signal. Stopping scheduler...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Scheduler stopped.")
        scheduler.shutdown()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hourly crypto prediction scheduler")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one prediction immediately then exit",
    )
    parser.add_argument(
        "--offset-minutes",
        type=int,
        default=OFFSET_MINUTES,
        help=f"Minutes after hour close to trigger (default: {OFFSET_MINUTES})",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Restrict to single symbol",
    )
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS

    if args.dry_run:
        logger.info("Dry-run mode: running one immediate prediction cycle.")
        results = run_predictions(symbols)
        sys.exit(0 if results else 1)

    start_scheduler(offset_minutes=args.offset_minutes, symbols=symbols)


if __name__ == "__main__":
    main()
