"""
Central configuration for the crypto probability forecasting system.
All constants, symbols, timeframes, and paths live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_v2"
MODELS_DIR = ROOT / "models_v2"
LOGS_DIR = ROOT / "logs"

for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Target assets ─────────────────────────────────────────────────────────────
SYMBOLS: List[str] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "DOGE/USDT",
]

SYMBOL_SHORT: Dict[str, str] = {s: s.split("/")[0] for s in SYMBOLS}

# ── Timeframes to fetch ───────────────────────────────────────────────────────
TIMEFRAMES: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]

# How many bars to keep per timeframe
FETCH_LIMIT: Dict[str, int] = {
    "1m":  20_000,   # ~14 days
    "5m":  20_000,   # ~70 days
    "15m": 10_000,   # ~104 days
    "1h":  8_760,    # 1 year
    "4h":  4_380,    # 2 years
    "1d":  1_095,    # 3 years
}

# ── Exchange ──────────────────────────────────────────────────────────────────
EXCHANGE_ID = "binance"
RATE_LIMIT_SLEEP = 0.2   # seconds between requests
MAX_RETRIES = 5

# ── Feature engineering ───────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    # RSI periods per timeframe
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])

    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR periods
    atr_periods: List[int] = field(default_factory=lambda: [7, 14])

    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3

    # ADX
    adx_period: int = 14

    # CCI
    cci_period: int = 20

    # Past return lags (in number of 1h bars)
    return_lags: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 168]
    )

    # Rolling volatility windows (1h bars)
    vol_windows: List[int] = field(default_factory=lambda: [6, 12, 24, 48, 168])

    # Cross-asset rolling correlation windows (1h bars)
    corr_windows: List[int] = field(default_factory=lambda: [24, 168])

    # Hurst exponent window
    hurst_window: int = 100


FEATURE_CONFIG = FeatureConfig()

# ── Training ──────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Walk-forward splits
    n_wf_splits: int = 6
    min_train_bars: int = 2_000   # minimum hourly bars in training window
    gap_bars: int = 24            # gap (hours) between train end and val start (purge)

    # Optuna
    n_optuna_trials: int = 80
    optuna_timeout_sec: int = 600  # 10 min per model

    # Feature selection
    max_features: int = 80

    # Calibration
    calibration_method: str = "isotonic"   # "isotonic" | "platt" | "beta" | "ensemble"
    cal_cv_folds: int = 5

    # Ensemble weights (learnt via meta-learner)
    use_meta_learner: bool = True

    # Random seed
    seed: int = 42


TRAIN_CONFIG = TrainConfig()

# ── Scheduler ─────────────────────────────────────────────────────────────────
SCHEDULER_TIMEZONE = "UTC"
PREDICTION_LOG = LOGS_DIR / "predictions.jsonl"
