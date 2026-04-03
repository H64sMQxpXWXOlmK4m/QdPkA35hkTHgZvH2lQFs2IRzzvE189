# Crypto Hourly Direction Forecaster

![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Models](https://img.shields.io/badge/models-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-FF6B35?style=flat-square)
![Assets](https://img.shields.io/badge/assets-BTC%20%7C%20ETH%20%7C%20SOL%20%7C%20XRP%20%7C%20BNB%20%7C%20DOGE-F7931A?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-22C55E?style=flat-square)

> At the top of every UTC hour, predict the probability that the next 1-hour candle closes **higher** than it opens — for 6 major crypto assets on Binance.

---

## Results

| Asset | Accuracy | ROC AUC | Brier Score | ECE |
|:------|:--------:|:-------:|:-----------:|:---:|
| BTC/USDT | 56.8% | 0.588 | 0.244 | **0.020** |
| ETH/USDT | 70.7% | 0.766 | 0.199 | 0.053 |
| SOL/USDT | 68.1% | 0.744 | 0.208 | 0.049 |
| XRP/USDT | 69.5% | 0.753 | 0.204 | 0.047 |
| BNB/USDT | **71.6%** | **0.776** | **0.193** | 0.040 |
| DOGE/USDT | 69.3% | 0.766 | 0.197 | 0.034 |
| *Random baseline* | *50.0%* | *0.500* | *0.250* | *—* |

*Evaluated on ~1,200 out-of-sample bars per asset. ECE = Expected Calibration Error (lower is better).*

---

## How It Works

```
Binance API  ──►  data/fetcher.py        paginated, incremental, rate-limited
                        │
                        ▼
              features/pipeline.py       330 features per bar, strict no-lookahead
              ┌─────────┬──────────┬─────────────┬──────────┐
              │technical│microstruc│ cross-asset │  regime  │
              │ RSI MACD│ taker buy│ BTC lead    │  Hurst   │
              │ BB ATR  │ OFI Amih.│ correlations│  vol reg │
              └─────────┴──────────┴─────────────┴──────────┘
                        │
                        ▼
              models/ensemble.py         stacked ensemble
              ┌──────────┬─────────┬──────────┐
              │ LightGBM │ XGBoost │ CatBoost │  ← Optuna-tuned
              └────┬─────┴────┬────┴─────┬────┘
                   └──────────┼──────────┘
                              ▼
                     Logistic Meta-Learner
                              │
                              ▼
              models/calibration.py      Platt + Isotonic + Beta + Temp scaling
                              │
                              ▼
                     P(close > open)     calibrated probability output
```

**Key design decisions:**
- All features are shifted by 1 bar — zero lookahead leakage
- Hyperparameter search uses expanding-window time-series CV with a 24-bar purge gap
- Calibrator is fit on a genuine OOS holdout, then base models are retrained on 100% of data
- Predictions fire at **HH:01 UTC** (1 min after candle close) to ensure data propagation

---

## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train all models** *(fetches fresh data automatically)*
```bash
python train.py --fetch
```

**3. Run the live scheduler** *(fires at HH:01 UTC every hour)*
```bash
python scheduler.py
```

**4. One-off prediction**
```bash
python predict.py
```

**5. Test everything works**
```bash
python scheduler.py --dry-run
```

---

## Sample Output

```
=================================================================
  HOURLY FORECASTS — 2026-04-03 06:01 UTC
=================================================================
  BTC/USDT     ▲ P(Up)=61.3%  Conf=22.6% [██        ]
  ETH/USDT     ▼ P(Up)=19.1%  Conf=61.7% [██████    ]
  SOL/USDT     ▼ P(Up)=38.0%  Conf=23.9% [██        ]
  XRP/USDT     ▲ P(Up)=53.2%  Conf=6.5%  [          ]
  BNB/USDT     ▼ P(Up)=37.5%  Conf=25.0% [██        ]
  DOGE/USDT    ▼ P(Up)=36.2%  Conf=27.6% [██        ]
=================================================================
```

Predictions are logged to `logs/predictions.jsonl` for record-keeping.

---

## Project Structure

```
.
├── src/
│   ├── config.py                  # Central config (symbols, paths, timeframes)
│   ├── data/
│   │   └── fetcher.py             # Binance data fetching & incremental caching
│   ├── features/
│   │   ├── technical.py           # RSI, MACD, BB, ATR, ADX, OBV, VWAP, ...
│   │   ├── microstructure.py      # Taker buy/sell, order-flow imbalance, Amihud
│   │   ├── cross_asset.py         # BTC leadership, correlations, relative strength
│   │   ├── regime.py              # Hurst exponent, vol regime, calendar features
│   │   └── pipeline.py            # Multi-timeframe assembly, no-lookahead shift
│   ├── models/
│   │   ├── ensemble.py            # LightGBM + XGBoost + CatBoost stacked ensemble
│   │   └── calibration.py         # 4-method ensemble calibrator
│   └── validation/
│       ├── metrics.py             # ECE, Brier, reliability diagrams
│       └── walk_forward.py        # Purged expanding-window cross-validation
│
├── train.py                       # Train / retrain all models
├── predict.py                     # Single prediction cycle
├── scheduler.py                   # APScheduler hourly runner
│
├── data_v2/                       # Cached parquet data (auto-updated)
│   └── {1m,5m,15m,1h,4h,1d}/
│       └── {ASSET}_USDT.parquet
│
├── models_v2/                     # Trained model artifacts
│   └── {ASSET}_USDT_ensemble.pkl
│
├── logs/
│   └── predictions.jsonl          # Append-only prediction log
│
├── docs/
│   └── SYSTEM_REPORT.md           # Full technical report & methodology
│
└── requirements.txt
```

---

## Feature Engineering (330 features per bar)

| Category | Features |
|:---------|:---------|
| **Technical** | RSI (7/14/21), MACD, Bollinger Bands, ATR, ADX, Stochastic, CCI, Williams %R, OBV slope, VWAP deviation, SMA/EMA deviations, log returns (10 lags), rolling volatility, candle body/shadow ratios |
| **Microstructure** | Taker buy ratio, volume imbalance, cumulative order flow, Amihud illiquidity, buy/sell z-scores, trade intensity |
| **Cross-asset** | BTC lagged returns, rolling correlations & betas to BTC, relative strength, market breadth, cross-sectional momentum rank |
| **Regime** | Hurst exponent (rolling 100-bar), volatility regime percentile, vol ratio (24h/168h), autocorrelations, calendar (hour/day/month sin-cos), session flags |

All computed across **1h / 4h / 1d / 5m / 15m** timeframes and aligned with a 1-bar shift to prevent lookahead.

---

## Scheduler Options

```bash
python scheduler.py                          # run indefinitely (fires at HH:01 UTC)
python scheduler.py --offset-minutes 2       # fire 2 min after candle close
python scheduler.py --symbol BTC/USDT        # single asset only
python scheduler.py --dry-run                # one immediate prediction, then exit
```

## Retrain Options

```bash
python train.py --fetch                      # full retrain, fetch fresh data first
python train.py --symbol SOL/USDT --fetch    # retrain single asset
python train.py --backtest                   # walk-forward backtest only
```

---

## Notes on Calibration

Calibration means: *when the model says 60%, does it happen 60% of the time?*

The answer is **yes, in the 40–65% range** where virtually all predictions land. ECE scores of 0.02–0.05 across all assets confirm this. Predictions at the extremes (>70%) are rare and slightly overconfident — treat those with some skepticism.

---

## License

MIT
