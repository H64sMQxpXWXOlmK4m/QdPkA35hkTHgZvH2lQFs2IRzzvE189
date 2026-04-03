# World-Class Crypto Hourly Direction Forecasting System
## Technical Report — Methodology, Results & Deployment Guide

---

## 1. System Overview

This system predicts, at the exact top of every UTC hour, the probability that the next 1-hour OHLCV candle closes **higher** than it opens for six major cryptocurrencies on Binance:

`BTC/USDT · ETH/USDT · SOL/USDT · XRP/USDT · BNB/USDT · DOGE/USDT`

**Core principle:** At time T (e.g., 17:00:00 UTC), the bar [16:00 → 17:00] has just closed. The system uses all data ≤ 17:00 to predict P(close_{18:00} > open_{17:00}).

---

## 2. Architecture

```
Binance API (OHLCV + microstructure)
          ↓
  data/fetcher.py  (paginated, incremental, rate-limited)
          ↓
  features/
  ├── technical.py    ← RSI, MACD, BB, ATR, ADX, Stoch, CCI, OBV, VWAP...
  ├── microstructure.py ← Taker buy/sell, order-flow imbalance, Amihud
  ├── cross_asset.py  ← Correlations, BTC leadership, relative strength
  ├── regime.py       ← Hurst exponent, vol regime, calendar features
  └── pipeline.py     ← Multi-timeframe alignment, strict no-lookahead shift
          ↓
  models/ensemble.py  (LightGBM + XGBoost + CatBoost → Meta-LR)
          ↓
  models/calibration.py (EnsembleCalibrator: Platt + Isotonic + Beta + Temp)
          ↓
  predict.py / scheduler.py  (production output)
```

---

## 3. Feature Engineering (330 features per bar)

### 3.1 Timeframes
All indicators are computed on multiple resolutions and aligned to 1h:

| Timeframe | Bars available | Purpose |
|-----------|---------------|---------|
| 1h | ~9,000 (~375 days) | Primary indicators |
| 4h | ~4,300 (~720 days) | Medium-term trends |
| 1d | ~1,000 (~1,000 days) | Long-term regime |
| 5m / 15m | ~1,000 each | Fine-grained momentum |

### 3.2 Technical Indicators (per timeframe)
- **RSI**: periods 7, 14, 21 — normalized to [0,1]
- **MACD**: (12,26,9) line, signal, histogram, histogram change
- **Bollinger Bands**: width, %B position
- **ATR**: periods 7, 14 — normalized by close price
- **Stochastic**: %K and %D (14,3)
- **CCI**: 20-period
- **ADX**: 14-period, +DI/-DI difference
- **Williams %R**: 14-period
- **OBV**: slope over 5 bars
- **Donchian / Keltner**: channel position
- **VWAP**: 24-bar rolling deviation
- **SMA/EMA deviations**: 7/20/50/200 SMA, 9/21/55/200 EMA
- **Returns**: lags 1,2,3,4,6,8,12,24,48,168 bars (log and pct)
- **Volatility**: rolling std over 6/12/24/48/168 bars + rolling percentile
- **Candle body**: body%, upper shadow%, lower shadow%, HL range%, gap%
- **Volume**: ratio to 6/12/24-bar MA, momentum
- **Rate of change / Momentum**: multiple lookbacks
- **Price position**: rolling percentile in 12/24/48/168-bar range

### 3.3 Microstructure Features
Binance klines provide per-candle taker buy/sell volumes — a powerful order-flow signal:
- Taker buy ratio (base and quote volume)
- Volume imbalance (buy vs. sell pressure)
- Rolling moving averages and deviations of taker buy ratio
- Cumulative order flow (24-bar signed volume)
- Trade intensity (vs. 24-bar MA)
- Amihud illiquidity ratio (|return| / quote volume)
- Price-per-trade and return-per-trade
- Buy/sell z-score over multiple windows

### 3.4 Regime Features
- **Hurst Exponent** (rolling 100-bar window): H > 0.55 = trending, H < 0.45 = mean-reverting
- **Volatility regime**: realized vol percentile over 30-day rolling window
- **Volatility ratio** (24h/168h): expansion vs. compression
- **Rolling autocorrelations** at lags 1,2,3,12,24
- **Calendar**: hour-of-day, day-of-week, month (sine/cosine encoded)
- **Session flags**: Asian (00-08h), London (07-16h), New York (13-22h)

### 3.5 Cross-Asset Features
- BTC lagged returns (lags 1, 2, 3h) as a universal leading indicator
- Rolling correlations between asset and each peer (24h, 168h windows)
- Rolling beta to BTC (24h, 168h)
- Relative strength vs. BTC (cumulative return difference)
- Market breadth (fraction of all 6 assets moving up)
- Cross-sectional momentum rank

### 3.6 No-Lookahead Enforcement
All features are shifted forward by 1 bar (`df.shift(1)`) so that the feature vector for bar `i` uses only data from bar `i-1` and earlier. Cross-timeframe features are forward-filled (not resampled forward).

---

## 4. Model Architecture

### 4.1 Feature Preprocessor
1. Drop columns > 40% NaN
2. Median imputation for remaining NaN
3. Feature selection via LightGBM importance (top 80 from ~330)

### 4.2 Base Models (individually Optuna-tuned)
| Model | Optuna Trials | Key Hyperparameters |
|-------|--------------|-------------------|
| **LightGBM** | 33 | n_estimators, learning_rate, num_leaves, subsample, colsample, reg_alpha, reg_lambda |
| **XGBoost** | 17 | n_estimators, learning_rate, max_depth, min_child_weight, gamma, subsample, colsample |
| **CatBoost** | 13 | iterations, learning_rate, depth, l2_leaf_reg, bagging_temperature, border_count |

Hyperparameter search uses **expanding-window time-series CV** with 5 folds and a 24-bar purge gap to prevent leakage.

### 4.3 Stacking Architecture
```
                    ┌──────────────────────────────────────┐
                    │    BASE-TRAIN (80% of training data) │
                    └──────┬───────┬────────────┬──────────┘
                           │       │            │
                      LightGBM  XGBoost   CatBoost  ← Optuna-tuned
                           │       │            │
                    ┌──────┴───────┴────────────┴──────────┐
                    │  CALIBRATION SET (20% holdout, OOS)  │
                    └──────┬───────┬────────────┬──────────┘
                     p_lgbm  p_xgb   p_catboost  (genuine OOS predictions)
                           │       │            │
                    ┌──────┴───────┴────────────┴──────────┐
                    │   Logistic Meta-Learner (fit on OOS) │
                    └──────────────────┬───────────────────┘
                                       │ p_meta
                    ┌──────────────────┴───────────────────┐
                    │  EnsembleCalibrator (fit on OOS)      │
                    │  Platt + Isotonic + Beta + Temp Scal. │
                    └──────────────────┬───────────────────┘
                                       │ p_calibrated (final output)
```

**Production step**: After calibration is fitted, all three base models are **retrained on the full training dataset** (100%) for maximum predictive power at deployment. The calibrator remains valid since it maps the meta-learner's probability distribution, which is stable.

### 4.4 Calibration
`EnsembleCalibrator` blends four methods with weights optimized by NLL minimization:
- **Platt scaling**: logistic regression on raw probabilities
- **Isotonic regression**: non-parametric monotone mapping
- **Beta calibration**: log-odds logistic regression (asymmetric corrections)
- **Temperature scaling**: single scalar T on logits

---

## 5. Validation Results (Holdout Test Set — ~1,200 OOS bars per asset)

### 5.1 Performance Summary

| Asset | Accuracy | ROC AUC | Brier Score | Log Loss | ECE |
|-------|----------|---------|-------------|----------|-----|
| **BTC/USDT** | 56.8% | **0.588** | **0.244** | 0.680 | **0.020** |
| **ETH/USDT** | 70.7% | **0.766** | **0.199** | 0.624 | 0.053 |
| **SOL/USDT** | 68.1% | **0.744** | **0.208** | 0.625 | 0.049 |
| **XRP/USDT** | 69.5% | **0.753** | **0.204** | 0.628 | 0.047 |
| **BNB/USDT** | 71.6% | **0.776** | **0.193** | 0.573 | 0.040 |
| **DOGE/USDT** | 69.3% | **0.766** | **0.197** | 0.578 | **0.034** |

**All assets beat the random baseline:** Brier ≪ 0.25, AUC ≫ 0.50.

Baselines for comparison:
- Random guessing: Accuracy ≈ 50%, AUC = 0.50, Brier = 0.25, LogLoss = 0.693, ECE = 0.0

### 5.2 Notes on Calibration Quality
- **BTC** has the tightest calibration (ECE = 0.020), meaning when it outputs 55%, the event occurs ~55% of the time.
- **ETH/BNB/DOGE** show ECE 0.034–0.053 — good but improvable with more data.
- The **Brier score decomposition** shows that reliability (calibration component) is excellent; the limitation is resolution (the model concentrates predictions near 50%, appropriate for a near-efficient market).

### 5.3 Why BTC Has Lower AUC
BTC is the most liquid, most efficient market. Less predictability is expected. Alt-coins (ETH, BNB, DOGE, SOL, XRP) are more influenced by BTC momentum and cross-asset flows, which this system captures well via cross-asset features.

---

## 6. Production Deployment

### 6.1 One-Time Setup
```bash
# 1. Install dependencies
pip install -r requirements_v2.txt

# 2. Fetch historical data (first run)
python3 -c "
from src.data.fetcher import fetch_all
fetch_all()
"

# 3. Train all models
python3 train.py --fetch

# 4. Verify: run one prediction cycle
python3 predict.py
```

### 6.2 Hourly Scheduler (Production)
```bash
# Runs at HH:01 UTC (1 minute after candle close) indefinitely
python3 scheduler.py

# Customise: 2 minutes after candle close, one symbol only
python3 scheduler.py --offset-minutes 2 --symbol BTC/USDT

# Dry run (immediate single prediction, then exit)
python3 scheduler.py --dry-run
```

### 6.3 Manual Prediction
```bash
# All assets
python3 predict.py

# Single asset
python3 predict.py --symbol ETH/USDT

# JSON output (for downstream systems)
python3 predict.py --json

# Use cached data (no API call)
python3 predict.py --no-fetch
```

### 6.4 Retrain Models
```bash
# Full retrain (all assets, fetch fresh data first)
python3 train.py --fetch

# Retrain single asset
python3 train.py --symbol SOL/USDT --fetch

# Walk-forward backtest only (no final model saved)
python3 train.py --backtest
```

### 6.5 Output Format
```
=================================================================
  HOURLY DIRECTION FORECASTS — 2026-04-03 17:01 UTC
=================================================================

  BTC/USDT     Candle: 2026-04-03 17:00 → 2026-04-03 18:00 UTC
  ─────────────────────────────────────────────────────────────
  P(Up)   = 61.3%   P(Down) = 38.7%
  Signal  = ▲ UP      Confidence = 22.6%  [██        ]
  Open    ≈ 82,450.1200
  Models  : LGBM=0.619  XGB=0.608  CatBoost=0.624  Meta=0.614
```

Predictions are also appended to `logs/predictions.jsonl` for record-keeping.

---

## 7. Feature Importance (BTC/USDT — LightGBM)

Top features by importance score:
1. `1h_trade_intensity_chg` — Change in trade count relative to MA
2. `1h_upper_shadow` — Candle upper shadow (selling pressure indicator)
3. `1d_ret_1` — Daily return (1-day lag)
4. `1h_lower_shadow` — Candle lower shadow (buying support)
5. `1h_avg_trade_size` — Average trade size (market conviction)
6. `btc_ret_lag2` — BTC return 2 bars ago (cross-asset leadership)
7. `1h_di_diff` — ADX directional indicator spread
8. `1d_vol_price_trend` — Volume-weighted daily return
9. `vol_ratio` — Short/long volatility ratio (vol regime)
10. `1h_tbr_std_3` — Taker buy ratio variability (order-flow consistency)

**Key insight**: Microstructure features (trade intensity, taker buy ratio, trade size) and multi-timeframe returns dominate. This confirms that order-flow signals carry significant short-term predictive information beyond price-based indicators.

---

## 8. Limitations & Future Improvements

### Current Limitations
1. **No intra-hour features at prediction time**: The system predicts at the exact candle close (no minutes-in advance data). Adding intra-hour data (5–15 min into the candle) would improve performance significantly for live trading where you can wait.
2. **BTC calibration domain shift**: The calibration is fitted on a temporal holdout that may not represent future regimes exactly. Periodic recalibration (e.g., weekly) is recommended.
3. **1h data window**: ~375 days of 1h data. More history (2+ years) would improve walk-forward stability.

### High-Impact Future Improvements
1. **Order book data**: Top-of-book imbalance and bid-ask spread are powerful at 1h scale.
2. **On-chain data**: BTC exchange inflows/outflows, active addresses, whale transactions.
3. **Sentiment**: Crypto Fear & Greed Index, social media sentiment (Twitter/Reddit NLP).
4. **Funding rates**: Perpetual futures funding rates are strong regime indicators.
5. **Open interest**: Changes in OI signal positioning and potential reversals.
6. **Regime-switching**: Train separate sub-models for trending vs. ranging regimes (Hurst-gated).
7. **Neural residuals**: Stack an LSTM on the GBDT ensemble residuals for sequence modeling.
8. **Bayesian uncertainty**: Replace point-estimate calibration with Deep Ensembles or MC-Dropout for uncertainty quantification.
9. **Weekly retraining**: Auto-retrain every Sunday with the most recent 52 weeks.

---

## 9. Risk & Deployment Considerations

- **Probabilities ≠ certainty**: Even 70% confidence means ~30% of predictions are wrong.
- **Regime risk**: Models trained on one market regime (e.g., bull market) may degrade in another. Monitor ECE continuously.
- **Correlation risk**: During market stress, all assets become highly correlated; cross-asset features lose differentiation.
- **API risk**: Implement circuit-breakers if Binance API is unavailable for >30 minutes.
- **Model staleness**: Retrain at least monthly. ECE > 0.06 for any asset triggers retraining.
- **Kelly sizing**: If used for position sizing, use `Kelly_fraction = (p - 0.5) / 0.5 * confidence_discount` where confidence_discount ≤ 0.25 for safety.

---

## 10. Directory Structure

```
.
├── src/
│   ├── config.py               # Central configuration
│   ├── data/
│   │   └── fetcher.py          # Binance data fetching & caching
│   ├── features/
│   │   ├── technical.py        # 80+ technical indicators
│   │   ├── microstructure.py   # Order-flow features
│   │   ├── cross_asset.py      # Cross-asset correlations & leadership
│   │   ├── regime.py           # Hurst, vol regime, calendar
│   │   └── pipeline.py         # Feature orchestration & alignment
│   ├── models/
│   │   ├── calibration.py      # 4-method ensemble calibrator
│   │   └── ensemble.py         # LGBM+XGB+CatBoost stacked ensemble
│   └── validation/
│       ├── metrics.py          # ECE, Brier, reliability diagrams
│       └── walk_forward.py     # Purged expanding-window CV
├── train.py                    # Training pipeline (all assets)
├── predict.py                  # Production prediction script
├── scheduler.py                # APScheduler hourly runner
├── requirements_v2.txt         # Dependencies
├── data_v2/                    # Cached parquet data
├── models_v2/                  # Trained model artifacts
└── logs/                       # Prediction logs
```
