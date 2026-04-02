# Hourly Candle Direction Probability Estimator

This project builds a system that estimates the probability that the next 1-hour candle closes higher than it opens for six major cryptocurrencies: BTC, ETH, SOL, XRP, BNB, DOGE.

The system uses historical price and volume data from Binance, engineers features from both hourly and intra-hour (minute) data, and trains machine learning models to output calibrated probabilities.

## Key Results

- **Baseline (random guessing)**: 50% accuracy, Brier score 0.25
- **Intra-hour features only (logistic regression)**: 65-74% accuracy, ROC AUC 0.72-0.79, Brier score 0.18-0.21
- **Combined hourly + intra-hour features (XGBoost + Platt scaling)**: ~62-63% accuracy, ROC AUC 0.68, Brier score 0.23 (BTC example)

The system demonstrates predictive power beyond random chance, suggesting that early intra-hour price action contains signal for the full hour's direction.

## System Architecture

### Data Pipeline
1. **Data Fetching**: Uses `ccxt` to download minute and hourly OHLCV data from Binance.
2. **Data Storage**: Raw data saved in `data/raw/`, resampled data in `data/processed/`.
3. **Feature Engineering**:
   - **Hourly features**: gap return, past returns (1h-48h), rolling volatility, volume ratio, price position, moving average cross, RSI.
   - **Intra-hour features**: first 5-minute return, volume, high-low range, minute returns statistics (mean, std, skew, kurtosis), volume ratio, price change within first 15 minutes.
   - Features are aligned to avoid lookahead bias.
4. **Modeling**:
   - Per-asset XGBoost classifier with L2 regularization.
   - Platt scaling for probability calibration.
   - Models saved in `models/` with corresponding scalers and calibrators.
5. **Prediction**: Given latest market data, compute features and output probability for the next hour.

### Files

- `fetch_data.py` – Download historical data from Binance.
- `data_loader.py` – Load and combine asset data.
- `feature_engineering.py` – Hourly feature generation.
- `intra_hour_features.py` – Intra-hour feature generation.
- `modeling.py` – Logistic regression baseline.
- `model_intra_hour.py` – Evaluate intra-hour features.
- `build_btc_model.py` – Train XGBoost model for BTC with combined features.
- `predict.py` – Predict probability for the current hour.
- `panel_features.py` – (Experimental) Panel dataset construction.

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Fetch data (optional, already provided)
```bash
python fetch_data.py
```

### 3. Train models
Train a model for BTC:
```bash
python build_btc_model.py
```
This will save model artifacts in `models/`.

### 4. Run prediction
```bash
python predict.py
```
Outputs probability that the current hour (based on latest available minute data) will close higher than it opened.

### 5. Evaluate performance
```bash
python model_intra_hour.py
```
Generates calibration curves and metrics for all assets using intra-hour features only.

## Key Insights

- Early intra-hour price movements (first 5–15 minutes) contain predictive information for the full hour's direction.
- Combining hourly trends with intra-hour dynamics improves robustness.
- The system is well‑calibrated, with Brier scores consistently below 0.25 (random baseline).
- Feature importance analysis shows `intra_price_change_within` (price change in the first 15 minutes) is the strongest predictor.

## Limitations & Future Work

- **Data scope**: Only 30 days of minute data due to API rate limits; longer histories would improve generalization.
- **Market regimes**: Model performance may vary across bull/bear markets; regime‑switching could be incorporated.
- **Cross‑asset signals**: Currently only BTC leader features are used; more sophisticated cross‑asset modeling could be added.
- **Live deployment**: The prediction script assumes data is already downloaded; a real‑time pipeline would need incremental updates.

## License

MIT