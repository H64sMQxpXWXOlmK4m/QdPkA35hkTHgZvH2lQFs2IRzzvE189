import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import os
from data_loader import DataLoader
from intra_hour_features import IntraHourFeatureEngineer

def load_and_merge_features(symbol='BTC/USDT', lookback_minutes=15):
    """
    Load hourly and minute data, compute features, merge.
    Returns DataFrame with features and labels.
    """
    loader = DataLoader()
    # Hourly data (90 days)
    panel = loader.get_hourly_panel([symbol])
    prefix = symbol.replace('/', '_')
    df_hourly = pd.DataFrame()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_hourly[col] = panel[col][f'{prefix}_{col}']
    
    # Compute hourly features (lagged)
    hourly_feats = pd.DataFrame(index=df_hourly.index)
    # Gap return
    hourly_feats['gap_return'] = df_hourly['open'] / df_hourly['close'].shift(1) - 1
    # Past returns
    for window in [1, 2, 4, 8, 12, 24, 48]:
        hourly_feats[f'ret_{window}h'] = df_hourly['close'].pct_change(window)
    # Rolling volatility (24h)
    returns = df_hourly['close'].pct_change()
    hourly_feats['volatility_24h'] = returns.rolling(24).std()
    # Volume ratio
    hourly_feats['volume_ratio'] = df_hourly['volume'] / df_hourly['volume'].rolling(24).mean()
    # Price position
    rolling_low = df_hourly['low'].rolling(24).min()
    rolling_high = df_hourly['high'].rolling(24).max()
    hourly_feats['price_position'] = (df_hourly['close'] - rolling_low) / (rolling_high - rolling_low + 1e-9)
    # SMA cross
    sma12 = df_hourly['close'].rolling(12).mean()
    sma24 = df_hourly['close'].rolling(24).mean()
    hourly_feats['ma_cross'] = sma12 / sma24 - 1
    # RSI
    delta = df_hourly['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    hourly_feats['rsi'] = 100 - (100 / (1 + rs))
    # Shift to avoid lookahead (except gap_return)
    shift_cols = [col for col in hourly_feats.columns if col != 'gap_return']
    hourly_feats[shift_cols] = hourly_feats[shift_cols].shift(1)
    
    # Minute data (30 days)
    df_minute = loader.load_asset(symbol, timeframe='1m')
    # Intra-hour features
    engineer = IntraHourFeatureEngineer(df_minute)
    hourly_labels = (df_hourly['close'] > df_hourly['open']).astype(int)
    intra_feats, _ = engineer.align_with_labels(hourly_labels)
    # Rename columns with prefix intra_
    intra_feats = intra_feats.add_prefix('intra_')
    
    # Merge on hour start
    merged = pd.concat([hourly_feats, intra_feats], axis=1, join='inner')
    # Label
    merged['label'] = (df_hourly['close'] > df_hourly['open']).astype(int)
    # Drop rows with NaN
    merged = merged.dropna()
    return merged

def train_test_split_temporal(df, test_ratio=0.3):
    """Split by time."""
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with calibration."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict probabilities (uncalibrated)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calibrate using Platt scaling (logistic regression)
    from sklearn.linear_model import LogisticRegression
    platt = LogisticRegression(C=1e9, solver='lbfgs')
    # Fit on train predictions
    train_proba = model.predict_proba(X_train_scaled)[:, 1].reshape(-1, 1)
    platt.fit(train_proba, y_train)
    # Calibrate test probabilities
    y_pred_proba_cal = platt.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
    
    # Evaluate
    metrics = {
        'accuracy': accuracy_score(y_test, (y_pred_proba >= 0.5).astype(int)),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'brier': brier_score_loss(y_test, y_pred_proba),
        'accuracy_cal': accuracy_score(y_test, (y_pred_proba_cal >= 0.5).astype(int)),
        'roc_auc_cal': roc_auc_score(y_test, y_pred_proba_cal),
        'brier_cal': brier_score_loss(y_test, y_pred_proba_cal),
    }
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_cal, n_bins=10)
    metrics['calibration_curve'] = (prob_true, prob_pred)
    
    return model, scaler, platt, metrics

def plot_calibration(prob_true, prob_pred, symbol):
    plt.figure(figsize=(8,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve for {symbol} (XGBoost + Platt)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'calibration_xgb_{symbol.replace("/", "_")}.png')
    plt.close()

def save_artifacts(model, scaler, platt, symbol):
    """Save model, scaler, and calibrator."""
    os.makedirs('models', exist_ok=True)
    prefix = symbol.replace('/', '_')
    with open(f'models/{prefix}_xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'models/{prefix}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'models/{prefix}_platt.pkl', 'wb') as f:
        pickle.dump(platt, f)
    print(f"Artifacts saved for {symbol}")

if __name__ == '__main__':
    symbol = 'BTC/USDT'
    print(f"Building features for {symbol}...")
    df = load_and_merge_features(symbol, lookback_minutes=15)
    print(f"Dataset shape: {df.shape}")
    
    # Split
    train, test = train_test_split_temporal(df, test_ratio=0.3)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    X_train = train.drop(columns=['label'])
    y_train = train['label']
    X_test = test.drop(columns=['label'])
    y_test = test['label']
    
    print("Training XGBoost with Platt scaling...")
    model, scaler, platt, metrics = train_xgboost(X_train, y_train, X_test, y_test)
    
    print("\n=== Metrics ===")
    print(f"Accuracy (uncal): {metrics['accuracy']:.4f}")
    print(f"ROC AUC (uncal): {metrics['roc_auc']:.4f}")
    print(f"Brier (uncal): {metrics['brier']:.4f}")
    print(f"Accuracy (cal): {metrics['accuracy_cal']:.4f}")
    print(f"ROC AUC (cal): {metrics['roc_auc_cal']:.4f}")
    print(f"Brier (cal): {metrics['brier_cal']:.4f}")
    
    # Plot calibration
    plot_calibration(*metrics['calibration_curve'], symbol)
    
    # Save artifacts
    save_artifacts(model, scaler, platt, symbol)
    
    # Feature importance
    importance = model.feature_importances_
    feat_names = X_train.columns
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importance}).sort_values('importance', ascending=False)
    print("\nTop 10 features:")
    print(imp_df.head(10))
    imp_df.to_csv(f'feature_importance_{symbol.replace("/", "_")}.csv', index=False)