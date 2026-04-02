import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from data_loader import DataLoader
from intra_hour_features import IntraHourFeatureEngineer

def load_model(symbol):
    prefix = symbol.replace('/', '_')
    with open(f'models/{prefix}_xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'models/{prefix}_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'models/{prefix}_platt.pkl', 'rb') as f:
        platt = pickle.load(f)
    return model, scaler, platt

def compute_hourly_features_for_time(df_hourly, hour_start):
    """
    Compute hourly features for a specific hour start using data up to previous hour.
    df_hourly: DataFrame with hourly OHLCV.
    hour_start: pd.Timestamp of the hour start.
    Returns Series of features.
    """
    # Ensure index is datetime
    df = df_hourly.copy()
    # We need data up to hour_start - 1 hour
    # Compute features using rolling windows up to previous hour close
    # We'll compute for all hours and then select the row for hour_start
    # For simplicity, compute features for all hours and locate
    feats = pd.DataFrame(index=df.index)
    feats['gap_return'] = df['open'] / df['close'].shift(1) - 1
    for window in [1, 2, 4, 8, 12, 24, 48]:
        feats[f'ret_{window}h'] = df['close'].pct_change(window)
    returns = df['close'].pct_change()
    feats['volatility_24h'] = returns.rolling(24).std()
    feats['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
    rolling_low = df['low'].rolling(24).min()
    rolling_high = df['high'].rolling(24).max()
    feats['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-9)
    sma12 = df['close'].rolling(12).mean()
    sma24 = df['close'].rolling(24).mean()
    feats['ma_cross'] = sma12 / sma24 - 1
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feats['rsi'] = 100 - (100 / (1 + rs))
    # Shift features except gap_return
    shift_cols = [col for col in feats.columns if col != 'gap_return']
    feats[shift_cols] = feats[shift_cols].shift(1)
    # Return features for hour_start
    if hour_start in feats.index:
        return feats.loc[hour_start]
    else:
        raise ValueError(f"No features for hour {hour_start}")

def compute_intra_hour_features_for_time(df_minute, hour_start, lookback_minutes=15):
    """
    Compute intra-hour features for a specific hour start using minute data.
    Assumes we have at least lookback_minutes of data within the hour.
    """
    engineer = IntraHourFeatureEngineer(df_minute)
    # We'll directly compute using the method compute_features_for_hour
    feats = engineer.compute_features_for_hour(hour_start, lookback_minutes)
    if feats.empty:
        raise ValueError(f"Insufficient minute data for hour {hour_start}")
    # Rename with intra_ prefix
    feats = feats.add_prefix('intra_')
    return feats

def predict_probability(symbol, hour_start, lookback_minutes=15):
    """
    Predict probability that hour closes higher than opens.
    hour_start: pd.Timestamp of the hour start.
    Returns probability (0-1).
    """
    # Load data
    loader = DataLoader()
    panel = loader.get_hourly_panel([symbol])
    prefix = symbol.replace('/', '_')
    df_hourly = pd.DataFrame()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_hourly[col] = panel[col][f'{prefix}_{col}']
    df_minute = loader.load_asset(symbol, timeframe='1m')
    
    # Compute features
    hourly_feats = compute_hourly_features_for_time(df_hourly, hour_start)
    intra_feats = compute_intra_hour_features_for_time(df_minute, hour_start, lookback_minutes)
    # Combine into Series with consistent column order
    # First ensure both are Series
    hourly_feats = hourly_feats.copy()
    intra_feats = intra_feats.copy()
    # Create a DataFrame with one row, columns in same order as training
    # Training columns: hourly features first, then intra features
    # We'll construct a dict
    feature_dict = {}
    for col in hourly_feats.index:
        feature_dict[col] = hourly_feats[col]
    for col in intra_feats.index:
        feature_dict[col] = intra_feats[col]
    X = pd.DataFrame([feature_dict])
    # Ensure column order matches training (should be same)
    # Load model
    model, scaler, platt = load_model(symbol)
    # Scale
    X_scaled = scaler.transform(X)
    # Predict uncalibrated probability
    prob_uncal = model.predict_proba(X_scaled)[:, 1]
    # Calibrate
    prob_cal = platt.predict_proba(prob_uncal.reshape(-1, 1))[:, 1]
    return prob_cal[0]

if __name__ == '__main__':
    symbol = 'BTC/USDT'
    # Use the latest hour that has at least 15 minutes of data (current hour)
    loader = DataLoader()
    df_min = loader.load_asset(symbol, '1m')
    latest_minute = df_min.index.max()
    hour_start = latest_minute.replace(minute=0, second=0, microsecond=0)
    print(f"Latest minute: {latest_minute}")
    print(f"Predicting for hour starting: {hour_start}")
    
    try:
        prob = predict_probability(symbol, hour_start, lookback_minutes=15)
        print(f"Predicted probability of higher close: {prob:.3f}")
        # For reference, show open price
        df_hourly = loader.load_asset(symbol, '1h')
        open_price = df_hourly.loc[hour_start]['open']
        print(f"Open price: {open_price}")
        # If hour has already closed (in historical data), we can compare
        if hour_start + timedelta(hours=1) <= df_hourly.index.max():
            close_price = df_hourly.loc[hour_start]['close']
            direction = 'UP' if close_price > open_price else 'DOWN'
            print(f"Actual close: {close_price} ({direction})")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()