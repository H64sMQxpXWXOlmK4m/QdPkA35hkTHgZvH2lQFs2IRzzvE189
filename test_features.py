#!/usr/bin/env python3
"""Test elite feature engineering."""
import sys
sys.path.append('.')
from elite_feature_engineer import EliteFeatureEngineer
import pandas as pd
import numpy as np

def test_basic():
    print("Testing EliteFeatureEngineer with BTC/USDT...")
    
    # Initialize
    engineer = EliteFeatureEngineer('BTC/USDT')
    
    # Print data info
    print(f"Hourly data shape: {engineer.df_hourly.shape}")
    print(f"15m data shape: {engineer.df_15m.shape}")
    print(f"5m data shape: {engineer.df_5m.shape}")
    print(f"1m data shape: {engineer.df_1m.shape}")
    
    # Pick a few hour starts
    hour_starts = pd.date_range(start='2026-03-01 00:00:00', periods=5, freq='h')
    
    for hs in hour_starts:
        print(f"\nHour start: {hs}")
        features = engineer.compute_features_at_hour_start(hs)
        if features.empty:
            print("  No features computed (insufficient data)")
        else:
            print(f"  Number of features: {len(features)}")
            # Show some key features
            keys = ['gap_return', 'return_24h', 'vol_24h', 'rsi', 'prev_hour_return', 'hour_sin']
            for k in keys:
                if k in features:
                    print(f"  {k}: {features[k]:.6f}")
    
    # Test dataset creation
    print("\nTesting dataset creation...")
    start = pd.Timestamp('2026-03-01 00:00:00')
    end = pd.Timestamp('2026-03-05 23:00:00')
    X, y = engineer.create_dataset(start, end)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    if X.shape[0] > 0:
        print(f"Feature columns: {list(X.columns[:10])}...")
        print(f"Target distribution: up={y.sum()}, down={len(y)-y.sum()}")
        print(f"Up rate: {y.mean():.3f}")
        
        # Check for missing values
        missing = X.isna().sum().sum()
        print(f"Missing values in X: {missing}")
        
        # Check feature ranges
        print("\nFeature statistics:")
        print(X.describe().T[['mean', 'std', 'min', 'max']].head(10))

if __name__ == '__main__':
    test_basic()