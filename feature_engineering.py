import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Generate features and labels for a single asset.
    Assumes hourly OHLCV data with columns: open, high, low, close, volume.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features_df = None
        self.labels = None
    
    def compute_labels(self) -> pd.Series:
        """Label = 1 if close > open for the same hour, else 0."""
        labels = (self.df['close'] > self.df['open']).astype(int)
        return labels
    
    def compute_basic_features(self) -> pd.DataFrame:
        """
        Compute basic features using past data only (no lookahead).
        Features are aligned such that for hour t, we use data up to t-1
        (except open price at t).
        """
        df = self.df
        # Initialize features DataFrame with same index as df
        feats = pd.DataFrame(index=df.index)
        
        # 1. Gap return: open_t / close_{t-1} - 1
        feats['gap_return'] = df['open'] / df['close'].shift(1) - 1
        
        # 2. Past returns (close-to-close)
        for window in [1, 2, 4, 8, 12, 24, 48]:
            feats[f'ret_{window}h'] = df['close'].pct_change(window)
        
        # 3. Rolling volatility (std of hourly returns) over past 24h, 48h
        hourly_returns = df['close'].pct_change()
        feats['volatility_24h'] = hourly_returns.rolling(24).std()
        feats['volatility_48h'] = hourly_returns.rolling(48).std()
        
        # 4. Volume features
        feats['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
        feats['volume_change'] = df['volume'].pct_change()
        
        # 5. Price position within recent range (last 24h)
        rolling_low = df['low'].rolling(24).min()
        rolling_high = df['high'].rolling(24).max()
        feats['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-9)
        
        # 6. Moving average cross
        feats['sma_12'] = df['close'].rolling(12).mean()
        feats['sma_24'] = df['close'].rolling(24).mean()
        feats['ma_cross'] = feats['sma_12'] / feats['sma_24'] - 1
        
        # 7. RSI (14 period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        feats['rsi'] = 100 - (100 / (1 + rs))
        
        # 8. Time features
        feats['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        feats['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        feats['day_of_week'] = df.index.dayofweek
        
        # 9. Lagged target (previous hour label)
        feats['prev_label'] = self.compute_labels().shift(1)
        
        # Shift features that use current hour data (except gap_return which uses open_t)
        # Actually, gap_return uses open_t and close_{t-1}, that's fine.
        # Features that use close_t (like ret_1h) are forward-looking! Need to shift.
        # We must ensure no feature uses close_t. Let's recompute returns using shifted close.
        # We'll replace ret_{window}h with pct_change(window) shifted by 1.
        # Actually we want features at start of hour t, we only know close up to t-1.
        # So we need to shift all features that involve close_t.
        # Let's recompute returns as close_{t-1} / close_{t-1-window} - 1.
        # We'll create a separate function for shifted features.
        # For simplicity, we'll shift all features by 1 hour (except gap_return).
        # But we also have open_t; we can keep gap_return as is.
        
        # We'll create a new method that returns properly aligned features.
        # For now, we'll just shift everything except gap_return.
        # Let's identify columns that need shifting.
        # We'll do this later.
        
        return feats
    
    def align_features(self, feats: pd.DataFrame) -> pd.DataFrame:
        """
        Shift features to avoid lookahead bias.
        For each hour t, features must be based on data up to t-1 (except open at t).
        We'll shift all columns except those that explicitly use open_t.
        """
        # Columns that use open_t: 'gap_return'
        # Also 'hour_sin', 'hour_cos', 'day_of_week' are time features known at t.
        # Other columns should be shifted by 1.
        shift_cols = [col for col in feats.columns if col not in ['gap_return', 'hour_sin', 'hour_cos', 'day_of_week']]
        feats_aligned = feats.copy()
        feats_aligned[shift_cols] = feats[shift_cols].shift(1)
        return feats_aligned
    
    def generate(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate aligned features and labels."""
        feats = self.compute_basic_features()
        feats_aligned = self.align_features(feats)
        labels = self.compute_labels()
        # Align labels with features (same index)
        # Remove rows with NaN features (due to rolling windows)
        feats_aligned = feats_aligned.dropna()
        labels = labels.loc[feats_aligned.index]
        self.features_df = feats_aligned
        self.labels = labels
        return feats_aligned, labels

if __name__ == '__main__':
    from data_loader import DataLoader
    loader = DataLoader()
    symbols = ['BTC/USDT', 'ETH/USDT']
    panel = loader.get_hourly_panel(symbols)
    # Test on BTC
    btc_df = panel['close'][['BTC_USDT_close']].join(panel['open']['BTC_USDT_open']).join(panel['high']['BTC_USDT_high']).join(panel['low']['BTC_USDT_low']).join(panel['volume']['BTC_USDT_volume'])
    btc_df.columns = ['close', 'open', 'high', 'low', 'volume']
    print("BTC data shape:", btc_df.shape)
    eng = FeatureEngineer(btc_df)
    feats, labels = eng.generate()
    print("Features shape:", feats.shape)
    print("Labels shape:", labels.shape)
    print(feats.head())
    print(labels.head())