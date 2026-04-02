import pandas as pd
import numpy as np
from typing import Optional, Tuple, List

class IntraHourFeatureEngineer:
    """
    Generate intra-hour features from minute data.
    Assumes minute OHLCV data with columns: open, high, low, close, volume.
    """
    def __init__(self, df_minute: pd.DataFrame):
        self.df = df_minute.copy()
        self.df.index = pd.to_datetime(self.df.index)
        # resample to 5-minute intervals
        self.df_5min = self.resample_5min()
        
    def resample_5min(self) -> pd.DataFrame:
        """Resample minute data to 5-minute OHLCV."""
        resampled = self.df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return resampled
    
    def compute_features_for_hour(self, hour_start: pd.Timestamp, lookback_minutes: int = 15) -> pd.Series:
        """
        Compute features for the hour starting at hour_start,
        using minute data up to lookback_minutes within the hour.
        Returns a Series of features.
        """
        # Slice minute data within this hour
        hour_end = hour_start + pd.Timedelta(hours=1)
        minute_in_hour = self.df.loc[hour_start: hour_end - pd.Timedelta(minutes=1)]
        # If not enough data, return NaN
        if len(minute_in_hour) < lookback_minutes:
            return pd.Series()
        
        # Use only first lookback_minutes minutes
        minute_subset = minute_in_hour.iloc[:lookback_minutes]
        
        # Hour open price (first minute open)
        hour_open = minute_subset.iloc[0]['open']
        
        features = {}
        
        # 1. First 5-minute return (using 5min resampled data)
        # Find the 5-minute block that starts at hour_start
        five_min_block = self.df_5min.loc[hour_start: hour_start + pd.Timedelta(minutes=5)]
        if not five_min_block.empty:
            first_5m_close = five_min_block.iloc[0]['close']
            features['first_5m_return'] = first_5m_close / hour_open - 1
            features['first_5m_volume'] = five_min_block.iloc[0]['volume']
            features['first_5m_high_low_range'] = (five_min_block.iloc[0]['high'] - five_min_block.iloc[0]['low']) / hour_open
        else:
            features['first_5m_return'] = np.nan
            features['first_5m_volume'] = np.nan
            features['first_5m_high_low_range'] = np.nan
        
        # 2. Statistics of minute returns within lookback window
        minute_returns = minute_subset['close'].pct_change().dropna()
        if len(minute_returns) > 1:
            features['minute_returns_mean'] = minute_returns.mean()
            features['minute_returns_std'] = minute_returns.std()
            features['minute_returns_skew'] = minute_returns.skew()
            features['minute_returns_kurt'] = minute_returns.kurtosis()
        else:
            features['minute_returns_mean'] = np.nan
            features['minute_returns_std'] = np.nan
            features['minute_returns_skew'] = np.nan
            features['minute_returns_kurt'] = np.nan
        
        # 3. Volume ratio: volume in first lookback minutes vs average volume per minute in previous hour
        prev_hour_start = hour_start - pd.Timedelta(hours=1)
        prev_hour_minutes = self.df.loc[prev_hour_start: hour_start - pd.Timedelta(minutes=1)]
        if len(prev_hour_minutes) > 0:
            avg_volume_prev = prev_hour_minutes['volume'].mean()
            features['volume_ratio'] = minute_subset['volume'].sum() / (avg_volume_prev * lookback_minutes) if avg_volume_prev > 0 else np.nan
        else:
            features['volume_ratio'] = np.nan
        
        # 4. Price change from open to minute lookback (close of last minute in subset)
        last_close = minute_subset.iloc[-1]['close']
        features['price_change_within'] = last_close / hour_open - 1
        
        # 5. High/low range within lookback window
        features['high_low_range'] = (minute_subset['high'].max() - minute_subset['low'].min()) / hour_open
        
        # 6. Time of day (redundant but keep)
        features['hour_of_day'] = hour_start.hour
        features['day_of_week'] = hour_start.dayofweek
        
        return pd.Series(features)
    
    def generate_features_for_all_hours(self, lookback_minutes: int = 15) -> pd.DataFrame:
        """
        Generate features for all hours where minute data is available.
        Returns DataFrame indexed by hour start timestamps.
        """
        # Determine all hour starts that have at least lookback_minutes of data
        # Use the 5min resampled index as reference (since we need first 5min block)
        hour_starts = self.df_5min.resample('1h').asfreq().index
        all_features = []
        for hs in hour_starts:
            feats = self.compute_features_for_hour(hs, lookback_minutes)
            if not feats.empty:
                feats.name = hs
                all_features.append(feats)
        features_df = pd.DataFrame(all_features)
        return features_df
    
    def align_with_labels(self, hourly_labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align intra-hour features with hourly labels (binary).
        hourly_labels: Series indexed by hour start, value 1 if close > open.
        Returns (features, labels) aligned.
        """
        features_df = self.generate_features_for_all_hours()
        # Align indices
        aligned_idx = features_df.index.intersection(hourly_labels.index)
        features_aligned = features_df.loc[aligned_idx]
        labels_aligned = hourly_labels.loc[aligned_idx]
        # Drop rows with NaN features
        features_aligned = features_aligned.dropna()
        labels_aligned = labels_aligned.loc[features_aligned.index]
        return features_aligned, labels_aligned

if __name__ == '__main__':
    # Test with BTC minute data
    from data_loader import DataLoader
    loader = DataLoader(data_dir='data/raw')
    # Load minute data
    btc_minute = loader.load_asset('BTC/USDT', timeframe='1m')
    print("BTC minute shape:", btc_minute.shape)
    
    # Load hourly labels (close > open)
    btc_hourly = loader.load_asset('BTC/USDT', timeframe='1h')
    hourly_labels = (btc_hourly['close'] > btc_hourly['open']).astype(int)
    
    # Generate intra-hour features
    engineer = IntraHourFeatureEngineer(btc_minute)
    features, labels = engineer.align_with_labels(hourly_labels)
    print("Features shape:", features.shape)
    print(features.head())
    print(labels.head())