import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from intra_hour_features import IntraHourFeatureEngineer
from regime_detection import RegimeDetector
from data_loader import DataLoader

class AdvancedFeatureEngineer:
    def __init__(self, symbol: str, lookback_minutes: int = 15):
        self.symbol = symbol
        self.lookback = lookback_minutes
        self.loader = DataLoader(data_dir='data/deep')
        # Load data
        self.df_hourly = self._load_hourly()
        self.df_minute = self._load_minute()
        self.intra_engineer = IntraHourFeatureEngineer(self.df_minute)
        self.regime_detector = RegimeDetector(window_bars=168)
    
    def _load_hourly(self) -> pd.DataFrame:
        """Load hourly data."""
        return self.loader.load_asset(self.symbol, timeframe='1h')
    
    def _load_minute(self) -> pd.DataFrame:
        """Load minute data."""
        return self.loader.load_asset(self.symbol, timeframe='1m')
    
    def compute_intra_path_features(self, minute_slice: pd.DataFrame, open_price: float) -> pd.Series:
        """
        Compute path-based features from minute data slice.
        minute_slice: DataFrame with columns open, high, low, close, volume.
        open_price: hour's open price.
        """
        feats = {}
        if len(minute_slice) < 2:
            return pd.Series()
        
        prices = minute_slice['close']
        highs = minute_slice['high']
        lows = minute_slice['low']
        volumes = minute_slice['volume']
        
        # 1. Return from open to last minute
        feats['intra_return'] = prices.iloc[-1] / open_price - 1
        
        # 2. Maximum gain and drawdown within window
        rel_high = highs.max()
        rel_low = lows.min()
        feats['max_gain'] = rel_high / open_price - 1
        feats['max_drawdown'] = rel_low / open_price - 1
        
        # 3. Time to max gain / max drawdown (normalized by window length)
        idx_max_gain = highs.idxmax()
        idx_min_low = lows.idxmin()
        # position within slice (0 to 1)
        feats['time_to_max_gain'] = (idx_max_gain - minute_slice.index[0]).total_seconds() / (60 * self.lookback)
        feats['time_to_max_dd'] = (idx_min_low - minute_slice.index[0]).total_seconds() / (60 * self.lookback)
        
        # 4. Number of crosses of opening price
        crosses = ((prices.shift(1) <= open_price) & (prices > open_price)) | \
                  ((prices.shift(1) >= open_price) & (prices < open_price))
        feats['crosses'] = crosses.sum()
        
        # 5. Momentum: linear regression slope of price path
        if len(prices) >= 3:
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices.values)
            feats['path_slope'] = slope / open_price  # normalized
            feats['path_r2'] = r_value ** 2
        else:
            feats['path_slope'] = np.nan
            feats['path_r2'] = np.nan
        
        # 6. Volatility of minute returns within window
        minute_returns = prices.pct_change().dropna()
        if len(minute_returns) > 1:
            feats['intra_minute_vol'] = minute_returns.std()
            feats['intra_minute_skew'] = minute_returns.skew()
            feats['intra_minute_kurt'] = minute_returns.kurtosis()
        else:
            feats['intra_minute_vol'] = np.nan
            feats['intra_minute_skew'] = np.nan
            feats['intra_minute_kurt'] = np.nan
        
        # 7. Volume concentration
        feats['volume_sum'] = volumes.sum()
        feats['volume_std'] = volumes.std() / (volumes.mean() + 1e-9)
        
        # 8. Price efficiency (close-to-close variance vs high-low range)
        if len(prices) > 1:
            close_var = prices.pct_change().var()
            hl_range = (highs - lows).mean() / open_price
            feats['price_efficiency'] = close_var / (hl_range + 1e-9)
        else:
            feats['price_efficiency'] = np.nan
        
        return pd.Series(feats)
    
    def compute_hourly_features(self) -> pd.DataFrame:
        """
        Compute hourly features (lagged).
        """
        df = self.df_hourly
        feats = pd.DataFrame(index=df.index)
        
        # Gap return
        feats['gap_return'] = df['open'] / df['close'].shift(1) - 1
        
        # Past returns
        for window in [1, 2, 4, 8, 12, 24, 48, 96, 168]:
            feats[f'ret_{window}h'] = df['close'].pct_change(window)
        
        # Rolling volatility (24h, 48h, 168h)
        returns = df['close'].pct_change()
        for window in [24, 48, 168]:
            feats[f'volatility_{window}h'] = returns.rolling(window).std()
        
        # Volume features
        feats['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
        feats['volume_change'] = df['volume'].pct_change()
        
        # Price position within recent range (24h)
        rolling_low = df['low'].rolling(24).min()
        rolling_high = df['high'].rolling(24).max()
        feats['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-9)
        
        # Moving average cross
        feats['sma_12'] = df['close'].rolling(12).mean()
        feats['sma_24'] = df['close'].rolling(24).mean()
        feats['sma_50'] = df['close'].rolling(50).mean()
        feats['sma_200'] = df['close'].rolling(200).mean()
        feats['ma_cross_12_24'] = feats['sma_12'] / feats['sma_24'] - 1
        feats['ma_cross_50_200'] = feats['sma_50'] / feats['sma_200'] - 1
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        feats['rsi'] = 100 - (100 / (1 + rs))
        
        # Shift to avoid lookahead (except gap_return)
        shift_cols = [col for col in feats.columns if col != 'gap_return']
        feats[shift_cols] = feats[shift_cols].shift(1)
        
        # Drop rows with NaN (due to rolling windows)
        feats = feats.dropna()
        return feats
    
    def compute_regime_features(self) -> pd.DataFrame:
        """Compute regime features."""
        regime_feats = self.regime_detector.compute_regime_features(self.df_hourly)
        return regime_feats
    
    def compute_intra_hour_features(self) -> pd.DataFrame:
        """
        Compute intra-hour features for all hours.
        Uses IntraHourFeatureEngineer for basic intra features,
        then adds path features.
        """
        hourly_labels = (self.df_hourly['close'] > self.df_hourly['open']).astype(int)
        intra_feats, labels_aligned = self.intra_engineer.align_with_labels(hourly_labels)
        # Add path features
        all_intra = []
        for hour_start in intra_feats.index:
            # Get minute slice for first lookback minutes
            hour_end = hour_start + pd.Timedelta(hours=1)
            minute_slice = self.df_minute.loc[hour_start: hour_end - pd.Timedelta(minutes=1)]
            minute_slice = minute_slice.iloc[:self.lookback]
            if len(minute_slice) < 2:
                continue
            open_price = self.df_hourly.loc[hour_start, 'open']
            path_feats = self.compute_intra_path_features(minute_slice, open_price)
            if path_feats.empty:
                continue
            # Combine with existing intra features
            base_feats = intra_feats.loc[hour_start]
            combined = pd.concat([base_feats, path_feats])
            combined.name = hour_start
            all_intra.append(combined)
        
        intra_df = pd.DataFrame(all_intra)
        return intra_df
    
    def compute_all_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute all features and labels.
        Returns (features, labels).
        """
        # Hourly features
        hourly_feats = self.compute_hourly_features()
        # Regime features
        regime_feats = self.compute_regime_features()
        # Intra-hour features
        intra_feats = self.compute_intra_hour_features()
        
        # Align indices (intersection)
        common_idx = hourly_feats.index.intersection(regime_feats.index).intersection(intra_feats.index)
        hourly_aligned = hourly_feats.loc[common_idx]
        regime_aligned = regime_feats.loc[common_idx]
        intra_aligned = intra_feats.loc[common_idx]
        
        # Combine
        features = pd.concat([hourly_aligned, regime_aligned, intra_aligned], axis=1)
        # Labels
        labels = (self.df_hourly['close'] > self.df_hourly['open']).astype(int)
        labels = labels.loc[common_idx]
        
        # Remove rows with NaN
        features = features.dropna()
        labels = labels.loc[features.index]
        
        return features, labels

def test_advanced_features():
    """Test the advanced feature engineering."""
    symbol = 'BTC/USDT'
    print(f"Testing advanced features for {symbol}")
    engineer = AdvancedFeatureEngineer(symbol, lookback_minutes=15)
    features, labels = engineer.compute_all_features()
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Feature columns: {list(features.columns)}")
    print(f"Label distribution:\n{labels.value_counts() / len(labels)}")
    
    # Save sample
    features.to_csv('advanced_features_sample.csv')
    return features, labels

if __name__ == '__main__':
    features, labels = test_advanced_features()