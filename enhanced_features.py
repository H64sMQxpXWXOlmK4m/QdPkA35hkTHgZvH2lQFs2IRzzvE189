import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from advanced_features import AdvancedFeatureEngineer
from data_loader import DataLoader
from regime_detection import RegimeDetector

class EnhancedFeatureEngineer(AdvancedFeatureEngineer):
    """
    Enhanced features for better regime robustness.
    Adds:
    - Volatility-normalized early returns
    - Technical indicators (ADX, Bollinger bandwidth, RSI divergence)
    - Interaction features (early_return * regime)
    - Mean reversion features for sideways markets
    """
    def __init__(self, symbol: str, lookback_minutes: int = 15):
        super().__init__(symbol, lookback_minutes)
    
    def compute_technical_indicators(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators from hourly data.
        All indicators are lagged to avoid lookahead.
        """
        df = df_hourly.copy()
        feats = pd.DataFrame(index=df.index)
        
        # 1. Bollinger Bands (manual calculation)
        window = 20
        std = 2
        rolling_mean = df['close'].rolling(window).mean()
        rolling_std = df['close'].rolling(window).std()
        bb_upper = rolling_mean + std * rolling_std
        bb_lower = rolling_mean - std * rolling_std
        feats['bb_width'] = (bb_upper - bb_lower) / rolling_mean
        feats['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
        
        # 2. MACD (manual)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        feats['macd'] = ema12 - ema26
        feats['macd_signal'] = feats['macd'].ewm(span=9, adjust=False).mean()
        feats['macd_hist'] = feats['macd'] - feats['macd_signal']
        
        # 3. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        feats['rsi'] = 100 - (100 / (1 + rs))
        
        # 4. ATR (manual)
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift()).abs()
        low_close_prev = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        feats['atr'] = tr.rolling(14).mean()
        
        # 5. Volume indicators
        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        feats['volume_obv'] = obv
        
        # 6. Price distance from moving averages
        feats['dist_sma_50'] = df['close'] / df['close'].rolling(50).mean() - 1
        feats['dist_sma_200'] = df['close'] / df['close'].rolling(200).mean() - 1
        
        # 7. Rolling volatility (different windows)
        returns = df['close'].pct_change()
        for window in [6, 12, 24, 48]:
            feats[f'volatility_{window}h'] = returns.rolling(window).std()
        
        # 8. Trend strength (simplified ADX)
        # Directional movement
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        
        pos_di = 100 * pd.Series(pos_dm, index=df.index).rolling(14).mean() / (atr14 + 1e-9)
        neg_di = 100 * pd.Series(neg_dm, index=df.index).rolling(14).mean() / (atr14 + 1e-9)
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-9)
        feats['adx'] = dx.rolling(14).mean()
        feats['adx_trend_strength'] = (feats['adx'] > 25).astype(float)
        
        # Shift all technical indicators by 1 hour (to avoid lookahead)
        feats = feats.shift(1)
        
        return feats
    
    def compute_enhanced_intra_features(self, minute_slice: pd.DataFrame, open_price: float,
                                        recent_volatility: float) -> pd.Series:
        """
        Enhanced intra-hour features.
        recent_volatility: rolling volatility from previous hour.
        """
        feats = {}
        
        # Basic path features from parent
        path_feats = self.compute_intra_path_features(minute_slice, open_price)
        for k, v in path_feats.items():
            feats[k] = v
        
        # Enhanced features
        if len(minute_slice) >= 2:
            prices = minute_slice['close']
            highs = minute_slice['high']
            lows = minute_slice['low']
            volumes = minute_slice['volume']
            
            # Normalize early returns by recent volatility
            early_return = feats.get('intra_return', np.nan)
            if not np.isnan(early_return) and recent_volatility > 0:
                feats['early_return_norm'] = early_return / (recent_volatility + 1e-9)
            
            # Volume acceleration
            if 'volume_sum' in feats and recent_volatility > 0:
                feats['volume_acceleration'] = feats['volume_sum'] / (recent_volatility + 1e-9)
            
            # Price efficiency (close-to-close variance vs high-low range)
            if 'price_efficiency' in feats:
                # Already computed
                pass
            
            # Mean reversion tendency (for sideways markets)
            # If price oscillates around opening frequently
            crosses = feats.get('crosses', 0)
            feats['mean_reversion_score'] = crosses / len(minute_slice) if len(minute_slice) > 0 else 0
            
            # Trend vs noise ratio
            if 'path_slope' in feats and 'intra_minute_vol' in feats:
                slope = feats['path_slope']
                minute_vol = feats['intra_minute_vol']
                if minute_vol > 0:
                    feats['trend_noise_ratio'] = abs(slope) / minute_vol
        
        return pd.Series(feats)
    
    def compute_regime_interaction_features(self, hourly_feats: pd.DataFrame, 
                                           regime_feats: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between early signals and regimes.
        """
        interactions = pd.DataFrame(index=hourly_feats.index)
        
        # Get early return column name
        early_cols = [col for col in hourly_feats.columns if 'first_5m_return' in col]
        if early_cols:
            early_col = early_cols[0]
            early_return = hourly_feats[early_col]
            
            # Interactions with volatility regime
            if 'vol_high' in regime_feats.columns:
                interactions['early_return_high_vol'] = early_return * regime_feats['vol_high']
                interactions['early_return_low_vol'] = early_return * regime_feats['vol_low']
            
            # Interactions with trend regime
            if 'trend_up' in regime_feats.columns:
                interactions['early_return_trend_up'] = early_return * regime_feats['trend_up']
                interactions['early_return_trend_down'] = early_return * regime_feats['trend_down']
                interactions['early_return_sideways'] = early_return * regime_feats['trend_sideways']
        
        return interactions
    
    def compute_all_features_enhanced(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute all enhanced features.
        """
        # 1. Basic features from parent
        hourly_feats = self.compute_hourly_features()
        regime_feats = self.compute_regime_features()
        
        # 2. Technical indicators
        tech_feats = self.compute_technical_indicators(self.df_hourly)
        
        # 3. Enhanced intra-hour features (need recent volatility)
        # Compute rolling volatility for normalization
        returns = self.df_hourly['close'].pct_change()
        recent_volatility = returns.rolling(24).std()
        
        # Get intra-hour features with enhancement
        hourly_labels = (self.df_hourly['close'] > self.df_hourly['open']).astype(int)
        intra_feats, labels_aligned = self.intra_engineer.align_with_labels(hourly_labels)
        
        enhanced_intra = []
        for hour_start in intra_feats.index:
            # Get minute slice
            hour_end = hour_start + pd.Timedelta(hours=1)
            minute_slice = self.df_minute.loc[hour_start: hour_end - pd.Timedelta(minutes=1)]
            minute_slice = minute_slice.iloc[:self.lookback]
            if len(minute_slice) < 2:
                continue
            
            open_price = self.df_hourly.loc[hour_start, 'open']
            # Recent volatility at start of hour (using data up to previous hour)
            vol = recent_volatility.loc[hour_start] if hour_start in recent_volatility.index else np.nan
            
            # Compute enhanced features
            enhanced = self.compute_enhanced_intra_features(minute_slice, open_price, vol)
            if enhanced.empty:
                continue
            
            # Combine with existing intra features
            base = intra_feats.loc[hour_start]
            combined = pd.concat([base, enhanced])
            combined.name = hour_start
            enhanced_intra.append(combined)
        
        intra_df = pd.DataFrame(enhanced_intra) if enhanced_intra else pd.DataFrame()
        
        # 4. Regime interaction features
        interaction_feats = self.compute_regime_interaction_features(hourly_feats, regime_feats)
        
        # 5. Combine all features
        # Align indices
        all_feats = [hourly_feats, regime_feats, tech_feats, interaction_feats]
        if not intra_df.empty:
            all_feats.append(intra_df)
        
        # Intersection of indices
        common_idx = hourly_feats.index
        for df in all_feats[1:]:
            if not df.empty:
                common_idx = common_idx.intersection(df.index)
        
        # Combine
        combined_feats = pd.concat([df.loc[common_idx] for df in all_feats if not df.empty], axis=1)
        
        # Labels
        labels = (self.df_hourly['close'] > self.df_hourly['open']).astype(int)
        labels = labels.loc[common_idx]
        
        # Remove rows with NaN
        combined_feats = combined_feats.dropna()
        labels = labels.loc[combined_feats.index]
        
        return combined_feats, labels
    
    def compute_simplified_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Compute a simplified feature set focusing on the most robust signals.
        """
        # Start with basic hourly features
        hourly = self.compute_hourly_features()
        
        # Select key features
        key_cols = [
            'gap_return',
            'ret_1h', 'ret_24h', 'ret_168h',
            'volatility_24h', 'volatility_168h',
            'volume_ratio',
            'price_position',
            'rsi',
        ]
        selected = hourly[key_cols].copy()
        
        # Add normalized early return
        early_cols = [col for col in hourly.columns if 'first_5m_return' in col]
        if early_cols:
            early_col = early_cols[0]
            selected['early_return'] = hourly[early_col]
            # Normalize by recent volatility
            selected['early_return_norm'] = selected['early_return'] / (hourly['volatility_24h'] + 1e-9)
        
        # Add regime (volatility)
        regime = self.compute_regime_features()
        regime_cols = ['vol_high', 'vol_low', 'vol_medium', 'trend_up', 'trend_down', 'trend_sideways']
        for col in regime_cols:
            if col in regime.columns:
                selected[col] = regime[col]
        
        # Interaction: early_return * vol_regime
        if 'early_return' in selected.columns and 'vol_high' in selected.columns:
            selected['early_return_high_vol'] = selected['early_return'] * selected['vol_high']
            selected['early_return_low_vol'] = selected['early_return'] * selected['vol_low']
        
        # Labels
        labels = (self.df_hourly['close'] > self.df_hourly['open']).astype(int)
        labels = labels.loc[selected.index]
        
        # Drop NaN
        selected = selected.dropna()
        labels = labels.loc[selected.index]
        
        return selected, labels

def test_enhanced_features():
    """Test the enhanced feature engineering."""
    symbol = 'BTC/USDT'
    print(f"Testing enhanced features for {symbol}")
    engineer = EnhancedFeatureEngineer(symbol, lookback_minutes=15)
    
    # Compute all enhanced features
    features, labels = engineer.compute_all_features_enhanced()
    print(f"Enhanced features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of feature columns: {len(features.columns)}")
    
    # Show some new columns
    new_cols = [col for col in features.columns if any(x in col for x in ['adx', 'bb_', 'macd', 'norm', 'interaction'])]
    print(f"New feature examples: {new_cols[:10]}")
    
    # Compute simplified features
    simple_features, simple_labels = engineer.compute_simplified_features()
    print(f"\nSimplified features shape: {simple_features.shape}")
    print(f"Simplified feature columns: {list(simple_features.columns)}")
    
    # Save samples
    features.to_csv('enhanced_features_sample.csv')
    simple_features.to_csv('simplified_features_sample.csv')
    
    return features, labels

if __name__ == '__main__':
    features, labels = test_enhanced_features()