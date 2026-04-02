import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RegimeDetector:
    def __init__(self, window_bars: int = 168):  # 7 days hourly
        self.window = window_bars
    
    def detect_all(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        Detect multiple regimes for hourly data.
        Returns DataFrame with regime columns.
        """
        df = df_hourly.copy()
        
        # 1. Volatility regime (based on ATR)
        df['atr'] = self._compute_atr(df)
        df['volatility_regime'] = self._volatility_regime(df['atr'])
        
        # 2. Trend regime (based on price slope)
        df['trend_regime'] = self._trend_regime(df['close'])
        
        # 3. Momentum regime (recent returns)
        df['momentum_regime'] = self._momentum_regime(df['close'])
        
        # 4. Volume regime (unusual volume)
        df['volume_regime'] = self._volume_regime(df['volume'])
        
        # 5. Market state (combined)
        df['market_state'] = self._combined_state(df)
        
        return df[['atr', 'volatility_regime', 'trend_regime', 'momentum_regime', 'volume_regime', 'market_state']]
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _volatility_regime(self, atr_series: pd.Series, q_high: float = 0.7, q_low: float = 0.3) -> pd.Series:
        """
        Label volatility regime: 1 = high, -1 = low, 0 = medium.
        Based on rolling percentile within window.
        """
        # rolling percentiles
        rolling_q_high = atr_series.rolling(self.window, min_periods=1).quantile(q_high)
        rolling_q_low = atr_series.rolling(self.window, min_periods=1).quantile(q_low)
        regime = pd.Series(0, index=atr_series.index)
        regime[atr_series > rolling_q_high] = 1
        regime[atr_series < rolling_q_low] = -1
        return regime
    
    def _trend_regime(self, price_series: pd.Series, q_high: float = 0.7, q_low: float = 0.3) -> pd.Series:
        """
        Trend regime: 1 = uptrend, -1 = downtrend, 0 = sideways.
        Based on rolling regression slope percentiles.
        """
        # Compute rolling slope
        def slope(arr):
            if len(arr) < 2:
                return 0
            x = np.arange(len(arr))
            return stats.linregress(x, arr).slope
        
        slopes = price_series.rolling(self.window, min_periods=2).apply(slope, raw=True)
        # Normalize slope by price level
        norm_slopes = slopes / price_series.rolling(self.window, min_periods=1).mean()
        # Rolling percentiles of normalized slope
        rolling_q_high = norm_slopes.rolling(self.window, min_periods=1).quantile(q_high)
        rolling_q_low = norm_slopes.rolling(self.window, min_periods=1).quantile(q_low)
        regime = pd.Series(0, index=price_series.index)
        regime[norm_slopes > rolling_q_high] = 1
        regime[norm_slopes < rolling_q_low] = -1
        return regime
    
    def _momentum_regime(self, price_series: pd.Series, lookback: int = 24,
                         q_high: float = 0.7, q_low: float = 0.3) -> pd.Series:
        """
        Momentum regime: 1 = positive momentum, -1 = negative, 0 = neutral.
        Based on rolling percentile of lookback returns.
        """
        returns = price_series.pct_change(lookback)
        rolling_q_high = returns.rolling(self.window, min_periods=1).quantile(q_high)
        rolling_q_low = returns.rolling(self.window, min_periods=1).quantile(q_low)
        regime = pd.Series(0, index=price_series.index)
        regime[returns > rolling_q_high] = 1
        regime[returns < rolling_q_low] = -1
        return regime
    
    def _volume_regime(self, volume_series: pd.Series, q_high: float = 0.9) -> pd.Series:
        """Volume regime: 1 = high volume, 0 = normal."""
        rolling_q_high = volume_series.rolling(self.window, min_periods=1).quantile(q_high)
        regime = (volume_series > rolling_q_high).astype(int)
        return regime
    
    def _combined_state(self, df: pd.DataFrame) -> pd.Series:
        """
        Combined market state:
        0 = low/medium vol, sideways
        1 = low/medium vol, trending
        2 = high vol, sideways
        3 = high vol, trending
        """
        vol_binary = (df['volatility_regime'] == 1).astype(int)  # high vol only
        trend_binary = (df['trend_regime'] != 0).astype(int)     # trending (up or down)
        state = vol_binary * 2 + trend_binary
        return state
    
    def compute_regime_features(self, df_hourly: pd.DataFrame) -> pd.DataFrame:
        """
        Compute regime features for modeling.
        Returns DataFrame with columns suitable as features.
        """
        regimes = self.detect_all(df_hourly)
        features = pd.DataFrame(index=df_hourly.index)
        
        # Volatility regime (three categories)
        features['vol_high'] = (regimes['volatility_regime'] == 1).astype(float)
        features['vol_low'] = (regimes['volatility_regime'] == -1).astype(float)
        features['vol_medium'] = (regimes['volatility_regime'] == 0).astype(float)
        
        # Trend regime (three categories)
        features['trend_up'] = (regimes['trend_regime'] == 1).astype(float)
        features['trend_down'] = (regimes['trend_regime'] == -1).astype(float)
        features['trend_sideways'] = (regimes['trend_regime'] == 0).astype(float)
        
        # Momentum regime (three categories)
        features['mom_up'] = (regimes['momentum_regime'] == 1).astype(float)
        features['mom_down'] = (regimes['momentum_regime'] == -1).astype(float)
        features['mom_neutral'] = (regimes['momentum_regime'] == 0).astype(float)
        
        # Volume regime (binary)
        features['vol_high_volume'] = (regimes['volume_regime'] == 1).astype(float)
        
        # Combined state (4 categories)
        for i in range(4):
            features[f'state_{i}'] = (regimes['market_state'] == i).astype(float)
        
        # Lag features (regime persistence)
        for col in features.columns:
            features[f'{col}_lag1'] = features[col].shift(1)
            features[f'{col}_lag2'] = features[col].shift(2)
        
        return features.dropna()

if __name__ == '__main__':
    # Test with BTC hourly data
    from data_loader import DataLoader
    loader = DataLoader()
    btc_hourly = loader.load_asset('BTC/USDT', timeframe='1h')
    print("BTC hourly shape:", btc_hourly.shape)
    
    detector = RegimeDetector(window_bars=168)
    regime_features = detector.compute_regime_features(btc_hourly)
    print("Regime features shape:", regime_features.shape)
    print(regime_features.head())
    print(regime_features.tail())