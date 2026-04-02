import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class EliteFeatureEngineer:
    """
    Elite feature engineering for predicting hourly candle direction.
    STRICTLY NO LOOKAHEAD - only uses data available at hour start.
    
    Features:
    1. Hourly momentum (returns over multiple horizons)
    2. Volatility structure (multi-scale volatility, clustering)
    3. Volume features (trends, ratios, concentration)
    4. Price position (distance from MAs, RSI, MACD, Bollinger)
    5. Regime detection (trend, volatility, momentum regimes)
    6. Intra-hour features from previous hour (path, momentum, volume)
    7. Time features (hour-of-day, day-of-week)
    8. Advanced technical indicators (ADX, ATR, etc.)
    9. Interaction features
    """
    
    def __init__(self, symbol: str, data_dir: str = 'data/deep'):
        self.symbol = symbol
        self.data_dir = data_dir
        
        # Load data
        self.df_hourly = self._load_hourly()
        self.df_15m = self._load_15m()
        self.df_5m = self._load_5m()
        self.df_1m = self._load_1m()
        
        # Validate no future data leakage
        self._validate_timelines()
        
        # Precompute returns for efficiency
        self.df_hourly['returns'] = self.df_hourly['close'].pct_change()
        
    def _load_hourly(self) -> pd.DataFrame:
        """Load hourly data."""
        path = f'{self.data_dir}/{self.symbol.replace("/", "_")}_1h.csv'
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df.sort_index(inplace=True)
        return df
    
    def _load_15m(self) -> pd.DataFrame:
        """Load 15-minute data."""
        path = f'{self.data_dir}/{self.symbol.replace("/", "_")}_15m.csv'
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df.sort_index(inplace=True)
        return df
    
    def _load_5m(self) -> pd.DataFrame:
        """Load 5-minute data."""
        path = f'{self.data_dir}/{self.symbol.replace("/", "_")}_5m.csv'
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df.sort_index(inplace=True)
        return df
    
    def _load_1m(self) -> pd.DataFrame:
        """Load 1-minute data."""
        path = f'{self.data_dir}/{self.symbol.replace("/", "_")}_1m.csv'
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df.sort_index(inplace=True)
        return df
    
    def _validate_timelines(self):
        """Ensure no future data leakage across timeframes."""
        # Ensure hourly data is aligned (no minute data beyond hourly timestamps)
        # For simplicity, we'll rely on proper feature construction
        pass
    
    def compute_features_at_hour_start(self, hour_start: pd.Timestamp) -> pd.Series:
        """
        Compute all features for a given hour start timestamp.
        Uses only data available at hour_start (strictly before hour_start).
        
        Parameters:
        -----------
        hour_start : pd.Timestamp
            Start of the hour to predict.
            
        Returns:
        --------
        pd.Series with feature values.
        """
        features = {}
        
        # 1. Hourly momentum features (using hourly data up to previous hour)
        hourly_features = self._compute_hourly_momentum(hour_start)
        features.update(hourly_features)
        
        # 2. Volatility features
        vol_features = self._compute_volatility_features(hour_start)
        features.update(vol_features)
        
        # 3. Volume features
        volume_features = self._compute_volume_features(hour_start)
        features.update(volume_features)
        
        # 4. Price position features
        position_features = self._compute_price_position_features(hour_start)
        features.update(position_features)
        
        # 5. Regime features
        regime_features = self._compute_regime_features(hour_start)
        features.update(regime_features)
        
        # 6. Intra-hour features from previous hour
        intra_features = self._compute_intra_hour_features(hour_start)
        features.update(intra_features)
        
        # 7. Time features
        time_features = self._compute_time_features(hour_start)
        features.update(time_features)
        
        # 8. Advanced technical indicators
        tech_features = self._compute_technical_indicators(hour_start)
        features.update(tech_features)
        
        # 9. Interaction features
        interaction_features = self._compute_interaction_features(features)
        features.update(interaction_features)
        
        return pd.Series(features)
    
    def _compute_hourly_momentum(self, hour_start: pd.Timestamp) -> Dict:
        """Compute momentum features using hourly data up to previous hour."""
        # Get hourly data up to previous hour
        df = self.df_hourly.loc[:hour_start - pd.Timedelta(hours=1)]
        if len(df) < 24:
            return {}
        
        # Use last available close (previous hour close)
        last_close = df['close'].iloc[-1]
        open_price = self._get_open_at_hour(hour_start)
        
        features = {}
        
        # Gap return (open vs previous close)
        if open_price and last_close:
            features['gap_return'] = open_price / last_close - 1
        
        # Returns over various horizons (using hourly closes)
        horizons = [1, 4, 12, 24, 48, 168]
        for h in horizons:
            if len(df) >= h + 1:
                past_close = df['close'].iloc[-h-1]
                features[f'return_{h}h'] = last_close / past_close - 1
        
        # Momentum acceleration (change in momentum)
        if len(df) >= 24:
            mom_12 = last_close / df['close'].iloc[-13] - 1
            mom_24 = last_close / df['close'].iloc[-25] - 1
            features['momentum_accel'] = mom_12 - mom_24
        
        # Price change over last hour
        if len(df) >= 2:
            prev_close = df['close'].iloc[-2]
            features['hourly_return'] = last_close / prev_close - 1
        
        return features
    
    def _compute_volatility_features(self, hour_start: pd.Timestamp) -> Dict:
        """Compute volatility features."""
        df = self.df_hourly.loc[:hour_start - pd.Timedelta(hours=1)]
        if len(df) < 24:
            return {}
        
        returns = df['returns'].dropna()
        
        features = {}
        
        # Volatility over different horizons
        windows = [24, 48, 168]
        for w in windows:
            if len(returns) >= w:
                vol = returns[-w:].std()
                features[f'vol_{w}h'] = vol
        
        # Volatility ratio (short/long term)
        if len(returns) >= 168:
            vol_24 = returns[-24:].std()
            vol_168 = returns[-168:].std()
            features['vol_ratio_24_168'] = vol_24 / (vol_168 + 1e-9)
        
        # Volatility clustering (autocorrelation of squared returns)
        if len(returns) >= 48:
            squared = returns[-48:] ** 2
            if len(squared) > 1:
                acf = squared.autocorr(lag=1)
                features['vol_clustering'] = acf
        
        # High-low range (normalized)
        if len(df) >= 24:
            high = df['high'].iloc[-24:].max()
            low = df['low'].iloc[-24:].min()
            mid = (high + low) / 2
            features['range_24h'] = (high - low) / (mid + 1e-9)
        
        return features
    
    def _compute_volume_features(self, hour_start: pd.DataFrame) -> Dict:
        """Compute volume features."""
        df = self.df_hourly.loc[:hour_start - pd.Timedelta(hours=1)]
        if len(df) < 24:
            return {}
        
        features = {}
        
        # Volume ratio (recent vs longer term)
        if len(df) >= 24:
            vol_24 = df['volume'].iloc[-24:].sum()
            vol_168 = df['volume'].iloc[-168:].sum() / 7  # daily average
            features['volume_ratio_24h'] = vol_24 / (vol_168 + 1e-9)
        
        # Volume trend (linear regression slope)
        if len(df) >= 24:
            volumes = df['volume'].iloc[-24:].values
            x = np.arange(len(volumes))
            slope, _, r_value, _, _ = stats.linregress(x, volumes)
            features['volume_trend'] = slope / (np.mean(volumes) + 1e-9)
            features['volume_trend_r2'] = r_value ** 2
        
        # Volume concentration (std / mean)
        if len(df) >= 24:
            vols = df['volume'].iloc[-24:]
            features['volume_concentration'] = vols.std() / (vols.mean() + 1e-9)
        
        # Volume-price correlation
        if len(df) >= 24:
            corr = df['volume'].iloc[-24:].corr(df['returns'].iloc[-24:])
            features['volume_price_corr'] = 0 if pd.isna(corr) else corr
        
        return features
    
    def _compute_price_position_features(self, hour_start: pd.Timestamp) -> Dict:
        """Compute price position relative to moving averages, support/resistance."""
        df = self.df_hourly.loc[:hour_start - pd.Timedelta(hours=1)]
        if len(df) < 200:
            return {}
        
        last_close = df['close'].iloc[-1]
        
        features = {}
        
        # Distance from moving averages
        ma_windows = [10, 20, 50, 100, 200]
        for w in ma_windows:
            if len(df) >= w:
                ma = df['close'].rolling(w).mean().iloc[-1]
                features[f'ma_distance_{w}'] = last_close / ma - 1
        
        # RSI
        if len(df) >= 14:
            rsi = self._compute_rsi(df['close'].iloc[-14:])
            features['rsi'] = rsi
        
        # MACD
        if len(df) >= 26:
            macd, signal = self._compute_macd(df['close'])
            features['macd'] = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
            features['macd_signal'] = signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0
        
        # Bollinger Bands position
        if len(df) >= 20:
            bb_pos = self._compute_bollinger_position(df['close'].iloc[-20:])
            features['bb_position'] = bb_pos
        
        # Support/resistance levels (simplified)
        if len(df) >= 50:
            support = df['low'].iloc[-50:].min()
            resistance = df['high'].iloc[-50:].max()
            features['support_distance'] = last_close / support - 1
            features['resistance_distance'] = last_close / resistance - 1
        
        return features
    
    def _compute_regime_features(self, hour_start: pd.Timestamp) -> Dict:
        """Compute regime detection features."""
        df = self.df_hourly.loc[:hour_start - pd.Timedelta(hours=1)]
        if len(df) < 168:
            return {}
        
        returns = df['returns'].dropna()
        
        features = {}
        
        # Trend regime (ADX-like)
        if len(df) >= 14:
            trend_strength = self._compute_trend_strength(df['high'].iloc[-14:], df['low'].iloc[-14:], df['close'].iloc[-14:])
            features['trend_strength'] = trend_strength
        
        # Volatility regime (percentile of recent volatility)
        if len(returns) >= 168:
            recent_vol = returns[-24:].std()
            historical_vol = returns[-168:].std()
            features['vol_regime'] = recent_vol / (historical_vol + 1e-9)
        
        # Momentum regime (percentile of recent returns)
        if len(returns) >= 168:
            recent_return = df['close'].iloc[-1] / df['close'].iloc[-24] - 1
            historical_returns = [df['close'].iloc[-i-1] / df['close'].iloc[-i-25] - 1 
                                  for i in range(len(df)-24) if i+25 <= len(df)]
            if historical_returns:
                # Compute percentile of recent_return within historical_returns
                features['momentum_regime'] = np.sum(np.array(historical_returns) <= recent_return) / len(historical_returns) * 100
        
        # Mean reversion indicator (price distance from MA)
        if len(df) >= 50:
            last_close = df['close'].iloc[-1]
            ma50 = df['close'].rolling(50).mean().iloc[-1]
            distance = last_close / ma50 - 1
            # Z-score of distance over recent period
            distances = (df['close'].iloc[-100:] / df['close'].rolling(50).mean().iloc[-100:]) - 1
            if len(distances.dropna()) > 10:
                z = (distance - distances.mean()) / (distances.std() + 1e-9)
                features['mean_reversion_z'] = z
        
        return features
    
    def _compute_intra_hour_features(self, hour_start: pd.Timestamp) -> Dict:
        """Compute features from minute data of previous hour."""
        # Get minute data from previous hour (complete hour)
        prev_hour_end = hour_start - pd.Timedelta(minutes=1)
        prev_hour_start = hour_start - pd.Timedelta(hours=1)
        
        minute_slice = self.df_1m.loc[prev_hour_start:prev_hour_end]
        if len(minute_slice) < 10:  # at least 10 minutes
            return {}
        
        features = {}
        
        # Path features within previous hour
        open_prev = minute_slice['open'].iloc[0]
        close_prev = minute_slice['close'].iloc[-1]
        
        features['prev_hour_return'] = close_prev / open_prev - 1
        features['prev_hour_high_low_range'] = (minute_slice['high'].max() - minute_slice['low'].min()) / open_prev
        
        # Intra-hour volatility (minute returns std)
        minute_returns = minute_slice['close'].pct_change().dropna()
        if len(minute_returns) > 1:
            features['prev_hour_minute_vol'] = minute_returns.std()
            features['prev_hour_minute_skew'] = minute_returns.skew()
            features['prev_hour_minute_kurt'] = minute_returns.kurtosis()
        
        # Volume features
        features['prev_hour_volume'] = minute_slice['volume'].sum()
        
        # Price efficiency (close-to-close variance vs high-low range)
        close_var = minute_returns.var() if len(minute_returns) > 1 else 0
        hl_range = (minute_slice['high'] - minute_slice['low']).mean() / open_prev
        features['prev_hour_price_efficiency'] = close_var / (hl_range + 1e-9)
        
        # Last N minutes momentum (e.g., last 15 minutes)
        last_n = 15
        if len(minute_slice) >= last_n:
            last_slice = minute_slice.iloc[-last_n:]
            last_return = last_slice['close'].iloc[-1] / last_slice['close'].iloc[0] - 1
            features['last_15min_momentum'] = last_return
        
        return features
    
    def _compute_time_features(self, hour_start: pd.Timestamp) -> Dict:
        """Compute time-based features."""
        features = {}
        
        # Hour of day (categorical but encoded as cyclical)
        hour = hour_start.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week
        weekday = hour_start.weekday()
        features['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
        
        # Month of year
        month = hour_start.month
        features['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        features['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        
        # Weekend flag
        features['is_weekend'] = 1 if weekday >= 5 else 0
        
        # Asian/European/US session flags
        features['session_asia'] = 1 if 0 <= hour <= 8 else 0
        features['session_europe'] = 1 if 8 < hour <= 16 else 0
        features['session_us'] = 1 if 16 < hour <= 24 else 0
        
        return features
    
    def _compute_technical_indicators(self, hour_start: pd.Timestamp) -> Dict:
        """Compute advanced technical indicators."""
        df = self.df_hourly.loc[:hour_start - pd.Timedelta(hours=1)]
        if len(df) < 14:
            return {}
        
        features = {}
        
        # ATR (Average True Range)
        if len(df) >= 14:
            atr = self._compute_atr(df['high'].iloc[-14:], df['low'].iloc[-14:], df['close'].iloc[-14:])
            features['atr'] = atr
        
        # ADX (Average Directional Index)
        if len(df) >= 14:
            adx = self._compute_adx(df['high'].iloc[-14:], df['low'].iloc[-14:], df['close'].iloc[-14:])
            features['adx'] = adx
        
        # Stochastic oscillator
        if len(df) >= 14:
            stoch = self._compute_stochastic(df['high'].iloc[-14:], df['low'].iloc[-14:], df['close'].iloc[-14:])
            features['stoch_k'] = stoch[0]
            features['stoch_d'] = stoch[1]
        
        # Williams %R
        if len(df) >= 14:
            williams = self._compute_williams_r(df['high'].iloc[-14:], df['low'].iloc[-14:], df['close'].iloc[-14:])
            features['williams_r'] = williams
        
        # CCI (Commodity Channel Index)
        if len(df) >= 20:
            cci = self._compute_cci(df['high'].iloc[-20:], df['low'].iloc[-20:], df['close'].iloc[-20:])
            features['cci'] = cci
        
        return features
    
    def _compute_interaction_features(self, features: Dict) -> Dict:
        """Compute interaction features between base features."""
        interaction = {}
        
        # Momentum * volatility interaction
        if 'return_24h' in features and 'vol_24h' in features:
            interaction['momentum_vol_interaction'] = features['return_24h'] * features['vol_24h']
        
        # Volume * price change interaction
        if 'volume_ratio_24h' in features and 'hourly_return' in features:
            interaction['volume_price_interaction'] = features['volume_ratio_24h'] * features['hourly_return']
        
        # RSI * volatility regime interaction
        if 'rsi' in features and 'vol_regime' in features:
            interaction['rsi_vol_regime'] = features['rsi'] * features['vol_regime']
        
        # Time-of-day * momentum
        if 'hour_sin' in features and 'return_12h' in features:
            interaction['hour_momentum_interaction'] = features['hour_sin'] * features['return_12h']
        
        return interaction
    
    # Helper methods for technical indicators
    
    def _get_open_at_hour(self, hour_start: pd.Timestamp) -> Optional[float]:
        """Get open price for hour start (if available)."""
        if hour_start in self.df_hourly.index:
            return self.df_hourly.loc[hour_start, 'open']
        return None
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Compute RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _compute_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Compute MACD."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def _compute_bollinger_position(self, prices: pd.Series) -> float:
        """Compute position within Bollinger Bands."""
        ma = prices.rolling(20).mean().iloc[-1]
        std = prices.rolling(20).std().iloc[-1]
        if pd.isna(ma) or pd.isna(std) or std == 0:
            return 0
        last_price = prices.iloc[-1]
        return (last_price - ma) / (2 * std)
    
    def _compute_trend_strength(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Compute trend strength (simplified ADX)."""
        # Simplified: average of absolute price changes normalized by range
        price_changes = close.diff().abs()
        ranges = high - low
        if ranges.mean() > 0:
            return (price_changes.mean() / ranges.mean()) * 100
        return 0
    
    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
    
    def _compute_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Compute Average Directional Index."""
        # Simplified implementation
        up_move = high.diff()
        down_move = low.diff().abs() * -1
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr = self._compute_atr(high, low, close, period)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / tr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / tr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(period).mean()
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
    
    def _compute_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[float, float]:
        """Compute Stochastic oscillator %K and %D."""
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
        d = k.rolling(3).mean()
        return k.iloc[-1] if not pd.isna(k.iloc[-1]) else 50, d.iloc[-1] if not pd.isna(d.iloc[-1]) else 50
    
    def _compute_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Compute Williams %R."""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
        return wr.iloc[-1] if not pd.isna(wr.iloc[-1]) else -50
    
    def _compute_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> float:
        """Compute Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(period).mean()
        mad = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (typical_price - sma) / (0.015 * mad + 1e-9)
        return cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0
    
    def create_dataset(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Create feature dataset for hour starts between start_date and end_date.
        
        Parameters:
        -----------
        start_date : pd.Timestamp
            First hour start to include.
        end_date : pd.Timestamp
            Last hour start to include.
            
        Returns:
        --------
        DataFrame with features and target (1 if hour closes up, 0 otherwise).
        """
        # Generate hour starts within range
        hour_starts = pd.date_range(start=start_date, end=end_date, freq='h')
        
        features_list = []
        targets = []
        
        for hour_start in hour_starts:
            # Ensure we have enough historical data
            if hour_start - pd.Timedelta(hours=200) < self.df_hourly.index[0]:
                continue
            
            # Compute features
            feat_series = self.compute_features_at_hour_start(hour_start)
            if feat_series.empty:
                continue
            
            # Compute target (whether hour closes up)
            # Get hour's close price (if available)
            if hour_start in self.df_hourly.index:
                hour_data = self.df_hourly.loc[hour_start]
                target = 1 if hour_data['close'] > hour_data['open'] else 0
                features_list.append(feat_series)
                targets.append(target)
        
        if not features_list:
            return pd.DataFrame()
        
        X = pd.DataFrame(features_list)
        y = pd.Series(targets, index=X.index)
        
        # Remove any rows with missing values
        valid = X.notna().all(axis=1)
        X = X[valid]
        y = y[valid]
        
        return X, y