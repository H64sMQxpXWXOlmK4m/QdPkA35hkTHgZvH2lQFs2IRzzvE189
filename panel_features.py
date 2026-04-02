import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from data_loader import DataLoader
from intra_hour_features import IntraHourFeatureEngineer
import warnings
warnings.filterwarnings('ignore')

class PanelFeatureEngineer:
    def __init__(self, symbols: List[str], data_dir='data/raw'):
        self.symbols = symbols
        self.loader = DataLoader(data_dir)
        self.panel_hourly = None  # dict of DataFrames per column
        self.minute_data = {}     # dict of minute DataFrames per symbol
        
    def load_all_data(self):
        """Load hourly panel and minute data."""
        # Hourly panel
        self.panel_hourly = self.loader.get_hourly_panel(self.symbols)
        # Minute data
        for sym in self.symbols:
            self.minute_data[sym] = self.loader.load_asset(sym, timeframe='1m')
    
    def compute_intra_hour_features(self, lookback_minutes: int = 15) -> Dict[str, pd.DataFrame]:
        """Compute intra-hour features for each symbol."""
        intra_features = {}
        for sym in self.symbols:
            engineer = IntraHourFeatureEngineer(self.minute_data[sym])
            # Get hourly labels for alignment (we'll compute later)
            hourly_labels = (self.panel_hourly['close'][f'{sym.replace("/", "_")}_close'] > 
                             self.panel_hourly['open'][f'{sym.replace("/", "_")}_open']).astype(int)
            feats, _ = engineer.align_with_labels(hourly_labels)
            intra_features[sym] = feats
        return intra_features
    
    def compute_hourly_features(self) -> Dict[str, pd.DataFrame]:
        """
        Compute hourly features for each symbol using hourly panel.
        Features include lagged returns, volatility, etc.
        """
        hourly_feats = {}
        for sym in self.symbols:
            # Extract columns for this symbol
            prefix = sym.replace('/', '_')
            df = pd.DataFrame()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = self.panel_hourly[col][f'{prefix}_{col}']
            
            # Compute features (similar to FeatureEngineer but panel-safe)
            feats = pd.DataFrame(index=df.index)
            # Gap return
            feats['gap_return'] = df['open'] / df['close'].shift(1) - 1
            # Past returns
            for window in [1, 2, 4, 8, 12, 24, 48]:
                feats[f'ret_{window}h'] = df['close'].pct_change(window)
            # Rolling volatility (24h)
            returns = df['close'].pct_change()
            feats['volatility_24h'] = returns.rolling(24).std()
            # Volume ratio
            feats['volume_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
            # Price position
            rolling_low = df['low'].rolling(24).min()
            rolling_high = df['high'].rolling(24).max()
            feats['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-9)
            # SMA cross
            sma12 = df['close'].rolling(12).mean()
            sma24 = df['close'].rolling(24).mean()
            feats['ma_cross'] = sma12 / sma24 - 1
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            feats['rsi'] = 100 - (100 / (1 + rs))
            # Shift to avoid lookahead (except gap_return)
            shift_cols = [col for col in feats.columns if col != 'gap_return']
            feats[shift_cols] = feats[shift_cols].shift(1)
            # Drop rows with NaN
            feats = feats.dropna()
            hourly_feats[sym] = feats
        return hourly_feats
    
    def compute_cross_asset_features(self, intra_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute cross-asset features (e.g., BTC leader).
        Returns DataFrame indexed by hour start with columns like 'BTC_first_5m_return'.
        """
        # Use intra-hour features of BTC as cross-asset features for all
        btc_sym = 'BTC/USDT'
        if btc_sym not in intra_features:
            raise ValueError("BTC not in symbols")
        btc_feats = intra_features[btc_sym].copy()
        # Rename columns with prefix 'btc_'
        btc_feats = btc_feats.add_prefix('btc_')
        # Also compute market-wide average of first_5m_return across assets
        # We'll need to align indices across assets
        # For simplicity, just return btc features for now
        return btc_feats
    
    def build_panel(self, lookback_minutes: int = 15) -> pd.DataFrame:
        """
        Build final panel dataset with rows for each (hour, asset).
        Columns: asset_id, features..., label.
        """
        self.load_all_data()
        intra_feats = self.compute_intra_hour_features(lookback_minutes)
        hourly_feats = self.compute_hourly_features()
        cross_feats = self.compute_cross_asset_features(intra_feats)
        
        # Determine common hour index across all assets and features
        # Intersection of indices where all assets have intra features
        common_idx = None
        for sym in self.symbols:
            if sym in intra_feats and not intra_feats[sym].empty:
                idx = intra_feats[sym].index
                if common_idx is None:
                    common_idx = idx
                else:
                    common_idx = common_idx.intersection(idx)
        if common_idx is None or len(common_idx) == 0:
            raise ValueError("No common hours across assets")
        
        # Build list of records
        records = []
        for hs in common_idx:
            for sym in self.symbols:
                # Get features for this symbol and hour
                intra = intra_feats[sym].loc[hs] if hs in intra_feats[sym].index else pd.Series()
                hourly = hourly_feats[sym].loc[hs] if hs in hourly_feats[sym].index else pd.Series()
                cross = cross_feats.loc[hs] if hs in cross_feats.index else pd.Series()
                # Combine
                combined = pd.concat([intra, hourly, cross])
                if combined.isna().any():
                    # skip if any missing feature
                    continue
                # Label
                prefix = sym.replace('/', '_')
                label = (self.panel_hourly['close'].loc[hs, f'{prefix}_close'] > 
                         self.panel_hourly['open'].loc[hs, f'{prefix}_open']).astype(int)
                combined['asset'] = sym
                combined['label'] = label
                records.append(combined)
        
        panel = pd.DataFrame(records)
        return panel

if __name__ == '__main__':
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    engineer = PanelFeatureEngineer(symbols)
    print("Building panel...")
    panel = engineer.build_panel(lookback_minutes=15)
    print(f"Panel shape: {panel.shape}")
    print(panel.head())
    # Save panel
    panel.to_csv('data/panel_dataset.csv', index=False)
    print("Panel saved to data/panel_dataset.csv")