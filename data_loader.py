import pandas as pd
import numpy as np
import os
from typing import List, Dict

class DataLoader:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = data_dir
    
    def load_asset(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Load single asset data from CSV."""
        symbol_fname = symbol.replace('/', '_')
        path = f"{self.data_dir}/{symbol_fname}_{timeframe}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        # ensure ascending index
        df = df.sort_index()
        return df
    
    def load_multiple_assets(self, symbols: List[str], timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """Load multiple assets into dict."""
        data = {}
        for sym in symbols:
            data[sym] = self.load_asset(sym, timeframe)
        return data
    
    def combine_assets(self, symbols: List[str], column: str = 'close', timeframe: str = '1h') -> pd.DataFrame:
        """
        Combine a specific column across assets into a single DataFrame.
        Returns DataFrame with columns like 'BTC_close', 'ETH_close', etc.
        """
        data = self.load_multiple_assets(symbols, timeframe)
        combined = pd.DataFrame()
        for sym, df in data.items():
            col_name = sym.replace('/', '_') + f'_{column}'
            combined[col_name] = df[column]
        # align timestamps (intersection)
        combined = combined.dropna()
        return combined
    
    def get_hourly_panel(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get panel data: dict with keys 'open', 'high', 'low', 'close', 'volume'
        each as DataFrame with assets as columns.
        """
        panel = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            panel[col] = self.combine_assets(symbols, column=col, timeframe='1h')
        return panel

if __name__ == '__main__':
    loader = DataLoader()
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    panel = loader.get_hourly_panel(symbols)
    for col, df in panel.items():
        print(f"{col}: shape {df.shape}")
        print(df.head())