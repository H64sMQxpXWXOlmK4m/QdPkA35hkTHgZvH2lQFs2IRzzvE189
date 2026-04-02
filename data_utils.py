import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional

def get_data_range(symbol: str, data_dir: str = 'data/deep') -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get overlapping date range where both hourly and minute data exist.
    
    Returns:
    --------
    (start_date, end_date) as pd.Timestamp
    """
    symbol_fname = symbol.replace('/', '_')
    
    # Load hourly data
    hourly_path = f'{data_dir}/{symbol_fname}_1h.csv'
    if not os.path.exists(hourly_path):
        raise FileNotFoundError(f"Hourly data not found: {hourly_path}")
    
    hourly = pd.read_csv(hourly_path, nrows=1)
    hourly_start = pd.to_datetime(hourly.iloc[0, 0]) if not hourly.empty else None
    hourly = pd.read_csv(hourly_path, usecols=[0])
    hourly_end = pd.to_datetime(hourly.iloc[-1, 0])
    
    # Load minute data
    minute_path = f'{data_dir}/{symbol_fname}_1m.csv'
    if not os.path.exists(minute_path):
        raise FileNotFoundError(f"Minute data not found: {minute_path}")
    
    minute = pd.read_csv(minute_path, nrows=1)
    minute_start = pd.to_datetime(minute.iloc[0, 0]) if not minute.empty else None
    minute = pd.read_csv(minute_path, usecols=[0])
    minute_end = pd.to_datetime(minute.iloc[-1, 0])
    
    # Determine overlapping range
    start = max(hourly_start, minute_start) if hourly_start and minute_start else (hourly_start or minute_start)
    end = min(hourly_end, minute_end)
    
    if start >= end:
        raise ValueError(f"No overlapping data for {symbol}")
    
    return start, end

def get_data_info(symbol: str) -> dict:
    """Get information about available data."""
    symbol_fname = symbol.replace('/', '_')
    info = {'symbol': symbol}
    
    for tf in ['1h', '15m', '5m', '1m']:
        path = f'data/deep/{symbol_fname}_{tf}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=[0])
            info[f'{tf}_rows'] = len(df)
            info[f'{tf}_start'] = pd.to_datetime(df.iloc[0, 0])
            info[f'{tf}_end'] = pd.to_datetime(df.iloc[-1, 0])
            info[f'{tf}_days'] = (info[f'{tf}_end'] - info[f'{tf}_start']).days
        else:
            info[f'{tf}_rows'] = 0
    
    return info

def print_data_summary(symbols=None):
    """Print summary of data availability."""
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    
    print("DATA SUMMARY")
    print("=" * 80)
    for sym in symbols:
        try:
            info = get_data_info(sym)
            print(f"\n{sym}")
            print(f"  Hourly: {info.get('1h_rows', 0):6} rows, {info.get('1h_days', 0):3} days")
            print(f"  1m:     {info.get('1m_rows', 0):6} rows, {info.get('1m_days', 0):3} days")
            print(f"  5m:     {info.get('5m_rows', 0):6} rows")
            print(f"  15m:    {info.get('15m_rows', 0):6} rows")
        except Exception as e:
            print(f"{sym}: ERROR - {e}")