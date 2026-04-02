import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_minute_chunk(symbol, start_dt, end_dt, timeframe='1m'):
    """Fetch minute data for a single chunk."""
    exchange = ccxt.binance({'enableRateLimit': True})
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    all_data = []
    current_since = start_ts
    
    while current_since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not ohlcv:
                break
            batch_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            batch_df = batch_df[batch_df['timestamp'] <= end_ts]
            if batch_df.empty:
                break
            all_data.append(batch_df)
            current_since = batch_df['timestamp'].iloc[-1] + 1
            if len(ohlcv) < 1000:
                break
            time.sleep(exchange.rateLimit / 1000)  # respect rate limit
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            time.sleep(5)
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.concat(all_data, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def extend_minute_data(symbol, days=180):
    """Extend minute data for a symbol."""
    symbol_fname = symbol.replace('/', '_')
    path = f'data/deep/{symbol_fname}_1m.csv'
    
    if not os.path.exists(path):
        logger.error(f"File {path} does not exist")
        return None
    
    existing = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    existing.sort_index(inplace=True)
    oldest = existing.index.min()
    newest = existing.index.max()
    target_oldest = datetime.now() - timedelta(days=days)
    
    if oldest <= target_oldest:
        logger.info(f"Already have {days} days of minute data (oldest: {oldest})")
        return existing
    
    logger.info(f"Fetching minute data from {target_oldest} to {oldest}")
    # Fetch in 7-day chunks to avoid rate limits and memory issues
    chunk_start = target_oldest
    chunk_end = oldest - timedelta(minutes=1)
    all_new = []
    
    chunk_days = 7  # 7 days per chunk
    while chunk_start < chunk_end:
        chunk_end_dt = min(chunk_start + timedelta(days=chunk_days), chunk_end)
        logger.info(f"Fetching chunk {chunk_start} to {chunk_end_dt}")
        chunk_df = fetch_minute_chunk(symbol, chunk_start, chunk_end_dt, timeframe='1m')
        if not chunk_df.empty:
            all_new.append(chunk_df)
        chunk_start = chunk_end_dt + timedelta(minutes=1)
        time.sleep(1)  # be nice
        
        # Safety break after too many chunks
        if len(all_new) >= 30:
            logger.warning("Reached safety limit of 30 chunks")
            break
    
    if not all_new:
        logger.warning("No new minute data fetched")
        return existing
    
    new_data = pd.concat(all_new)
    combined = pd.concat([new_data, existing])
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Save
    combined.to_csv(path)
    logger.info(f"Extended minute data: {len(existing)} -> {len(combined)} rows")
    return combined

def resample_to_timeframes(symbol):
    """Resample minute data to 5m, 15m, 1h."""
    symbol_fname = symbol.replace('/', '_')
    path = f'data/deep/{symbol_fname}_1m.csv'
    
    if not os.path.exists(path):
        logger.error(f"File {path} does not exist")
        return
    
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    df.sort_index(inplace=True)
    
    for resample_tf, rule in [('5m', '5min'), ('15m', '15min'), ('1h', '1h')]:
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Remove last row if incomplete (current candle)
        now = pd.Timestamp.now()
        last_time = resampled.index[-1]
        if last_time + pd.Timedelta(rule) > now:
            resampled = resampled.iloc[:-1]
        
        out_path = f'data/deep/{symbol_fname}_{resample_tf}.csv'
        resampled.to_csv(out_path)
        logger.info(f"Resampled to {resample_tf}: {len(resampled)} rows")

def process_all_assets(days=180):
    """Process all six assets."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    
    for sym in symbols:
        logger.info(f"Processing {sym}")
        # Extend minute data
        df = extend_minute_data(sym, days=days)
        if df is not None:
            logger.info(f"{sym}: {len(df)} minute rows from {df.index.min()} to {df.index.max()}")
        
        # Resample
        resample_to_timeframes(sym)
        
        # Sleep between assets
        time.sleep(5)
    
    logger.info("All assets processed.")

if __name__ == '__main__':
    # Extend to 180 days of minute data
    process_all_assets(days=180)