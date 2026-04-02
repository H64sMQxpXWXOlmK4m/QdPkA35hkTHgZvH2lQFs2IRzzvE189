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

def fetch_hourly_chunk(symbol, start_dt, end_dt):
    """Fetch hourly data for a single chunk."""
    exchange = ccxt.binance({'enableRateLimit': True})
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    all_data = []
    current_since = start_ts
    
    while current_since < end_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
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
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.error(f"Error: {e}")
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

def extend_btc_hourly():
    """Extend BTC hourly data to 2 years."""
    symbol = 'BTC/USDT'
    path = 'data/deep/BTC_USDT_1h.csv'
    
    if not os.path.exists(path):
        logger.error(f"File {path} does not exist")
        return
    
    existing = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    existing.sort_index(inplace=True)
    oldest = existing.index.min()
    newest = existing.index.max()
    target_oldest = datetime.now() - timedelta(days=365*2)
    
    if oldest <= target_oldest:
        logger.info(f"Already have 2 years of data (oldest: {oldest})")
        return existing
    
    logger.info(f"Fetching data from {target_oldest} to {oldest}")
    # Fetch in 3-month chunks to avoid rate limits
    chunk_start = target_oldest
    chunk_end = oldest - timedelta(hours=1)
    all_new = []
    
    while chunk_start < chunk_end:
        chunk_end_dt = min(chunk_start + timedelta(days=90), chunk_end)
        logger.info(f"Fetching chunk {chunk_start} to {chunk_end_dt}")
        chunk_df = fetch_hourly_chunk(symbol, chunk_start, chunk_end_dt)
        if not chunk_df.empty:
            all_new.append(chunk_df)
        chunk_start = chunk_end_dt + timedelta(hours=1)
        time.sleep(1)
    
    if not all_new:
        logger.warning("No new data fetched")
        return existing
    
    new_data = pd.concat(all_new)
    combined = pd.concat([new_data, existing])
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Save
    combined.to_csv(path)
    logger.info(f"Extended data: {len(existing)} -> {len(combined)} rows")
    return combined

if __name__ == '__main__':
    df = extend_btc_hourly()
    if df is not None:
        print(f"Data range: {df.index.min()} to {df.index.max()}")
