import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
import logging
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepDataCollector:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        self.rate_limit = self.exchange.rateLimit / 1000  # seconds
        self.max_retries = 3
    
    def fetch_ohlcv_paginated(self, symbol: str, timeframe: str, 
                               start_dt: datetime, end_dt: Optional[datetime] = None,
                               limit_per_request: int = 1000) -> pd.DataFrame:
        """
        Fetch OHLCV data from start_dt to end_dt (or until now) with pagination.
        """
        if end_dt is None:
            end_dt = datetime.now()
        
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        all_ohlcv = []
        current_since = start_ts
        retry_count = 0
        
        while current_since < end_ts:
            try:
                # Fetch batch
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, 
                                                   since=current_since, 
                                                   limit=limit_per_request)
                if not ohlcv:
                    logger.info(f"No more data for {symbol}")
                    break
                
                # Convert to DataFrame for filtering
                batch_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                # Filter out beyond end_ts
                batch_df = batch_df[batch_df['timestamp'] <= end_ts]
                if batch_df.empty:
                    break
                
                all_ohlcv.append(batch_df)
                
                # Update since to last timestamp + 1ms
                current_since = batch_df['timestamp'].iloc[-1] + 1
                
                # If we got less than limit, we've reached the end
                if len(ohlcv) < limit_per_request:
                    break
                
                # Respect rate limit
                time.sleep(self.rate_limit)
                retry_count = 0  # reset retry count on success
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Max retries reached for {symbol}")
                    break
                time.sleep(5 * retry_count)  # exponential backoff
        
        if not all_ohlcv:
            return pd.DataFrame()
        
        df = pd.concat(all_ohlcv, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        # Drop duplicates (should not happen)
        df = df[~df.index.duplicated(keep='first')]
        return df
    
    def fetch_hourly_data(self, symbol: str, years: int = 1) -> pd.DataFrame:
        """Fetch hourly data for given number of years."""
        start_dt = datetime.now() - timedelta(days=365 * years)
        logger.info(f"Fetching {symbol} hourly data from {start_dt}")
        df = self.fetch_ohlcv_paginated(symbol, '1h', start_dt)
        logger.info(f"Fetched {len(df)} hourly rows for {symbol}")
        return df
    
    def fetch_minute_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Fetch minute data for given number of days."""
        start_dt = datetime.now() - timedelta(days=days)
        logger.info(f"Fetching {symbol} minute data from {start_dt}")
        df = self.fetch_ohlcv_paginated(symbol, '1m', start_dt)
        logger.info(f"Fetched {len(df)} minute rows for {symbol}")
        return df
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, base_dir: str):
        """Save DataFrame to CSV, creating directory if needed."""
        symbol_fname = symbol.replace('/', '_')
        os.makedirs(base_dir, exist_ok=True)
        path = f"{base_dir}/{symbol_fname}_{timeframe}.csv"
        df.to_csv(path)
        logger.info(f"Saved {len(df)} rows to {path}")
    
    def load_or_fetch(self, symbol: str, timeframe: str, days: int = None, years: int = None,
                      base_dir: str = 'data/deep') -> pd.DataFrame:
        """
        Load data from disk if exists, otherwise fetch and save.
        """
        symbol_fname = symbol.replace('/', '_')
        path = f"{base_dir}/{symbol_fname}_{timeframe}.csv"
        if os.path.exists(path):
            logger.info(f"Loading existing data from {path}")
            df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
            df = df.sort_index()
            # Check if we need more recent data
            latest = df.index.max()
            cutoff = datetime.now() - timedelta(hours=24)  # allow 1 day stale
            if latest >= cutoff:
                logger.info(f"Data is recent enough (latest: {latest})")
                return df
            else:
                logger.info(f"Data is stale (latest: {latest}), fetching updates...")
                # Fetch data from latest to now
                start_dt = latest + timedelta(milliseconds=1)
                new_df = self.fetch_ohlcv_paginated(symbol, timeframe, start_dt)
                if not new_df.empty:
                    df = pd.concat([df, new_df]).sort_index()
                    df = df[~df.index.duplicated(keep='first')]
                    self.save_data(df, symbol, timeframe, base_dir)
                return df
        else:
            # Fetch fresh
            if timeframe == '1h' and years is not None:
                df = self.fetch_hourly_data(symbol, years=years)
            elif timeframe == '1m' and days is not None:
                df = self.fetch_minute_data(symbol, days=days)
            else:
                raise ValueError("Invalid timeframe or missing days/years")
            self.save_data(df, symbol, timeframe, base_dir)
            return df

def collect_all_assets(symbols: List[str], hourly_years: int = 1, minute_days: int = 90):
    """Collect data for all symbols."""
    collector = DeepDataCollector()
    os.makedirs('data/deep', exist_ok=True)
    
    for sym in symbols:
        logger.info(f"Processing {sym}")
        # Hourly
        hourly = collector.load_or_fetch(sym, '1h', years=hourly_years)
        # Minute
        minute = collector.load_or_fetch(sym, '1m', days=minute_days)
        # Resample minute to 5min, 15min, 1h for intra-hour features
        for resample_tf, rule in [('5min', '5min'), ('15min', '15min'), ('1h', '1h')]:
            resampled = minute.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            collector.save_data(resampled, sym, resample_tf, base_dir='data/deep/processed')
        
        # Sleep between assets to avoid rate limits
        time.sleep(2)
    
    logger.info("All assets collected.")

if __name__ == '__main__':
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    # Start with hourly only for now (minute will take time)
    collect_all_assets(symbols, hourly_years=1, minute_days=30)  # 30 days minute already exists