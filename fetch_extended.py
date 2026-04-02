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

class ExtendedDataCollector:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        self.rate_limit = self.exchange.rateLimit / 1000  # seconds
    
    def fetch_ohlcv_range(self, symbol, timeframe, start_dt, end_dt):
        """
        Fetch OHLCV data between start_dt and end_dt.
        Returns DataFrame.
        """
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)
        
        all_data = []
        current_since = start_ts
        
        while current_since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=1000
                )
                if not ohlcv:
                    break
                
                batch_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                # Filter out beyond end_ts
                batch_df = batch_df[batch_df['timestamp'] <= end_ts]
                if batch_df.empty:
                    break
                
                all_data.append(batch_df)
                current_since = batch_df['timestamp'].iloc[-1] + 1
                
                if len(ohlcv) < 1000:
                    break
                
                time.sleep(self.rate_limit)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
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
    
    def extend_hourly_data(self, symbol, years=2):
        """
        Ensure we have at least `years` of hourly data.
        """
        symbol_fname = symbol.replace('/', '_')
        path = f'data/deep/{symbol_fname}_1h.csv'
        
        if os.path.exists(path):
            existing = pd.read_csv(path, index_col='timestamp', parse_dates=True)
            existing.sort_index(inplace=True)
            oldest = existing.index.min()
            newest = existing.index.max()
            target_oldest = datetime.now() - timedelta(days=365*years)
            
            if oldest <= target_oldest:
                logger.info(f"Already have {years} years of data (from {oldest})")
                return existing
            
            logger.info(f"Extending hourly data back to {target_oldest}")
            # Fetch older data
            older_data = self.fetch_ohlcv_range(
                symbol, '1h', target_oldest, oldest - timedelta(milliseconds=1)
            )
            if not older_data.empty:
                combined = pd.concat([older_data, existing])
                combined.sort_index(inplace=True)
                combined = combined[~combined.index.duplicated(keep='first')]
                combined.to_csv(path)
                logger.info(f"Added {len(older_data)} older rows, total {len(combined)}")
                return combined
            else:
                logger.warning("Could not fetch older data")
                return existing
        else:
            # Fetch fresh
            start_dt = datetime.now() - timedelta(days=365*years)
            df = self.fetch_ohlcv_range(symbol, '1h', start_dt, datetime.now())
            df.to_csv(path)
            logger.info(f"Saved {len(df)} rows to {path}")
            return df
    
    def extend_minute_data(self, symbol, days=60):
        """
        Ensure we have at least `days` of minute data.
        """
        symbol_fname = symbol.replace('/', '_')
        path = f'data/deep/{symbol_fname}_1m.csv'
        
        if os.path.exists(path):
            existing = pd.read_csv(path, index_col='timestamp', parse_dates=True)
            existing.sort_index(inplace=True)
            oldest = existing.index.min()
            newest = existing.index.max()
            target_oldest = datetime.now() - timedelta(days=days)
            
            if oldest <= target_oldest:
                logger.info(f"Already have {days} days of minute data (from {oldest})")
                return existing
            
            logger.info(f"Extending minute data back to {target_oldest}")
            # Fetch older data (minute data API may have limits)
            # We'll fetch in chunks to avoid rate limits
            chunk_days = 10
            all_older = []
            current_end = oldest - timedelta(milliseconds=1)
            current_start = max(target_oldest, current_end - timedelta(days=chunk_days))
            
            while current_start < current_end:
                logger.info(f"Fetching minute chunk from {current_start} to {current_end}")
                chunk = self.fetch_ohlcv_range(
                    symbol, '1m', current_start, current_end
                )
                if not chunk.empty:
                    all_older.append(chunk)
                
                # Move window back
                current_end = current_start - timedelta(milliseconds=1)
                current_start = max(target_oldest, current_end - timedelta(days=chunk_days))
                time.sleep(1)  # be nice
                
                if len(all_older) >= 5:  # safety break
                    break
            
            if all_older:
                older_data = pd.concat(all_older)
                combined = pd.concat([older_data, existing])
                combined.sort_index(inplace=True)
                combined = combined[~combined.index.duplicated(keep='first')]
                combined.to_csv(path)
                logger.info(f"Added {len(older_data)} older minute rows, total {len(combined)}")
                return combined
            else:
                logger.warning("Could not fetch older minute data")
                return existing
        else:
            # Fetch fresh
            start_dt = datetime.now() - timedelta(days=days)
            df = self.fetch_ohlcv_range(symbol, '1m', start_dt, datetime.now())
            df.to_csv(path)
            logger.info(f"Saved {len(df)} rows to {path}")
            return df
    
    def process_all_assets(self, symbols, hourly_years=2, minute_days=60):
        """Extend data for all assets."""
        for sym in symbols:
            logger.info(f"Processing {sym}")
            # Hourly
            hourly = self.extend_hourly_data(sym, years=hourly_years)
            # Minute
            minute = self.extend_minute_data(sym, days=minute_days)
            # Resample minute to other timeframes
            self.resample_minute_data(sym, minute)
            # Sleep between assets
            time.sleep(2)
        
        logger.info("All assets processed.")
    
    def resample_minute_data(self, symbol, df_minute):
        """Resample minute data to 5min, 15min, 1h."""
        symbol_fname = symbol.replace('/', '_')
        for resample_tf, rule in [('5min', '5min'), ('15min', '15min'), ('1h', '1h')]:
            resampled = df_minute.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            path = f'data/deep/processed/{symbol_fname}_{resample_tf}.csv'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            resampled.to_csv(path)
            logger.info(f"Resampled to {resample_tf}: {len(resampled)} rows")

def main():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    collector = ExtendedDataCollector()
    collector.process_all_assets(symbols, hourly_years=2, minute_days=60)

if __name__ == '__main__':
    main()