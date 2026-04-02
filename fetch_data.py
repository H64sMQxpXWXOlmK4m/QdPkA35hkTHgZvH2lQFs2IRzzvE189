import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, exchange_id='binance'):
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        self.rate_limit = self.exchange.rateLimit / 1000  # seconds
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', 
                    since: int = None, limit: int = 1000):
        """
        Fetch OHLCV data with pagination.
        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        """
        all_ohlcv = []
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not ohlcv:
                    logger.info(f"No more data for {symbol}")
                    break
                all_ohlcv.extend(ohlcv)
                # update since to last timestamp + 1ms
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < limit:
                    break
                # respect rate limit
                time.sleep(self.rate_limit)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                # wait and retry
                time.sleep(5)
        if not all_ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    def fetch_since_days(self, symbol: str, timeframe: str = '1m', days: int = 30):
        """Fetch data from now - days."""
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        logger.info(f"Fetching {symbol} {timeframe} since {since} ({days} days back)")
        return self.fetch_ohlcv(symbol, timeframe, since=since)
    
    def save_df(self, df: pd.DataFrame, symbol: str, timeframe: str, base_dir='data/raw'):
        """Save DataFrame to CSV."""
        symbol_fname = symbol.replace('/', '_')
        path = f"{base_dir}/{symbol_fname}_{timeframe}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)
        logger.info(f"Saved {len(df)} rows to {path}")

def fetch_all_assets(assets: List[Tuple[str, str]], timeframes: List[str], days: int = 30):
    """
    assets: list of (symbol, exchange_id)
    """
    for symbol, exchange_id in assets:
        fetcher = DataFetcher(exchange_id)
        for tf in timeframes:
            df = fetcher.fetch_since_days(symbol, tf, days)
            if df.empty:
                logger.warning(f"No data for {symbol} {tf}")
                continue
            fetcher.save_df(df, symbol, tf)
            # resample to higher timeframes if needed
            if tf == '1m':
                # create 5m, 15m, 1h resampled data
                for resample_tf, rule in [('5m', '5min'), ('15m', '15min'), ('1h', '1h')]:
                    resampled = df.resample(rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    fetcher.save_df(resampled, symbol, resample_tf, base_dir='data/processed')
        # sleep between assets to avoid rate limits
        time.sleep(2)
    logger.info("All assets fetched.")

if __name__ == '__main__':
    assets = [
        ('BTC/USDT', 'binance'),
        ('ETH/USDT', 'binance'),
        ('SOL/USDT', 'binance'),
        ('XRP/USDT', 'binance'),
        ('BNB/USDT', 'binance'),
        ('DOGE/USDT', 'binance'),
    ]
    # timeframe: days to fetch
    timeframe_days = {
        '1m': 7,
        '1h': 90,
    }
    for tf, days in timeframe_days.items():
        logger.info(f"Fetching {tf} data for {days} days")
        fetch_all_assets(assets, [tf], days=days)
    logger.info("Data fetching complete.")