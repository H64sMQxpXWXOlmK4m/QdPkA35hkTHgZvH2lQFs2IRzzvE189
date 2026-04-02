import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_ohlcv(symbol, timeframe='1m', since=None, limit=1000, exchange_id='binance'):
    """
    Fetch OHLCV data from exchange.
    """
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        }
    })
    
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            # update since to last timestamp + 1ms to avoid duplicates
            since = ohlcv[-1][0] + 1
            # if we got less than limit, we've reached the end
            if len(ohlcv) < limit:
                break
            # be nice to rate limits
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            break
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_multiple_assets(assets, timeframe='1m', days_back=90):
    """
    Fetch data for multiple assets.
    assets: list of tuples (symbol, exchange_id)
    """
    since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    data = {}
    for symbol, exchange_id in assets:
        logger.info(f"Fetching {symbol} from {exchange_id}")
        df = fetch_ohlcv(symbol, timeframe, since=since, exchange_id=exchange_id)
        data[symbol] = df
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        # save raw data
        raw_path = f"data/raw/{symbol.replace('/', '_')}_{timeframe}.csv"
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path)
    return data

if __name__ == '__main__':
    # Define assets: symbol, exchange_id
    assets = [
        ('BTC/USDT', 'binance'),
        ('ETH/USDT', 'binance'),
        ('SOL/USDT', 'binance'),
        ('XRP/USDT', 'binance'),
        ('BNB/USDT', 'binance'),
        ('DOGE/USDT', 'binance'),
    ]
    data = fetch_multiple_assets(assets, timeframe='1m', days_back=180)  # 6 months