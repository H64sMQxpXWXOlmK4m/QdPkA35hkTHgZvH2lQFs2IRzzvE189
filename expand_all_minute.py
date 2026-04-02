#!/usr/bin/env python3
"""
Expand minute data for all assets to 90 days.
"""
import sys
sys.path.append('.')
from expand_minute_data import extend_minute_data, resample_to_timeframes
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def expand_asset(symbol, days=90):
    """Expand minute data for a single asset."""
    logger.info(f"Expanding minute data for {symbol} to {days} days")
    
    # Extend minute data
    df = extend_minute_data(symbol, days=days)
    if df is None:
        logger.error(f"Failed to extend minute data for {symbol}")
        return False
    
    logger.info(f"{symbol}: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # Resample to other timeframes
    logger.info(f"Resampling {symbol} to 5m, 15m, 1h")
    try:
        resample_to_timeframes(symbol)
    except Exception as e:
        logger.error(f"Failed to resample {symbol}: {e}")
        return False
    
    return True

def main():
    symbols = ['ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    days = 90
    
    for sym in symbols:
        logger.info(f"Processing {sym}")
        success = expand_asset(sym, days=days)
        if success:
            logger.info(f"Completed {sym}")
        else:
            logger.warning(f"Failed {sym}, moving to next")
        
        # Sleep between assets to avoid rate limits
        time.sleep(30)
    
    logger.info("All assets processed.")

if __name__ == '__main__':
    main()