#!/usr/bin/env python3
"""
Expand BTC minute data to 180 days.
"""
import sys
sys.path.append('.')
from expand_minute_data import extend_minute_data, resample_to_timeframes
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    symbol = 'BTC/USDT'
    days = 180  # target total days
    
    logger.info(f"Expanding minute data for {symbol} to {days} days")
    
    # Extend minute data
    df = extend_minute_data(symbol, days=days)
    if df is None:
        logger.error("Failed to extend minute data")
        return
    
    logger.info(f"Extended minute data: {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # Resample to other timeframes
    logger.info("Resampling to 5m, 15m, 1h")
    resample_to_timeframes(symbol)
    
    logger.info("BTC minute data expansion complete")

if __name__ == '__main__':
    main()