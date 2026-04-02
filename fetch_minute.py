import sys
sys.path.append('.')
from fetch_data import fetch_all_assets
import time

assets = [
    ('BTC/USDT', 'binance'),
    ('ETH/USDT', 'binance'),
    ('SOL/USDT', 'binance'),
    ('XRP/USDT', 'binance'),
    ('BNB/USDT', 'binance'),
    ('DOGE/USDT', 'binance'),
]

print("Fetching 1m data for 30 days...")
fetch_all_assets(assets, ['1m'], days=30)
print("Done.")