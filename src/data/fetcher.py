"""
Multi-timeframe Binance data fetcher.

Fetches OHLCV + extended fields (taker buy volumes, num trades) for all
target assets and timeframes. Stores results as parquet files and handles
incremental updates, rate limits, and retries.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    EXCHANGE_ID,
    FETCH_LIMIT,
    MAX_RETRIES,
    RATE_LIMIT_SLEEP,
    SYMBOLS,
    TIMEFRAMES,
)

logger = logging.getLogger(__name__)

# Binance kline columns (in order returned by the API)
_KLINE_COLS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

_KEEP_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote",
]


def _make_exchange() -> ccxt.Exchange:
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    ex.load_markets()
    return ex


def _parquet_path(symbol: str, tf: str) -> Path:
    safe = symbol.replace("/", "_")
    return DATA_DIR / tf / f"{safe}.parquet"


def _fetch_ohlcv_raw(
    exchange: ccxt.Exchange,
    symbol: str,
    tf: str,
    since_ms: Optional[int],
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch raw klines from Binance using the /klines endpoint via ccxt's
    fetch_ohlcv plus fetch_ohlcv with extra params for additional columns.
    Falls back to standard fetch_ohlcv if extended not available.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Use binance-specific klines for full column data
            if hasattr(exchange, "fetch_ohlcv"):
                rows = exchange.fetch_ohlcv(
                    symbol, timeframe=tf, since=since_ms, limit=limit,
                    params={"price": "mark"} if False else {}
                )
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return df
        except ccxt.RateLimitExceeded:
            wait = 2 ** attempt * 2
            logger.warning(f"Rate limit hit, sleeping {wait}s")
            time.sleep(wait)
        except Exception as e:
            logger.error(f"Fetch error ({attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(1)
    return pd.DataFrame()


def _fetch_binance_klines_extended(
    exchange: ccxt.Exchange,
    symbol: str,
    tf: str,
    since_ms: Optional[int],
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch full kline data including taker buy volumes via Binance REST.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # Use ccxt's underlying request for full kline data
            market_id = exchange.market_id(symbol)
            params = {"symbol": market_id, "interval": tf, "limit": min(limit, 1000)}
            if since_ms is not None:
                params["startTime"] = since_ms

            resp = exchange.publicGetKlines(params)
            if not resp:
                return pd.DataFrame()

            df = pd.DataFrame(resp, columns=_KLINE_COLS)
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            for col in _KEEP_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df[_KEEP_COLS]

        except ccxt.RateLimitExceeded:
            wait = 2 ** attempt * 2
            logger.warning(f"Rate limit, sleeping {wait}s")
            time.sleep(wait)
        except Exception as e:
            logger.warning(f"Extended kline fetch failed ({attempt+1}): {e}")
            # fallback
            return _fetch_ohlcv_raw(exchange, symbol, tf, since_ms, limit)

    return pd.DataFrame()


def fetch_full_history(
    symbol: str,
    tf: str,
    exchange: Optional[ccxt.Exchange] = None,
    max_bars: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch full history for symbol/timeframe via paginated requests.
    Returns DataFrame sorted ascending by timestamp.
    """
    if exchange is None:
        exchange = _make_exchange()

    limit = FETCH_LIMIT.get(tf, 5000) if max_bars is None else max_bars
    batch = 1000
    all_frames = []
    since_ms = None

    while True:
        df = _fetch_binance_klines_extended(exchange, symbol, tf, since_ms, batch)
        if df.empty:
            break
        all_frames.append(df)
        fetched = len(df)
        logger.info(f"  {symbol} {tf}: fetched {fetched} bars (total so far: {sum(len(f) for f in all_frames)})")

        if fetched < batch:
            break

        last_ts = df.index[-1]
        # advance one candle
        since_ms = int(last_ts.timestamp() * 1000) + 1

        total = sum(len(f) for f in all_frames)
        if total >= limit:
            break

        time.sleep(RATE_LIMIT_SLEEP)

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames)
    result = result[~result.index.duplicated(keep="last")]
    result.sort_index(inplace=True)

    # Keep only last `limit` bars
    if len(result) > limit:
        result = result.iloc[-limit:]

    return result


def load_parquet(symbol: str, tf: str) -> pd.DataFrame:
    p = _parquet_path(symbol, tf)
    if p.exists():
        df = pd.read_parquet(p)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    return pd.DataFrame()


def save_parquet(df: pd.DataFrame, symbol: str, tf: str) -> None:
    p = _parquet_path(symbol, tf)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)


def update_data(
    symbol: str,
    tf: str,
    exchange: Optional[ccxt.Exchange] = None,
) -> pd.DataFrame:
    """
    Incrementally update stored data for symbol/timeframe.
    Fetches only new bars since the last stored timestamp.
    """
    if exchange is None:
        exchange = _make_exchange()

    existing = load_parquet(symbol, tf)

    if existing.empty:
        logger.info(f"No cached data for {symbol} {tf}, fetching full history...")
        df = fetch_full_history(symbol, tf, exchange)
    else:
        last_ts = existing.index[-1]
        since_ms = int(last_ts.timestamp() * 1000) + 1
        logger.info(f"Updating {symbol} {tf} from {last_ts}...")
        new_data = _fetch_binance_klines_extended(exchange, symbol, tf, since_ms, 1000)
        if new_data.empty:
            return existing
        df = pd.concat([existing, new_data])
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        # Trim to FETCH_LIMIT
        limit = FETCH_LIMIT.get(tf, 5000)
        if len(df) > limit:
            df = df.iloc[-limit:]

    if not df.empty:
        save_parquet(df, symbol, tf)

    return df


def fetch_all(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    update_only: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Fetch/update data for all symbols and timeframes.
    Returns nested dict: data[symbol][tf] = DataFrame.
    """
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    exchange = _make_exchange()
    data: Dict[str, Dict[str, pd.DataFrame]] = {}

    for sym in symbols:
        data[sym] = {}
        for tf in timeframes:
            logger.info(f"Fetching {sym} {tf}...")
            try:
                if update_only:
                    df = update_data(sym, tf, exchange)
                else:
                    df = fetch_full_history(sym, tf, exchange)
                    save_parquet(df, sym, tf)
                data[sym][tf] = df
                logger.info(f"  {sym} {tf}: {len(df)} bars, "
                            f"{df.index.min()} → {df.index.max()}")
            except Exception as e:
                logger.error(f"Failed {sym} {tf}: {e}")
                data[sym][tf] = pd.DataFrame()
            time.sleep(RATE_LIMIT_SLEEP)

    return data


def load_all(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all data from cache (no API calls)."""
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        data[sym] = {}
        for tf in timeframes:
            data[sym][tf] = load_parquet(sym, tf)
    return data
