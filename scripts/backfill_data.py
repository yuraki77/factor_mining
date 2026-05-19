"""Backfill historical Binance kline and funding rate data to parquet warehouse.

Downloads 5m klines and funding rates from Binance public API, saves in the
partition layout expected by factor_mining.data.loader:

  data/parquet/market={market}/dataset={dataset}/symbol={symbol}/interval={interval}/year={year}/month={month}/data.parquet

Usage:
  python scripts/backfill_data.py              # download all missing data
  python scripts/backfill_data.py --dry-run    # show what would be downloaded
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import httpx
import zipfile
import io

UTC = timezone.utc
PARQUET_DIR = Path("data/parquet")

# Data availability dates (Binance listing dates)
SPOT_START = {
    "BTCUSDT": "2017-08-17",
    "ETHUSDT": "2017-08-17",
}
FUTURES_START = {
    "BTCUSDT": "2019-09-08",
    "ETHUSDT": "2019-11-27",
}
FUNDING_START = FUTURES_START  # same as futures


def _month_ranges(start_date: str, end_date: str | None = None) -> list[tuple[int, int]]:
    """Return list of (year, month) tuples from start_date to end_date."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.now(UTC) if end_date is None else datetime.strptime(end_date, "%Y-%m-%d")
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _parquet_path(market: str, dataset: str, symbol: str, interval: str, year: int, month: int) -> Path:
    return (
        PARQUET_DIR
        / f"market={market}"
        / f"dataset={dataset}"
        / f"symbol={symbol}"
        / f"interval={interval}"
        / f"year={year}"
        / f"month={month:02d}"
        / "data.parquet"
    )


def fetch_klines(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    *,
    market: str,
) -> pd.DataFrame:
    """Fetch klines from data.binance.vision monthly zip files."""
    market_path = "futures/um" if market == "um_futures" else "spot"
    # URL format: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2022-01.zip
    url = f"https://data.binance.vision/data/{market_path}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"

    resp = httpx.get(url, timeout=60)
    if resp.status_code == 404:
        return pd.DataFrame()  # Not generated yet or missing
    resp.raise_for_status()

    # Read zip in memory
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        # Get the first csv file in the zip
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            # Check if there is a header by reading the first line
            first_line = f.readline().decode('utf-8').strip()
            has_header = "open" in first_line.lower() or "time" in first_line.lower()
            
            # Reset file pointer
            f.seek(0)
            
            # Binance CSV format for klines doesn't always have headers.
            names=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trade_count",
                "taker_buy_volume", "taker_buy_quote_volume", "ignore",
            ]
            
            if has_header:
                df = pd.read_csv(f, header=0, names=names)
            else:
                df = pd.read_csv(f, header=None, names=names)

    df = df.drop(columns=["ignore"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df["trade_count"] = df["trade_count"].astype("int64")
    df["market"] = market
    df["data_quality_degraded"] = False
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df


def fetch_funding_rates(
    symbol: str,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Fetch funding rate history from data.binance.vision monthly zip files."""
    # URL format: https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2022-01.zip
    url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"

    resp = httpx.get(url, timeout=60)
    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    # The CSV headers can vary: calc_time, funding_rate, symbol
    result = pd.DataFrame()
    
    # Try different possible column names
    time_col = next((c for c in df.columns if c.lower() in ("calctime", "fundingtime", "calc_time")), None)
    if time_col is None:
        # Maybe no header?
        time_col = 0
        df = pd.read_csv(f, header=None)

    result["calc_time"] = pd.to_numeric(df[time_col], errors="coerce").astype("int64")
    
    rate_col = next((c for c in df.columns if "funding" in str(c).lower() and "rate" in str(c).lower()), None)
    if rate_col is None:
        rate_col = 1 if time_col == 0 else "fundingRate"
        
    result["last_funding_rate"] = pd.to_numeric(df[rate_col], errors="coerce")
    result["funding_interval_hours"] = 8  # Default for UM futures
    
    result = result.dropna(subset=["calc_time", "last_funding_rate"]).drop_duplicates("calc_time")
    result = result.sort_values("calc_time").reset_index(drop=True)
    return result


def backfill_klines(
    symbol: str,
    market: str,
    interval: str = "5m",
    *,
    dry_run: bool = False,
) -> int:
    """Download missing kline months for a symbol/market pair."""
    start_dates = FUTURES_START if market == "um_futures" else SPOT_START
    start_date = start_dates.get(symbol)
    if start_date is None:
        print(f"  Skip {symbol}/{market}: no known start date")
        return 0

    months = _month_ranges(start_date)
    downloaded = 0

    for year, month in months:
        path = _parquet_path(market, "klines", symbol, interval, year, month)
        if path.exists():
            continue

        if dry_run:
            print(f"  WOULD download {symbol}/{market}/{interval} {year}/{month:02d}")
            downloaded += 1
            continue

        print(f"  Downloading {symbol}/{market}/{interval} {year}/{month:02d}...", end="", flush=True)

        try:
            df = fetch_klines(symbol, interval, year, month, market=market)
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        if df.empty:
            print(f" (no data)")
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        print(f" {len(df):,} bars")
        downloaded += 1

    return downloaded


def backfill_funding(
    symbol: str,
    *,
    dry_run: bool = False,
) -> int:
    """Download missing funding rate months for a symbol."""
    start_date = FUNDING_START.get(symbol)
    if start_date is None:
        print(f"  Skip {symbol} funding: no known start date")
        return 0

    months = _month_ranges(start_date)
    downloaded = 0

    for year, month in months:
        path = _parquet_path("um_futures", "fundingRate", symbol, "funding", year, month)
        if path.exists():
            continue

        if dry_run:
            print(f"  WOULD download {symbol}/funding {year}/{month:02d}")
            downloaded += 1
            continue

        print(f"  Downloading {symbol}/funding {year}/{month:02d}...", end="", flush=True)

        try:
            df = fetch_funding_rates(symbol, year, month)
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        if df.empty:
            print(f" (no data)")
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, engine="pyarrow")
        print(f" {len(df)} snapshots")
        downloaded += 1

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Backfill Binance historical data from data.binance.vision")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()

    symbols = ["BTCUSDT", "ETHUSDT"]
    total = 0

    # Spot klines
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Spot klines: {symbol}")
        print(f"{'='*60}")
        total += backfill_klines(symbol, "spot", dry_run=args.dry_run)

    # Futures klines
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Futures klines: {symbol}")
        print(f"{'='*60}")
        total += backfill_klines(symbol, "um_futures", dry_run=args.dry_run)

    # Funding rates
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Funding rates: {symbol}")
        print(f"{'='*60}")
        total += backfill_funding(symbol, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    action = "Would download" if args.dry_run else "Downloaded"
    print(f"{action} {total} files total.")


if __name__ == "__main__":
    main()
