"""Load and standardize kline data from parquet files."""

from __future__ import annotations

import glob
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factor_mining.config import Settings


def _to_unix_ms(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert DatetimeIndex to Unix milliseconds (int64), handling any resolution."""
    return (index.asi8 // _resolution_divisor(index)).astype(np.int64)


def _resolution_divisor(index: pd.DatetimeIndex) -> int:
    """Return the divisor to convert asi8 to milliseconds."""
    unit = str(index.dtype).replace("datetime64[", "").replace("]", "")
    if unit == "ns":
        return 1_000_000
    if unit == "us":
        return 1_000
    if unit == "ms":
        return 1
    if unit == "s":
        raise ValueError("Second-resolution timestamps not supported")
    return 1_000_000  # default: assume ns


def _date_start_ms(value: str | None) -> int | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp() * 1000)


def resolve_frame_path(settings: Settings, symbol: str | None = None, market: str | None = None) -> Path:
    """Find the first parquet data file from config or default locations."""
    return resolve_frame_paths(settings, symbol=symbol, market=market)[0]


def resolve_frame_paths(settings: Settings, symbol: str | None = None, market: str | None = None) -> list[Path]:
    """Find parquet data files from flat files or the partitioned warehouse."""
    symbol = symbol or settings.data.symbols[0]
    market = market or ("um_futures" if "um_futures" in settings.data.markets else settings.data.markets[0])
    parquet_dir = settings.data.parquet_dir
    for candidate in [
        parquet_dir / f"{symbol}_{settings.data.default_interval}.parquet",
        parquet_dir / f"{symbol}_{settings.data.default_interval}_5y.parquet",
        parquet_dir / "btc_5m_5y.parquet",
        Path("data/parquet/btc_5m_5y.parquet"),
        Path("../btc_5m_5y.parquet"),
    ]:
        if candidate.exists():
            return [candidate.resolve()]

    partition_pattern = (
        parquet_dir
        / f"market={market}"
        / "dataset=klines"
        / f"symbol={symbol}"
        / f"interval={settings.data.default_interval}"
        / "year=*"
        / "month=*"
        / "data.parquet"
    )
    partition_files = [Path(path).resolve() for path in sorted(glob.glob(str(partition_pattern)))]
    if partition_files:
        return partition_files

    raise FileNotFoundError(
        f"No parquet found for {symbol} {settings.data.default_interval}. "
        f"Checked flat files in {parquet_dir}, partition path {partition_pattern}, "
        "data/parquet/, ../btc_5m_5y.parquet"
    )


def load_frame(
    settings: Settings,
    symbol: str | None = None,
    *,
    market: str | None = None,
    tail: int | None = None,
    end_ms: int | None = None,
) -> pd.DataFrame:
    """Load standardized kline DataFrame from parquet.

    Returns a DataFrame with columns: open_time, open, high, low, close,
    volume, close_time, quote_volume, trade_count, taker_buy_volume,
    taker_buy_quote_volume, market.
    """
    paths = resolve_frame_paths(settings, symbol=symbol, market=market)
    frames = [pd.read_parquet(path) for path in paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    if "open_time" in df.columns:
        frame = df.copy()
    else:
        frame = pd.DataFrame({
            "open_time": _to_unix_ms(df.index),
            "open": df["open"].to_numpy(dtype=float),
            "high": df["high"].to_numpy(dtype=float),
            "low": df["low"].to_numpy(dtype=float),
            "close": df["close"].to_numpy(dtype=float),
            "volume": df["volume"].to_numpy(dtype=float),
            "close_time": _to_unix_ms(df.index),
            "quote_volume": (df.get("volume", df["volume"]) * df["close"]).to_numpy(dtype=float),
            "trade_count": 0,
            "taker_buy_volume": (df["volume"] * 0.5).to_numpy(dtype=float),
            "taker_buy_quote_volume": (df["volume"] * df["close"] * 0.5).to_numpy(dtype=float),
            "market": "spot",
        })

    frame = frame.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    if "market" not in frame.columns:
        frame["market"] = market or "spot"
    if "data_quality_degraded" not in frame.columns:
        frame["data_quality_degraded"] = False

    start_ms = _date_start_ms(settings.data.start_date)
    if start_ms is not None:
        frame = frame[pd.to_numeric(frame["open_time"], errors="coerce") >= start_ms].reset_index(drop=True)

    # Pin the data extent to a reproduce snapshot: truncate to bars at or
    # before `end_ms` so a faithful re-run sees the same history (and hence
    # the same last-20% OOS window) the original mining run did.
    if end_ms is not None:
        frame = frame[pd.to_numeric(frame["open_time"], errors="coerce") <= int(end_ms)].reset_index(drop=True)

    if tail is not None:
        frame = frame.iloc[-tail:].reset_index(drop=True)

    return frame


def load_funding(settings: Settings, symbol: str = "BTCUSDT") -> pd.DataFrame | None:
    """Load funding rate data from synced Binance parquets.

    Returns DataFrame with columns: calc_time, funding_interval_hours,
    last_funding_rate when available.
    Returns None if no funding data is available.
    """
    pattern = str(
        settings.data.parquet_dir
        / "market=um_futures"
        / "dataset=fundingRate"
        / f"symbol={symbol}"
        / "interval=funding"
        / "year=*"
        / "month=*"
        / "data.parquet"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    frames = [pd.read_parquet(f) for f in files]
    funding = pd.concat(frames, ignore_index=True)
    funding = funding.sort_values("calc_time").reset_index(drop=True)
    cols = ["calc_time"]
    if "funding_interval_hours" in funding.columns:
        cols.append("funding_interval_hours")
    cols.append("last_funding_rate")
    return funding[cols]


def load_partitioned_dataset(
    settings: Settings,
    *,
    market: str,
    dataset: str,
    symbol: str,
    interval: str | None = None,
) -> pd.DataFrame | None:
    interval = interval or settings.data.default_interval
    pattern = str(
        settings.data.parquet_dir
        / f"market={market}"
        / f"dataset={dataset}"
        / f"symbol={symbol}"
        / f"interval={interval}"
        / "year=*"
        / "month=*"
        / "data.parquet"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    time_col = "open_time" if "open_time" in frame.columns else "timestamp"
    return frame.sort_values(time_col).drop_duplicates(time_col).reset_index(drop=True)


def load_supplemental_features(
    settings: Settings,
    frame: pd.DataFrame,
    *,
    symbol: str,
    market: str,
    interval: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Load optional Binance USD-M supplemental datasets as aligned features."""
    if market != "um_futures":
        return pd.DataFrame(index=frame.index), {}

    interval = interval or settings.data.default_interval
    data = {
        name: load_partitioned_dataset(settings, market=market, dataset=name, symbol=symbol, interval=interval)
        for name in [
            "markPriceKlines",
            "indexPriceKlines",
            "premiumIndexKlines",
            "openInterestHist",
            "globalLongShortAccountRatio",
            "topLongShortAccountRatio",
            "topLongShortPositionRatio",
            "takerlongshortRatio",
            "basis",
        ]
    }
    features: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    spot_klines = load_partitioned_dataset(settings, market="spot", dataset="klines", symbol=symbol, interval=interval)
    spot_close = _aligned_dataset_series(frame, spot_klines, "open_time", "close")
    if spot_close is not None and "close" in frame.columns:
        perp_close = pd.Series(frame["close"].to_numpy(dtype=float), index=frame.index)
        spot_perp_basis = (perp_close / spot_close.replace(0, np.nan)) - 1.0
        _add_feature(features, meta, "spot_perp_basis", spot_perp_basis, "funding_basis", "negative_when_high")
        _add_feature(features, meta, "spot_perp_basis_z_288", _rolling_zscore(spot_perp_basis, 288), "funding_basis", "negative_when_high")
        _add_feature(features, meta, "spot_perp_basis_chg_12", spot_perp_basis.diff(12), "funding_basis", "negative_when_high")

    mark_close = _aligned_dataset_series(frame, data["markPriceKlines"], "open_time", "close")
    index_close = _aligned_dataset_series(frame, data["indexPriceKlines"], "open_time", "close")
    premium_close = _aligned_dataset_series(frame, data["premiumIndexKlines"], "open_time", "close")
    if mark_close is not None:
        _add_feature(features, meta, "mark_price_ret_12", mark_close.pct_change(12), "funding_basis", "positive")
    if index_close is not None:
        _add_feature(features, meta, "index_price_ret_12", index_close.pct_change(12), "trend_following", "positive")
    if mark_close is not None and index_close is not None:
        mark_index_basis = (mark_close / index_close.replace(0, np.nan)) - 1.0
        _add_feature(features, meta, "mark_index_basis", mark_index_basis, "funding_basis", "negative_when_high")
        _add_feature(features, meta, "mark_index_basis_z_288", _rolling_zscore(mark_index_basis, 288), "funding_basis", "negative_when_high")
    if premium_close is not None:
        _add_feature(features, meta, "premium_index", premium_close, "funding_basis", "negative_when_high")
        _add_feature(features, meta, "premium_index_z_288", _rolling_zscore(premium_close, 288), "funding_basis", "negative_when_high")
        _add_feature(features, meta, "premium_index_chg_12", premium_close.diff(12), "funding_basis", "negative_when_high")

    oi = data["openInterestHist"]
    if oi is not None:
        oi_value = _aligned_dataset_series(frame, oi, "timestamp", "sumOpenInterestValue")
        oi_qty = _aligned_dataset_series(frame, oi, "timestamp", "sumOpenInterest")
        if oi_value is not None:
            _add_feature(features, meta, "open_interest_value_z_288", _rolling_zscore(oi_value, 288), "volume_confirmation", "positive")
            _add_feature(features, meta, "open_interest_value_chg_12", oi_value.pct_change(12), "volume_confirmation", "positive")
        if oi_qty is not None:
            _add_feature(features, meta, "open_interest_chg_12", oi_qty.pct_change(12), "volume_confirmation", "positive")

    for dataset, prefix in [
        ("globalLongShortAccountRatio", "global_ls_account"),
        ("topLongShortAccountRatio", "top_ls_account"),
        ("topLongShortPositionRatio", "top_ls_position"),
    ]:
        ratio = _aligned_dataset_series(frame, data[dataset], "timestamp", "longShortRatio")
        if ratio is not None:
            _add_feature(features, meta, f"{prefix}_ratio", ratio, "funding_basis", "negative_when_high")
            _add_feature(features, meta, f"{prefix}_ratio_z_288", _rolling_zscore(ratio, 288), "funding_basis", "negative_when_high")
        long_account = _aligned_dataset_series(frame, data[dataset], "timestamp", "longAccount")
        short_account = _aligned_dataset_series(frame, data[dataset], "timestamp", "shortAccount")
        if long_account is not None and short_account is not None:
            imbalance = long_account - short_account
            _add_feature(features, meta, f"{prefix}_imbalance", imbalance, "funding_basis", "negative_when_high")

    taker = data["takerlongshortRatio"]
    if taker is not None:
        buy_sell = _aligned_dataset_series(frame, taker, "timestamp", "buySellRatio")
        buy_vol = _aligned_dataset_series(frame, taker, "timestamp", "buyVol")
        sell_vol = _aligned_dataset_series(frame, taker, "timestamp", "sellVol")
        if buy_sell is not None:
            _add_feature(features, meta, "taker_buy_sell_ratio", buy_sell, "volume_confirmation", "positive")
            _add_feature(features, meta, "taker_buy_sell_ratio_z_288", _rolling_zscore(buy_sell, 288), "volume_confirmation", "positive")
        if buy_vol is not None and sell_vol is not None:
            imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol).replace(0, np.nan)
            _add_feature(features, meta, "taker_buy_sell_imbalance", imbalance, "volume_confirmation", "positive")

    basis = data["basis"]
    if basis is not None:
        basis_rate = _aligned_dataset_series(frame, basis, "timestamp", "basisRate")
        basis_abs = _aligned_dataset_series(frame, basis, "timestamp", "basis")
        if basis_rate is not None:
            _add_feature(features, meta, "perp_basis_rate", basis_rate, "funding_basis", "negative_when_high")
            _add_feature(features, meta, "perp_basis_rate_z_288", _rolling_zscore(basis_rate, 288), "funding_basis", "negative_when_high")
        if basis_abs is not None:
            _add_feature(features, meta, "perp_basis_chg_12", basis_abs.diff(12), "funding_basis", "negative_when_high")

    if not features:
        return pd.DataFrame(index=frame.index), {}
    return pd.DataFrame(features, index=frame.index), meta


def merge_funding_to_frame(frame: pd.DataFrame, funding: pd.DataFrame | None) -> pd.Series:
    """Merge 8h funding rate into 5m frame via forward fill.

    Returns a pd.Series of funding_rate aligned to frame.index.
    """
    if funding is None or funding.empty:
        return pd.Series(0.0, index=frame.index)

    fr = pd.Series(
        funding["last_funding_rate"].to_numpy(dtype=float),
        index=funding["calc_time"].to_numpy(dtype="int64"),
    )
    # Reindex to frame's open_time with forward fill
    aligned = fr.reindex(frame["open_time"].to_numpy(dtype="int64"), method="ffill")
    # Backfill initial NaNs
    aligned = aligned.bfill().fillna(0.0)
    aligned.index = frame.index
    return aligned


def funding_event_zscore_to_frame(
    frame: pd.DataFrame,
    funding: pd.DataFrame | None,
    *,
    lookback_events: int = 21,
) -> pd.Series:
    """Convert 8h funding events into an event-window z-score aligned to bars.

    The rolling statistics are computed on funding snapshots, not repeated 5m
    bars. The result is forward-filled after each known event and remains 0
    before the first event to avoid looking ahead.
    """
    if funding is None or funding.empty:
        return pd.Series(0.0, index=frame.index)

    funding = funding.sort_values("calc_time").drop_duplicates("calc_time")
    rates = pd.Series(
        funding["last_funding_rate"].to_numpy(dtype=float),
        index=funding["calc_time"].to_numpy(dtype="int64"),
    )
    min_periods = min(6, max(2, lookback_events))
    mean = rates.rolling(lookback_events, min_periods=min_periods).mean()
    std = rates.rolling(lookback_events, min_periods=min_periods).std().replace(0, np.nan)
    z = ((rates - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    aligned = z.reindex(frame["open_time"].to_numpy(dtype="int64"), method="ffill").fillna(0.0)
    aligned.index = frame.index
    return aligned


def _aligned_dataset_series(
    frame: pd.DataFrame,
    dataset: pd.DataFrame | None,
    time_col: str,
    value_col: str,
) -> pd.Series | None:
    if dataset is None or dataset.empty or time_col not in dataset.columns or value_col not in dataset.columns:
        return None
    series = pd.Series(
        pd.to_numeric(dataset[value_col], errors="coerce").to_numpy(dtype=float),
        index=dataset[time_col].astype("int64").to_numpy(),
    ).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    aligned = series.reindex(frame["open_time"].to_numpy(dtype="int64"), method="ffill")
    aligned.index = frame.index
    return aligned.replace([np.inf, -np.inf], np.nan)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(12, window // 6)).mean()
    std = series.rolling(window, min_periods=max(12, window // 6)).std().replace(0, np.nan)
    return (series - mean) / std


def _add_feature(
    features: dict[str, np.ndarray],
    meta: dict[str, dict],
    name: str,
    series: pd.Series,
    family: str,
    direction: str,
) -> None:
    clean = series.replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() < 10:
        return
    features[name] = clean.to_numpy(dtype=float)
    meta[name] = {"family": family, "direction": direction, "regime": "any", "source": "binance_supplemental"}
