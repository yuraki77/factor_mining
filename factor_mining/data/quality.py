from __future__ import annotations

import pandas as pd

from factor_mining.models import DataQualityNote


def normalize_timestamp_ms(value: int | float) -> int:
    value_int = int(value)
    if value_int >= 10**15:
        return value_int // 1000
    return value_int


def normalize_kline_frame(frame: pd.DataFrame, *, market: str) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    frame = frame.iloc[:, :12].copy()
    if str(frame.iloc[0, 0]).lower() == "open_time":
        frame = frame.iloc[1:].reset_index(drop=True)
    frame.columns = columns
    frame["open_time"] = frame["open_time"].map(normalize_timestamp_ms)
    frame["close_time"] = frame["close_time"].map(normalize_timestamp_ms)
    numeric = [col for col in columns if col not in {"open_time", "close_time", "ignore"}]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["market"] = market
    frame = frame.drop(columns=["ignore"])
    return frame.sort_values("open_time").reset_index(drop=True)


def normalize_funding_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    frame = frame.iloc[:, :3].copy()
    if str(frame.iloc[0, 0]).lower() == "calc_time":
        frame = frame.iloc[1:].reset_index(drop=True)
    frame.columns = columns
    frame["calc_time"] = frame["calc_time"].map(normalize_timestamp_ms)
    frame["funding_interval_hours"] = pd.to_numeric(frame["funding_interval_hours"], errors="coerce")
    frame["last_funding_rate"] = pd.to_numeric(frame["last_funding_rate"], errors="coerce")
    frame["symbol"] = symbol
    return frame.sort_values("calc_time").reset_index(drop=True)


def kline_quality_notes(frame: pd.DataFrame, *, interval_ms: int, scope: str) -> list[DataQualityNote]:
    notes: list[DataQualityNote] = []
    total = max(len(frame), 1)
    duplicates = frame["open_time"].duplicated().sum()
    if duplicates:
        notes.append(DataQualityNote(scope=scope, message="duplicate timestamps detected", degraded_ratio=duplicates / total))

    numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
    negative_rows = (frame[numeric] < 0).any(axis=1).sum()
    if negative_rows:
        notes.append(DataQualityNote(scope=scope, severity="fail", message="negative OHLCV values detected", degraded_ratio=negative_rows / total))

    invalid_high = (frame["high"] < frame[["open", "close"]].max(axis=1)).sum()
    invalid_low = (frame["low"] > frame[["open", "close"]].min(axis=1)).sum()
    if invalid_high or invalid_low:
        notes.append(
            DataQualityNote(
                scope=scope,
                severity="fail",
                message="OHLC invariant failed",
                degraded_ratio=(invalid_high + invalid_low) / total,
            )
        )

    zero_volume_ratio = float((frame["volume"] == 0).sum() / total)
    if zero_volume_ratio > 0.01:
        notes.append(DataQualityNote(scope=scope, message="zero-volume bar ratio exceeded 1%", degraded_ratio=zero_volume_ratio))

    price_jump_ratio = float((frame["close"].pct_change().abs() > 0.10).sum() / total)
    if price_jump_ratio > 0:
        notes.append(DataQualityNote(scope=scope, message="adjacent close jump exceeded 10%", degraded_ratio=price_jump_ratio))

    gaps = frame["open_time"].diff().dropna()
    if not gaps.empty:
        gap_rows = (gaps > interval_ms).sum()
        if gap_rows:
            notes.append(DataQualityNote(scope=scope, message="coverage continuity gaps detected", degraded_ratio=float(gap_rows / total)))
    return notes


def interval_to_ms(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60_000
    if interval.endswith("h"):
        return int(interval[:-1]) * 3_600_000
    if interval.endswith("d"):
        return int(interval[:-1]) * 86_400_000
    raise ValueError(f"Unsupported interval: {interval}")
