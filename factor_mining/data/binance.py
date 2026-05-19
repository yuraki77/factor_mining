from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from factor_mining.config import Settings
from factor_mining.data.quality import (
    interval_to_ms,
    kline_quality_notes,
    normalize_funding_frame,
    normalize_kline_frame,
    normalize_timestamp_ms,
)
from factor_mining.models import DataCoverageRecord
from factor_mining.storage import MetadataStore


BINANCE_DATA_BASE = "https://data.binance.vision/data"
BINANCE_SPOT_REST_BASE = "https://api.binance.com"
BINANCE_UM_FUTURES_REST_BASE = "https://fapi.binance.com"
REST_KLINE_LIMIT = 1000
REST_FUNDING_LIMIT = 1000
REST_PRICE_KLINE_LIMIT = 1500
REST_FUTURES_DATA_LIMIT = 500
FUTURES_DATA_RECENT_WINDOW_MS = 30 * 86_400_000

PRICE_KLINE_DATASETS: dict[str, tuple[str, str]] = {
    "markPriceKlines": ("/fapi/v1/markPriceKlines", "symbol"),
    "indexPriceKlines": ("/fapi/v1/indexPriceKlines", "pair"),
    "premiumIndexKlines": ("/fapi/v1/premiumIndexKlines", "symbol"),
}

FUTURES_DATASETS: dict[str, tuple[str, str, dict[str, str]]] = {
    "openInterestHist": ("/futures/data/openInterestHist", "symbol", {}),
    "globalLongShortAccountRatio": ("/futures/data/globalLongShortAccountRatio", "symbol", {}),
    "topLongShortAccountRatio": ("/futures/data/topLongShortAccountRatio", "symbol", {}),
    "topLongShortPositionRatio": ("/futures/data/topLongShortPositionRatio", "symbol", {}),
    "takerlongshortRatio": ("/futures/data/takerlongshortRatio", "symbol", {}),
    "basis": ("/futures/data/basis", "pair", {"contractType": "PERPETUAL"}),
}

DEFAULT_SUPPLEMENTAL_DATASETS: tuple[str, ...] = (
    "markPriceKlines",
    "indexPriceKlines",
    "premiumIndexKlines",
    "openInterestHist",
    "globalLongShortAccountRatio",
    "topLongShortAccountRatio",
    "topLongShortPositionRatio",
    "takerlongshortRatio",
    "basis",
)


@dataclass(frozen=True)
class ArchiveAsset:
    market: str
    dataset: str
    symbol: str
    interval: str | None
    year: int
    month: int

    @property
    def filename(self) -> str:
        month = f"{self.year:04d}-{self.month:02d}"
        if self.dataset == "fundingRate":
            return f"{self.symbol}-fundingRate-{month}.zip"
        if self.interval is None:
            raise ValueError("interval is required for kline archives")
        return f"{self.symbol}-{self.interval}-{month}.zip"

    @property
    def path(self) -> str:
        if self.market == "spot":
            if self.interval is None:
                raise ValueError("spot archive requires interval")
            return f"spot/monthly/klines/{self.symbol}/{self.interval}/{self.filename}"
        if self.dataset == "fundingRate":
            return f"futures/um/monthly/fundingRate/{self.symbol}/{self.filename}"
        if self.interval is None:
            raise ValueError("futures archive requires interval")
        return f"futures/um/monthly/klines/{self.symbol}/{self.interval}/{self.filename}"

    @property
    def url(self) -> str:
        return f"{BINANCE_DATA_BASE}/{self.path}"

    @property
    def checksum_url(self) -> str:
        return f"{self.url}.CHECKSUM"


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    cursor = date(start.year, start.month, 1)
    stop = date(end.year, end.month, 1)
    out: list[tuple[int, int]] = []
    while cursor <= stop:
        out.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return out


def verify_checksum(payload: bytes, checksum_text: str) -> bool:
    expected = checksum_text.strip().split()[0]
    actual = hashlib.sha256(payload).hexdigest()
    return actual == expected


def _date_start_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def _date_end_ms(day: date) -> int:
    return _date_start_ms(day) + 86_400_000 - 1


class BinanceArchiveClient:
    def __init__(self, settings: Settings, store: MetadataStore | None = None) -> None:
        self.settings = settings
        self.store = store

    def sync(
        self,
        *,
        symbols: list[str] | None = None,
        markets: list[str] | None = None,
        interval: str | None = None,
        start: date | None = None,
        end: date | None = None,
        dry_run: bool = False,
    ) -> list[DataCoverageRecord]:
        symbols = symbols or self.settings.data.symbols
        markets = markets or self.settings.data.markets
        interval = interval or self.settings.data.default_interval
        start = start or datetime.fromisoformat(self.settings.data.start_date).date()
        end = end or datetime.utcnow().date().replace(day=1)
        records: list[DataCoverageRecord] = []
        for symbol in symbols:
            for year, month in month_range(start, end):
                for market in markets:
                    records.append(self._sync_asset(ArchiveAsset(market, "klines", symbol, interval, year, month), dry_run=dry_run))
                records.append(self._sync_asset(ArchiveAsset("um_futures", "fundingRate", symbol, None, year, month), dry_run=dry_run))
        return records

    def sync_rest(
        self,
        *,
        symbols: list[str] | None = None,
        markets: list[str] | None = None,
        interval: str | None = None,
        start: date | None = None,
        end: date | None = None,
        supplemental: bool = False,
        supplemental_datasets: list[str] | None = None,
        dry_run: bool = False,
    ) -> list[DataCoverageRecord]:
        """Sync klines, funding, and optional USD-M supplemental market data."""
        symbols = symbols or self.settings.data.symbols
        markets = markets or self.settings.data.markets
        interval = interval or self.settings.data.default_interval
        start = start or datetime.fromisoformat(self.settings.data.start_date).date()
        end = end or datetime.now(timezone.utc).date()
        if end < start:
            raise ValueError(f"end date {end.isoformat()} is before start date {start.isoformat()}")

        start_ms = _date_start_ms(start)
        end_ms = _date_end_ms(end)
        records: list[DataCoverageRecord] = []
        for symbol in symbols:
            for market in markets:
                records.extend(self._sync_rest_klines(symbol, market, interval, start_ms, end_ms, dry_run=dry_run))
            if "um_futures" in markets:
                records.extend(self._sync_rest_funding(symbol, start_ms, end_ms, dry_run=dry_run))
                if supplemental:
                    datasets = supplemental_datasets or list(DEFAULT_SUPPLEMENTAL_DATASETS)
                    records.extend(self._sync_rest_supplemental(symbol, interval, start_ms, end_ms, datasets, dry_run=dry_run))
        return records

    def _sync_asset(self, asset: ArchiveAsset, *, dry_run: bool) -> DataCoverageRecord:
        record = DataCoverageRecord(
            market=asset.market,
            dataset=asset.dataset,
            symbol=asset.symbol,
            interval=asset.interval,
            year=asset.year,
            month=asset.month,
            source_url=asset.url,
            checksum_verified=False,
        )
        if dry_run:
            return record
        try:
            payload = self._download(asset.url)
            checksum_text = self._download(asset.checksum_url).decode("utf-8")
            record.checksum_verified = verify_checksum(payload, checksum_text)
            if not record.checksum_verified:
                record.status = "failed"
                record.message = "checksum mismatch"
                return self._persist_record(record)
            raw_path = self._raw_path(asset)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(payload)
            frame = self._read_zip_csv(payload, asset)
            parquet_path = self._write_parquet(frame, asset)
            record.parquet_path = str(parquet_path)
            record.row_count = len(frame)
            record.status = "normalized"
            return self._persist_record(record)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                record.status = "missing"
                record.message = "archive file does not exist"
            else:
                record.status = "failed"
                record.message = str(exc)
            return self._persist_record(record)
        except Exception as exc:
            record.status = "failed"
            record.message = str(exc)
            return self._persist_record(record)

    def _download(self, url: str) -> bytes:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content

    def _sync_rest_klines(
        self,
        symbol: str,
        market: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        dry_run: bool,
    ) -> list[DataCoverageRecord]:
        base_url, endpoint = self._kline_rest_endpoint(market)
        if dry_run:
            return self._planned_rest_records(market, "klines", symbol, interval, base_url + endpoint, start_ms, end_ms)

        try:
            rows = self._fetch_rest_kline_rows(base_url, endpoint, symbol, interval, start_ms, end_ms)
            if not rows:
                return [self._persist_record(self._missing_rest_record(market, "klines", symbol, interval, base_url + endpoint, start_ms))]

            frame = normalize_kline_frame(pd.DataFrame(rows), market=market)
            frame = frame[(frame["open_time"] >= start_ms) & (frame["open_time"] <= end_ms)]
            scope = f"{market}:{symbol}:{interval}:rest"
            notes = kline_quality_notes(frame, interval_ms=interval_to_ms(interval), scope=scope)
            frame["data_quality_degraded"] = any(note.severity != "info" for note in notes)
            return self._write_rest_partitions(
                frame,
                market=market,
                dataset="klines",
                symbol=symbol,
                interval=interval,
                time_col="open_time",
                source_url=base_url + endpoint,
            )
        except Exception as exc:
            return [self._persist_record(self._failed_rest_record(market, "klines", symbol, interval, base_url + endpoint, start_ms, exc))]

    def _sync_rest_funding(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        *,
        dry_run: bool,
    ) -> list[DataCoverageRecord]:
        base_url = BINANCE_UM_FUTURES_REST_BASE
        endpoint = "/fapi/v1/fundingRate"
        if dry_run:
            return self._planned_rest_records("um_futures", "fundingRate", symbol, "funding", base_url + endpoint, start_ms, end_ms)

        try:
            rows = self._fetch_rest_funding_rows(symbol, start_ms, end_ms)
            if not rows:
                return [self._persist_record(self._missing_rest_record("um_futures", "fundingRate", symbol, "funding", base_url + endpoint, start_ms))]

            frame = pd.DataFrame({
                "calc_time": [int(row["fundingTime"]) for row in rows],
                "funding_interval_hours": [8.0 for _ in rows],
                "last_funding_rate": [float(row["fundingRate"]) for row in rows],
                "symbol": symbol,
            })
            frame = frame[(frame["calc_time"] >= start_ms) & (frame["calc_time"] <= end_ms)]
            frame = frame.sort_values("calc_time").drop_duplicates("calc_time").reset_index(drop=True)
            return self._write_rest_partitions(
                frame,
                market="um_futures",
                dataset="fundingRate",
                symbol=symbol,
                interval="funding",
                time_col="calc_time",
                source_url=base_url + endpoint,
            )
        except Exception as exc:
            return [self._persist_record(self._failed_rest_record("um_futures", "fundingRate", symbol, "funding", base_url + endpoint, start_ms, exc))]

    def _sync_rest_supplemental(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        datasets: list[str],
        *,
        dry_run: bool,
    ) -> list[DataCoverageRecord]:
        records: list[DataCoverageRecord] = []
        for dataset in datasets:
            if dataset in PRICE_KLINE_DATASETS:
                records.extend(self._sync_rest_price_klines(dataset, symbol, interval, start_ms, end_ms, dry_run=dry_run))
            elif dataset in FUTURES_DATASETS:
                records.extend(self._sync_rest_futures_data(dataset, symbol, interval, start_ms, end_ms, dry_run=dry_run))
            else:
                raise ValueError(f"Unsupported Binance supplemental dataset: {dataset}")
        return records

    def _sync_rest_price_klines(
        self,
        dataset: str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        dry_run: bool,
    ) -> list[DataCoverageRecord]:
        endpoint, symbol_param = PRICE_KLINE_DATASETS[dataset]
        source_url = BINANCE_UM_FUTURES_REST_BASE + endpoint
        if dry_run:
            return self._planned_rest_records("um_futures", dataset, symbol, interval, source_url, start_ms, end_ms)

        try:
            rows = self._fetch_rest_price_kline_rows(endpoint, symbol_param, symbol, interval, start_ms, end_ms)
            if not rows:
                return [self._persist_record(self._missing_rest_record("um_futures", dataset, symbol, interval, source_url, start_ms))]
            frame = normalize_kline_frame(pd.DataFrame(rows), market="um_futures")
            frame["symbol"] = symbol
            frame = frame[(frame["open_time"] >= start_ms) & (frame["open_time"] <= end_ms)]
            return self._write_rest_partitions(
                frame,
                market="um_futures",
                dataset=dataset,
                symbol=symbol,
                interval=interval,
                time_col="open_time",
                source_url=source_url,
            )
        except Exception as exc:
            return [self._persist_record(self._failed_rest_record("um_futures", dataset, symbol, interval, source_url, start_ms, exc))]

    def _sync_rest_futures_data(
        self,
        dataset: str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        dry_run: bool,
    ) -> list[DataCoverageRecord]:
        endpoint, symbol_param, extra_params = FUTURES_DATASETS[dataset]
        source_url = BINANCE_UM_FUTURES_REST_BASE + endpoint
        recent_start_ms = max(start_ms, end_ms - FUTURES_DATA_RECENT_WINDOW_MS + 1)
        if dry_run:
            return self._planned_rest_records("um_futures", dataset, symbol, interval, source_url, recent_start_ms, end_ms)

        try:
            rows = self._fetch_rest_futures_data_rows(endpoint, symbol_param, symbol, interval, recent_start_ms, end_ms, extra_params)
            if not rows:
                return [self._persist_record(self._missing_rest_record("um_futures", dataset, symbol, interval, source_url, recent_start_ms))]
            frame = self._normalize_futures_data_rows(pd.DataFrame(rows), symbol=symbol)
            frame = frame[(frame["timestamp"] >= recent_start_ms) & (frame["timestamp"] <= end_ms)]
            return self._write_rest_partitions(
                frame,
                market="um_futures",
                dataset=dataset,
                symbol=symbol,
                interval=interval,
                time_col="timestamp",
                source_url=source_url,
            )
        except Exception as exc:
            return [self._persist_record(self._failed_rest_record("um_futures", dataset, symbol, interval, source_url, recent_start_ms, exc))]

    def _fetch_rest_kline_rows(
        self,
        base_url: str,
        endpoint: str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list:
        interval_ms = interval_to_ms(interval)
        cursor = start_ms
        rows: list = []
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            while cursor <= end_ms:
                payload = self._request_json(
                    client,
                    base_url,
                    endpoint,
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": REST_KLINE_LIMIT,
                    },
                )
                if not isinstance(payload, list):
                    raise RuntimeError(f"Unexpected Binance kline response for {symbol}: {payload!r}")
                if not payload:
                    break
                rows.extend(payload)
                last_open = int(payload[-1][0])
                next_cursor = last_open + interval_ms
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                if len(payload) < REST_KLINE_LIMIT:
                    break
        return rows

    def _fetch_rest_funding_rows(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        cursor = start_ms
        rows: list[dict] = []
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            while cursor <= end_ms:
                payload = self._request_json(
                    client,
                    BINANCE_UM_FUTURES_REST_BASE,
                    "/fapi/v1/fundingRate",
                    {
                        "symbol": symbol,
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": REST_FUNDING_LIMIT,
                    },
                )
                if not isinstance(payload, list):
                    raise RuntimeError(f"Unexpected Binance funding response for {symbol}: {payload!r}")
                if not payload:
                    break
                rows.extend(payload)
                last_time = int(payload[-1]["fundingTime"])
                next_cursor = last_time + 1
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                if len(payload) < REST_FUNDING_LIMIT:
                    break
        return rows

    def _fetch_rest_price_kline_rows(
        self,
        endpoint: str,
        symbol_param: str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list:
        interval_ms = interval_to_ms(interval)
        cursor = start_ms
        rows: list = []
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            while cursor <= end_ms:
                payload = self._request_json(
                    client,
                    BINANCE_UM_FUTURES_REST_BASE,
                    endpoint,
                    {
                        symbol_param: symbol,
                        "interval": interval,
                        "startTime": cursor,
                        "endTime": end_ms,
                        "limit": REST_PRICE_KLINE_LIMIT,
                    },
                )
                if not isinstance(payload, list):
                    raise RuntimeError(f"Unexpected Binance price kline response for {symbol}: {payload!r}")
                if not payload:
                    break
                rows.extend(payload)
                last_open = int(payload[-1][0])
                next_cursor = last_open + interval_ms
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                if len(payload) < REST_PRICE_KLINE_LIMIT:
                    break
        return rows

    def _fetch_rest_futures_data_rows(
        self,
        endpoint: str,
        symbol_param: str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        extra_params: dict[str, str],
    ) -> list[dict]:
        period_ms = interval_to_ms(interval)
        cursor = start_ms
        rows: list[dict] = []
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            while cursor <= end_ms:
                params = {
                    symbol_param: symbol,
                    "period": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": REST_FUTURES_DATA_LIMIT,
                    **extra_params,
                }
                payload = self._request_json(client, BINANCE_UM_FUTURES_REST_BASE, endpoint, params)
                if not isinstance(payload, list):
                    raise RuntimeError(f"Unexpected Binance futures data response for {symbol}: {payload!r}")
                if not payload:
                    break
                rows.extend(payload)
                last_time = int(payload[-1]["timestamp"])
                next_cursor = last_time + period_ms
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
                if len(payload) < REST_FUTURES_DATA_LIMIT:
                    break
        return rows

    def _normalize_futures_data_rows(self, frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
        frame = frame.copy()
        frame["symbol"] = frame.get("symbol", symbol)
        if "pair" in frame.columns:
            frame["pair"] = frame["pair"].fillna(symbol)
        for col in frame.columns:
            if col in {"symbol", "pair", "contractType"}:
                continue
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["timestamp"] = frame["timestamp"].map(normalize_timestamp_ms)
        return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    def _request_json(self, client: httpx.Client, base_url: str, endpoint: str, params: dict) -> object:
        response = client.get(f"{base_url}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()

    def _kline_rest_endpoint(self, market: str) -> tuple[str, str]:
        if market == "spot":
            return BINANCE_SPOT_REST_BASE, "/api/v3/klines"
        if market == "um_futures":
            return BINANCE_UM_FUTURES_REST_BASE, "/fapi/v1/klines"
        raise ValueError(f"Unsupported Binance market for REST sync: {market}")

    def _read_zip_csv(self, payload: bytes, asset: ArchiveAsset) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv_name = archive.namelist()[0]
            with archive.open(csv_name) as handle:
                frame = pd.read_csv(handle, header=None)
        if asset.dataset == "fundingRate":
            return normalize_funding_frame(frame, symbol=asset.symbol)
        normalized = normalize_kline_frame(frame, market=asset.market)
        scope = f"{asset.market}:{asset.symbol}:{asset.interval}:{asset.year}-{asset.month:02d}"
        notes = kline_quality_notes(normalized, interval_ms=interval_to_ms(asset.interval or "5m"), scope=scope)
        normalized["data_quality_degraded"] = any(note.severity != "info" for note in notes)
        return normalized

    def _raw_path(self, asset: ArchiveAsset) -> Path:
        return self.settings.data.raw_dir / asset.path

    def _write_parquet(self, frame: pd.DataFrame, asset: ArchiveAsset) -> Path:
        interval = asset.interval or "funding"
        path = (
            self.settings.data.parquet_dir
            / f"market={asset.market}"
            / f"dataset={asset.dataset}"
            / f"symbol={asset.symbol}"
            / f"interval={interval}"
            / f"year={asset.year:04d}"
            / f"month={asset.month:02d}"
            / "data.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path

    def _write_rest_partitions(
        self,
        frame: pd.DataFrame,
        *,
        market: str,
        dataset: str,
        symbol: str,
        interval: str,
        time_col: str,
        source_url: str,
    ) -> list[DataCoverageRecord]:
        if frame.empty:
            return []
        timestamps = pd.to_datetime(frame[time_col].astype("int64"), unit="ms", utc=True)
        records: list[DataCoverageRecord] = []
        for (year, month), idx in frame.groupby([timestamps.dt.year, timestamps.dt.month], sort=True).groups.items():
            partition = frame.loc[idx].sort_values(time_col).drop_duplicates(time_col).reset_index(drop=True)
            path = (
                self.settings.data.parquet_dir
                / f"market={market}"
                / f"dataset={dataset}"
                / f"symbol={symbol}"
                / f"interval={interval}"
                / f"year={year:04d}"
                / f"month={month:02d}"
                / "data.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            partition.to_parquet(path, index=False)
            records.append(self._persist_record(DataCoverageRecord(
                market=market,
                dataset=dataset,
                symbol=symbol,
                interval=interval,
                year=int(year),
                month=int(month),
                source_url=source_url,
                checksum_verified=False,
                parquet_path=str(path),
                row_count=len(partition),
                status="normalized",
                message="REST source; checksum not available",
            )))
        return records

    def _planned_rest_records(
        self,
        market: str,
        dataset: str,
        symbol: str,
        interval: str,
        source_url: str,
        start_ms: int,
        end_ms: int,
    ) -> list[DataCoverageRecord]:
        start_day = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()
        end_day = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).date()
        records = []
        for year, month in month_range(start_day, end_day):
            records.append(DataCoverageRecord(
                market=market,
                dataset=dataset,
                symbol=symbol,
                interval=interval,
                year=year,
                month=month,
                source_url=source_url,
                checksum_verified=False,
            ))
        return records

    def _missing_rest_record(
        self,
        market: str,
        dataset: str,
        symbol: str,
        interval: str,
        source_url: str,
        start_ms: int,
    ) -> DataCoverageRecord:
        start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        return DataCoverageRecord(
            market=market,
            dataset=dataset,
            symbol=symbol,
            interval=interval,
            year=start_dt.year,
            month=start_dt.month,
            source_url=source_url,
            checksum_verified=False,
            status="missing",
            message="REST API returned no rows",
        )

    def _failed_rest_record(
        self,
        market: str,
        dataset: str,
        symbol: str,
        interval: str,
        source_url: str,
        start_ms: int,
        exc: Exception,
    ) -> DataCoverageRecord:
        start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        message = str(exc)
        if isinstance(exc, httpx.HTTPStatusError):
            body = exc.response.text[:240].replace("\n", " ")
            message = f"HTTP {exc.response.status_code}: {body}"
        return DataCoverageRecord(
            market=market,
            dataset=dataset,
            symbol=symbol,
            interval=interval,
            year=start_dt.year,
            month=start_dt.month,
            source_url=source_url,
            checksum_verified=False,
            status="failed",
            message=message,
        )

    def _persist_record(self, record: DataCoverageRecord) -> DataCoverageRecord:
        if self.store is not None:
            coverage_id = f"{record.market}:{record.dataset}:{record.symbol}:{record.interval}:{record.year}:{record.month}"
            self.store.save_coverage(coverage_id, record.model_dump(mode="json"))
        return record
