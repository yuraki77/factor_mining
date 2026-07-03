from datetime import date, datetime, timezone

import pandas as pd

from factor_mining.config import DataConfig, Settings
from factor_mining.data.binance import BinanceArchiveClient
from factor_mining.data.loader import funding_event_zscore_to_frame, load_frame, load_funding, load_supplemental_features
from factor_mining.data.quality import interval_to_ms, kline_quality_notes, normalize_kline_frame, normalize_timestamp_ms


def test_spot_microsecond_timestamp_is_normalized_to_milliseconds() -> None:
    assert normalize_timestamp_ms(1_735_689_600_000_000) == 1_735_689_600_000
    assert normalize_timestamp_ms(1_735_689_600_000) == 1_735_689_600_000


def test_kline_quality_checks_detect_bad_rows() -> None:
    frame = pd.DataFrame(
        [
            [1_000_000, 100, 101, 99, 100, 1, 1_299_999, 100, 1, 1, 100, 0],
            [1_300_000, 100, 90, 99, 100, 0, 1_599_999, 0, 1, 1, 100, 0],
            [1_300_000, -1, 101, 99, 100, 1, 1_599_999, 100, 1, 1, 100, 0],
        ]
    )
    normalized = normalize_kline_frame(frame, market="spot")
    notes = kline_quality_notes(normalized, interval_ms=interval_to_ms("5m"), scope="test")
    messages = {note.message for note in notes}
    assert "duplicate timestamps detected" in messages
    assert "negative OHLCV values detected" in messages
    assert "OHLC invariant failed" in messages


def test_kline_normalizer_handles_optional_header_row() -> None:
    frame = pd.DataFrame(
        [
            ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"],
            [1_735_689_600_000_000, 100, 101, 99, 100, 1, 1_735_689_899_999_999, 100, 1, 1, 100, 0],
        ]
    )
    normalized = normalize_kline_frame(frame, market="spot")
    assert len(normalized) == 1
    assert normalized.loc[0, "open_time"] == 1_735_689_600_000


def test_load_frame_reads_partitioned_parquet_warehouse(tmp_path) -> None:
    parquet_root = tmp_path / "parquet"
    partition = (
        parquet_root
        / "market=um_futures"
        / "dataset=klines"
        / "symbol=BTCUSDT"
        / "interval=5m"
        / "year=2026"
        / "month=04"
    )
    partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "open_time": [300_000, 0],
            "open": [101.0, 100.0],
            "high": [102.0, 101.0],
            "low": [100.0, 99.0],
            "close": [101.5, 100.5],
            "volume": [2.0, 1.0],
            "quote_volume": [203.0, 100.5],
            "market": ["um_futures", "um_futures"],
        }
    ).to_parquet(partition / "data.parquet", index=False)

    settings = Settings(data=DataConfig(symbols=["BTCUSDT"], parquet_dir=parquet_root, start_date="1970-01-01"))
    loaded = load_frame(settings, symbol="BTCUSDT", market="um_futures", tail=1)

    assert loaded.loc[0, "open_time"] == 300_000
    assert loaded.loc[0, "market"] == "um_futures"
    assert not bool(loaded.loc[0, "data_quality_degraded"])


def test_load_frame_filters_rows_before_configured_start_date(tmp_path) -> None:
    parquet_root = tmp_path / "parquet"
    partition = (
        parquet_root
        / "market=spot"
        / "dataset=klines"
        / "symbol=BTCUSDT"
        / "interval=5m"
        / "year=2019"
        / "month=12"
    )
    partition.mkdir(parents=True)
    cutoff_ms = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    pd.DataFrame(
        {
            "open_time": [cutoff_ms - 300_000, cutoff_ms, cutoff_ms + 300_000],
            "open": [99.0, 100.0, 101.0],
            "high": [100.0, 101.0, 102.0],
            "low": [98.0, 99.0, 100.0],
            "close": [99.5, 100.5, 101.5],
            "volume": [1.0, 1.0, 1.0],
            "quote_volume": [99.5, 100.5, 101.5],
            "market": ["spot", "spot", "spot"],
        }
    ).to_parquet(partition / "data.parquet", index=False)

    settings = Settings(data=DataConfig(symbols=["BTCUSDT"], markets=["spot"], parquet_dir=parquet_root, start_date="2020-01-01"))
    loaded = load_frame(settings, symbol="BTCUSDT", market="spot")

    assert loaded["open_time"].tolist() == [cutoff_ms, cutoff_ms + 300_000]


def test_funding_event_zscore_uses_8h_events_not_repeated_bars() -> None:
    eight_hours = 8 * 60 * 60 * 1000
    five_minutes = 5 * 60 * 1000
    frame = pd.DataFrame({"open_time": [-five_minutes, 0, eight_hours, eight_hours + five_minutes, 2 * eight_hours]})
    funding = pd.DataFrame({
        "calc_time": [0, eight_hours, 2 * eight_hours],
        "funding_interval_hours": [8.0, 8.0, 8.0],
        "last_funding_rate": [0.0, 0.01, -0.01],
    })

    aligned = funding_event_zscore_to_frame(frame, funding, lookback_events=2)

    assert aligned.iloc[0] == 0.0
    assert aligned.iloc[2] > 0
    assert aligned.iloc[3] == aligned.iloc[2]
    assert aligned.iloc[4] < 0


def test_rest_sync_writes_partitioned_klines_and_8h_funding(tmp_path, monkeypatch) -> None:
    start_ms = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    settings = Settings(data=DataConfig(
        symbols=["BTCUSDT"],
        markets=["um_futures"],
        default_interval="5m",
        raw_dir=tmp_path / "raw",
        parquet_dir=tmp_path / "parquet",
        sqlite_path=tmp_path / "factor_mining.sqlite3",
    ))
    client = BinanceArchiveClient(settings)

    def fake_request_json(_http_client, _base_url, endpoint, _params):
        if endpoint == "/fapi/v1/klines":
            return [
                [start_ms, "100", "101", "99", "100.5", "10", start_ms + 299_999, "1005", 12, "5", "502.5", "0"],
                [start_ms + 300_000, "100.5", "102", "100", "101", "11", start_ms + 599_999, "1111", 14, "6", "606", "0"],
            ]
        if endpoint == "/fapi/v1/fundingRate":
            return [
                {"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingTime": start_ms},
                {"symbol": "BTCUSDT", "fundingRate": "-0.0002", "fundingTime": start_ms + 28_800_000},
            ]
        raise AssertionError(endpoint)

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    records = client.sync_rest(
        symbols=["BTCUSDT"],
        markets=["um_futures"],
        interval="5m",
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
    )

    assert {(r.dataset, r.interval, r.row_count) for r in records} == {
        ("klines", "5m", 2),
        ("fundingRate", "funding", 2),
    }
    loaded = load_frame(settings, symbol="BTCUSDT", market="um_futures")
    funding = load_funding(settings, symbol="BTCUSDT")

    assert len(loaded) == 2
    assert funding is not None
    assert funding["funding_interval_hours"].tolist() == [8.0, 8.0]


def test_rest_sync_writes_supplemental_features(tmp_path, monkeypatch) -> None:
    start_ms = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    interval_ms = 300_000
    settings = Settings(data=DataConfig(
        symbols=["BTCUSDT"],
        markets=["um_futures"],
        default_interval="5m",
        raw_dir=tmp_path / "raw",
        parquet_dir=tmp_path / "parquet",
        sqlite_path=tmp_path / "factor_mining.sqlite3",
    ))
    client = BinanceArchiveClient(settings)

    def kline_rows(base: float) -> list[list]:
        rows = []
        for i in range(40):
            open_time = start_ms + i * interval_ms
            close = base + i
            rows.append([open_time, str(close - 0.2), str(close + 0.4), str(close - 0.5), str(close), "10", open_time + interval_ms - 1, "1000", 12, "5", "500", "0"])
        return rows

    def futures_rows(endpoint: str) -> list[dict]:
        rows = []
        for i in range(40):
            timestamp = start_ms + i * interval_ms
            if endpoint == "/futures/data/openInterestHist":
                rows.append({"symbol": "BTCUSDT", "sumOpenInterest": str(1000 + i), "sumOpenInterestValue": str(10_000 + i * 50), "timestamp": timestamp})
            elif endpoint in {
                "/futures/data/globalLongShortAccountRatio",
                "/futures/data/topLongShortAccountRatio",
                "/futures/data/topLongShortPositionRatio",
            }:
                rows.append({"symbol": "BTCUSDT", "longShortRatio": str(1.0 + i * 0.01), "longAccount": "0.55", "shortAccount": "0.45", "timestamp": timestamp})
            elif endpoint == "/futures/data/takerlongshortRatio":
                rows.append({"buySellRatio": str(1.1 + i * 0.01), "buyVol": str(100 + i), "sellVol": str(80 + i), "timestamp": timestamp})
            elif endpoint == "/futures/data/basis":
                rows.append({"pair": "BTCUSDT", "contractType": "PERPETUAL", "basisRate": str(0.0001 * i), "basis": str(i), "indexPrice": "100", "futuresPrice": "101", "annualizedBasisRate": "", "timestamp": timestamp})
            else:
                raise AssertionError(endpoint)
        return rows

    def fake_request_json(_http_client, _base_url, endpoint, _params):
        if endpoint == "/fapi/v1/klines":
            return kline_rows(100.0)
        if endpoint == "/fapi/v1/fundingRate":
            return [{"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingTime": start_ms}]
        if endpoint == "/fapi/v1/markPriceKlines":
            return kline_rows(101.0)
        if endpoint == "/fapi/v1/indexPriceKlines":
            return kline_rows(100.0)
        if endpoint == "/fapi/v1/premiumIndexKlines":
            return kline_rows(0.001)
        return futures_rows(endpoint)

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    records = client.sync_rest(
        symbols=["BTCUSDT"],
        markets=["um_futures"],
        interval="5m",
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        supplemental=True,
    )

    normalized = {(record.dataset, record.status) for record in records}
    assert ("markPriceKlines", "normalized") in normalized
    assert ("openInterestHist", "normalized") in normalized
    assert ("globalLongShortAccountRatio", "normalized") in normalized
    frame = load_frame(settings, symbol="BTCUSDT", market="um_futures")
    supplemental, meta = load_supplemental_features(settings, frame, symbol="BTCUSDT", market="um_futures")

    assert "mark_index_basis" in supplemental
    assert "global_ls_account_ratio" in supplemental
    assert "taker_buy_sell_imbalance" in supplemental
    assert meta["mark_index_basis"]["family"] == "funding_basis"
    assert meta["taker_buy_sell_imbalance"]["family"] == "volume_confirmation"


def test_supplemental_features_include_spot_perp_basis_from_existing_klines(tmp_path) -> None:
    parquet_root = tmp_path / "parquet"
    start_ms = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    interval_ms = 300_000

    def write_klines(market: str, close_offset: float) -> None:
        rows = []
        for idx in range(320):
            open_time = start_ms + idx * interval_ms
            close = 100.0 + idx * 0.01 + close_offset
            rows.append({
                "open_time": open_time,
                "open": close - 0.01,
                "high": close + 0.02,
                "low": close - 0.02,
                "close": close,
                "volume": 10.0,
                "close_time": open_time + interval_ms - 1,
                "quote_volume": close * 10.0,
                "trade_count": 12,
                "taker_buy_volume": 5.0,
                "taker_buy_quote_volume": close * 5.0,
                "market": market,
                "data_quality_degraded": False,
            })
        partition = (
            parquet_root
            / f"market={market}"
            / "dataset=klines"
            / "symbol=BTCUSDT"
            / "interval=5m"
            / "year=2026"
            / "month=01"
        )
        partition.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(partition / "data.parquet", index=False)

    write_klines("spot", close_offset=0.0)
    write_klines("um_futures", close_offset=0.5)
    settings = Settings(data=DataConfig(symbols=["BTCUSDT"], markets=["spot", "um_futures"], parquet_dir=parquet_root))
    futures = load_frame(settings, symbol="BTCUSDT", market="um_futures")

    supplemental, meta = load_supplemental_features(settings, futures, symbol="BTCUSDT", market="um_futures")

    assert "spot_perp_basis" in supplemental
    assert "spot_perp_basis_z_288" in supplemental
    assert "spot_perp_basis_chg_12" in supplemental
    assert meta["spot_perp_basis"]["family"] == "funding_basis"
    assert meta["spot_perp_basis"]["direction"] == "negative_when_high"


def test_data_extent_content_hash_detects_resynced_warehouse(tmp_path) -> None:
    """WHY (I3): archives pinned reproduction to row count + open_time extent
    only, so a silently re-synced parquet with identical shape but different
    prices passed the manifest. The content hash must change when bytes
    change even though rows and extent are identical."""
    import pandas as pd

    from factor_mining.config import Settings
    from factor_mining.data.loader import data_extent

    settings = Settings()
    settings.data.parquet_dir = tmp_path
    settings.data.symbols = ["TESTUSDT"]
    path = tmp_path / f"TESTUSDT_{settings.data.default_interval}.parquet"

    frame = pd.DataFrame({"open_time": [0, 300, 600], "close": [1.0, 2.0, 3.0]})
    frame.to_parquet(path)
    first = data_extent(settings, symbol="TESTUSDT", market="um_futures")

    frame_resynced = frame.assign(close=[1.0, 2.5, 3.0])
    frame_resynced.to_parquet(path)
    second = data_extent(settings, symbol="TESTUSDT", market="um_futures")

    assert first["rows"] == second["rows"]
    assert first["data_start_ms"] == second["data_start_ms"]
    assert first["data_end_ms"] == second["data_end_ms"]
    assert first["content_sha256"] != second["content_sha256"]


def test_checkpoint_fingerprint_detects_content_change_with_identical_shape() -> None:
    """WHY (I3): resuming a run against a re-synced warehouse with the same
    row count and extent must invalidate stage checkpoints (fail-loud on
    resume) rather than silently mixing datasets."""
    import pandas as pd

    from factor_mining.config import Settings
    from factor_mining.pipeline import _checkpoint_fingerprint

    frame = pd.DataFrame({
        "open_time": [0, 300, 600],
        "open": [1.0, 2.0, 3.0],
        "close": [1.0, 2.0, 3.0],
    })
    resynced = frame.assign(close=[1.0, 2.1, 3.0])
    kwargs = {"run_args": {}, "symbol": "BTCUSDT", "market": "um_futures"}
    original = _checkpoint_fingerprint(Settings(), frame=frame, **kwargs)
    changed = _checkpoint_fingerprint(Settings(), frame=resynced, **kwargs)

    assert original["row_count"] == changed["row_count"]
    assert original["open_time_min"] == changed["open_time_min"]
    assert original["open_time_max"] == changed["open_time_max"]
    assert original != changed


def test_supplemental_alignment_never_uses_future_stamped_records() -> None:
    """G2 (P2-5): the bar at open_time T must receive the last supplemental
    record stamped <= T. A record stamped after T leaking backward would be
    lookahead no signal lag can repair."""
    import pandas as pd

    from factor_mining.data.loader import _aligned_dataset_series

    frame = pd.DataFrame({"open_time": [0, 300, 600, 900]})
    dataset = pd.DataFrame({"timestamp": [299, 601], "value": [1.0, 2.0]})

    aligned = _aligned_dataset_series(frame, dataset, "timestamp", "value")

    # Bar 0: nothing stamped yet. Bar 300: sees the 299 record. Bar 600: the
    # 601 record is in the future — must still carry 1.0. Bar 900: sees 2.0.
    assert pd.isna(aligned.iloc[0])
    assert aligned.iloc[1] == 1.0
    assert aligned.iloc[2] == 1.0
    assert aligned.iloc[3] == 2.0


def test_funding_event_zscore_changes_only_at_event_boundaries() -> None:
    """G2 (P2-5): funding statistics are computed on event snapshots, not on
    forward-filled per-bar copies — and a bar before an event's calc_time
    must not reflect that event."""
    import pandas as pd

    from factor_mining.data.loader import funding_event_zscore_to_frame

    frame = pd.DataFrame({"open_time": list(range(0, 3000, 300))})
    funding = pd.DataFrame({
        "calc_time": [500, 1500, 2500],
        "last_funding_rate": [0.01, 0.03, -0.02],
    })

    z = funding_event_zscore_to_frame(frame, funding, lookback_events=3)

    # Bars before the first event stay 0 (no backfill of future events).
    assert (z.iloc[:2] == 0.0).all()
    # Value is constant between events (event-indexed stats, ffilled).
    assert z.iloc[2] == z.iloc[3] == z.iloc[4]
    assert z.iloc[6] == z.iloc[7] == z.iloc[8]
