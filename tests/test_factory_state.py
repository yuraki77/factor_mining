"""Factory arrival detection must gate discovery on genuine new OOS data.

The factory's whole premise is that trial budgets are only spent when fresh
out-of-sample bars have arrived — if these triggers misfire, the factory
degenerates back into re-mining frozen data (misfire = wasted N inflating the
DSR penalty; missed fire = the factory never runs).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factor_mining import factory
from factor_mining.config import DataConfig, Settings
from factor_mining.data.loader import data_end_ms
from factor_mining.storage import MetadataStore

_DAY_MS = 86_400_000


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data=DataConfig(
            symbols=["BTCUSDT"],
            markets=["spot"],
            raw_dir=tmp_path / "raw",
            parquet_dir=tmp_path / "parquet",
            sqlite_path=tmp_path / "meta.sqlite3",
        )
    )


def _write_parquet(settings: Settings, open_times: list[int]) -> None:
    parquet_dir = settings.data.parquet_dir
    parquet_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "open_time": open_times,
            "open": [1.0] * len(open_times),
            "high": [1.0] * len(open_times),
            "low": [1.0] * len(open_times),
            "close": [1.0] * len(open_times),
            "volume": [1.0] * len(open_times),
        }
    )
    frame.to_parquet(parquet_dir / f"BTCUSDT_{settings.data.default_interval}.parquet")


def test_data_end_ms_is_max_open_time_without_hashing(tmp_path: Path) -> None:
    """The nightly poll must see the newest bar; rows may be unordered on disk."""
    settings = _settings(tmp_path)
    _write_parquet(settings, [100, 300, 200])
    assert data_end_ms(settings, symbol="BTCUSDT", market="spot") == 300


def test_data_end_ms_missing_parquet_reads_as_no_data(tmp_path: Path) -> None:
    """A missing warehouse must read as 'no data yet' (0), not crash the worker."""
    settings = _settings(tmp_path)
    assert data_end_ms(settings, symbol="BTCUSDT", market="spot") == 0


def test_discovery_fires_at_go_live_when_no_extent_recorded() -> None:
    """First factory run has no baseline — it must fire once to establish one."""
    assert factory.discovery_due({"BTCUSDT/spot": 500 * _DAY_MS}, {}, min_new_days=1.0)


def test_discovery_waits_for_min_new_days() -> None:
    last = {"BTCUSDT/spot": 100 * _DAY_MS}
    below = {"BTCUSDT/spot": 100 * _DAY_MS + _DAY_MS // 2}
    at = {"BTCUSDT/spot": 101 * _DAY_MS}
    assert not factory.discovery_due(below, last, min_new_days=1.0)
    assert factory.discovery_due(at, last, min_new_days=1.0)


def test_lagging_feed_holds_discovery_back() -> None:
    """min-over-keys: one stale feed must block the round, otherwise the budget
    is spent on data that is fresh for one symbol and frozen for the other."""
    last = {"BTCUSDT/spot": 100 * _DAY_MS, "ETHUSDT/spot": 100 * _DAY_MS}
    current = {"BTCUSDT/spot": 102 * _DAY_MS, "ETHUSDT/spot": 100 * _DAY_MS + _DAY_MS // 4}
    assert not factory.discovery_due(current, last, min_new_days=1.0)


def test_no_data_at_all_never_fires() -> None:
    assert not factory.discovery_due({"BTCUSDT/spot": 0}, {}, min_new_days=1.0)


def test_factory_state_roundtrip_and_discovery_recording(tmp_path: Path) -> None:
    """Extent baseline must survive process restarts — it lives in the store,
    not in worker memory, so a crashed supervisor cannot double-trigger."""
    store = MetadataStore(tmp_path / "meta.sqlite3")
    state = factory.load_factory_state(store)
    assert state["last_discovery_extent_ms"] == {}

    factory.record_discovery_completed(
        store, run_id="run_1", extents_ms={"BTCUSDT/spot": 123, "ETHUSDT/spot": 0}
    )
    reloaded = factory.load_factory_state(store)
    # zero-extent keys are dropped: recording "no data" as a baseline would
    # make the next real bar look like an arbitrarily large arrival
    assert reloaded["last_discovery_extent_ms"] == {"BTCUSDT/spot": 123}
    assert reloaded["last_discovery_run_id"] == "run_1"

    factory.record_recheck_completed(store, run_id="run_2")
    reloaded = factory.load_factory_state(store)
    assert reloaded["last_recheck_run_id"] == "run_2"
    assert reloaded["last_discovery_run_id"] == "run_1"
