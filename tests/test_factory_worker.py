"""Supervisor decision + safety rails.

The worker is what makes the factory continuous without making it reckless:
it must never start a second ~20GB pipeline (shared store with the
backtest_master lab daemon), never let a dead run's 'running' row wedge the
gate forever, and always leave a crashed discovery resumable in the pre-sync
window where its checkpoints still match the data.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta

from factor_mining import factory, worker
from factor_mining.config import DataConfig, FactoryConfig, Settings
from factor_mining.storage import MetadataStore


def _settings(tmp_path, **factory_overrides) -> Settings:
    return Settings(
        data=DataConfig(
            symbols=["BTCUSDT"],
            markets=["spot"],
            raw_dir=tmp_path / "raw",
            parquet_dir=tmp_path / "parquet",
            sqlite_path=tmp_path / "meta.sqlite3",
        ),
        factory=FactoryConfig(enabled=True, **factory_overrides),
    )


def _store(settings: Settings) -> MetadataStore:
    return MetadataStore(settings.data.sqlite_path)


def test_decide_skips_when_any_pipeline_is_active(tmp_path) -> None:
    """The gate covers foreign runs too — the lab daemon shares this store,
    and two concurrent pipelines is the machine-killing failure mode."""
    settings = _settings(tmp_path)
    store = _store(settings)
    store.create_pipeline_run("foreign_run", {})
    assert worker.decide_action(store, settings) == "skip_active_run"


def test_decide_prefers_discovery_when_new_data_arrived(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, min_new_days=1.0)
    store = _store(settings)
    day_ms = 86_400_000
    factory.record_discovery_completed(store, run_id="r0", extents_ms={"BTCUSDT/spot": 100 * day_ms})
    monkeypatch.setattr(factory, "current_extents_ms", lambda _s: {"BTCUSDT/spot": 102 * day_ms})
    assert worker.decide_action(store, settings) == "discovery"


def test_decide_falls_back_to_recheck_then_idle(tmp_path, monkeypatch) -> None:
    """No new data → recheck survivors, but not more often than the
    configured interval: rechecks are cheap, not free."""
    settings = _settings(tmp_path, min_new_days=1.0, recheck_interval_days=1)
    store = _store(settings)
    day_ms = 86_400_000
    factory.record_discovery_completed(store, run_id="r0", extents_ms={"BTCUSDT/spot": 100 * day_ms})
    monkeypatch.setattr(factory, "current_extents_ms", lambda _s: {"BTCUSDT/spot": 100 * day_ms})

    assert worker.decide_action(store, settings) == "recheck"
    factory.record_recheck_completed(store, run_id="r1")
    assert worker.decide_action(store, settings) == "idle"
    later = datetime.now(UTC) + timedelta(days=2)
    assert worker.decide_action(store, settings, now=later) == "recheck"


def test_stale_running_row_is_failed_not_left_wedging_the_gate(tmp_path) -> None:
    settings = _settings(tmp_path, max_run_hours=1.0)
    store = _store(settings)
    store.create_pipeline_run("dead_run", {})
    stale = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    import sqlite3

    with sqlite3.connect(settings.data.sqlite_path) as conn:
        conn.execute("update pipeline_runs set started_at = ? where run_id = ?", (stale, "dead_run"))

    worker._mark_stale_run_failed(store, settings)
    assert store.pipeline_run_status("dead_run") == "failed"
    assert worker.decide_action(store, settings) != "skip_active_run"


def test_escalate_stops_child_and_fails_the_run(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = _store(settings)
    store.create_pipeline_run("hung_run", {})
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    code = worker._escalate(process, store, run_id="hung_run", reason="max_run_hours", grace_seconds=0.2)

    assert process.poll() is not None, "child must be dead after escalation"
    assert code != 0
    # request_pipeline_stop flipped it to 'stopping'; the child died without
    # recording an end state, so the supervisor must fail it explicitly.
    assert store.pipeline_run_status("hung_run") == "failed"


def test_failed_discovery_is_queued_for_resume_and_success_records_extents(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    store = _store(settings)
    day_ms = 86_400_000
    monkeypatch.setattr(factory, "current_extents_ms", lambda _s: {"BTCUSDT/spot": 200 * day_ms})

    monkeypatch.setattr(worker, "_run_supervised_child", lambda args, s, st: (1, "crashed_run"))
    assert worker.run_discovery(settings, store) == 1
    state = factory.load_factory_state(store)
    assert state["pending_resume_run_id"] == "crashed_run"
    assert state["last_discovery_extent_ms"] == {}, "failed round must not advance the baseline"

    monkeypatch.setattr(worker, "_run_supervised_child", lambda args, s, st: (0, "good_run"))
    assert worker.run_discovery(settings, store) == 0
    state = factory.load_factory_state(store)
    assert state["last_discovery_run_id"] == "good_run"
    assert state["last_discovery_extent_ms"] == {"BTCUSDT/spot": 200 * day_ms}


def test_discovery_child_mines_llm_hypotheses_unless_disabled(tmp_path, monkeypatch) -> None:
    """Production discovery uses LLM-generated hypotheses; the knob exists so
    an environment without DEEPSEEK_API_KEY can fall back to deterministic
    defaults instead of failing every round."""
    calls: list[list[str]] = []
    monkeypatch.setattr(worker, "_run_supervised_child", lambda args, s, st: calls.append(args) or (0, "r"))
    monkeypatch.setattr(factory, "current_extents_ms", lambda _s: {})

    settings = _settings(tmp_path)  # use_llm defaults True
    worker.run_discovery(settings, _store(settings))
    assert "--llm" in calls[0]

    calls.clear()
    settings = _settings(tmp_path, use_llm=False)
    worker.run_discovery(settings, _store(settings))
    assert "--llm" not in calls[0]


def test_worker_cap_flows_to_both_child_kinds(tmp_path, monkeypatch) -> None:
    """max_workers is the lever that lowers the RAM peak (fewer pool
    processes) rather than merely detecting it; both discovery and recheck
    children must inherit it, and no flag may leak when unset."""
    calls: list[list[str]] = []
    monkeypatch.setattr(worker, "_run_supervised_child", lambda args, s, st: calls.append(args) or (0, "r"))
    monkeypatch.setattr(factory, "current_extents_ms", lambda _s: {})

    settings = _settings(tmp_path, max_workers=4)
    worker.run_discovery(settings, _store(settings))
    worker.run_recheck(settings, _store(settings))
    assert all("--workers" in c and "4" in c for c in calls)

    calls.clear()
    settings = _settings(tmp_path)  # max_workers unset
    worker.run_discovery(settings, _store(settings))
    assert "--workers" not in calls[0]


class _FakeProc:
    def __init__(self, rss: int, children: list["_FakeProc"] | None = None, dying: bool = False) -> None:
        self._rss = rss
        self._children = children or []
        self._dying = dying

    def memory_info(self):
        if self._dying:
            raise RuntimeError("process already gone")
        from types import SimpleNamespace

        return SimpleNamespace(rss=self._rss)

    def children(self, recursive: bool = False):
        return list(self._children)


def test_tree_rss_counts_the_pool_not_just_the_parent() -> None:
    """The pipeline's memory lives in its ProcessPoolExecutor workers; a
    parent-only reading under-measures by roughly the whole pool and the
    watchdog would never fire before the machine swap-storms."""
    parent = _FakeProc(1_000, children=[_FakeProc(7_000), _FakeProc(9_000), _FakeProc(5_000, dying=True)])
    assert worker._tree_rss_bytes(parent) == 17_000  # dying worker skipped, not fatal


def test_supervise_trips_on_process_tree_rss(tmp_path, monkeypatch) -> None:
    settings = _settings(tmp_path, max_rss_gb=0.001)
    store = _store(settings)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    monkeypatch.setattr(worker, "_tree_rss_bytes", lambda proc: 10**12)
    seen: dict = {}

    def fake_escalate(proc, st, *, run_id, reason, grace_seconds=0.1):
        seen["reason"] = reason
        proc.kill()
        proc.wait()
        return -9

    monkeypatch.setattr(worker, "_escalate", fake_escalate)
    code = worker._supervise(process, store, settings, run_id=None)
    assert seen["reason"] == "max_rss_gb"
    assert code == -9


def test_abandon_after_sync_prunes_only_that_runs_checkpoints(tmp_path) -> None:
    """Post-sync the I3 fingerprint can never match again — the marker and
    checkpoints must go. Underscores in run ids are LIKE wildcards; a sloppy
    prefix delete would also erase a sibling run's checkpoints."""
    settings = _settings(tmp_path)
    store = _store(settings)
    worker._set_pending_resume(store, "run_1")
    store.save_artifact("checkpoint:run_1:stage_a", "pipeline_checkpoint", {"x": 1})
    store.save_artifact("checkpoint:runX1:stage_a", "pipeline_checkpoint", {"x": 2})

    worker._abandon_pending_resume(store, reason="test")

    assert factory.load_factory_state(store).get("pending_resume_run_id") is None
    assert store.load_artifact("checkpoint:run_1:stage_a") is None
    assert store.load_artifact("checkpoint:runX1:stage_a") is not None


def test_resume_is_attempted_only_in_pre_sync_window_states(tmp_path, monkeypatch) -> None:
    """A pending run that is still 'running' belongs to a live child —
    resuming it would double-run the pipeline."""
    settings = _settings(tmp_path)
    store = _store(settings)
    store.create_pipeline_run("busy_run", {})
    worker._set_pending_resume(store, "busy_run")
    calls: list[list[str]] = []
    monkeypatch.setattr(worker, "_run_supervised_child", lambda args, s, st: calls.append(args) or (0, "x"))

    worker._attempt_pending_resume(settings, store)
    assert calls == []

    store.update_pipeline_run("busy_run", "failed", error="crash")
    monkeypatch.setattr(factory, "current_extents_ms", lambda _s: {})
    worker._attempt_pending_resume(settings, store)
    assert calls == [["mine", "run", "--resume", "busy_run"]]
    assert factory.load_factory_state(store).get("pending_resume_run_id") is None
