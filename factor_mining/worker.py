"""Factory supervisor: sync nightly, mine only when new data arrives.

Discovery spends a fixed trial budget and runs only when a coherent increment
of fresh OOS data has landed; between rounds the worker just rechecks
survivors. All pipeline work runs in child processes so an OOM or crash takes
the child, not the scheduler — checkpoints plus ``--resume`` make child death
recoverable. Resume is attempted BEFORE the nightly sync: the I3 frame
fingerprint fails any resume loudly once the warehouse changes, so after a
sync the pending run is abandoned and its checkpoints pruned instead.

With ``factory.enabled=False`` (default) this keeps the pre-factory behavior:
a nightly data sync and nothing else.
"""

from __future__ import annotations

import fcntl
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from apscheduler.schedulers.blocking import BlockingScheduler

from factor_mining import factory
from factor_mining.config import Settings, load_settings
from factor_mining.data.binance import BinanceArchiveClient
from factor_mining.storage import MetadataStore

logger = logging.getLogger(__name__)

_WATCHDOG_POLL_SECONDS = 60.0
_STOP_GRACE_SECONDS = 120.0
_CHILD_RUN_ID_WAIT_SECONDS = 30.0


def run_worker() -> None:
    settings = load_settings()
    store = MetadataStore(settings.data.sqlite_path)
    client = BinanceArchiveClient(settings, store)

    if not settings.factory.enabled:
        # Pre-factory behavior: nightly data sync only.
        scheduler = BlockingScheduler(timezone="UTC")

        @scheduler.scheduled_job("cron", hour=3, minute=15)
        def daily_data_sync() -> None:
            client.sync()

        scheduler.start()
        return

    logging.basicConfig(level=logging.INFO)
    lock_handle = _acquire_singleton_lock(settings)
    try:
        scheduler = BlockingScheduler(timezone="UTC")

        @scheduler.scheduled_job(
            "cron",
            hour=settings.factory.sync_hour_utc,
            minute=settings.factory.sync_minute_utc,
            misfire_grace_time=3600,
            coalesce=True,
            max_instances=1,
        )
        def nightly_factory_job() -> None:
            factory_tick(settings, store, client)

        # A discovery that died with the supervisor resumes immediately at
        # startup — this is the pre-sync window where checkpoints still match.
        _attempt_pending_resume(settings, store)
        scheduler.start()
    finally:
        lock_handle.close()


def factory_tick(settings: Settings, store: MetadataStore, client: BinanceArchiveClient) -> None:
    """One nightly cycle: resume → sync → abandon stale resume → decide → act."""
    _attempt_pending_resume(settings, store)
    client.sync()
    _abandon_pending_resume(store, reason="data_synced_checkpoints_stale")
    _mark_stale_run_failed(store, settings)

    action = decide_action(store, settings)
    logger.info("factory tick: %s", action)
    if action == "discovery":
        run_discovery(settings, store)
    elif action == "recheck":
        run_recheck(settings, store)

    state = factory.load_factory_state(store)
    if not state.get("pending_resume_run_id"):
        removed = store.prune_artifacts(kind="pipeline_checkpoint", max_unprotected_rows=500)
        if removed:
            logger.info("pruned %d old pipeline checkpoints", removed)


def decide_action(store: MetadataStore, settings: Settings, *, now: datetime | None = None) -> str:
    """'skip_active_run' | 'discovery' | 'recheck' | 'idle'.

    The active-run gate covers foreign pipelines too (backtest_master's lab
    daemon shares this store): 2×~20GB concurrent runs is the failure mode."""
    if store.active_pipeline_run() is not None:
        return "skip_active_run"
    state = factory.load_factory_state(store)
    current = factory.current_extents_ms(settings)
    if factory.discovery_due(current, state["last_discovery_extent_ms"], settings.factory.min_new_days):
        return "discovery"
    now = (now or datetime.now(UTC)).astimezone(UTC)
    last_recheck = state.get("last_recheck_at")
    if last_recheck is not None:
        elapsed_days = (now - datetime.fromisoformat(last_recheck)).total_seconds() / 86400.0
        if elapsed_days < float(settings.factory.recheck_interval_days):
            return "idle"
    return "recheck"


def run_discovery(settings: Settings, store: MetadataStore) -> int:
    # Snapshot extents BEFORE the child runs: bars arriving mid-run belong to
    # the next round's arrival check, not this one's baseline.
    extents = factory.current_extents_ms(settings)
    args = ["mine", "run", "--trial-budget", str(settings.factory.trial_budget_per_round)]
    if settings.factory.use_llm:
        args.append("--llm")
    args.extend(_worker_cap_args(settings))
    code, run_id = _run_supervised_child(args, settings, store)
    if code == 0:
        factory.record_discovery_completed(store, run_id=run_id or "unknown", extents_ms=extents)
    elif run_id:
        _set_pending_resume(store, run_id)
        logger.warning("discovery run %s exited %d; queued for pre-sync resume", run_id, code)
    return code


def run_recheck(settings: Settings, store: MetadataStore) -> int:
    args = ["mine", "verify-survivors", *_worker_cap_args(settings)]
    code, run_id = _run_supervised_child(args, settings, store)
    if code == 0:
        factory.record_recheck_completed(store, run_id=run_id or "unknown")
    else:
        logger.warning("recheck run %s exited %d", run_id, code)
    return code


def _attempt_pending_resume(settings: Settings, store: MetadataStore) -> None:
    state = factory.load_factory_state(store)
    run_id = state.get("pending_resume_run_id")
    if not run_id:
        return
    if store.pipeline_run_status(run_id) in {"running", "stopping"}:
        return
    logger.info("resuming interrupted discovery %s (pre-sync window)", run_id)
    extents = factory.current_extents_ms(settings)
    code, resumed_run_id = _run_supervised_child(["mine", "run", "--resume", run_id], settings, store)
    if code == 0:
        _clear_pending_resume(store)
        factory.record_discovery_completed(store, run_id=resumed_run_id or run_id, extents_ms=extents)
    # nonzero: keep the pending marker — tonight's post-sync abandonment
    # decides whether the checkpoints are still worth anything.


def _set_pending_resume(store: MetadataStore, run_id: str) -> None:
    state = factory.load_factory_state(store)
    state["pending_resume_run_id"] = run_id
    factory.save_factory_state(store, state)


def _clear_pending_resume(store: MetadataStore) -> None:
    state = factory.load_factory_state(store)
    if state.pop("pending_resume_run_id", None) is not None:
        factory.save_factory_state(store, state)


def _abandon_pending_resume(store: MetadataStore, *, reason: str) -> None:
    """After a sync the I3 frame fingerprint no longer matches: the pending
    run can never resume, so drop the marker and its dead checkpoints."""
    state = factory.load_factory_state(store)
    run_id = state.pop("pending_resume_run_id", None)
    if not run_id:
        return
    factory.save_factory_state(store, state)
    removed = store.delete_artifacts_with_prefix(f"checkpoint:{run_id}")
    logger.warning("abandoned resume of %s (%s); pruned %d checkpoints", run_id, reason, removed)


def _mark_stale_run_failed(store: MetadataStore, settings: Settings) -> None:
    """A 'running' row with no live child (supervisor died mid-run) would
    block every future cycle through the active-run gate."""
    active = store.active_pipeline_run()
    if active is None:
        return
    try:
        started = datetime.fromisoformat(str(active["started_at"]))
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
    except (KeyError, ValueError):
        return
    age_hours = (datetime.now(UTC) - started).total_seconds() / 3600.0
    if age_hours > 2.0 * float(settings.factory.max_run_hours):
        store.update_pipeline_run(str(active["run_id"]), "failed", error="stale_run_no_live_child")
        logger.warning("marked stale run %s failed (age %.1fh)", active["run_id"], age_hours)


def _worker_cap_args(settings: Settings) -> list[str]:
    if settings.factory.max_workers is None:
        return []
    return ["--workers", str(int(settings.factory.max_workers))]


def _tree_rss_bytes(proc: Any) -> int:
    """RSS of the child and its whole process tree. The pipeline fans out
    into a ProcessPoolExecutor, so the pool workers — not the direct child —
    hold most of the memory; a parent-only check under-measures by roughly
    the entire pool."""
    total = int(proc.memory_info().rss)
    for child in proc.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except Exception:  # noqa: BLE001 - pool workers come and go mid-scan
            continue
    return total


def _run_supervised_child(
    cli_args: list[str], settings: Settings, store: MetadataStore
) -> tuple[int, str | None]:
    process = subprocess.Popen([sys.executable, "-m", "factor_mining.cli", *cli_args])
    run_id = _wait_for_child_run_id(store, process)
    code = _supervise(process, store, settings, run_id=run_id)
    return code, run_id


def _wait_for_child_run_id(store: MetadataStore, process: subprocess.Popen) -> str | None:
    """The child mints its own run id; the decide gate guaranteed no active
    run beforehand, so the first row to appear is the child's."""
    deadline = time.monotonic() + _CHILD_RUN_ID_WAIT_SECONDS
    while time.monotonic() < deadline:
        active = store.active_pipeline_run()
        if active is not None:
            return str(active["run_id"])
        if process.poll() is not None:
            return None
        time.sleep(1.0)
    return None


def _supervise(
    process: subprocess.Popen,
    store: MetadataStore,
    settings: Settings,
    *,
    run_id: str | None,
) -> int:
    deadline = time.monotonic() + float(settings.factory.max_run_hours) * 3600.0
    rss_cap_bytes: float | None = None
    proc_info = None
    if settings.factory.max_rss_gb:
        try:
            import psutil

            proc_info = psutil.Process(process.pid)
            rss_cap_bytes = float(settings.factory.max_rss_gb) * (1024.0**3)
        except Exception:  # noqa: BLE001 - psutil optional: degrade to wall-clock only
            proc_info = None
            logger.info("psutil unavailable; RSS watchdog disabled")
    while True:
        code = process.poll()
        if code is not None:
            return code
        breach = None
        if time.monotonic() > deadline:
            breach = "max_run_hours"
        elif proc_info is not None and rss_cap_bytes is not None:
            try:
                if _tree_rss_bytes(proc_info) > rss_cap_bytes:
                    breach = "max_rss_gb"
            except Exception:  # noqa: BLE001 - child may be exiting under us
                proc_info = None
        if breach:
            return _escalate(process, store, run_id=run_id, reason=breach)
        time.sleep(_WATCHDOG_POLL_SECONDS)


def _escalate(
    process: subprocess.Popen,
    store: MetadataStore,
    *,
    run_id: str | None,
    reason: str,
    grace_seconds: float = _STOP_GRACE_SECONDS,
) -> int:
    """Stop request → SIGTERM → SIGKILL. Checkpoints make death recoverable,
    but the polite path lets the child record a clean stop."""
    logger.warning("watchdog breach (%s) on run %s; escalating", reason, run_id)
    if run_id:
        store.request_pipeline_stop(run_id)
    code = _wait(process, grace_seconds)
    if code is None:
        process.terminate()
        code = _wait(process, 2.0 * grace_seconds)
    if code is None:
        process.kill()
        code = _wait(process, grace_seconds)
        if code is None:
            code = -9
    if run_id and store.pipeline_run_status(run_id) in {"running", "stopping"}:
        store.update_pipeline_run(run_id, "failed", error=f"watchdog_{reason}")
    return int(code)


def _wait(process: subprocess.Popen, timeout_s: float) -> int | None:
    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None


def _acquire_singleton_lock(settings: Settings) -> IO[str]:
    lock_path = Path(settings.data.sqlite_path).parent / "factory.lock"
    handle = open(lock_path, "w")  # noqa: SIM115 - held for process lifetime
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit(f"factory supervisor already running ({lock_path} is locked)")
    return handle
