from __future__ import annotations

import json
import math
import threading
import time
import traceback
import uuid
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from factor_mining.config import Settings, load_settings
from factor_mining.pipeline import _PipelineCancelled, run_pipeline
from factor_mining.registry import METHOD_REGISTRY, schedulable_methods
from factor_mining.storage import MetadataStore


UTC = timezone.utc
STATIC_DIR = Path(__file__).with_name("static")
_RUN_THREADS: dict[str, threading.Thread] = {}
_STOP_EVENTS: dict[str, threading.Event] = {}
_INTERRUPTED_RUN_ERROR = "Dashboard worker interrupted before completion."
_STOP_REQUESTED_ERROR = "Stop requested from dashboard."
_RUN_DISPLAY_ARTIFACT_IDS = {
    "latest_hypotheses",
    "initial_candidates",
    "latest_candidates",
    "latest_backtests",
    "latest_gatechecks",
    "latest_gatecheck_diagnostics",
    "latest_hardscores",
    "latest_factor_evidence",
    "latest_research_gate",
    "latest_near_misses",
    "latest_research_survivors",
    "latest_research_survivor_store",
    "latest_optimization_history",
    "latest_detail_index",
}


class _DashboardRunCancelled(BaseException):
    # Pipeline event sinks intentionally swallow regular Exception instances.
    # This internal signal must pass through that safety wrapper to stop the run.
    pass

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Factor Mining - Local Dashboard</title>
  <link rel="stylesheet" href="/dashboard.css" />
</head>
<body>
  <div id="app" class="boot">Loading Factor Mining...</div>
  <script src="/dashboard.js" defer></script>
</body>
</html>
"""


import sys

class SafeStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        try:
            self.stream.write(data)
        except BrokenPipeError:
            pass
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except BrokenPipeError:
            pass
        except Exception:
            pass

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def run_dashboard(host: str = "127.0.0.1", port: int = 8501, *, open_browser: bool = True) -> None:
    """Run the local dashboard."""
    sys.stdout = SafeStream(sys.stdout)
    sys.stderr = SafeStream(sys.stderr)
    settings = load_settings()
    recovered = _recover_interrupted_runs(settings)
    httpd = _bind_server(host, port, _make_handler(settings))
    actual_port = int(httpd.server_address[1])
    url = f"http://{host}:{actual_port}/"
    if recovered:
        print(f"Recovered {recovered} interrupted run(s).", flush=True)
    print(f"Factor Mining dashboard: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def build_dashboard_state(settings: Settings | None = None, store: MetadataStore | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    store = store or MetadataStore(settings.data.sqlite_path)
    bundle = _load_latest_bundle(store)
    experiments = _experiment_rows(bundle)
    runs = store.list_pipeline_runs(limit=20)
    latest_run = runs[0] if runs else None
    active_run = store.active_pipeline_run()
    events = store.load_pipeline_events(latest_run["run_id"], limit=500) if latest_run else []
    coverage = store.list_coverage(limit=500)

    return {
        "settings": {
            "symbols": settings.data.symbols,
            "markets": settings.data.markets,
            "default_interval": settings.data.default_interval,
            "sqlite_path": str(settings.data.sqlite_path),
            "parquet_dir": str(settings.data.parquet_dir),
            "trial_window_days": settings.trial_ledger.window_days,
        },
        "bundle": bundle,
        "experiments": experiments,
        "runs": runs,
        "active_run": active_run,
        "latest_run": latest_run,
        "events": events,
        "coverage": coverage,
        "methods": [method.model_dump(mode="json") for method in METHOD_REGISTRY],
        "schedulable_method_count": len(schedulable_methods(2)),
        "ledger": _trial_ledger_rows(settings, store, experiments),
        "recent_trials": _load_recent_trials(store),
        "server_time": datetime.now(UTC).isoformat(),
    }


def _bind_server(host: str, port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    errors: list[OSError] = []
    for candidate in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            errors.append(exc)
    raise RuntimeError(f"Could not bind dashboard on {host}:{port}-{port + 19}") from errors[-1]


def _make_handler(settings: Settings) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                store = MetadataStore(settings.data.sqlite_path)
                self._send_json(build_dashboard_state(settings, store))
                return
            if parsed.path == "/api/detail":
                query = parse_qs(parsed.query)
                experiment_id = (query.get("id") or [""])[0]
                self._send_json(_load_experiment_detail(settings, experiment_id))
                return
            if parsed.path == "/api/archives":
                self._send_json({"archives": _load_archives()})
                return
            if parsed.path in {"/dashboard.css", "/dashboard.js"}:
                self._send_static(parsed.path.removeprefix("/"))
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            payload = self._read_json()
            if parsed.path == "/api/run":
                ok, message = _start_dashboard_run(settings, _run_args(payload))
                self._send_json({"ok": ok, "message": message})
                return
            if parsed.path == "/api/hosted":
                ok, message = _start_hosted_run(settings, _run_args(payload))
                self._send_json({"ok": ok, "message": message})
                return
            if parsed.path == "/api/stop":
                store = MetadataStore(settings.data.sqlite_path)
                active = store.active_pipeline_run()
                run_id = str(payload.get("run_id") or (active or {}).get("run_id") or "")
                ok, message = _request_run_stop(settings, run_id)
                self._send_json({"ok": ok, "message": message})
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}

        def _send_json(self, payload: dict[str, Any]) -> None:
            self._send_bytes(
                json.dumps(_json_safe(payload), default=str, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send_static(self, filename: str) -> None:
            path = (STATIC_DIR / filename).resolve()
            if not path.is_file() or STATIC_DIR.resolve() not in path.parents:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            content_type = "text/css; charset=utf-8" if path.suffix == ".css" else "text/javascript; charset=utf-8"
            self._send_bytes(path.read_bytes(), content_type)

        def _send_bytes(self, data: bytes, content_type: str) -> None:
            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(data)))
                self.send_header("cache-control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            except BrokenPipeError:
                pass  # client disconnected before response was sent

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            data = json.dumps({"ok": False, "message": message}).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DashboardHandler


def _run_args(payload: dict[str, Any]) -> dict[str, Any]:
    tail = _as_int(payload.get("tail"), 50_000)
    return {
        "use_llm": bool(payload.get("use_llm", False)),
        "iterations": max(1, min(_as_int(payload.get("iterations"), 1), 7)),
        "hypothesis_count": max(1, min(_as_int(payload.get("hypothesis_count"), 5), 10)),
        "max_workers": max(1, _as_int(payload.get("max_workers"), 1)),
        "tail": None if tail <= 0 else tail,
        "archive_top": max(0, min(_as_int(payload.get("archive_top"), 3), 10)),
        "research_brief": str(payload.get("research_brief") or "").strip() or None,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _start_dashboard_run(settings: Settings, args: dict[str, Any]) -> tuple[bool, str]:
    _recover_interrupted_runs(settings)
    store = MetadataStore(settings.data.sqlite_path)
    active = store.active_pipeline_run()
    if active is not None:
        return False, f"Run {active['run_id']} is already active."

    args = {**args, "mode": "single"}
    run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    cleared = _clear_run_display_cache(store)
    store.create_pipeline_run(run_id, args)
    store.append_pipeline_event(
        run_id,
        phase="ui",
        level="info",
        message="Dashboard run queued.",
        payload={**args, "cleared_display_artifacts": cleared},
    )

    def worker() -> None:
        worker_store = MetadataStore(settings.data.sqlite_path)
        stop_event = threading.Event()
        _STOP_EVENTS[run_id] = stop_event

        def sink(phase: str, level: str, message: str, payload: dict[str, Any] | None = None) -> None:
            if stop_event.is_set():
                raise _DashboardRunCancelled(_STOP_REQUESTED_ERROR)
            _cancel_if_stop_requested(worker_store, run_id)
            worker_store.append_pipeline_event(run_id, phase=phase, level=level, message=message, payload=payload)

        try:
            worker_store.append_pipeline_event(run_id, phase="ui", level="info", message="Background worker started.")
            run_pipeline(
                settings,
                use_llm=args["use_llm"],
                max_workers=args["max_workers"],
                tail=args["tail"],
                archive_top=args["archive_top"],
                research_brief=args["research_brief"],
                hypothesis_count=args["hypothesis_count"],
                iterations=args["iterations"],
                store=worker_store,
                event_sink=sink,
                stop_event=stop_event,
            )
            if worker_store.pipeline_run_status(run_id) == "stopping":
                worker_store.append_pipeline_event(run_id, phase="ui", level="warn", message="Run cancelled after stop request.")
                worker_store.update_pipeline_run(run_id, "cancelled", error=_STOP_REQUESTED_ERROR)
            else:
                worker_store.append_pipeline_event(run_id, phase="ui", level="info", message="Run completed.")
                worker_store.update_pipeline_run(run_id, "completed")
        except (_DashboardRunCancelled, _PipelineCancelled):
            worker_store.append_pipeline_event(run_id, phase="ui", level="warn", message="Run cancelled at pipeline checkpoint.")
            worker_store.update_pipeline_run(run_id, "cancelled", error=_STOP_REQUESTED_ERROR)
        except Exception as exc:
            worker_store.append_pipeline_event(
                run_id,
                phase="error",
                level="error",
                message=str(exc),
                payload={"traceback": traceback.format_exc(limit=12)},
            )
            worker_store.update_pipeline_run(run_id, "failed", error=str(exc))
        finally:
            _RUN_THREADS.pop(run_id, None)
            _STOP_EVENTS.pop(run_id, None)

    thread = threading.Thread(target=worker, name=f"factor-mining-{run_id}", daemon=True)
    _RUN_THREADS[run_id] = thread
    thread.start()
    return True, f"Started {run_id}."


def _start_hosted_run(settings: Settings, args: dict[str, Any]) -> tuple[bool, str]:
    _recover_interrupted_runs(settings)
    store = MetadataStore(settings.data.sqlite_path)
    active = store.active_pipeline_run()
    if active is not None:
        return False, f"Run {active['run_id']} is already active."

    args = {**args, "mode": "hosted"}
    run_id = f"hosted_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    cleared = _clear_run_display_cache(store)
    store.create_pipeline_run(run_id, args)
    store.append_pipeline_event(
        run_id,
        phase="ui",
        level="info",
        message="Hosted run queued.",
        payload={**args, "cleared_display_artifacts": cleared},
    )

    def worker() -> None:
        worker_store = MetadataStore(settings.data.sqlite_path)

        def sink(phase: str, level: str, message: str, payload: dict[str, Any] | None = None) -> None:
            worker_store.append_pipeline_event(run_id, phase=phase, level=level, message=message, payload=payload)

        cycle = 1
        try:
            worker_store.append_pipeline_event(run_id, phase="hosted", level="info", message="Background worker started.")
            while worker_store.pipeline_run_status(run_id) == "running":
                worker_store.append_pipeline_event(
                    run_id,
                    phase="hosted",
                    level="info",
                    message="Hosted cycle started.",
                    payload={"cycle": cycle, "archive_top": args["archive_top"]},
                )
                result = run_pipeline(
                    settings,
                    use_llm=args["use_llm"],
                    max_workers=args["max_workers"],
                    tail=args["tail"],
                    archive_top=args["archive_top"],
                    research_brief=args["research_brief"],
                    hypothesis_count=args["hypothesis_count"],
                    iterations=args["iterations"],
                    store=worker_store,
                    event_sink=sink,
                )
                if worker_store.pipeline_run_status(run_id) == "stopping":
                    worker_store.append_pipeline_event(
                        run_id,
                        phase="hosted",
                        level="warn",
                        message="Hosted cycle completed after stop request.",
                        payload={
                            "cycle": cycle,
                            "backtests": len(result.backtests),
                            "gatecheck_passed": result.n_gatecheck_passed,
                            "positive_hardscores": sum(1 for score in result.hardscores if score.score > 0),
                            "elapsed_s": round(result.elapsed_s, 3),
                        },
                    )
                else:
                    worker_store.append_pipeline_event(
                        run_id,
                        phase="hosted",
                        level="info",
                        message="Hosted cycle completed.",
                        payload={
                            "cycle": cycle,
                            "backtests": len(result.backtests),
                            "gatecheck_passed": result.n_gatecheck_passed,
                            "positive_hardscores": sum(1 for score in result.hardscores if score.score > 0),
                            "elapsed_s": round(result.elapsed_s, 3),
                        },
                    )
                cycle += 1
                time.sleep(1)

            worker_store.append_pipeline_event(run_id, phase="hosted", level="info", message="Hosted run stopped.")
            worker_store.update_pipeline_run(run_id, "stopped")
        except Exception as exc:
            worker_store.append_pipeline_event(
                run_id,
                phase="error",
                level="error",
                message=str(exc),
                payload={"traceback": traceback.format_exc(limit=12), "cycle": cycle},
            )
            worker_store.update_pipeline_run(run_id, "failed", error=str(exc))
        finally:
            _RUN_THREADS.pop(run_id, None)

    thread = threading.Thread(target=worker, name=f"factor-mining-{run_id}", daemon=True)
    _RUN_THREADS[run_id] = thread
    thread.start()
    return True, f"Started {run_id}."


def _request_hosted_stop(settings: Settings, run_id: str) -> tuple[bool, str]:
    return _request_run_stop(settings, run_id)


def _request_run_stop(settings: Settings, run_id: str) -> tuple[bool, str]:
    if not run_id:
        return False, "No active run."
    store = MetadataStore(settings.data.sqlite_path)
    status = store.pipeline_run_status(run_id)
    if status not in {"running", "stopping"}:
        return False, f"Run {run_id} is not active."
    mode = _pipeline_run_mode(store, run_id)
    if status == "stopping":
        if _run_thread_alive(run_id):
            if mode == "hosted":
                return True, "Stop already requested. The hosted worker will stop after the current cycle."
            return True, "Stop already requested. The worker will cancel at the next checkpoint."
        store.append_pipeline_event(run_id, phase="ui", level="warn", message="Finalized interrupted stop request.")
        if mode == "hosted":
            store.update_pipeline_run(run_id, "stopped")
        else:
            store.update_pipeline_run(run_id, "cancelled", error=_STOP_REQUESTED_ERROR)
        return True, "Stopped stale run."
    if status == "running":
        store.request_pipeline_stop(run_id)
        store.append_pipeline_event(run_id, phase="ui", level="warn", message="Stop requested.")
        stop_event = _STOP_EVENTS.get(run_id)
        if stop_event is not None:
            stop_event.set()
    if mode == "hosted":
        return True, "Stop requested. The hosted worker will stop after the current cycle."
    return True, "Stop requested. The worker will cancel at the next checkpoint."


def _run_thread_alive(run_id: str) -> bool:
    thread = _RUN_THREADS.get(run_id)
    return bool(thread and thread.is_alive())


def _clear_run_display_cache(store: MetadataStore) -> int:
    return store.delete_artifacts(_RUN_DISPLAY_ARTIFACT_IDS)


def _cancel_if_stop_requested(store: MetadataStore, run_id: str) -> None:
    if store.pipeline_run_status(run_id) != "stopping":
        return
    store.append_pipeline_event(run_id, phase="ui", level="warn", message="Cancellation checkpoint reached.")
    raise _DashboardRunCancelled(_STOP_REQUESTED_ERROR)


def _pipeline_run_mode(store: MetadataStore, run_id: str) -> str:
    with store.connect() as conn:
        row = conn.execute(
            "select args_json from pipeline_runs where run_id = ?",
            (run_id,),
        ).fetchone()
    if row is None:
        return ""
    try:
        args = json.loads(row["args_json"])
    except json.JSONDecodeError:
        return ""
    return str(args.get("mode") or "")


def _recover_interrupted_stop_requests(settings: Settings) -> int:
    return _recover_interrupted_runs(settings, recover_running=False)


def _recover_interrupted_runs(settings: Settings, *, recover_running: bool = True) -> int:
    store = MetadataStore(settings.data.sqlite_path)
    statuses = ("running", "stopping") if recover_running else ("stopping",)
    placeholders = ",".join("?" for _ in statuses)
    with store.connect() as conn:
        rows = conn.execute(
            f"""
            select run_id, status from pipeline_runs
            where status in ({placeholders}) and ended_at is null
            order by started_at asc
            """,
            statuses,
        ).fetchall()
    recovered = 0
    for row in rows:
        run_id = str(row["run_id"])
        if _run_thread_alive(run_id):
            continue
        status = str(row["status"])
        if status == "stopping":
            store.append_pipeline_event(
                run_id,
                phase="ui",
                level="warn",
                message="Recovered interrupted stop request on dashboard startup.",
            )
            store.update_pipeline_run(run_id, "stopped")
        elif status == "running" and recover_running:
            store.append_pipeline_event(
                run_id,
                phase="ui",
                level="error",
                message="Recovered interrupted dashboard run on dashboard startup.",
            )
            store.update_pipeline_run(run_id, "failed", error=_INTERRUPTED_RUN_ERROR)
        recovered += 1
    return recovered


def _load_experiment_detail(settings: Settings, experiment_id: str) -> dict[str, Any]:
    store = MetadataStore(settings.data.sqlite_path)
    bundle = _load_latest_bundle(store)
    row = next((item for item in _experiment_rows(bundle) if item["experiment_id"] == experiment_id), None)
    detail = store.load_artifact(f"experiment_detail_{experiment_id}") if experiment_id else None
    candidates = {item.get("candidate_id"): item for item in bundle["candidates"]}
    hypotheses = {item.get("hypothesis_id"): item for item in bundle["hypotheses"]}
    candidate = candidates.get((row or {}).get("candidate_id"), {})
    hypothesis = hypotheses.get(candidate.get("hypothesis_id") or (row or {}).get("hypothesis_id"), {})
    return {
        "row": row,
        "detail": detail,
        "candidate": candidate,
        "hypothesis": hypothesis,
    }


def _load_archives() -> list[dict[str, Any]]:
    archives_root = Path("archives")
    if not archives_root.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for archive_dir in sorted(archives_root.iterdir(), reverse=True):
        if not archive_dir.is_dir():
            continue
        manifest_path = archive_dir / "manifest.json"
        hardscore_path = archive_dir / "hardscore.json"
        gatecheck_path = archive_dir / "gatecheck.json"
        result_path = archive_dir / "backtest_result.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            hardscore = json.loads(hardscore_path.read_text(encoding="utf-8")) if hardscore_path.exists() else {}
            gatecheck = json.loads(gatecheck_path.read_text(encoding="utf-8")) if gatecheck_path.exists() else {}
            integrity = "valid"
            if result_path.exists():
                from factor_mining.archive import stable_hash
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                if stable_hash(result_payload) != manifest.get("result_hash"):
                    integrity = "invalid"
            else:
                integrity = "incomplete"
            results.append({
                "experiment_id": manifest.get("experiment_id", archive_dir.name),
                "created_at": manifest.get("created_at", ""),
                "git_sha": manifest.get("git_sha"),
                "hardscore": hardscore.get("score"),
                "sharpe": hardscore.get("sharpe"),
                "dsr": hardscore.get("dsr"),
                "gate_passed": gatecheck.get("passed", False),
                "risk_tier": gatecheck.get("risk_tier", "unknown"),
                "integrity": integrity,
            })
        except (json.JSONDecodeError, OSError):
            results.append({
                "experiment_id": archive_dir.name,
                "created_at": "",
                "hardscore": None,
                "sharpe": None,
                "integrity": "corrupt",
            })
    return results


def _load_latest_bundle(store: MetadataStore) -> dict[str, Any]:
    bundle = {
        "hypotheses": _artifact_items(store.load_artifact("latest_hypotheses")),
        "candidates": _artifact_items(store.load_artifact("latest_candidates") or store.load_artifact("initial_candidates")),
        "backtests": _artifact_items(store.load_artifact("latest_backtests")),
        "gatechecks": _artifact_items(store.load_artifact("latest_gatechecks")),
        "hardscores": _artifact_items(store.load_artifact("latest_hardscores")),
        "factor_evidence": _artifact_items(store.load_artifact("latest_factor_evidence")),
        "research_gate": _artifact_items(store.load_artifact("latest_research_gate")),
        "near_misses": _artifact_items(store.load_artifact("latest_near_misses")),
        "optimization_history": _artifact_items(store.load_artifact("latest_optimization_history")),
        "research_survivors": _artifact_items(store.load_artifact("latest_research_survivors")),
        "research_survivor_store": _artifact_items(store.load_artifact("latest_research_survivor_store")),
        "gatecheck_diagnostics": store.load_artifact("latest_gatecheck_diagnostics") or {},
    }
    if not (bundle["gatecheck_diagnostics"].get("rows") if isinstance(bundle["gatecheck_diagnostics"], dict) else None):
        bundle["gatecheck_diagnostics"] = _fallback_gatecheck_diagnostics(bundle)
    return bundle


def _artifact_items(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    items = payload.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def _fallback_gatecheck_diagnostics(bundle: dict[str, Any]) -> dict[str, Any]:
    candidates = {item.get("candidate_id"): item for item in bundle.get("candidates", [])}
    gates = {item.get("experiment_id"): item for item in bundle.get("gatechecks", [])}
    failure_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for backtest in bundle.get("backtests", []):
        experiment_id = backtest.get("experiment_id")
        candidate = candidates.get(backtest.get("candidate_id"), {})
        gate = gates.get(experiment_id, {})
        failures, warnings = _gate_failures_warnings(gate)
        failure_counts.update(failures)
        warning_counts.update(warnings)
        metrics = backtest.get("metrics_primary") or {}
        gross_metrics = backtest.get("metrics_gross") or {}
        gross_sharpe = _as_optional_float(gross_metrics.get("sharpe"))
        net_sharpe = _as_float(metrics.get("sharpe"))
        break_even = _as_float(backtest.get("break_even_cost_bps"))
        actual_cost = _as_float(backtest.get("actual_cost_bps"))
        cost_margin = break_even - (2.0 * actual_cost)
        params = candidate.get("params") or {}
        rows.append({
            "candidate_id": backtest.get("candidate_id"),
            "experiment_id": experiment_id,
            "hypothesis_family": backtest.get("hypothesis_family") or candidate.get("hypothesis_family") or "unknown",
            "method_id": backtest.get("method_id") or candidate.get("method_id") or "unknown",
            "symbol": backtest.get("symbol") or candidate.get("symbol") or "unknown",
            "search_variant": params.get("search_variant") or "unknown",
            "signal_source": params.get("signal_source"),
            "net_sharpe": net_sharpe,
            "gross_sharpe": gross_sharpe,
            "cost_drag_sharpe": None if gross_sharpe is None else gross_sharpe - net_sharpe,
            "annualized_return": _as_float(metrics.get("annualized_return")),
            "max_drawdown": _as_float(metrics.get("max_drawdown")),
            "ic_tstat_nw": _as_float(backtest.get("ic_tstat_nw")),
            "rankic_tstat_nw": _as_float(backtest.get("rankic_tstat_nw")),
            "permutation_pvalue": _as_float(backtest.get("permutation_test_pvalue")),
            "pbo": _as_optional_float(backtest.get("pbo")),
            "factor_turnover": _as_float(backtest.get("factor_turnover")),
            "oos_trade_count": int(backtest.get("oos_trade_count") or 0),
            "break_even_cost_bps": break_even,
            "actual_cost_bps": actual_cost,
            "cost_margin_bps": cost_margin,
            "failures": failures,
            "warnings": warnings,
        })

    return {
        "generated_by": "ui_fallback",
        "total": len(bundle.get("gatechecks", [])),
        "passed": sum(1 for gate in bundle.get("gatechecks", []) if gate.get("passed")),
        "failure_counts": [
            {"rule_id": rule_id, "count": count}
            for rule_id, count in failure_counts.most_common()
        ],
        "warning_counts": [
            {"rule_id": rule_id, "count": count}
            for rule_id, count in warning_counts.most_common()
        ],
        "metric_summary": {
            "net_sharpe": _numeric_summary(row["net_sharpe"] for row in rows),
            "gross_sharpe": _numeric_summary(row["gross_sharpe"] for row in rows),
            "cost_drag_sharpe": _numeric_summary(row["cost_drag_sharpe"] for row in rows),
            "factor_turnover": _numeric_summary(row["factor_turnover"] for row in rows),
            "break_even_cost_bps": _numeric_summary(row["break_even_cost_bps"] for row in rows),
            "actual_cost_bps": _numeric_summary(row["actual_cost_bps"] for row in rows),
            "cost_margin_bps": _numeric_summary(row["cost_margin_bps"] for row in rows),
            "oos_trade_count": _numeric_summary(row["oos_trade_count"] for row in rows),
        },
        "top_by_net_sharpe": sorted(rows, key=lambda row: row["net_sharpe"], reverse=True)[:10],
        "top_by_cost_margin": sorted(rows, key=lambda row: row["cost_margin_bps"], reverse=True)[:10],
        "rows": rows,
    }


def _numeric_summary(values) -> dict[str, float | int | None]:
    clean = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not clean:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "count": len(clean),
        "min": clean[0],
        "p25": clean[int((len(clean) - 1) * 0.25)],
        "median": clean[int((len(clean) - 1) * 0.50)],
        "p75": clean[int((len(clean) - 1) * 0.75)],
        "max": clean[-1],
    }


def _experiment_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {item.get("candidate_id"): item for item in bundle["candidates"]}
    hypotheses = {item.get("hypothesis_id"): item for item in bundle["hypotheses"]}
    gatechecks = {item.get("experiment_id"): item for item in bundle["gatechecks"]}
    hardscores = {item.get("experiment_id"): item for item in bundle["hardscores"]}
    diagnostics = bundle.get("gatecheck_diagnostics") or {}
    diag_by_exp = {
        item.get("experiment_id"): item
        for item in diagnostics.get("rows", [])
        if isinstance(item, dict)
    }

    rows: list[dict[str, Any]] = []
    for backtest in bundle["backtests"]:
        experiment_id = str(backtest.get("experiment_id", ""))
        candidate_id = backtest.get("candidate_id")
        candidate = candidates.get(candidate_id, {})
        hypothesis = hypotheses.get(candidate.get("hypothesis_id") or backtest.get("hypothesis_id"), {})
        metrics = backtest.get("metrics_primary") or {}
        gate = gatechecks.get(experiment_id)
        score = hardscores.get(experiment_id, {})
        diag = diag_by_exp.get(experiment_id, {})
        failures, warnings = _gate_failures_warnings(gate)
        rows.append(
            {
                "experiment_id": experiment_id,
                "candidate_id": candidate_id,
                "hypothesis_id": candidate.get("hypothesis_id") or hypothesis.get("hypothesis_id"),
                "hypothesis": _short_text(hypothesis.get("economic_mechanism"), 120),
                "family": backtest.get("hypothesis_family") or candidate.get("hypothesis_family") or "unknown",
                "method": backtest.get("method_id") or candidate.get("method_id") or "unknown",
                "symbol": backtest.get("symbol") or candidate.get("symbol") or "unknown",
                "market": backtest.get("market") or candidate.get("market") or "unknown",
                "interval": backtest.get("interval") or candidate.get("interval") or "",
                "params": candidate.get("params") or {},
                "sharpe": _as_float(metrics.get("sharpe")),
                "total_return": _as_float(metrics.get("total_return")),
                "ann_return": _as_float(metrics.get("annualized_return")),
                "max_drawdown": _as_float(metrics.get("max_drawdown")),
                "calmar": _as_float(metrics.get("calmar")),
                "trades": int(metrics.get("trade_count") or 0),
                "ic_tstat": _as_float(backtest.get("ic_tstat_nw")),
                "rank_ic_tstat": _as_float(backtest.get("rankic_tstat_nw")),
                "dsr": _as_float(backtest.get("deflated_sharpe")),
                "psr": _as_float(backtest.get("probabilistic_sharpe")),
                "pbo": _as_float(backtest.get("pbo")),
                "perm_p": _as_float(backtest.get("permutation_test_pvalue")),
                "fdr_p": _as_float(score.get("fdr_adjusted_pvalue")),
                "hardscore": _as_float(score.get("score")),
                "haircut_sharpe": _as_float(score.get("haircut_sharpe")),
                "gate": _gate_status(gate),
                "failures": failures,
                "warnings": warnings,
                "capacity_usd": _as_float(backtest.get("estimated_capacity_usd")),
                "break_even_cost_bps": _as_float(backtest.get("break_even_cost_bps")),
                "actual_cost_bps": _as_float(backtest.get("actual_cost_bps")),
                "gross_sharpe": _as_optional_float(diag.get("gross_sharpe")),
                "cost_drag_sharpe": _as_optional_float(diag.get("cost_drag_sharpe")),
                "cost_margin_bps": _as_float(diag.get("cost_margin_bps")),
                "factor_turnover": _as_float(diag.get("factor_turnover")),
                "search_variant": diag.get("search_variant") or (candidate.get("params") or {}).get("search_variant") or "unknown",
                "signal_source": diag.get("signal_source") or (candidate.get("params") or {}).get("signal_source"),
                "exit": _exit_summary(candidate.get("params") or {}),
                "created_at": backtest.get("created_at"),
            }
        )

    rows.sort(key=lambda item: (item["hardscore"], item["sharpe"]), reverse=True)
    return rows


def _trial_ledger_rows(settings: Settings, store: MetadataStore, experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({row["family"] for row in experiments if row.get("family")}) or [
        "momentum",
        "mean_reversion",
        "volatility",
        "funding_basis",
        "volume_confirmation",
    ]
    rows = []
    for family in families:
        family_count, rolling_count, global_count = store.trial_counts(
            family,
            window_days=settings.trial_ledger.window_days,
        )
        rows.append(
            {
                "family": family,
                "family_trials": family_count,
                "rolling_trials": rolling_count,
                "global_trials": global_count,
                "effective_trials": max(family_count, rolling_count),
            }
        )
    return rows


def _load_recent_trials(store: MetadataStore, limit: int = 200) -> list[dict[str, Any]]:
    with store.connect() as conn:
        rows = conn.execute(
            """
            select trial_id, candidate_id, experiment_id, hypothesis_family, method_id, evaluated_at
            from trials
            order by evaluated_at desc
            limit ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def _gate_failures_warnings(gate: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not gate:
        return [], []
    items = gate.get("items") or []
    failures = [str(item.get("rule_id")) for item in items if item.get("status") == "fail"]
    warnings = [str(item.get("rule_id")) for item in items if item.get("status") == "warn"]
    if not failures:
        failures = [str(item.get("rule_id")) for item in gate.get("failures", []) if item.get("rule_id")]
    if not warnings:
        warnings = [str(item.get("rule_id")) for item in gate.get("warnings", []) if item.get("rule_id")]
    return failures, warnings


def _gate_status(gate: dict[str, Any] | None) -> str:
    if not gate:
        return "missing"
    failures, warnings = _gate_failures_warnings(gate)
    if failures or gate.get("passed") is False:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _exit_summary(params: dict[str, Any]) -> dict[str, Any]:
    keys = ("stop_loss_pct", "max_hold_bars", "tp_tiers", "trailing_stop_pct", "trailing_after_first_tp")
    return {k: params[k] for k in keys if k in params}


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _short_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
