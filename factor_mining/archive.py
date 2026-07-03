from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from factor_mining.config import Settings
from factor_mining.models import (
    ArchiveManifest,
    BacktestResult,
    CandidateStrategySpec,
    GateCheckResult,
    HardScoreReport,
)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def archive_experiment(
    *,
    result: BacktestResult,
    gatecheck: GateCheckResult,
    hardscore: HardScoreReport,
    settings: Settings,
    candidate: CandidateStrategySpec | None = None,
    data_manifest: dict[str, Any] | None = None,
    root: Path = Path("archives"),
) -> ArchiveManifest:
    archive_dir = root / result.experiment_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    result_payload = result.model_dump(mode="json")
    gate_payload = gatecheck.model_dump(mode="json")
    score_payload = hardscore.model_dump(mode="json")
    config_payload = settings.model_dump(mode="json")
    _write_json(archive_dir / "backtest_result.json", result_payload)
    _write_json(archive_dir / "gatecheck.json", gate_payload)
    _write_json(archive_dir / "hardscore.json", score_payload)
    _write_json(archive_dir / "config.json", config_payload)
    if candidate is not None:
        # The full candidate spec (params, lookback, type, …) so a later reproduce
        # can reconstruct the exact CandidateStrategySpec — BacktestResult omits
        # params and is insufficient to re-run the candidate on its own.
        _write_json(archive_dir / "candidate.json", candidate.model_dump(mode="json"))
    manifest = ArchiveManifest(
        experiment_id=result.experiment_id,
        git_sha=current_git_sha(),
        data_manifest=data_manifest or {},
        config_hash=stable_hash(config_payload),
        result_hash=stable_hash(result_payload),
    )
    _write_json(archive_dir / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def verify_archive(experiment_id: str, *, root: Path = Path("archives")) -> dict[str, Any]:
    archive_dir = root / experiment_id
    manifest_path = archive_dir / "manifest.json"
    result_path = archive_dir / "backtest_result.json"
    if not manifest_path.exists() or not result_path.exists():
        return {"status": "invalid", "reason": "archive files missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    reproduced_hash = stable_hash(result_payload)
    original_hash = manifest["result_hash"]
    if reproduced_hash == original_hash:
        return {"status": "valid", "relative_deviation": 0.0}
    return {"status": "invalid", "relative_deviation": 1.0, "reason": "result hash changed"}


_RERUN_COMPARED_FIELDS = ("total_return", "sharpe", "max_drawdown")


def rerun_verify_archive(
    experiment_id: str,
    settings: Settings,
    *,
    root: Path = Path("archives"),
    rel_tolerance: float = 1e-6,
    abs_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Re-run the archived candidate and compare metrics_primary (H1/P1-8).

    ``verify_archive`` only re-hashes the stored JSON — file integrity, not
    reproducibility. This re-runs ``reproduce_candidate`` with the archived
    data pin and trial counts, compares the reproducible fields
    (total_return / sharpe / max_drawdown) within tolerance, and stores the
    verdict as ``verify_verdict.json`` in the archive dir — so "verified"
    means the archived claim was actually re-produced, not merely
    un-corrupted. Archives predating a deliberate engine-semantics change
    (e.g. F3 warm-up state) will honestly report ``mismatch``.
    """
    from datetime import datetime, timezone

    from factor_mining.pipeline import reproduce_candidate

    archive_dir = root / experiment_id
    paths = {
        "candidate": archive_dir / "candidate.json",
        "result": archive_dir / "backtest_result.json",
        "manifest": archive_dir / "manifest.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {"status": "invalid", "reason": f"archive files missing: {', '.join(missing)}"}

    spec = CandidateStrategySpec.model_validate(json.loads(paths["candidate"].read_text(encoding="utf-8")))
    archived = json.loads(paths["result"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    data_end_ms = (manifest.get("data_manifest") or {}).get("data_end_ms")
    trial_counts = {
        "effective_trials_count": int(archived.get("effective_trials_at_eval") or 1),
        "global_cumulative_trials_count": int(archived.get("global_trials_at_eval") or 1),
    }
    try:
        reproduced = reproduce_candidate(
            spec,
            settings,
            data_end_ms=int(data_end_ms) if data_end_ms else None,
            trial_counts=trial_counts,
        )
    except FileNotFoundError as exc:
        return {"status": "data_unavailable", "reason": str(exc)}

    archived_primary = archived.get("metrics_primary") or {}
    reproduced_primary = reproduced.metrics_primary.model_dump(mode="json")
    fields: dict[str, dict[str, float]] = {}
    all_within = True
    for field in _RERUN_COMPARED_FIELDS:
        archived_value = float(archived_primary.get(field) or 0.0)
        reproduced_value = float(reproduced_primary.get(field) or 0.0)
        deviation = abs(reproduced_value - archived_value)
        within = deviation <= max(abs_tolerance, rel_tolerance * abs(archived_value))
        all_within = all_within and within
        fields[field] = {
            "archived": archived_value,
            "reproduced": reproduced_value,
            "deviation": deviation,
            "within_tolerance": within,
        }

    verdict = {
        "status": "verified" if all_within else "mismatch",
        "fields": fields,
        "data_end_ms": data_end_ms,
        "pinned": bool(data_end_ms),
        "trial_counts": trial_counts,
        "rel_tolerance": rel_tolerance,
        "abs_tolerance": abs_tolerance,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(archive_dir / "verify_verdict.json", verdict)
    return verdict


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

