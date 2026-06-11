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


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

