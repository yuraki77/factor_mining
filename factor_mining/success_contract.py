"""Success contract report — a verifiable A/B comparison of evolution
operators against the seed baseline, from stored trajectory artifacts.

Read-only. Per-operator promotion-rate uplift is measured against the
explicit baseline operator set (default: the seed operators) with a seeded
bootstrap confidence interval; operators without enough samples report
``insufficient_data`` instead of a point estimate.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from factor_mining.config import Settings
from factor_mining.storage import MetadataStore

BASELINE_OPERATORS = frozenset({"SEED", "SEED_INFORMED"})
_UPLIFT_MIN_SAMPLES = 20
_UPLIFT_RESAMPLES = 2000


def build_success_contract_report(
    *,
    store: MetadataStore,
    settings: Settings,
    experiment_ids: list[str] | None = None,
    baseline_operators: frozenset[str] | set[str] | None = None,
    min_samples: int = _UPLIFT_MIN_SAMPLES,
    n_resamples: int = _UPLIFT_RESAMPLES,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate trajectory statistics plus the operator A/B contract.

    Returns a dictionary suitable for serialisation as a dashboard artifact
    or CLI report.  When *experiment_ids* is passed the report is scoped to
    those experiments; otherwise all stored trajectories are included.

    Metrics:
    - ``total_candidates_evaluated``
    - ``trajectory_counts_by_operator`` — how many candidates each operator produced
    - ``production_rate_by_operator`` — fraction reaching production_passed
    - ``mean_sharpe_by_operator``
    - ``gatecheck_pass_rate``
    - ``research_survivor_rate``
    - ``baseline_operators`` / ``operator_uplift`` — the A/B contract: each
      non-baseline operator's promoted-rate uplift vs the pooled baseline,
      with a seeded bootstrap 95% CI and a verdict
      (``positive``/``negative``/``inconclusive``/``insufficient_data``).
      Records are resampled iid; round-level clustering is not modelled.
    - ``lineage_depth_stats`` — min / max / mean of parent chain depth
    """
    baseline_ops = frozenset(baseline_operators) if baseline_operators is not None else BASELINE_OPERATORS
    records = store.list_trajectories(limit=2000)

    if experiment_ids is not None:
        experiment_set = set(experiment_ids)
        records = [r for r in records if _scoped(r, experiment_set)]

    if not records:
        return _empty_report()

    total = len(records)
    operator_counter: Counter[str] = Counter()
    production_counter: Counter[str] = Counter()
    sharpe_sums: dict[str, float] = {}
    sharpe_counts: dict[str, int] = {}
    depth_values: list[int] = []

    for record in records:
        op = record.operator
        operator_counter[op] += 1
        if _is_production_passed(record):
            production_counter[op] += 1

        sr = _extract_sharpe(record)
        if sr is not None:
            sharpe_sums[op] = sharpe_sums.get(op, 0.0) + sr
            sharpe_counts[op] = sharpe_counts.get(op, 0) + 1

        depth_values.append(len(record.parent_ids))

    production_count = sum(1 for r in records if _is_production_passed(r))
    survivor_count = sum(1 for r in records if _is_research_survivor(r))
    promoted_count = production_count + survivor_count

    # Per-operator derived metrics
    production_rate_by_operator: dict[str, float] = {}
    research_survivor_rate_by_operator: dict[str, float] = {}
    promoted_rate_by_operator: dict[str, float] = {}
    mean_sharpe_by_operator: dict[str, float | None] = {}
    for op in sorted(operator_counter):
        n = operator_counter[op]
        op_production_count = production_counter[op]
        op_survivor_count = sum(
            1 for r in records
            if r.operator == op and _is_research_survivor(r)
        )
        production_rate_by_operator[op] = op_production_count / n if n > 0 else 0.0
        research_survivor_rate_by_operator[op] = op_survivor_count / n if n > 0 else 0.0
        promoted_rate_by_operator[op] = (
            (op_production_count + op_survivor_count) / n if n > 0 else 0.0
        )
        mean_sharpe_by_operator[op] = (
            sharpe_sums.get(op, 0.0) / sharpe_counts[op] if sharpe_counts.get(op, 0) > 0 else None
        )

    depth_sorted = sorted(depth_values)
    lineage_depth_stats: dict[str, float] = {
        "min": float(depth_sorted[0]) if depth_sorted else 0.0,
        "max": float(depth_sorted[-1]) if depth_sorted else 0.0,
        "mean": float(sum(depth_values) / len(depth_values)) if depth_values else 0.0,
    }

    return {
        "total_candidates_evaluated": total,
        "trajectory_counts_by_operator": dict(operator_counter),
        "production_rate_by_operator": production_rate_by_operator,
        "research_survivor_rate_by_operator": research_survivor_rate_by_operator,
        "promoted_rate_by_operator": promoted_rate_by_operator,
        "mean_sharpe_by_operator": mean_sharpe_by_operator,
        "baseline_operators": sorted(baseline_ops),
        "operator_uplift": _operator_uplift(
            records,
            baseline_ops=baseline_ops,
            min_samples=min_samples,
            n_resamples=n_resamples,
            seed=seed,
        ),
        "gatecheck_pass_rate": production_count / total if total > 0 else 0.0,
        "research_survivor_rate": survivor_count / total if total > 0 else 0.0,
        "promoted_rate": promoted_count / total if total > 0 else 0.0,
        "lineage_depth_stats": lineage_depth_stats,
    }


def _operator_uplift(
    records: list[Any],
    *,
    baseline_ops: frozenset[str],
    min_samples: int,
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Promoted-rate uplift of each non-baseline operator vs the pooled baseline.

    ``verdict`` is ``positive``/``negative`` when the bootstrap 95% CI of the
    rate difference excludes zero, ``inconclusive`` when it straddles zero,
    and ``insufficient_data`` (no CI) when either arm has fewer than
    ``min_samples`` records — a point estimate on a handful of trajectories
    is noise dressed as a comparison."""
    promoted_flags: dict[str, list[int]] = {}
    for record in records:
        flag = 1 if (_is_production_passed(record) or _is_research_survivor(record)) else 0
        promoted_flags.setdefault(record.operator, []).append(flag)

    baseline = np.array(
        [flag for op in sorted(baseline_ops) for flag in promoted_flags.get(op, [])],
        dtype=float,
    )
    uplift: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(seed)
    for op in sorted(promoted_flags):
        if op in baseline_ops:
            continue
        arm = np.array(promoted_flags[op], dtype=float)
        entry: dict[str, Any] = {
            "n": int(arm.size),
            "baseline_n": int(baseline.size),
            "promoted_rate": float(arm.mean()) if arm.size else 0.0,
            "baseline_promoted_rate": float(baseline.mean()) if baseline.size else 0.0,
        }
        if arm.size < min_samples or baseline.size < min_samples:
            entry["verdict"] = "insufficient_data"
            uplift[op] = entry
            continue
        entry["uplift"] = entry["promoted_rate"] - entry["baseline_promoted_rate"]
        arm_idx = rng.integers(0, arm.size, size=(n_resamples, arm.size))
        base_idx = rng.integers(0, baseline.size, size=(n_resamples, baseline.size))
        diffs = arm[arm_idx].mean(axis=1) - baseline[base_idx].mean(axis=1)
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        entry["ci_low"] = float(ci_low)
        entry["ci_high"] = float(ci_high)
        if ci_low > 0.0:
            entry["verdict"] = "positive"
        elif ci_high < 0.0:
            entry["verdict"] = "negative"
        else:
            entry["verdict"] = "inconclusive"
        uplift[op] = entry
    return uplift


def _scoped(record: Any, experiment_ids: set[str]) -> bool:
    """Check whether a trajectory belongs to the requested experiment scope."""
    direct_eid = getattr(record, "experiment_id", None)
    if direct_eid:
        return str(direct_eid) in experiment_ids
    backtest = getattr(record, "backtest_result", None)
    if isinstance(backtest, dict):
        eid = backtest.get("experiment_id", "")
        return str(eid) in experiment_ids
    return False


def _is_production_passed(record: Any) -> bool:
    if getattr(record, "classification", None) == "production_passed":
        return True
    research_gate = getattr(record, "research_gate_snapshot", None)
    return isinstance(research_gate, dict) and research_gate.get("status") == "production_passed"


def _is_research_survivor(record: Any) -> bool:
    if _is_production_passed(record):
        return False
    if getattr(record, "classification", None) == "research_survivor":
        return True
    research_gate = getattr(record, "research_gate_snapshot", None)
    return isinstance(research_gate, dict) and research_gate.get("status") == "research_survivor"


def _extract_sharpe(record: Any) -> float | None:
    backtest = getattr(record, "backtest_result", None)
    if not isinstance(backtest, dict):
        return None
    primary = backtest.get("metrics_primary")
    if not isinstance(primary, dict):
        return None
    sr = primary.get("sharpe")
    return float(sr) if isinstance(sr, int | float) else None


def _empty_report() -> dict[str, Any]:
    return {
        "total_candidates_evaluated": 0,
        "trajectory_counts_by_operator": {},
        "production_rate_by_operator": {},
        "research_survivor_rate_by_operator": {},
        "promoted_rate_by_operator": {},
        "mean_sharpe_by_operator": {},
        "baseline_operators": sorted(BASELINE_OPERATORS),
        "operator_uplift": {},
        "gatecheck_pass_rate": 0.0,
        "research_survivor_rate": 0.0,
        "promoted_rate": 0.0,
        "lineage_depth_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
    }
