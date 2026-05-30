"""Success contract report — compare current optimizer performance against
evolutionary wrapper baseline using existing trajectory artifacts.

This report is pure read-only aggregation.  It reads stored trajectories
and summarises pipeline efficiency without changing any production behaviour.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from factor_mining.config import Settings
from factor_mining.storage import MetadataStore


def build_success_contract_report(
    *,
    store: MetadataStore,
    settings: Settings,
    experiment_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate trajectory statistics for the current pipeline run.

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
    - ``operator_effectiveness`` — composite (production_rate + 2 * survivor_rate) / 3
    - ``lineage_depth_stats`` — min / max / mean of parent chain depth
    """
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
    effectiveness_by_operator: dict[str, float] = {}
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
        effectiveness_by_operator[op] = (
            production_rate_by_operator[op] + 2.0 * research_survivor_rate_by_operator[op]
        ) / 3.0

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
        "operator_effectiveness": effectiveness_by_operator,
        "gatecheck_pass_rate": production_count / total if total > 0 else 0.0,
        "research_survivor_rate": survivor_count / total if total > 0 else 0.0,
        "promoted_rate": promoted_count / total if total > 0 else 0.0,
        "lineage_depth_stats": lineage_depth_stats,
    }


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
        "operator_effectiveness": {},
        "gatecheck_pass_rate": 0.0,
        "research_survivor_rate": 0.0,
        "promoted_rate": 0.0,
        "lineage_depth_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
    }
