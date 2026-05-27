"""
Deterministic strategy optimizer.

The optimizer reads candidate diagnostics, Research Gate classifications, and
near-miss repair hints, then produces bounded candidate mutations for the next
round. It deliberately avoids model-generated tuning so optimization remains
reproducible and auditable.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from factor_mining.config import Settings
from factor_mining.mining import normalize_family
from factor_mining.models import BacktestResult, CandidateStrategySpec, GateCheckResult, NearMissAnalysis, ResearchGateResult
from factor_mining.near_miss import repair_adjustments_from_near_misses


_NEXT_HYPOTHESIS_LOOKBACKS: dict[str, list[int]] = {
    "momentum": [12, 48],
    "mean_reversion": [6, 24],
    "volatility": [24, 48],
    "funding_basis": [12, 96],
    "volume_confirmation": [6, 12],
}

_EVOLUTION_LOOKBACKS = [6, 12, 24, 48, 96]
_EVOLUTION_PARENT_LIMIT = 16
_EVOLUTION_MAX_PER_PARENT = 3
_COMPOSITE_POOL_LIMIT = 24
_COMPOSITE_MAX_COMBOS = 12
_COMPOSITE_GRID_WEIGHTS = (0.20, 0.40, 0.60, 0.80)
_EXIT_PARENT_LIMIT = 8
_EXIT_MAX_PER_PARENT = 3
_EXIT_TOTAL_LIMIT = 24
_HILL_CLIMB_ZSCORE_WINDOWS = (96, 288, 576)
_HILL_CLIMB_TANH_SCALES = (1.0, 2.0, 3.0)
_PROPOSAL_PARAM_KEYS = {
    "smooth_span",
    "signal_threshold",
    "position_buffer",
    "factor_lookback",
    "lookback",
    "regime_filter",
    "funding_state_filter",
    "funding_trend_filter",
    "side_mode",
    "signal_role",
    "zscore_window",
    "tanh_scale",
}


def build_optimization_context(
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
    gatechecks: list[GateCheckResult],
    iteration: int,
    previous_actions: list[dict] | None = None,
    research_gates: list[ResearchGateResult] | None = None,
    near_misses: list[NearMissAnalysis] | None = None,
) -> dict:
    """Build the context object the optimizer needs to understand the current state."""
    factor_summaries = []
    research_gate_by_experiment = {
        gate.experiment_id: gate
        for gate in (research_gates or [])
    }
    near_miss_by_experiment = {
        item.experiment_id: item
        for item in (near_misses or [])
    }
    for c, r, g in zip(candidates, results, gatechecks):
        research_gate = research_gate_by_experiment.get(r.experiment_id)
        near_miss = near_miss_by_experiment.get(r.experiment_id)
        research_score = research_gate.research_score if research_gate is not None else _research_score(r)
        summary = {
            "candidate_id": c.candidate_id,
            "parent_candidate_id": c.parent_candidate_id,
            "hypothesis_family": c.hypothesis_family,
            "method_id": c.method_id,
            "symbol": c.symbol,
            "params": c.params,
            "sharpe": r.metrics_primary.sharpe,
            "gross_sharpe": r.metrics_gross.sharpe if r.metrics_gross is not None else None,
            "cost_drag_sharpe": (
                r.metrics_gross.sharpe - r.metrics_primary.sharpe
                if r.metrics_gross is not None else None
            ),
            "max_dd": r.metrics_primary.max_drawdown,
            "ann_return": r.metrics_primary.annualized_return,
            "calmar": r.metrics_primary.calmar,
            "trade_count": r.metrics_primary.trade_count,
            "ic_tstat": r.ic_tstat_nw,
            "rankic_tstat": r.rankic_tstat_nw,
            "deflated_sr": r.deflated_sharpe,
            "probabilistic_sr": r.probabilistic_sharpe,
            "permutation_p": r.permutation_test_pvalue,
            "pbo": r.pbo,
            "factor_turnover": r.factor_turnover,
            "break_even_cost_bps": r.break_even_cost_bps,
            "actual_cost_bps": r.actual_cost_bps,
            "avg_holding_bars": r.avg_holding_period_bars,
            "oos_trade_count": r.oos_trade_count,
            "research_score": research_score,
            "research_gate_status": research_gate.status if research_gate is not None else None,
            "research_gate_reasons": research_gate.reasons if research_gate is not None else [],
            "near_miss_reason": near_miss.primary_reason if near_miss is not None else None,
            "near_miss_reasons": near_miss.reasons if near_miss is not None else [],
            "repair_actions": near_miss.repair_actions if near_miss is not None else [],
            "suggested_repair_params": near_miss.suggested_params if near_miss is not None else {},
            "suggested_repair_param_variants": near_miss.suggested_param_variants if near_miss is not None else [],
            "gatecheck_passed": g.passed,
            "gatecheck_raw_passed": g.raw_passed,
            "risk_tier": g.risk_tier,
            "allocation_multiplier": g.allocation_multiplier,
            "review_after_days": g.review_after_days,
            "factor_evidence_level": g.factor_evidence_level,
            "gatecheck_failures": [item.rule_id for item in g.failures],
            "gatecheck_warnings": [item.rule_id for item in g.warnings],
            "regime_metrics": {
                regime: {
                    "sharpe": block.sharpe,
                    "pnl": block.pnl,
                }
                for regime, block in r.regime_conditional_metrics.items()
            },
            "exit": _exit_params(c.params),
            "exit_indicators": {
                "max_dd": r.metrics_primary.max_drawdown,
                "avg_holding_bars": r.avg_holding_period_bars,
                "trade_count": r.metrics_primary.trade_count,
                "cost_drag_sharpe": (
                    r.metrics_gross.sharpe - r.metrics_primary.sharpe
                    if r.metrics_gross is not None else None
                ),
            },
        }
        outcome = _optimizer_outcome(c, summary)
        if outcome:
            summary["optimizer_outcome"] = outcome
            summary["optimizer_proposal_signature"] = c.params.get("optimizer_proposal_signature")
        summary["survivor_reason"] = (
            ",".join(research_gate.reasons)
            if research_gate is not None and research_gate.reasons
            else _survivor_reason(summary)
        )
        factor_summaries.append(summary)

    if research_gates is not None:
        research_survivors = _select_formal_research_survivors(factor_summaries)
    else:
        research_survivors = _select_research_survivors(factor_summaries)
    survivor_ids = {item["candidate_id"] for item in research_survivors}
    for factor in factor_summaries:
        factor["research_survivor"] = factor["candidate_id"] in survivor_ids
    optimizer_outcomes = [
        factor["optimizer_outcome"]
        for factor in factor_summaries
        if factor.get("optimizer_outcome")
    ]
    optimizer_outcome_counts = dict(Counter(str(item.get("status", "unknown")) for item in optimizer_outcomes))

    return {
        "iteration": iteration,
        "num_candidates": len(candidates),
        "num_gatecheck_passed": sum(1 for g in gatechecks if g.passed),
        "num_research_survivors": len(research_survivors),
        "survivor_criteria": (
            "Formal ResearchGateResult status is production_passed or research_survivor. "
            "GateCheck pass includes full_pass and conditional_pass risk tiers."
            if research_gates is not None else
            "GateCheck pass, or strong discovery evidence from research_score/gross Sharpe/IC/cost margin. "
            "PBO and bootstrap Sharpe are treated as final-strategy gates, not raw-factor stops."
        ),
        "research_survivors": research_survivors,
        "near_misses": [item.model_dump(mode="json") for item in (near_misses or [])],
        "repair_adjustments": repair_adjustments_from_near_misses(near_misses or []),
        "factors": factor_summaries,
        "optimizer_outcomes": optimizer_outcomes,
        "optimizer_outcome_counts": optimizer_outcome_counts,
        "previous_actions": previous_actions or [],
    }

def optimize_traditionally(context: dict, mode: str = "full") -> dict:
    """Produce deterministic, bounded optimizer suggestions."""
    if mode == "exit_params":
        return optimize_exits_traditionally(context)
    passed = [f for f in context["factors"] if f["gatecheck_passed"]]
    survivors = context.get("research_survivors") or []
    failed = [f for f in context["factors"] if not f["gatecheck_passed"]]

    if not passed:
        # If nothing passes final GateCheck, keep the best discovery candidates
        # so the next round can optimize costs/combination instead of stopping.
        passed = survivors or _select_research_survivors(failed, limit=3)

    selected = passed[:8]
    factor_ids = [f["candidate_id"] for f in selected]
    n = len(factor_ids)
    source = "GateCheck-passing factors" if context.get("num_gatecheck_passed", 0) else "research survivors"
    composite_pool = _select_composite_pool(context.get("factors", []), selected)

    result = {
        "action": "traditional_survivor_low_turnover_combo",
        "reasoning": f"Deterministic optimizer selected {n} {source} with conservative turnover controls.",
        "combinations": _composite_combinations(selected, composite_pool),
        "adjustments": [],
        "next_hypotheses": [],
        "proposal_counts": {
            "combination": 0,
            "repair": 0,
            "evolution": 0,
            "hill_climb": 0,
            "memory_skipped": 0,
        },
    }
    result["proposal_counts"]["combination"] = len(result["combinations"])
    factors_by_id = {factor["candidate_id"]: factor for factor in context.get("factors", [])}
    failed_signatures = _failed_proposal_signatures(context)
    repairs, repair_skips = _prepare_adjustments(
        context.get("repair_adjustments", []),
        factors_by_id,
        failed_signatures,
        default_kind="repair",
    )
    result["adjustments"].extend(repairs)
    result["proposal_counts"]["repair"] += len(repairs)
    result["proposal_counts"]["memory_skipped"] += repair_skips

    evolutions, evolution_skips = _survivor_evolution_adjustments(context, failed_signatures)
    result["adjustments"].extend(evolutions)
    result["proposal_counts"]["evolution"] += len(evolutions)
    result["proposal_counts"]["memory_skipped"] += evolution_skips

    hill_climbs, hill_skips = _hill_climb_adjustments(context, failed_signatures)
    result["adjustments"].extend(hill_climbs)
    result["proposal_counts"]["hill_climb"] += len(hill_climbs)
    result["proposal_counts"]["memory_skipped"] += hill_skips

    if n < 2:
        for factor_id in factor_ids:
            adjustment = {
                "candidate_id": factor_id,
                "param": "turnover_controls",
                "current": "raw",
                "suggested": {
                    "smooth_span": 48,
                    "signal_threshold": 0.25,
                    "position_buffer": 0.20,
                },
                "rationale": "Only one research survivor is available; continue by lowering turnover before final GateCheck.",
                "proposal_kind": "turnover_control",
                "variant_key": "single_survivor_low_turnover",
            }
            prepared, skipped = _prepare_adjustments([adjustment], factors_by_id, failed_signatures, default_kind="turnover_control")
            result["adjustments"].extend(prepared)
            result["proposal_counts"]["memory_skipped"] += skipped

    # Drop factors with low OOS/IS IC ratio (suggest as adjustments)
    selected_ids = set(factor_ids)
    for f in [item for item in failed if item["candidate_id"] not in selected_ids][:5]:
        result["adjustments"].append({
            "candidate_id": f["candidate_id"],
            "param": "status",
            "current": "active",
            "suggested": "paused",
            "rationale": f"GateCheck failed: {f.get('gatecheck_failures', [])}"
        })

    return result


def _select_composite_pool(factors: list[dict], selected: list[dict]) -> list[dict]:
    pool: list[dict] = []
    seen: set[str] = set()

    def add(factor: dict) -> None:
        cid = str(factor.get("candidate_id") or "")
        if not cid or cid in seen:
            return
        seen.add(cid)
        pool.append(factor)

    for factor in selected:
        add(factor)
    viable = [
        factor for factor in factors
        if (
            factor.get("gatecheck_passed")
            or factor.get("research_gate_status") in {"production_passed", "research_survivor"}
            or factor.get("research_survivor")
            or _has_discovery_evidence(factor)
        )
    ]
    for factor in sorted(viable, key=_survivor_sort_key, reverse=True):
        add(factor)
        if len(pool) >= _COMPOSITE_POOL_LIMIT:
            break
    return pool


def _composite_combinations(selected: list[dict], pool: list[dict]) -> list[dict]:
    combos: list[dict] = []
    seen: set[str] = set()

    def add_combo(
        factors: list[dict],
        weights: list[float],
        *,
        subset_strategy: str,
        weighting_scheme: str,
        rationale: str,
    ) -> None:
        if len(combos) >= _COMPOSITE_MAX_COMBOS:
            return
        factor_ids = [str(factor.get("candidate_id")) for factor in factors if factor.get("candidate_id")]
        if not factor_ids or len(weights) != len(factor_ids):
            return
        signature = _combo_signature(factor_ids, weights)
        if signature in seen:
            return
        seen.add(signature)
        search_variant = (
            "survivor_combo_low_turnover"
            if subset_strategy == "top8" and weighting_scheme == "traditional"
            else f"survivor_combo_{subset_strategy}_{weighting_scheme}"
        )
        combos.append({
            "factor_ids": factor_ids,
            "weights": weights,
            "horizon": "5m",
            "signal_threshold": 0.25,
            "smooth_span": 48,
            "position_buffer": 0.20,
            "subset_strategy": subset_strategy,
            "weighting_scheme": weighting_scheme,
            "search_variant": search_variant,
            "rationale": rationale,
            "expected_improvement": "Diversify survivor combinations and let subsequent backtests select robust subsets.",
        })

    add_combo(
        selected,
        _traditional_weights(selected),
        subset_strategy="top8",
        weighting_scheme="traditional",
        rationale="Deterministic fallback: combine research survivors and damp signal churn before final GateCheck.",
    )
    if len(pool) < 2:
        return combos

    top4 = pool[:4]
    add_combo(
        top4,
        _equal_weights(top4),
        subset_strategy="top4",
        weighting_scheme="equal",
        rationale="Compact top-survivor composite with equal weights as a low-variance benchmark.",
    )
    add_combo(
        top4,
        _inverse_turnover_weights(top4),
        subset_strategy="top4",
        weighting_scheme="inverse_turnover",
        rationale="Compact top-survivor composite weighted toward lower-turnover components.",
    )

    diverse = _diverse_subset(pool, limit=8)
    add_combo(
        diverse,
        _traditional_weights(diverse),
        subset_strategy="diverse8",
        weighting_scheme="traditional",
        rationale="Family-diverse survivor composite to reduce single-theme concentration.",
    )

    low_turnover = sorted(pool, key=_turnover_sort_key)[:8]
    add_combo(
        low_turnover,
        _inverse_turnover_weights(low_turnover),
        subset_strategy="low_turnover8",
        weighting_scheme="inverse_turnover",
        rationale="Low-turnover survivor composite for cost-sensitive follow-up testing.",
    )

    high_ic = sorted(pool, key=_ic_strength, reverse=True)[:8]
    add_combo(
        high_ic,
        _ic_strength_weights(high_ic),
        subset_strategy="high_ic8",
        weighting_scheme="ic_strength",
        rationale="IC-weighted survivor composite to test concentrated signal-evidence allocation.",
    )

    for pair in _top_pairs(pool[:4]):
        for weight in _COMPOSITE_GRID_WEIGHTS:
            add_combo(
                pair,
                [float(weight), float(1.0 - weight)],
                subset_strategy="pair_grid",
                weighting_scheme=f"grid_{int(weight * 100)}_{int((1.0 - weight) * 100)}",
                rationale="Small pairwise weight grid for bounded composite allocation search.",
            )
            if len(combos) >= _COMPOSITE_MAX_COMBOS:
                return combos
    return combos


def _combo_signature(factor_ids: list[str], weights: list[float]) -> str:
    payload = {
        "factor_ids": factor_ids,
        "weights": [round(float(weight), 8) for weight in weights],
    }
    return _params_signature(payload)


def _equal_weights(factors: list[dict]) -> list[float]:
    if not factors:
        return []
    return [1.0 / len(factors)] * len(factors)


def _inverse_turnover_weights(factors: list[dict]) -> list[float]:
    raw = [1.0 / max(0.02, min(float(factor.get("factor_turnover") or 0.10), 1.0)) for factor in factors]
    return _normalize_positive_weights(raw)


def _ic_strength_weights(factors: list[dict]) -> list[float]:
    raw = [max(0.25, _ic_strength(factor)) for factor in factors]
    return _normalize_positive_weights(raw)


def _normalize_positive_weights(raw: list[float]) -> list[float]:
    if not raw:
        return []
    total = sum(max(0.0, value) for value in raw)
    if total <= 0:
        return [1.0 / len(raw)] * len(raw)
    return [max(0.0, value) / total for value in raw]


def _diverse_subset(pool: list[dict], *, limit: int) -> list[dict]:
    selected: list[dict] = []
    seen_ids: set[str] = set()
    seen_families: set[str] = set()
    for factor in pool:
        family = str(factor.get("hypothesis_family") or "")
        cid = str(factor.get("candidate_id") or "")
        if not cid or cid in seen_ids or family in seen_families:
            continue
        selected.append(factor)
        seen_ids.add(cid)
        seen_families.add(family)
        if len(selected) >= limit:
            return selected
    for factor in pool:
        cid = str(factor.get("candidate_id") or "")
        if cid and cid not in seen_ids:
            selected.append(factor)
            seen_ids.add(cid)
            if len(selected) >= limit:
                break
    return selected


def _top_pairs(pool: list[dict]) -> list[list[dict]]:
    pairs: list[list[dict]] = []
    for idx, left in enumerate(pool):
        for right in pool[idx + 1:]:
            pairs.append([left, right])
    return pairs


def _turnover_sort_key(factor: dict) -> tuple[float, float]:
    turnover = float(factor.get("factor_turnover") if factor.get("factor_turnover") is not None else 999.0)
    return (turnover, -float(factor.get("research_score") or 0.0))


def _ic_strength(factor: dict) -> float:
    return max(abs(float(factor.get("ic_tstat") or 0.0)), abs(float(factor.get("rankic_tstat") or 0.0)))


def _prepare_adjustments(
    adjustments: list[dict],
    factors_by_id: dict[str, dict],
    failed_signatures: set[str],
    *,
    default_kind: str,
) -> tuple[list[dict], int]:
    prepared: list[dict] = []
    skipped = 0
    for adjustment in adjustments:
        cid = str(adjustment.get("candidate_id") or "")
        factor = factors_by_id.get(cid)
        suggested = adjustment.get("suggested")
        if factor is None or not isinstance(suggested, dict):
            prepared.append(adjustment)
            continue
        kind = str(adjustment.get("proposal_kind") or default_kind)
        param_diff = _optimizer_param_diff(factor.get("params") or {}, suggested)
        if not param_diff:
            continue
        root_parent_id = _optimizer_root_parent_id(factor)
        signature = _proposal_signature(kind, root_parent_id, param_diff)
        if signature in failed_signatures:
            skipped += 1
            continue
        enriched = dict(adjustment)
        enriched["proposal_kind"] = kind
        enriched["param_diff"] = param_diff
        enriched["optimizer_root_parent_id"] = root_parent_id
        enriched["proposal_signature"] = signature
        enriched["proposal_id"] = _proposal_id(signature)
        enriched.setdefault("variant_key", _variant_key(kind, param_diff))
        prepared.append(enriched)
    return prepared, skipped


def _survivor_evolution_adjustments(context: dict, failed_signatures: set[str]) -> tuple[list[dict], int]:
    factors_by_id = {factor["candidate_id"]: factor for factor in context.get("factors", [])}
    parents = _select_evolution_parents(context.get("factors", []))
    adjustments: list[dict] = []
    skipped = 0
    seen: set[str] = set()
    for factor in parents[:_EVOLUTION_PARENT_LIMIT]:
        per_parent = 0
        for variant_key, params, rationale in _evolution_variants(factor):
            if per_parent >= _EVOLUTION_MAX_PER_PARENT:
                break
            adjustment = {
                "candidate_id": factor["candidate_id"],
                "param": "evolution_params",
                "current": "survivor_candidate",
                "suggested": params,
                "proposal_kind": "evolution",
                "variant_key": variant_key,
                "rationale": rationale,
            }
            prepared, skipped_count = _prepare_adjustments(
                [adjustment],
                factors_by_id,
                failed_signatures,
                default_kind="evolution",
            )
            skipped += skipped_count
            for item in prepared:
                signature = str(item.get("proposal_signature") or "")
                if signature in seen:
                    continue
                seen.add(signature)
                adjustments.append(item)
                per_parent += 1
    return adjustments, skipped


def _hill_climb_adjustments(context: dict, failed_signatures: set[str]) -> tuple[list[dict], int]:
    factors_by_id = {factor["candidate_id"]: factor for factor in context.get("factors", [])}
    adjustments: list[dict] = []
    skipped = 0
    seen: set[str] = set()
    for outcome in context.get("optimizer_outcomes", [])[:_EVOLUTION_PARENT_LIMIT]:
        if outcome.get("status") != "improved":
            continue
        cid = str(outcome.get("candidate_id") or "")
        factor = factors_by_id.get(cid)
        if factor is None:
            continue
        params = _hill_climb_params(factor, outcome)
        if not params:
            continue
        adjustment = {
            "candidate_id": cid,
            "param": "evolution_params",
            "current": "improved_optimizer_variant",
            "suggested": params,
            "proposal_kind": "hill_climb",
            "variant_key": "hill_climb_neighborhood",
            "rationale": "Continue one bounded deterministic step from an improved optimizer proposal.",
        }
        prepared, skipped_count = _prepare_adjustments(
            [adjustment],
            factors_by_id,
            failed_signatures,
            default_kind="hill_climb",
        )
        skipped += skipped_count
        for item in prepared:
            signature = str(item.get("proposal_signature") or "")
            if signature in seen:
                continue
            seen.add(signature)
            adjustments.append(item)
    return adjustments[:_EVOLUTION_PARENT_LIMIT], skipped


def _hill_climb_params(factor: dict, outcome: dict) -> dict[str, Any]:
    params = factor.get("params") or {}
    param_diff = outcome.get("param_diff") if isinstance(outcome.get("param_diff"), dict) else {}
    delta_sharpe = outcome.get("delta_sharpe")
    delta_turnover = outcome.get("delta_turnover")
    sharpe_ok = delta_sharpe is None or float(delta_sharpe) >= -0.05
    turnover_improved = delta_turnover is not None and float(delta_turnover) <= -0.01

    if {"smooth_span", "signal_threshold", "position_buffer"} & param_diff.keys() and sharpe_ok and turnover_improved:
        next_controls = _next_turnover_control_step(params)
        if next_controls:
            return next_controls

    if "factor_lookback" in param_diff and (delta_sharpe is None or float(delta_sharpe) >= 0.05):
        return _neighbor_lookback_params(factor)

    if "regime_filter" in param_diff and (delta_sharpe is None or float(delta_sharpe) >= 0.05):
        return _survivor_low_turnover_params(factor)
    if "zscore_window" in param_diff and (delta_sharpe is None or float(delta_sharpe) >= 0.05):
        return _neighbor_grid_value_params(params, "zscore_window", _HILL_CLIMB_ZSCORE_WINDOWS, int)
    if "tanh_scale" in param_diff and (delta_sharpe is None or float(delta_sharpe) >= 0.05):
        return _neighbor_grid_value_params(params, "tanh_scale", _HILL_CLIMB_TANH_SCALES, float)
    return {}


def _neighbor_grid_value_params(
    params: dict[str, Any],
    key: str,
    ladder: tuple[Any, ...],
    caster: Any,
) -> dict[str, Any]:
    try:
        current = caster(params.get(key))
    except (TypeError, ValueError):
        return {}
    values = [caster(value) for value in ladder]
    if current not in values:
        return {}
    idx = values.index(current)
    if idx + 1 < len(values):
        return {key: values[idx + 1]}
    if idx > 0:
        return {key: values[idx - 1]}
    return {}


def _next_turnover_control_step(params: dict[str, Any]) -> dict[str, float | int]:
    try:
        smooth_span = int(params.get("smooth_span", 1))
        signal_threshold = float(params.get("signal_threshold", 0.0))
        position_buffer = float(params.get("position_buffer", 0.05))
    except (TypeError, ValueError):
        return {}
    ladder = (
        {"smooth_span": 12, "signal_threshold": 0.10, "position_buffer": 0.08},
        {"smooth_span": 24, "signal_threshold": 0.20, "position_buffer": 0.15},
        {"smooth_span": 48, "signal_threshold": 0.30, "position_buffer": 0.25},
        {"smooth_span": 96, "signal_threshold": 0.40, "position_buffer": 0.30},
    )
    for rung in ladder:
        if (
            smooth_span < int(rung["smooth_span"])
            or signal_threshold < float(rung["signal_threshold"])
            or position_buffer < float(rung["position_buffer"])
        ):
            return dict(rung)
    return {}


def _select_evolution_parents(factors: list[dict]) -> list[dict]:
    viable = [
        factor for factor in factors
        if (
            factor.get("gatecheck_passed")
            or factor.get("research_gate_status") in {"production_passed", "research_survivor"}
            or factor.get("research_survivor")
        )
    ]
    return sorted(viable, key=_survivor_sort_key, reverse=True)


def _evolution_variants(factor: dict) -> list[tuple[str, dict[str, Any], str]]:
    variants: list[tuple[str, dict[str, Any], str]] = []
    low_turnover = _survivor_low_turnover_params(factor)
    if low_turnover:
        variants.append((
            "survivor_evolve_low_turnover",
            low_turnover,
            "Evolve survivor with a protective low-turnover signal-control variant.",
        ))
    lookback = _neighbor_lookback_params(factor)
    if lookback:
        variants.append((
            "survivor_evolve_lookback",
            lookback,
            "Evolve survivor by testing the nearest factor lookback neighbor.",
        ))
    regime = _regime_protection_params(factor)
    if regime:
        variants.append((
            "survivor_evolve_regime_filter",
            regime,
            "Evolve survivor with a protective regime filter from conditional performance.",
        ))
    return variants


def _survivor_low_turnover_params(factor: dict) -> dict[str, float | int]:
    params = factor.get("params") or {}
    target = (
        {"smooth_span": 48, "signal_threshold": 0.25, "position_buffer": 0.20}
        if float(factor.get("factor_turnover") or 0.0) >= 0.15 or float(factor.get("cost_drag_sharpe") or 0.0) >= 0.5
        else {"smooth_span": 24, "signal_threshold": 0.15, "position_buffer": 0.12}
    )
    return {
        key: value
        for key, value in target.items()
        if float(params.get(key, 0.0) or 0.0) < float(value)
    }


def _neighbor_lookback_params(factor: dict) -> dict[str, int]:
    params = factor.get("params") or {}
    current = params.get("factor_lookback") or params.get("lookback")
    if current is None:
        return {}
    try:
        current_int = int(current)
    except (TypeError, ValueError):
        return {}
    if current_int not in _EVOLUTION_LOOKBACKS:
        return {}
    idx = _EVOLUTION_LOOKBACKS.index(current_int)
    if idx < len(_EVOLUTION_LOOKBACKS) - 1:
        return {"factor_lookback": _EVOLUTION_LOOKBACKS[idx + 1]}
    if idx > 0:
        return {"factor_lookback": _EVOLUTION_LOOKBACKS[idx - 1]}
    return {}


def _regime_protection_params(factor: dict) -> dict[str, list[str]]:
    params = factor.get("params") or {}
    if params.get("regime_filter"):
        return {}
    regime_metrics = factor.get("regime_metrics")
    if not isinstance(regime_metrics, dict) or not regime_metrics:
        return {}
    best_label = None
    best_sharpe = -999.0
    for label, metrics in regime_metrics.items():
        if not isinstance(metrics, dict):
            continue
        sharpe = float(metrics.get("sharpe") or 0.0)
        if sharpe > best_sharpe:
            best_label = str(label)
            best_sharpe = sharpe
    if best_label is None:
        return {}
    net_sharpe = float(factor.get("sharpe") or 0.0)
    if best_sharpe >= max(0.4, net_sharpe + 0.5):
        return {"regime_filter": [best_label]}
    return {}


def optimize_exits_traditionally(context: dict, *, limit: int = _EXIT_PARENT_LIMIT) -> dict:
    """Suggest a small bounded exit grid from observed risk diagnostics."""
    selected = context.get("research_survivors") or [f for f in context["factors"] if f["gatecheck_passed"]]
    adjustments: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for factor in selected[:limit]:
        per_parent = 0
        for adjustment in _traditional_exit_adjustments(factor):
            signature = (str(adjustment.get("candidate_id") or ""), _params_signature(_exit_adjustment_params(adjustment)))
            if signature in seen:
                continue
            seen.add(signature)
            adjustments.append(adjustment)
            per_parent += 1
            if per_parent >= _EXIT_MAX_PER_PARENT or len(adjustments) >= _EXIT_TOTAL_LIMIT:
                break
        if len(adjustments) >= _EXIT_TOTAL_LIMIT:
            break
    return {
        "action": "traditional_exit_grid",
        "reasoning": "Deterministic bounded exit variants from drawdown, holding-period, and cost-drag diagnostics.",
        "exit_adjustments": adjustments,
    }


def _traditional_weights(factors: list[dict]) -> list[float]:
    if not factors:
        return []
    raw: list[float] = []
    for factor in factors:
        ic_strength = max(abs(factor.get("ic_tstat") or 0.0), abs(factor.get("rankic_tstat") or 0.0))
        evidence = max(0.25, min(4.0, ic_strength or factor.get("research_score") or factor.get("gross_sharpe") or 1.0))
        turnover = factor.get("factor_turnover")
        turnover_penalty = 1.0 / max(0.02, min(float(turnover) if turnover is not None else 0.10, 1.0))
        raw.append(evidence * turnover_penalty)
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(factors)] * len(factors)
    return [value / total for value in raw]


def _traditional_exit_adjustment(factor: dict) -> dict | None:
    adjustments = _traditional_exit_adjustments(factor)
    return adjustments[0] if adjustments else None


def _traditional_exit_adjustments(factor: dict) -> list[dict]:
    cid = factor.get("candidate_id")
    if not cid:
        return []
    adjustments: list[dict] = []
    seen: set[str] = set()
    max_dd = float(factor.get("max_dd") or 0.0)
    avg_holding = factor.get("avg_holding_bars")
    cost_drag = factor.get("cost_drag_sharpe")
    trade_count = int(factor.get("trade_count") or 0)

    def add(params: dict[str, object], variant_key: str, reasons: list[str]) -> None:
        if not params:
            return
        signature = _params_signature(params)
        if signature in seen:
            return
        seen.add(signature)
        adjustments.append({
            "candidate_id": cid,
            **params,
            "proposal_kind": "exit",
            "variant_key": variant_key,
            "rationale": ",".join(reasons),
        })

    if max_dd <= -0.12:
        add(
            {"stop_loss_pct": -0.03, "trailing_stop_pct": 0.02},
            "exit_drawdown_balanced",
            ["drawdown_control"],
        )
        add(
            {"stop_loss_pct": -0.02, "trailing_stop_pct": 0.015},
            "exit_drawdown_tight",
            ["drawdown_control", "tight_stop"],
        )
    elif max_dd <= -0.08:
        add(
            {"stop_loss_pct": -0.04, "trailing_stop_pct": 0.025},
            "exit_drawdown_moderate",
            ["drawdown_control"],
        )
    if avg_holding is not None and float(avg_holding) > 500:
        add(
            {"max_hold_bars": 500},
            "exit_max_hold_500",
            ["bounded_holding_period"],
        )
        add(
            {"max_hold_bars": 250},
            "exit_max_hold_250",
            ["bounded_holding_period", "faster_recycle"],
        )
    if cost_drag is not None and float(cost_drag) > 0.5 and trade_count >= 50:
        add(
            {"tp_tiers": [[0.02, 0.50]], "trailing_after_first_tp": True},
            "exit_profit_lock_balanced",
            ["cost_drag_profit_lock"],
        )
        add(
            {"tp_tiers": [[0.01, 0.40]], "trailing_stop_pct": 0.015, "trailing_after_first_tp": True},
            "exit_profit_lock_fast",
            ["cost_drag_profit_lock", "fast_partial"],
        )
    return adjustments[:_EXIT_MAX_PER_PARENT]


def _exit_adjustment_params(adjustment: dict) -> dict[str, object]:
    return {key: adjustment[key] for key in _EXIT_PARAM_KEYS if key in adjustment}


def _select_research_survivors(factors: list[dict], *, limit: int = 8) -> list[dict]:
    viable = [factor for factor in factors if _has_discovery_evidence(factor)]
    ranked = sorted(viable, key=_survivor_sort_key, reverse=True)
    return ranked[:limit]


def _select_formal_research_survivors(factors: list[dict], *, limit: int = 8) -> list[dict]:
    viable = [
        factor for factor in factors
        if factor.get("research_gate_status") in {"production_passed", "research_survivor"}
    ]
    ranked = sorted(
        viable,
        key=lambda factor: (
            factor.get("research_gate_status") == "production_passed",
            factor.get("research_score", -999),
            factor.get("gross_sharpe") if factor.get("gross_sharpe") is not None else -999,
            -float(factor.get("factor_turnover") if factor.get("factor_turnover") is not None else 999.0),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _has_discovery_evidence(factor: dict) -> bool:
    if factor.get("gatecheck_passed"):
        return True
    if (factor.get("research_score") or 0.0) >= 1.5:
        return True
    gross = factor.get("gross_sharpe")
    if gross is not None and gross >= 0.5:
        return True
    if max(abs(factor.get("ic_tstat") or 0.0), abs(factor.get("rankic_tstat") or 0.0)) >= 1.5:
        return True
    if (factor.get("permutation_p") or 1.0) <= 0.20:
        return True
    cost_margin = _cost_margin_bps(factor)
    return cost_margin is not None and cost_margin > 0.0 and (factor.get("sharpe") or 0.0) > 0.0


def _survivor_sort_key(factor: dict) -> tuple[float, float, float, float, float]:
    gross = factor.get("gross_sharpe")
    turnover = factor.get("factor_turnover")
    return (
        float(factor.get("gatecheck_passed") is True),
        float(factor.get("research_score") or -999.0),
        float(gross if gross is not None else -999.0),
        float(_cost_margin_bps(factor) or -999.0),
        -float(turnover if turnover is not None else 999.0),
    )


def _survivor_reason(factor: dict) -> str | None:
    if factor.get("gatecheck_passed"):
        return "final_gatecheck_pass"
    reasons: list[str] = []
    gross = factor.get("gross_sharpe")
    cost_margin = _cost_margin_bps(factor)
    ic_strength = max(abs(factor.get("ic_tstat") or 0.0), abs(factor.get("rankic_tstat") or 0.0))
    if (factor.get("research_score") or 0.0) >= 1.5:
        reasons.append("research_score")
    if gross is not None and gross >= 0.5:
        reasons.append("gross_sharpe")
    if ic_strength >= 1.5:
        reasons.append("ic_strength")
    if (factor.get("permutation_p") or 1.0) <= 0.20:
        reasons.append("permutation_p")
    if cost_margin is not None and cost_margin > 0.0:
        reasons.append("cost_margin")
    return ",".join(reasons) if reasons else None


def _research_score(result: BacktestResult) -> float:
    gross_sharpe = result.metrics_gross.sharpe if result.metrics_gross is not None else result.metrics_primary.sharpe
    cost_margin = result.break_even_cost_bps - 2.0 * result.actual_cost_bps
    ic_strength = max(abs(result.ic_tstat_nw), abs(result.rankic_tstat_nw))
    score = 0.0
    score += max(-2.0, min(4.0, gross_sharpe))
    score += 1.0 if result.metrics_primary.sharpe > 0 else 0.0
    score += 1.0 if ic_strength >= 2.0 else 0.0
    score += 1.0 if result.permutation_test_pvalue <= 0.10 else 0.0
    score += 1.0 if cost_margin > 0 else 0.0
    score -= 1.0 if result.oos_trade_count == 0 else 0.0
    return float(score)


def _cost_margin_bps(factor: dict) -> float | None:
    break_even = factor.get("break_even_cost_bps")
    actual = factor.get("actual_cost_bps")
    if break_even is None or actual is None:
        return None
    return float(break_even) - 2.0 * float(actual)


def _optimizer_root_parent_id(factor: dict) -> str:
    params = factor.get("params") or {}
    return str(
        params.get("optimizer_root_parent_id")
        or params.get("parent_id")
        or factor.get("parent_candidate_id")
        or factor.get("candidate_id")
    )


def _optimizer_param_diff(current_params: dict, suggested: dict) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, value in suggested.items():
        if key not in _PROPOSAL_PARAM_KEYS:
            continue
        if current_params.get(key) != value:
            diff[key] = value
    return diff


def _proposal_signature(kind: str, root_parent_id: str, param_diff: dict[str, Any]) -> str:
    payload = {
        "kind": kind,
        "root_parent_id": root_parent_id,
        "param_diff": param_diff,
    }
    return hashlib.sha256(_params_signature(payload).encode("utf-8")).hexdigest()


def _proposal_id(signature: str) -> str:
    return f"opt_{signature[:12]}"


def _variant_key(kind: str, param_diff: dict[str, Any]) -> str:
    parts = [f"{key}={param_diff[key]}" for key in sorted(param_diff)]
    return f"{kind}_{'_'.join(parts)}" if parts else kind


def _parent_metrics_for(candidate_id: str, result_by_candidate: dict[str, BacktestResult]) -> dict[str, float | None]:
    result = result_by_candidate.get(candidate_id)
    if result is None:
        return {
            "sharpe": None,
            "gross_sharpe": None,
            "max_dd": None,
            "factor_turnover": None,
            "cost_margin_bps": None,
        }
    gross = result.metrics_gross.sharpe if result.metrics_gross is not None else None
    return {
        "sharpe": result.metrics_primary.sharpe,
        "gross_sharpe": gross,
        "max_dd": result.metrics_primary.max_drawdown,
        "factor_turnover": result.factor_turnover,
        "cost_margin_bps": result.break_even_cost_bps - 2.0 * result.actual_cost_bps,
    }


def _apply_optimizer_lineage(
    candidate: CandidateStrategySpec,
    source: CandidateStrategySpec,
    adj: dict,
    suggested: dict,
    result_by_candidate: dict[str, BacktestResult],
    *,
    default_kind: str,
) -> None:
    param_diff = adj.get("param_diff")
    if not isinstance(param_diff, dict):
        param_diff = _optimizer_param_diff(source.params, suggested)
    root_parent_id = str(
        adj.get("optimizer_root_parent_id")
        or source.params.get("optimizer_root_parent_id")
        or source.params.get("parent_id")
        or source.parent_candidate_id
        or source.candidate_id
    )
    kind = str(adj.get("proposal_kind") or default_kind)
    signature = str(adj.get("proposal_signature") or _proposal_signature(kind, root_parent_id, param_diff))
    candidate.params["optimizer_proposal_id"] = str(adj.get("proposal_id") or _proposal_id(signature))
    candidate.params["optimizer_proposal_signature"] = signature
    candidate.params["optimizer_proposal_kind"] = kind
    candidate.params["optimizer_root_parent_id"] = root_parent_id
    candidate.params["optimizer_param_diff"] = param_diff
    candidate.params["optimizer_reason"] = adj.get("rationale", "")
    candidate.params["optimizer_variant_key"] = adj.get("variant_key") or _variant_key(kind, param_diff)
    candidate.params["optimizer_parent_metrics"] = _parent_metrics_for(source.candidate_id, result_by_candidate)


def _optimizer_outcome(candidate: CandidateStrategySpec, summary: dict) -> dict[str, Any] | None:
    params = candidate.params
    parent_metrics = params.get("optimizer_parent_metrics")
    if not isinstance(parent_metrics, dict) or not params.get("optimizer_proposal_signature"):
        return None
    current = {
        "sharpe": summary.get("sharpe"),
        "gross_sharpe": summary.get("gross_sharpe"),
        "max_dd": summary.get("max_dd"),
        "factor_turnover": summary.get("factor_turnover"),
        "cost_margin_bps": _cost_margin_bps(summary),
    }
    deltas = {
        "delta_sharpe": _delta(current.get("sharpe"), parent_metrics.get("sharpe")),
        "delta_turnover": _delta(current.get("factor_turnover"), parent_metrics.get("factor_turnover")),
        "delta_max_dd": _delta(current.get("max_dd"), parent_metrics.get("max_dd")),
        "delta_cost_margin_bps": _delta(current.get("cost_margin_bps"), parent_metrics.get("cost_margin_bps")),
    }
    status = _outcome_status(deltas)
    return {
        "candidate_id": candidate.candidate_id,
        "parent_id": params.get("parent_id") or candidate.parent_candidate_id,
        "root_parent_id": params.get("optimizer_root_parent_id"),
        "proposal_id": params.get("optimizer_proposal_id"),
        "proposal_signature": params.get("optimizer_proposal_signature"),
        "proposal_kind": params.get("optimizer_proposal_kind"),
        "variant_key": params.get("optimizer_variant_key"),
        "param_diff": params.get("optimizer_param_diff") or {},
        "status": status,
        "parent_metrics": parent_metrics,
        "current_metrics": current,
        **deltas,
    }


def _delta(current: Any, previous: Any) -> float | None:
    if current is None or previous is None:
        return None
    return float(current) - float(previous)


def _outcome_status(deltas: dict[str, float | None]) -> str:
    delta_sharpe = deltas.get("delta_sharpe")
    delta_turnover = deltas.get("delta_turnover")
    delta_cost = deltas.get("delta_cost_margin_bps")
    delta_dd = deltas.get("delta_max_dd")
    sharpe_ok = delta_sharpe is not None and delta_sharpe >= 0.05
    turnover_ok = delta_turnover is not None and delta_turnover <= -0.01
    cost_ok = delta_cost is not None and delta_cost >= 0.5
    drawdown_ok = delta_dd is not None and delta_dd >= 0.01
    sharpe_not_bad = delta_sharpe is None or delta_sharpe >= -0.05
    if sharpe_ok or ((turnover_ok or cost_ok or drawdown_ok) and sharpe_not_bad):
        return "improved"
    if (
        (delta_sharpe is not None and delta_sharpe <= -0.05 and not (turnover_ok or cost_ok))
        or (delta_turnover is not None and delta_turnover >= 0.02 and not sharpe_ok)
    ):
        return "failed"
    return "neutral"


def _failed_proposal_signatures(context: dict) -> set[str]:
    failed = {
        str(outcome.get("proposal_signature"))
        for outcome in context.get("optimizer_outcomes", [])
        if outcome.get("status") == "failed" and outcome.get("proposal_signature")
    }
    for history in context.get("previous_actions", []):
        failed.update(_failed_proposal_signatures_from_history(history))
    return failed


def _failed_proposal_signatures_from_history(history: Any) -> set[str]:
    failed: set[str] = set()
    if isinstance(history, dict):
        for outcome in history.get("optimizer_outcomes", []) or []:
            if isinstance(outcome, dict) and outcome.get("status") == "failed" and outcome.get("proposal_signature"):
                failed.add(str(outcome["proposal_signature"]))
        for child in history.get("children", []) or []:
            failed.update(_failed_proposal_signatures_from_history(child))
    return failed


def apply_optimization_result(
    optimization: dict,
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
    *,
    allow_repairs: bool = True,
    allow_next_hypotheses: bool = True,
) -> tuple[list[CandidateStrategySpec], dict]:
    """Apply deterministic optimizer suggestions to create new candidate specs.

    Returns (new_candidates, optimization_summary).
    """
    new_candidates: list[CandidateStrategySpec] = []
    summary = {
        "combinations_created": 0,
        "adjustments_applied": 0,
        "repairs_created": 0,
        "repairs_suppressed": 0,
        "evolutions_created": 0,
        "hypotheses_suggested": 0,
        "hypotheses_suppressed": 0,
        "status_adjustments_recorded": 0,
    }

    # Create new candidates from suggested combinations
    base_candidate = candidates[0] if candidates else None
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    result_by_candidate = {result.candidate_id: result for result in results}
    for combo in optimization.get("combinations", []):
        factor_ids = _combo_factor_ids(combo)
        weights = combo.get("weights", [])
        if len(factor_ids) >= 2 and len(weights) == len(factor_ids):
            components = [
                candidate_by_id[factor_id].model_dump(mode="json")
                for factor_id in factor_ids
                if factor_id in candidate_by_id
            ]
            if len(components) < 2:
                continue
            turnover_controls = _combo_turnover_controls(combo, components, result_by_candidate)
            # Create a combined candidate
            import uuid
            new_candidates.append(CandidateStrategySpec(
                candidate_id=f"c_opt_{uuid.uuid4().hex[:12]}",
                hypothesis_id="h_optimized_composite",
                method_id="condition_combination_search",
                hypothesis_family="composite",
                symbol=base_candidate.symbol if base_candidate else "BTCUSDT",
                market=base_candidate.market if base_candidate else "um_futures",
                interval=base_candidate.interval if base_candidate else "5m",
                candidate_type="composite",
                params={
                    "factor_ids": factor_ids,
                    "weights": weights,
                    "components": components,
                    "horizon": combo.get("horizon", "5m"),
                    "rationale": combo.get("rationale", ""),
                    "search_variant": combo.get("search_variant", "survivor_combo_low_turnover"),
                    "generated_by": "traditional_survivor_composite",
                    "turnover_objective": "reduce_churn_before_final_gate",
                    "subset_strategy": combo.get("subset_strategy", "top8"),
                    "weighting_scheme": combo.get("weighting_scheme", "traditional"),
                    **turnover_controls,
                },
            ))
            summary["combinations_created"] += 1

    # Apply parameter adjustments to spawn new candidates
    import uuid
    repair_signatures: set[tuple[str, str]] = set()
    for adj in optimization.get("adjustments", []):
        cid, param_name, suggested = _normalized_adjustment(adj)
        for c in candidates:
            if c.candidate_id == cid and param_name and suggested is not None:
                if param_name == "status":
                    summary["status_adjustments_recorded"] += 1
                elif param_name == "turnover_controls" and isinstance(suggested, dict):
                    new_c = c.model_copy(deep=True)
                    new_c.candidate_id = f"c_adj_{uuid.uuid4().hex[:12]}"
                    new_c.candidate_type = "optimizer"
                    new_c.parent_candidate_id = cid
                    for key in ("smooth_span", "signal_threshold", "position_buffer"):
                        if key in suggested:
                            new_c.params[key] = suggested[key]
                    new_c.params["parent_id"] = cid
                    new_c.params["rationale"] = adj.get("rationale", "")
                    new_c.params["search_variant"] = "survivor_low_turnover"
                    new_c.params["generated_by"] = "traditional_survivor_adjustment"
                    _apply_optimizer_lineage(new_c, c, adj, suggested, result_by_candidate, default_kind="turnover_control")
                    new_candidates.append(new_c)
                    summary["adjustments_applied"] += 1
                elif param_name == "repair_params" and isinstance(suggested, dict):
                    if not allow_repairs:
                        summary["repairs_suppressed"] += 1
                        continue
                    signature = (cid, _params_signature(suggested))
                    if signature in repair_signatures:
                        continue
                    repair_signatures.add(signature)
                    new_c = c.model_copy(deep=True)
                    new_c.candidate_id = f"c_rep_{uuid.uuid4().hex[:12]}"
                    new_c.candidate_type = "repair"
                    new_c.parent_candidate_id = cid
                    reason = suggested.get("near_miss_reason") or "optimizer_repair"
                    repair_meta = {
                        "parent_id": cid,
                        "rationale": adj.get("rationale", ""),
                        "generated_by": "near_miss_repair" if suggested.get("near_miss_reason") else "optimizer_repair",
                        "search_variant": f"repair_{reason}",
                    }
                    for key, value in suggested.items():
                        new_c.params[key] = value
                    new_c.params.update(repair_meta)
                    _apply_optimizer_lineage(new_c, c, adj, suggested, result_by_candidate, default_kind="repair")
                    new_candidates.append(new_c)
                    summary["adjustments_applied"] += 1
                    summary["repairs_created"] += 1
                elif param_name == "evolution_params" and isinstance(suggested, dict):
                    new_c = c.model_copy(deep=True)
                    new_c.candidate_id = f"c_evo_{uuid.uuid4().hex[:12]}"
                    new_c.candidate_type = "optimizer"
                    new_c.parent_candidate_id = cid
                    for key, value in suggested.items():
                        new_c.params[key] = value
                    new_c.params["parent_id"] = cid
                    new_c.params["rationale"] = adj.get("rationale", "")
                    new_c.params["search_variant"] = str(adj.get("variant_key") or "survivor_evolution")
                    new_c.params["generated_by"] = "traditional_survivor_evolution"
                    _apply_optimizer_lineage(new_c, c, adj, suggested, result_by_candidate, default_kind="evolution")
                    new_candidates.append(new_c)
                    summary["adjustments_applied"] += 1
                    summary["evolutions_created"] += 1
                else:
                    # Spawn a new candidate with the tweaked parameter
                    new_c = c.model_copy(deep=True)
                    new_c.candidate_id = f"c_adj_{uuid.uuid4().hex[:12]}"
                    new_c.candidate_type = "optimizer"
                    new_c.parent_candidate_id = cid
                    new_c.params[param_name] = suggested
                    new_c.params["parent_id"] = cid
                    new_c.params["rationale"] = adj.get("rationale", "")
                    if adj.get("proposal_kind"):
                        _apply_optimizer_lineage(new_c, c, adj, {param_name: suggested}, result_by_candidate, default_kind=str(adj["proposal_kind"]))
                    new_candidates.append(new_c)
                    summary["adjustments_applied"] += 1

    next_hypotheses = optimization.get("next_hypotheses", [])
    if not allow_next_hypotheses:
        summary["hypotheses_suppressed"] = len(next_hypotheses)
        next_hypotheses = []

    for hypothesis in next_hypotheses:
        if not isinstance(hypothesis, dict) or base_candidate is None:
            continue
        raw_family = str(hypothesis.get("family", ""))
        canonical = normalize_family(raw_family)
        if canonical is None:
            continue
        expected_ic = _expected_ic_mid(hypothesis.get("expected_ic"))
        for lookback in _NEXT_HYPOTHESIS_LOOKBACKS.get(canonical, [12]):
            new_candidates.append(CandidateStrategySpec(
                candidate_id=f"c_hyp_{uuid.uuid4().hex[:12]}",
                hypothesis_id=f"h_traditional_{uuid.uuid4().hex[:8]}",
                method_id="factor_scoring",
                hypothesis_family=canonical,
                symbol=base_candidate.symbol,
                market=base_candidate.market,
                interval=base_candidate.interval,
                candidate_type="optimizer",
                params={
                    "signal_source": "factor_signal",
                    "factor_family": canonical,
                    "factor_lookback": lookback,
                    "transform": "raw_clip",
                    "expected_ic_mid": expected_ic,
                    "source_family": raw_family,
                    "mechanism": hypothesis.get("mechanism", ""),
                    "generated_by": "traditional_next_hypothesis",
                },
            ))

    summary["hypotheses_suggested"] = len(next_hypotheses)

    return new_candidates, summary


def apply_exit_adjustments(
    optimization: dict,
    candidates: list[CandidateStrategySpec],
    settings: Settings,
) -> list[CandidateStrategySpec]:
    """Apply bounded exit parameter adjustments onto existing candidates.

    Deduplicates by (parent_id, exit_params) signature so the optimizer cannot
    inflate the candidate pool with near-identical exit variants.
    """
    import uuid as _uuid
    bounds = settings.exit_bounds
    exit_adjustments = optimization.get("exit_adjustments", [])
    if not exit_adjustments:
        return candidates
    candidate_by_id = {c.candidate_id: c for c in candidates}
    new_candidates = list(candidates)
    seen: set[tuple] = {
        _exit_variant_signature(candidate, _effective_exit_params(candidate.params, settings))
        for candidate in candidates
    }
    for adj in exit_adjustments:
        cid = adj.get("candidate_id", "")
        target = candidate_by_id.get(cid)
        if target is None:
            continue
        clamped = _clamp_exit_params(adj, bounds)
        if not clamped:
            continue
        effective = _effective_exit_params(target.params, settings)
        effective.update(clamped)
        sig = _exit_variant_signature(target, effective)
        if sig in seen:
            continue
        seen.add(sig)
        new_c = target.model_copy(deep=True)
        new_c.candidate_id = f"c_exit_{_uuid.uuid4().hex[:12]}"
        new_c.candidate_type = "optimizer"
        new_c.parent_candidate_id = cid
        new_c.params.update(clamped)
        new_c.params["parent_id"] = cid
        new_c.params["generated_by"] = "traditional_exit_adjustment"
        new_c.params["exit_rationale"] = adj.get("rationale", "")
        if adj.get("variant_key"):
            new_c.params["exit_variant_key"] = adj.get("variant_key")
        if adj.get("proposal_kind"):
            new_c.params["exit_proposal_kind"] = adj.get("proposal_kind")
        new_candidates.append(new_c)
    return new_candidates


def _combo_turnover_controls(combo: dict, components: list[dict], result_by_candidate: dict[str, BacktestResult]) -> dict:
    component_ids = [item.get("candidate_id") for item in components]
    component_results = [result_by_candidate[cid] for cid in component_ids if cid in result_by_candidate]
    max_turnover = max((result.factor_turnover for result in component_results), default=0.0)
    worst_cost_drag = max(
        (
            result.metrics_gross.sharpe - result.metrics_primary.sharpe
            for result in component_results
            if result.metrics_gross is not None
        ),
        default=0.0,
    )

    turnover_control = combo.get("turnover_control") if isinstance(combo.get("turnover_control"), dict) else {}
    smooth_span = _combo_int(combo, "smooth_span", default=_combo_int(turnover_control, "smooth_span", default=24))
    signal_threshold = _combo_float(
        combo,
        "signal_threshold",
        default=_combo_float(turnover_control, "signal_threshold", default=0.20),
    )
    position_buffer = _combo_float(
        combo,
        "position_buffer",
        default=_combo_float(turnover_control, "position_buffer", default=0.15),
    )

    if max_turnover > 0.20 or worst_cost_drag > 1.0:
        smooth_span = max(smooth_span, 48)
        signal_threshold = max(signal_threshold, 0.25)
        position_buffer = max(position_buffer, 0.20)

    return {
        "smooth_span": smooth_span,
        "signal_threshold": signal_threshold,
        "position_buffer": position_buffer,
    }


def _combo_factor_ids(combo: dict) -> list[str]:
    raw_ids = combo.get("factor_ids")
    if not raw_ids:
        raw_ids = combo.get("components")
    if not isinstance(raw_ids, list):
        return []
    factor_ids: list[str] = []
    for item in raw_ids:
        if isinstance(item, str):
            factor_ids.append(item)
        elif isinstance(item, dict) and item.get("candidate_id"):
            factor_ids.append(str(item["candidate_id"]))
    return factor_ids


def _normalized_adjustment(adj: dict) -> tuple[str | None, str | None, object | None]:
    cid = adj.get("candidate_id")
    param_name = adj.get("param")
    suggested = adj.get("suggested")
    if param_name and suggested is not None:
        return cid, str(param_name), suggested

    suggested_param = adj.get("suggested_param")
    if isinstance(suggested_param, str):
        parsed = _parse_param_assignment(suggested_param)
        if parsed is not None:
            return cid, parsed[0], parsed[1]

    if isinstance(suggested, dict):
        return cid, "repair_params", {
            **suggested,
            "optimizer_repair_source": "traditional_adjustment",
        }
    return cid, None, None


def _params_signature(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))


def _parse_param_assignment(value: str) -> tuple[str, object] | None:
    if ":" not in value:
        return None
    key, raw = value.split(":", 1)
    key = key.strip()
    raw = raw.strip()
    if not key:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        items = [item.strip().strip("'").strip('"') for item in raw.strip("[]").split(",") if item.strip()]
        return key, items
    try:
        if "." in raw:
            return key, float(raw)
        return key, int(raw)
    except ValueError:
        return key, raw.strip("'").strip('"')


def _combo_int(combo: dict, key: str, *, default: int) -> int:
    try:
        return max(1, int(combo.get(key, default)))
    except (TypeError, ValueError):
        return default


def _combo_float(combo: dict, key: str, *, default: float) -> float:
    try:
        return max(0.0, float(combo.get(key, default)))
    except (TypeError, ValueError):
        return default


_EXIT_PARAM_KEYS = {"stop_loss_pct", "max_hold_bars", "tp_tiers", "trailing_stop_pct", "trailing_after_first_tp"}


def _exit_params(params: dict) -> dict:
    return {key: params.get(key) for key in _EXIT_PARAM_KEYS if key in params}


def _clamp_exit_params(params: dict, bounds) -> dict:
    """Clamp exit parameters to allowed ranges from ExitBoundsConfig."""
    clamped: dict[str, object] = {}
    if "stop_loss_pct" in params:
        sl = float(params["stop_loss_pct"])
        clamped["stop_loss_pct"] = 0.0 if sl >= 0.0 else float(max(bounds.stop_loss_pct_min, min(bounds.stop_loss_pct_max, sl)))
    if "max_hold_bars" in params:
        mh = int(params["max_hold_bars"])
        clamped["max_hold_bars"] = 0 if mh <= 0 else int(max(bounds.max_hold_bars_min, min(bounds.max_hold_bars_max, mh)))
    if "tp_tiers" in params:
        raw = params["tp_tiers"]
        if isinstance(raw, list):
            clamped_tiers: list[list[float]] = []
            for tier in raw[:bounds.max_tp_tiers]:
                if isinstance(tier, list | tuple) and len(tier) >= 2:
                    pct = max(bounds.tp_tier_pct_min, min(bounds.tp_tier_pct_max, float(tier[0])))
                    frac = max(bounds.tp_tier_fraction_min, min(bounds.tp_tier_fraction_max, float(tier[1])))
                    clamped_tiers.append([pct, frac])
            clamped["tp_tiers"] = clamped_tiers
    if "trailing_stop_pct" in params:
        tr = float(params["trailing_stop_pct"])
        clamped["trailing_stop_pct"] = 0.0 if tr <= 0.0 else float(max(bounds.trailing_stop_pct_min, min(bounds.trailing_stop_pct_max, tr)))
    if "trailing_after_first_tp" in params:
        clamped["trailing_after_first_tp"] = bool(params["trailing_after_first_tp"])
    return clamped


def _effective_exit_params(params: dict, settings: Settings) -> dict:
    ex = settings.exit
    return {
        "stop_loss_pct": float(params.get("stop_loss_pct", ex.stop_loss_pct)),
        "max_hold_bars": int(params.get("max_hold_bars", ex.max_hold_bars)),
        "tp_tiers": _normalize_tp_tiers(params.get("tp_tiers", ex.tp_tiers)),
        "trailing_stop_pct": float(params.get("trailing_stop_pct", ex.trailing_stop_pct)),
        "trailing_after_first_tp": bool(params.get("trailing_after_first_tp", ex.trailing_after_first_tp)),
    }


def _normalize_tp_tiers(raw) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list | tuple):
        return ()
    tiers: list[tuple[float, float]] = []
    for tier in raw:
        if isinstance(tier, list | tuple) and len(tier) >= 2:
            tiers.append((float(tier[0]), float(tier[1])))
    return tuple(sorted(tiers, key=lambda tier: tier[0]))


def _exit_signature(params: dict) -> tuple:
    return (
        round(float(params.get("stop_loss_pct", 0.0)), 10),
        int(params.get("max_hold_bars", 0)),
        _normalize_tp_tiers(params.get("tp_tiers", ())),
        round(float(params.get("trailing_stop_pct", 0.0)), 10),
        bool(params.get("trailing_after_first_tp", True)),
    )


def _exit_variant_signature(candidate: CandidateStrategySpec, effective_exit: dict) -> tuple:
    return (
        str(candidate.params.get("parent_id") or candidate.candidate_id),
        _exit_signature(effective_exit),
    )


def _expected_ic_mid(value) -> float:
    if isinstance(value, int | float):
        return abs(float(value)) or 0.02
    if isinstance(value, list | tuple) and value:
        numeric = [abs(float(item)) for item in value if isinstance(item, int | float)]
        if numeric:
            return float(sum(numeric) / len(numeric))
    return 0.02


def optimization_loop(
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
    gatechecks: list[GateCheckResult],
    settings: Settings,
    *,
    max_iterations: int = 5,
) -> list[dict]:
    """Run the full optimization loop.

    Each iteration: optimize → apply → (user runs backtests on new candidates) → repeat.

    Returns a history of optimization actions.
    """
    history = []
    for i in range(max_iterations):
        context = build_optimization_context(candidates, results, gatechecks, i, history)
        opt_result = optimize_traditionally(context, mode="full")
        new_candidates, summary = apply_optimization_result(opt_result, candidates, results)

        history.append({
            "iteration": i,
            "optimization": opt_result,
            "summary": summary,
            "new_candidates_count": len(new_candidates),
        })

        if summary["combinations_created"] == 0 and summary["adjustments_applied"] == 0 and summary["hypotheses_suggested"] == 0:
            break

    return history
