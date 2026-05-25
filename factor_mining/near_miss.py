from __future__ import annotations

import json
from typing import Any

from factor_mining.models import (
    BacktestResult,
    CandidateStrategySpec,
    FactorEvidenceReport,
    GateCheckResult,
    NearMissAnalysis,
    ResearchGateResult,
)


def analyze_near_misses(
    *,
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
    gatechecks: list[GateCheckResult],
    evidence_reports: list[FactorEvidenceReport],
    research_gates: list[ResearchGateResult],
) -> list[NearMissAnalysis]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    evidence_by_exp = {report.experiment_id: report for report in evidence_reports}
    research_by_exp = {gate.experiment_id: gate for gate in research_gates}
    out: list[NearMissAnalysis] = []
    for result, gatecheck in zip(results, gatechecks, strict=False):
        out.append(
            analyze_near_miss(
                candidate=candidate_by_id.get(result.candidate_id),
                result=result,
                gatecheck=gatecheck,
                evidence=evidence_by_exp.get(result.experiment_id),
                research_gate=research_by_exp.get(result.experiment_id),
            )
        )
    return out


def analyze_near_miss(
    *,
    candidate: CandidateStrategySpec | None,
    result: BacktestResult,
    gatecheck: GateCheckResult,
    evidence: FactorEvidenceReport | None,
    research_gate: ResearchGateResult | None,
) -> NearMissAnalysis:
    if gatecheck.passed:
        return NearMissAnalysis(
            experiment_id=result.experiment_id,
            candidate_id=result.candidate_id,
            primary_reason="production_passed",
            reasons=["production_gate_passed"],
            diagnostics=_diagnostics(candidate, result, gatecheck, evidence, research_gate),
        )

    diagnostics = _diagnostics(candidate, result, gatecheck, evidence, research_gate)
    reasons: list[str] = []
    suggested: dict[str, Any] = {}
    actions: list[str] = []

    is_underpowered_survivor = _statistically_underpowered_survivor(diagnostics)
    if is_underpowered_survivor:
        reasons.append("statistically_underpowered_survivor")
        actions.append("accumulate_oos_evidence")

    if _cost_destroyed_edge(diagnostics):
        reasons.append("cost_destroyed_edge")
        if _suggest_low_turnover_params(candidate, result, suggested):
            actions.append("reduce_turnover")
        else:
            reasons.append("turnover_repair_saturated")

    if _excess_turnover(diagnostics):
        reasons.append("excess_turnover")
        if _suggest_low_turnover_params(candidate, result, suggested):
            actions.append("reduce_turnover")
        else:
            reasons.append("turnover_repair_saturated")

    best_horizon = diagnostics.get("best_horizon_bars")
    current_lookback = diagnostics.get("current_lookback")
    if best_horizon is not None and current_lookback is not None and int(best_horizon) != int(current_lookback):
        if float(diagnostics.get("max_abs_ic") or 0.0) >= 0.01:
            reasons.append("horizon_mismatch")
            suggested["factor_lookback"] = int(best_horizon)
            suggested["evidence_horizon_bars"] = int(best_horizon)
            actions.append("use_best_horizon")

    best_regime = diagnostics.get("best_regime")
    if best_regime and _conditional_edge(diagnostics, "max_abs_regime_ic"):
        reasons.append("regime_mixing")
        suggested["regime_filter"] = [str(best_regime)]
        actions.append("add_regime_filter")

    best_funding_key = diagnostics.get("best_funding_key")
    if best_funding_key and _conditional_edge(diagnostics, "max_abs_funding_ic"):
        reasons.append("funding_state_mixing")
        if str(best_funding_key).startswith("state:"):
            suggested["funding_state_filter"] = [str(best_funding_key).split(":", 1)[1]]
        elif str(best_funding_key).startswith("trend:"):
            suggested["funding_trend_filter"] = [str(best_funding_key).split(":", 1)[1]]
        actions.append("add_funding_filter")

    best_side = diagnostics.get("best_side")
    if best_side and _side_asymmetry(diagnostics):
        reasons.append("long_short_asymmetry")
        suggested["side_mode"] = best_side
        actions.append("split_long_short")

    failures = set(diagnostics.get("gate_failures", "").split(","))
    if not is_underpowered_survivor and ("G7" in failures or int(diagnostics.get("oos_trade_count") or 0) < 100):
        reasons.append("insufficient_trades")
        suggested.setdefault("signal_threshold", 0.10)
        suggested.setdefault("position_buffer", 0.05)
        actions.append("broaden_entries")

    if "G2" in failures or "G5" in failures:
        reasons.append("overfit_or_unstable")
        suggested.setdefault("smooth_span", 24)
        suggested.setdefault("signal_threshold", 0.20)
        suggested["repair_complexity"] = "simplify"
        actions.append("simplify_signal")

    if not reasons and (research_gate and research_gate.status == "research_survivor"):
        reasons.append("weak_but_stable_ic")
        suggested.setdefault("signal_role", "filter")
        actions.append("use_as_filter")

    if not reasons:
        reasons.append("no_evidence")

    suggested_variants = _suggested_param_variants(candidate, suggested)
    actionable = bool(suggested_variants) and reasons[0] != "no_evidence"
    if actionable:
        suggested = dict(suggested_variants[0])
    else:
        suggested = {}
    return NearMissAnalysis(
        experiment_id=result.experiment_id,
        candidate_id=result.candidate_id,
        primary_reason=reasons[0],
        reasons=_dedupe(reasons),
        actionable=actionable,
        suggested_params=suggested,
        suggested_param_variants=suggested_variants if actionable else [],
        repair_actions=_dedupe(actions),
        diagnostics=diagnostics,
    )


def repair_adjustments_from_near_misses(
    near_misses: list[NearMissAnalysis],
    *,
    limit: int = 48,
    max_per_parent: int = 4,
) -> list[dict]:
    adjustments: list[dict] = []
    seen: set[tuple[str, str]] = set()
    per_parent: dict[str, int] = {}
    for miss in near_misses:
        if not miss.actionable:
            continue
        variants = miss.suggested_param_variants or ([miss.suggested_params] if miss.suggested_params else [])
        for idx, variant in enumerate(variants):
            if per_parent.get(miss.candidate_id, 0) >= max_per_parent:
                break
            signature = (miss.candidate_id, _params_signature(variant))
            if signature in seen:
                continue
            seen.add(signature)
            per_parent[miss.candidate_id] = per_parent.get(miss.candidate_id, 0) + 1
            adjustments.append({
                "candidate_id": miss.candidate_id,
                "param": "repair_params",
                "current": "failed_candidate",
                "suggested": {
                    **variant,
                    "near_miss_reason": miss.primary_reason,
                    "near_miss_reasons": miss.reasons,
                    "repair_actions": miss.repair_actions,
                },
                "proposal_kind": "near_miss_repair",
                "variant_key": _variant_key(miss.primary_reason, variant, idx),
                "param_diff": variant,
                "rationale": f"Near-miss repair for {miss.primary_reason}: {', '.join(miss.repair_actions)}",
            })
            if len(adjustments) >= limit:
                return adjustments
    return adjustments


def _diagnostics(
    candidate: CandidateStrategySpec | None,
    result: BacktestResult,
    gatecheck: GateCheckResult,
    evidence: FactorEvidenceReport | None,
    research_gate: ResearchGateResult | None,
) -> dict[str, float | int | str | bool | None]:
    gross = result.metrics_gross
    gross_sharpe = gross.sharpe if gross is not None else None
    cost_drag = None if gross_sharpe is None else gross_sharpe - result.metrics_primary.sharpe
    max_abs_ic, best_horizon = _best_abs(evidence.ic_by_horizon if evidence else {})
    max_abs_rankic, _ = _best_abs(evidence.rankic_by_horizon if evidence else {})
    max_abs_regime_ic, best_regime = _best_nested_abs(evidence.regime_conditional_ic if evidence else {})
    max_abs_funding_ic, best_funding_key = _best_nested_abs(evidence.funding_conditional_ic if evidence else {})
    long_sharpe = evidence.long_only_metrics.sharpe if evidence else 0.0
    short_sharpe = evidence.short_only_metrics.sharpe if evidence else 0.0
    best_side = "long_only" if long_sharpe >= short_sharpe else "short_only"
    params = candidate.params if candidate is not None else {}
    current_lookback = params.get("factor_lookback") or params.get("lookback")
    failures = [item.rule_id for item in gatecheck.failures]
    warnings = [item.rule_id for item in gatecheck.warnings]
    g3_status = _gate_item_status(gatecheck, "G3")
    g7_status = _gate_item_status(gatecheck, "G7")
    return {
        "production_gate_passed": gatecheck.passed,
        "gate_failure_count": len(failures),
        "gross_sharpe": gross_sharpe,
        "net_sharpe": result.metrics_primary.sharpe,
        "deflated_sharpe": result.deflated_sharpe,
        "probabilistic_sharpe": result.probabilistic_sharpe,
        "permutation_pvalue": result.permutation_test_pvalue,
        "fdr_adjusted_pvalue": _gate_item_value(gatecheck, "G3"),
        "pbo": result.pbo,
        "cost_drag_sharpe": cost_drag,
        "cost_margin_bps": result.break_even_cost_bps - 2.0 * result.actual_cost_bps,
        "break_even_cost_bps": result.break_even_cost_bps,
        "actual_cost_bps": result.actual_cost_bps,
        "factor_turnover": result.factor_turnover,
        "avg_holding_period_bars": result.avg_holding_period_bars,
        "oos_trade_count": result.oos_trade_count,
        "trade_count": result.metrics_primary.trade_count,
        "max_abs_ic": max_abs_ic,
        "max_abs_rankic": max_abs_rankic,
        "best_horizon_bars": int(best_horizon) if best_horizon is not None else None,
        "current_lookback": int(current_lookback) if _is_int_like(current_lookback) else None,
        "max_abs_regime_ic": max_abs_regime_ic,
        "best_regime": best_regime,
        "max_abs_funding_ic": max_abs_funding_ic,
        "best_funding_key": best_funding_key,
        "long_sharpe": long_sharpe,
        "short_sharpe": short_sharpe,
        "best_side": best_side,
        "research_gate_status": research_gate.status if research_gate is not None else None,
        "research_gate_reasons": ",".join(research_gate.reasons) if research_gate is not None else "",
        "gate_failures": ",".join(failures),
        "gate_warnings": ",".join(warnings),
        "g3_fdr_not_passed": g3_status in {"fail", "warn"},
        "g7_trades_not_passed": g7_status in {"fail", "warn"},
    }


def _cost_destroyed_edge(diagnostics: dict) -> bool:
    gross = diagnostics.get("gross_sharpe")
    cost_drag = diagnostics.get("cost_drag_sharpe")
    return (
        gross is not None
        and float(gross) >= 0.5
        and (
            float(diagnostics.get("net_sharpe") or 0.0) <= 0.0
            or (cost_drag is not None and float(cost_drag) >= 0.75)
            or float(diagnostics.get("cost_margin_bps") or 0.0) < 0.0
        )
    )


def _statistically_underpowered_survivor(diagnostics: dict) -> bool:
    return (
        bool(diagnostics.get("g3_fdr_not_passed"))
        and bool(diagnostics.get("g7_trades_not_passed"))
        and float(diagnostics.get("deflated_sharpe") or 0.0) > 0.0
        and float(diagnostics.get("net_sharpe") or 0.0) > 0.0
        and float(diagnostics.get("cost_margin_bps") or 0.0) > 0.0
    )


def _excess_turnover(diagnostics: dict) -> bool:
    return float(diagnostics.get("factor_turnover") or 0.0) >= 0.15


def _conditional_edge(diagnostics: dict, key: str) -> bool:
    overall = max(float(diagnostics.get("max_abs_ic") or 0.0), 0.005)
    conditional = float(diagnostics.get(key) or 0.0)
    return conditional >= max(0.015, overall * 1.5)


def _side_asymmetry(diagnostics: dict) -> bool:
    best = max(float(diagnostics.get("long_sharpe") or 0.0), float(diagnostics.get("short_sharpe") or 0.0))
    net = float(diagnostics.get("net_sharpe") or 0.0)
    return best >= 0.4 and best - net >= 0.5


_LOW_TURNOVER_LADDER: tuple[dict[str, float | int], ...] = (
    {"smooth_span": 24, "signal_threshold": 0.20, "position_buffer": 0.15},
    {"smooth_span": 48, "signal_threshold": 0.30, "position_buffer": 0.25},
    {"smooth_span": 96, "signal_threshold": 0.40, "position_buffer": 0.30},
)

_LOW_TURNOVER_GRID: tuple[dict[str, float | int], ...] = (
    {"smooth_span": 12, "signal_threshold": 0.10, "position_buffer": 0.08},
    {"smooth_span": 24, "signal_threshold": 0.20, "position_buffer": 0.15},
    {"smooth_span": 48, "signal_threshold": 0.30, "position_buffer": 0.25},
    {"smooth_span": 96, "signal_threshold": 0.40, "position_buffer": 0.30},
)

_LOW_TURNOVER_KEYS = {"smooth_span", "signal_threshold", "position_buffer"}


def _suggested_param_variants(
    candidate: CandidateStrategySpec | None,
    suggested: dict[str, Any],
) -> list[dict[str, Any]]:
    if not suggested:
        return []
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_variant(params: dict[str, Any]) -> None:
        signature = _params_signature(params)
        if signature in seen:
            return
        seen.add(signature)
        variants.append(dict(params))

    add_variant(suggested)
    if _LOW_TURNOVER_KEYS & suggested.keys():
        additive = {key: value for key, value in suggested.items() if key not in _LOW_TURNOVER_KEYS}
        for params in _low_turnover_grid_params(candidate):
            add_variant({**additive, **params})
    return variants[:4]


def _low_turnover_grid_params(
    candidate: CandidateStrategySpec | None,
) -> list[dict[str, float | int]]:
    current_idx = _current_low_turnover_grid_index(candidate.params if candidate is not None else {})
    if current_idx >= len(_LOW_TURNOVER_GRID) - 1:
        return []
    if current_idx >= 0:
        return [dict(item) for item in _LOW_TURNOVER_GRID[current_idx + 1:]]
    return [dict(item) for item in _LOW_TURNOVER_GRID]


def _suggest_low_turnover_params(
    candidate: CandidateStrategySpec | None,
    result: BacktestResult,
    suggested: dict[str, Any],
) -> bool:
    params = _low_turnover_params(candidate, result)
    if params is None:
        return False
    suggested.update(params)
    return True


def _low_turnover_params(
    candidate: CandidateStrategySpec | None,
    result: BacktestResult,
) -> dict[str, float | int] | None:
    current_idx = _current_low_turnover_ladder_index(candidate.params if candidate is not None else {})
    target_idx = 1 if result.factor_turnover >= 0.20 else 0
    next_idx = max(current_idx + 1, target_idx)
    if next_idx >= len(_LOW_TURNOVER_LADDER):
        return None
    return dict(_LOW_TURNOVER_LADDER[next_idx])


def _current_low_turnover_ladder_index(params: dict[str, Any]) -> int:
    return _current_low_turnover_index(params, _LOW_TURNOVER_LADDER)


def _current_low_turnover_grid_index(params: dict[str, Any]) -> int:
    return _current_low_turnover_index(params, _LOW_TURNOVER_GRID)


def _current_low_turnover_index(params: dict[str, Any], ladder: tuple[dict[str, float | int], ...]) -> int:
    try:
        smooth_span = int(params.get("smooth_span", 1))
        signal_threshold = float(params.get("signal_threshold", 0.0))
        position_buffer = float(params.get("position_buffer", 0.05))
    except (TypeError, ValueError):
        return -1
    current_idx = -1
    for idx, rung in enumerate(ladder):
        if (
            smooth_span >= int(rung["smooth_span"])
            and signal_threshold >= float(rung["signal_threshold"])
            and position_buffer >= float(rung["position_buffer"])
        ):
            current_idx = idx
    return current_idx


def _variant_key(reason: str, variant: dict[str, Any], idx: int) -> str:
    controls = []
    for key in ("smooth_span", "signal_threshold", "position_buffer", "factor_lookback", "side_mode"):
        if key in variant:
            controls.append(f"{key}={variant[key]}")
    if "regime_filter" in variant:
        controls.append(f"regime={','.join(str(item) for item in variant['regime_filter'])}")
    if "funding_state_filter" in variant:
        controls.append(f"funding_state={','.join(str(item) for item in variant['funding_state_filter'])}")
    if "funding_trend_filter" in variant:
        controls.append(f"funding_trend={','.join(str(item) for item in variant['funding_trend_filter'])}")
    suffix = "_".join(controls) if controls else f"variant={idx}"
    return f"{reason}_{suffix}"


def _params_signature(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))


def _best_abs(values: dict[str, float]) -> tuple[float, str | None]:
    if not values:
        return 0.0, None
    key, value = max(values.items(), key=lambda item: abs(float(item[1])))
    return abs(float(value)), key


def _best_nested_abs(values: dict[str, dict[str, float]]) -> tuple[float, str | None]:
    best_value = 0.0
    best_key: str | None = None
    for label, nested in values.items():
        for value in nested.values():
            abs_value = abs(float(value))
            if abs_value > best_value:
                best_value = abs_value
                best_key = label
    return best_value, best_key


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _gate_item_value(gatecheck: GateCheckResult, rule_id: str) -> float | str | None:
    for item in gatecheck.items:
        if item.rule_id == rule_id:
            return item.value
    return None


def _gate_item_status(gatecheck: GateCheckResult, rule_id: str) -> str | None:
    for item in gatecheck.items:
        if item.rule_id == rule_id:
            return item.status
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out
