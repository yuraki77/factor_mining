"""
Deterministic strategy optimizer.

The optimizer reads candidate diagnostics, Research Gate classifications, and
near-miss repair hints, then produces bounded candidate mutations for the next
round. It deliberately avoids model-generated tuning so optimization remains
reproducible and auditable.
"""
from __future__ import annotations

import json

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
    weights = _traditional_weights(selected)
    source = "GateCheck-passing factors" if context.get("num_gatecheck_passed", 0) else "research survivors"

    result = {
        "action": "traditional_survivor_low_turnover_combo",
        "reasoning": f"Deterministic optimizer selected {n} {source} with conservative turnover controls.",
        "combinations": [{
            "factor_ids": factor_ids,
            "weights": weights,
            "horizon": "5m",
            "signal_threshold": 0.25,
            "smooth_span": 48,
            "position_buffer": 0.20,
            "rationale": "Deterministic fallback: combine research survivors and damp signal churn before final GateCheck.",
            "expected_improvement": "Lower turnover and better diversification before applying final-strategy gates."
        }],
        "adjustments": [],
        "next_hypotheses": [],
    }
    result["adjustments"].extend(context.get("repair_adjustments", []))

    if n < 2:
        for factor_id in factor_ids:
            result["adjustments"].append({
                "candidate_id": factor_id,
                "param": "turnover_controls",
                "current": "raw",
                "suggested": {
                    "smooth_span": 48,
                    "signal_threshold": 0.25,
                    "position_buffer": 0.20,
                },
                "rationale": "Only one research survivor is available; continue by lowering turnover before final GateCheck.",
            })

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


def optimize_exits_traditionally(context: dict, *, limit: int = 3) -> dict:
    """Suggest a small bounded exit grid from observed risk diagnostics."""
    selected = context.get("research_survivors") or [f for f in context["factors"] if f["gatecheck_passed"]]
    adjustments: list[dict] = []
    for factor in selected[:limit]:
        adjustment = _traditional_exit_adjustment(factor)
        if adjustment:
            adjustments.append(adjustment)
    return {
        "action": "traditional_exit_grid",
        "reasoning": "Deterministic bounded exit adjustments from drawdown, holding-period, and cost-drag diagnostics.",
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
    cid = factor.get("candidate_id")
    if not cid:
        return None
    adjustment: dict[str, object] = {"candidate_id": cid}
    reasons: list[str] = []
    max_dd = float(factor.get("max_dd") or 0.0)
    avg_holding = factor.get("avg_holding_bars")
    cost_drag = factor.get("cost_drag_sharpe")
    trade_count = int(factor.get("trade_count") or 0)

    if max_dd <= -0.12:
        adjustment["stop_loss_pct"] = -0.03
        adjustment["trailing_stop_pct"] = 0.02
        reasons.append("drawdown_control")
    if avg_holding is not None and float(avg_holding) > 500:
        adjustment["max_hold_bars"] = 500
        reasons.append("bounded_holding_period")
    if cost_drag is not None and float(cost_drag) > 0.5 and trade_count >= 50:
        adjustment["tp_tiers"] = [[0.02, 0.50]]
        adjustment["trailing_after_first_tp"] = True
        reasons.append("cost_drag_profit_lock")
    if len(adjustment) == 1:
        return None
    adjustment["rationale"] = ",".join(reasons)
    return adjustment


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


def apply_optimization_result(
    optimization: dict,
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
) -> tuple[list[CandidateStrategySpec], dict]:
    """Apply deterministic optimizer suggestions to create new candidate specs.

    Returns (new_candidates, optimization_summary).
    """
    new_candidates: list[CandidateStrategySpec] = []
    summary = {
        "combinations_created": 0,
        "adjustments_applied": 0,
        "repairs_created": 0,
        "hypotheses_suggested": 0,
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
                    "search_variant": "survivor_combo_low_turnover",
                    "generated_by": "traditional_survivor_composite",
                    "turnover_objective": "reduce_churn_before_final_gate",
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
                    new_candidates.append(new_c)
                    summary["adjustments_applied"] += 1
                elif param_name == "repair_params" and isinstance(suggested, dict):
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
                    new_candidates.append(new_c)
                    summary["adjustments_applied"] += 1
                    summary["repairs_created"] += 1
                else:
                    # Spawn a new candidate with the tweaked parameter
                    new_c = c.model_copy(deep=True)
                    new_c.candidate_id = f"c_adj_{uuid.uuid4().hex[:12]}"
                    new_c.candidate_type = "optimizer"
                    new_c.parent_candidate_id = cid
                    new_c.params[param_name] = suggested
                    new_c.params["parent_id"] = cid
                    new_c.params["rationale"] = adj.get("rationale", "")
                    new_candidates.append(new_c)
                    summary["adjustments_applied"] += 1

    for hypothesis in optimization.get("next_hypotheses", []):
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

    summary["hypotheses_suggested"] = len(optimization.get("next_hypotheses", []))

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
