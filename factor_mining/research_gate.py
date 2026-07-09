from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from factor_mining.config import Settings
from factor_mining.models import BacktestResult, FactorEvidenceReport, GateCheckResult, ResearchGateResult, ResearchSurvivorRecord, UTC
from factor_mining.stats.metrics import combined_ic_tstat_pvalue


def apply_research_gate(
    results: list[BacktestResult],
    gatechecks: list[GateCheckResult],
    evidence_reports: list[FactorEvidenceReport],
) -> list[ResearchGateResult]:
    """Classify factors for research without weakening Production GateCheck."""
    evidence_by_experiment = {report.experiment_id: report for report in evidence_reports}
    out: list[ResearchGateResult] = []
    for result, gate in zip(results, gatechecks, strict=False):
        out.append(
            evaluate_research_gate(
                result=result,
                gatecheck=gate,
                evidence=evidence_by_experiment.get(result.experiment_id),
            )
        )
    return out


def evaluate_research_gate(
    *,
    result: BacktestResult,
    gatecheck: GateCheckResult,
    evidence: FactorEvidenceReport | None,
) -> ResearchGateResult:
    flags = _evidence_flags(result, gatecheck, evidence)
    reasons: list[str] = []
    score = 0.0

    if gatecheck.passed:
        if gatecheck.risk_tier == "conditional_pass":
            reasons.append("conditional_gate_passed")
            score += 5.0
        else:
            reasons.append("production_gate_passed")
            score += 10.0

    if flags["max_abs_ic"] >= 0.01:
        reasons.append("ic_signal")
        score += 1.0
    if flags["max_abs_rankic"] >= 0.01:
        reasons.append("rankic_signal")
        score += 1.0
    if abs(float(flags["best_quantile_spread_bps"] or 0.0)) >= 1.0:
        reasons.append("quantile_spread")
        score += 1.0
    if flags["max_abs_regime_ic"] >= 0.015:
        reasons.append("regime_conditional_signal")
        score += 1.5
    if flags["max_abs_funding_ic"] >= 0.015:
        reasons.append("funding_conditional_signal")
        score += 1.5
    if flags["gross_sharpe"] is not None and flags["gross_sharpe"] >= 0.5:
        reasons.append("gross_edge")
        score += 1.0
    if flags["net_sharpe"] > 0.0:
        reasons.append("positive_net_sharpe")
        score += 0.5
    if flags["best_side_sharpe"] >= 0.4:
        reasons.append("long_short_asymmetry")
        score += 1.0
    if flags["cost_margin_bps"] is not None and flags["cost_margin_bps"] > 0.0:
        reasons.append("cost_margin")
        score += 0.5
    if _statistically_underpowered(flags):
        reasons.append("statistically_underpowered_survivor")
        score += 1.0

    if gatecheck.passed:
        status = "production_passed"
    elif score >= 1.5:
        status = "research_survivor"
    else:
        status = "rejected"

    return ResearchGateResult(
        experiment_id=result.experiment_id,
        candidate_id=result.candidate_id,
        status=status,
        production_gate_passed=gatecheck.passed,
        production_gate_failures=[item.rule_id for item in gatecheck.failures],
        research_score=float(score),
        reasons=reasons,
        evidence_flags=flags,
    )


def research_survivor_payloads(
    candidates_by_id: dict[str, object],
    results: list[BacktestResult],
    research_gates: list[ResearchGateResult],
) -> list[dict]:
    """Create dashboard/optimizer-friendly survivor rows from formal ResearchGate results."""
    result_by_candidate = {result.candidate_id: result for result in results}
    rows: list[dict] = []
    for gate in research_gates:
        if gate.status not in {"production_passed", "research_survivor"}:
            continue
        result = result_by_candidate.get(gate.candidate_id)
        candidate = candidates_by_id.get(gate.candidate_id)
        rows.append({
            "candidate_id": gate.candidate_id,
            "experiment_id": gate.experiment_id,
            "hypothesis_family": getattr(candidate, "hypothesis_family", None) or (result.hypothesis_family if result else None),
            "method_id": getattr(candidate, "method_id", None) or (result.method_id if result else None),
            "status": gate.status,
            "research_score": gate.research_score,
            "survivor_reason": ",".join(gate.reasons),
            "reasons": gate.reasons,
            "sharpe": result.metrics_primary.sharpe if result else None,
            "gross_sharpe": result.metrics_gross.sharpe if result and result.metrics_gross is not None else None,
            "factor_turnover": result.factor_turnover if result else None,
            "cost_margin_bps": (
                result.break_even_cost_bps - 2.0 * result.actual_cost_bps
                if result else None
            ),
            "production_gate_failures": gate.production_gate_failures,
            "evidence_flags": gate.evidence_flags,
        })
    rows.sort(key=lambda row: (row["status"] == "production_passed", row["research_score"]), reverse=True)
    return rows


def build_research_survivor_records(
    *,
    candidates_by_id: dict[str, object],
    results: list[BacktestResult],
    research_gates: list[ResearchGateResult],
    fdr_map: dict[str, float],
    settings: Settings,
    now: datetime | None = None,
) -> list[ResearchSurvivorRecord]:
    """Create persistent active records for statistically underpowered survivors."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    result_by_candidate = {result.candidate_id: result for result in results}
    min_trades = int(settings.gatecheck.min_oos_trades)
    promotion_fdr = float(settings.gatecheck.research_survivor_promotion_fdr_p)
    required_days = int(settings.gatecheck.research_survivor_min_oos_days)
    records: list[ResearchSurvivorRecord] = []

    for gate in research_gates:
        if gate.status != "research_survivor":
            continue
        result = result_by_candidate.get(gate.candidate_id)
        if result is None:
            continue
        candidate = candidates_by_id.get(gate.candidate_id)
        candidate_payload = candidate.model_dump(mode="json") if hasattr(candidate, "model_dump") else {}
        # Strictly the OOS count: zero OOS trades means zero promotion progress.
        # (The old falsy `or` fallback substituted the full-slice trade count.)
        current_trades = int(result.oos_trade_count)
        required_additional = max(0, min_trades - current_trades)
        fdr_pvalue = float(fdr_map.get(result.experiment_id, combined_ic_tstat_pvalue(result.ic_tstat_nw, result.rankic_tstat_nw)))
        # This record's paper-trade clock starts at `now`; the store's upsert
        # preserves an older stored clock, and the authoritative days check
        # runs against it in _update_research_survivor_store. A freshly built
        # record therefore can never be promotion_ready when days are required.
        paper_trade_start = now
        elapsed_days = (now - paper_trade_start).days
        promotion_ready = (
            fdr_pvalue < promotion_fdr
            and current_trades >= min_trades
            and elapsed_days >= required_days
        )
        records.append(
            ResearchSurvivorRecord(
                candidate_id=gate.candidate_id,
                experiment_id=gate.experiment_id,
                status="active",
                candidate_payload=candidate_payload,
                paper_trade_start_date=paper_trade_start,
                last_evaluated_at=now,
                current_trades=current_trades,
                required_additional_trades=required_additional,
                required_oos_days=required_days,
                recheck_trigger=_recheck_trigger(required_additional),
                promotion_criteria=f"NW FDR P < {promotion_fdr:.2f} AND trades >= {min_trades} AND paper_days >= {required_days}",
                promotion_ready=promotion_ready,
                survivor_reason=",".join(gate.reasons),
                research_score=gate.research_score,
                fdr_pvalue=fdr_pvalue,
                sharpe=result.metrics_primary.sharpe,
                dsr=result.deflated_sharpe,
                # gross break-even vs 2× realized (turnover-weighted) cost since the
                # 2026-07 meter fix — see engine._break_even_cost_bps / _strategy_returns
                cost_margin_bps=result.break_even_cost_bps - 2.0 * result.actual_cost_bps,
                production_gate_failures=gate.production_gate_failures,
                evidence_flags=gate.evidence_flags,
                updated_at=now,
            )
        )

    return records


def _evidence_flags(
    result: BacktestResult,
    gatecheck: GateCheckResult,
    evidence: FactorEvidenceReport | None,
) -> dict[str, float | int | str | bool | None]:
    gross = result.metrics_gross
    gross_sharpe = gross.sharpe if gross is not None else None
    cost_margin = result.break_even_cost_bps - 2.0 * result.actual_cost_bps
    failure_rules = [item.rule_id for item in gatecheck.failures]
    g3_status = _gate_item_status(gatecheck, "G3")
    g7_status = _gate_item_status(gatecheck, "G7")
    flags: dict[str, float | int | str | bool | None] = {
        "production_gate_passed": gatecheck.passed,
        "gate_raw_passed": gatecheck.raw_passed,
        "risk_tier": gatecheck.risk_tier,
        "factor_evidence_level": gatecheck.factor_evidence_level,
        "allocation_multiplier": gatecheck.allocation_multiplier,
        "review_after_days": gatecheck.review_after_days,
        "production_gate_failure_count": len(failure_rules),
        "production_gate_failures": ",".join(failure_rules),
        "failed_g2_pbo": "G2" in failure_rules,
        "failed_g3_fdr": g3_status in {"fail", "warn"},
        "failed_g5_bootstrap": "G5" in failure_rules,
        "failed_g7_trades": g7_status in {"fail", "warn"},
        "failed_g8_cost": "G8" in failure_rules,
        "fdr_adjusted_pvalue": _gate_item_value(gatecheck, "G3"),
        "max_abs_ic": 0.0,
        "max_abs_rankic": 0.0,
        "best_quantile_spread_bps": 0.0,
        "max_abs_regime_ic": 0.0,
        "max_abs_funding_ic": 0.0,
        "best_horizon_bars": None,
        "long_sharpe": 0.0,
        "short_sharpe": 0.0,
        "best_side_sharpe": 0.0,
        "gross_sharpe": gross_sharpe,
        "net_sharpe": result.metrics_primary.sharpe,
        "deflated_sharpe": result.deflated_sharpe,
        "deflated_sharpe_prob": result.deflated_sharpe_prob,
        "probabilistic_sharpe": result.probabilistic_sharpe,
        "permutation_pvalue": result.permutation_test_pvalue,
        "pbo": result.pbo,
        "cost_margin_bps": cost_margin,
        "break_even_cost_bps": result.break_even_cost_bps,
        "actual_cost_bps": result.actual_cost_bps,
        "factor_turnover": result.factor_turnover,
        "avg_holding_period_bars": result.avg_holding_period_bars,
        "oos_trade_count": result.oos_trade_count,
        "trade_count": result.metrics_primary.trade_count,
    }
    if evidence is None:
        return flags

    flags["max_abs_ic"] = _max_abs(evidence.ic_by_horizon.values())
    flags["max_abs_rankic"] = _max_abs(evidence.rankic_by_horizon.values())
    flags["best_quantile_spread_bps"] = _max_abs_signed(evidence.quantile_spread_by_horizon.values())
    flags["max_abs_regime_ic"] = _max_nested_abs(evidence.regime_conditional_ic)
    flags["max_abs_funding_ic"] = _max_nested_abs(evidence.funding_conditional_ic)
    flags["best_horizon_bars"] = evidence.best_horizon_bars
    flags["long_sharpe"] = evidence.long_only_metrics.sharpe
    flags["short_sharpe"] = evidence.short_only_metrics.sharpe
    flags["best_side_sharpe"] = max(evidence.long_only_metrics.sharpe, evidence.short_only_metrics.sharpe)
    return flags


def _statistically_underpowered(flags: dict[str, float | int | str | bool | None]) -> bool:
    # "Promising but underpowered": more likely than not to beat the expected
    # max of the search (DSR prob >= 0.5), short of the G1 bar. The haircut
    # DSR is unusable here — annualized on intraday bars it is negative for
    # every honest candidate.
    return (
        bool(flags.get("failed_g3_fdr"))
        and bool(flags.get("failed_g7_trades"))
        and float(flags.get("deflated_sharpe_prob") or 0.0) >= 0.5
        and float(flags.get("net_sharpe") or 0.0) > 0.0
        and float(flags.get("cost_margin_bps") or 0.0) > 0.0
    )


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


def _recheck_trigger(required_additional_trades: int) -> str:
    if required_additional_trades > 0:
        return f"on_next_round_if_new_trades >= {required_additional_trades}"
    return "on_next_round_after_required_oos_days"


def _max_abs(values: Iterable[float]) -> float:
    clean = [abs(float(value)) for value in values if value is not None]
    return max(clean, default=0.0)


def _max_abs_signed(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return 0.0
    return max(clean, key=lambda value: abs(value))


def _max_nested_abs(values: dict[str, dict[str, float]]) -> float:
    return max(
        (abs(float(value)) for nested in values.values() for value in nested.values() if value is not None),
        default=0.0,
    )
