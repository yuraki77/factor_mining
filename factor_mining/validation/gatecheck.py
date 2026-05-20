from __future__ import annotations

from factor_mining.config import Settings
from factor_mining.models import BacktestResult, FactorEvidenceReport, GateCheckItem, GateCheckResult
from factor_mining.registry import MethodSpec
from factor_mining.stats.metrics import benjamini_hochberg, combined_ic_tstat_pvalue


_FULL_ALLOCATION = 1.0
_CONDITIONAL_ALLOCATION = 0.25
_CONDITIONAL_REVIEW_DAYS = 60
_BLOCKING_RULES = {"G1", "G2", "G5", "G8", "G10", "G11", "G14", "G15"}
_FDR_EFFECTIVE_FAMILY_MIN = 10


def apply_fdr(results: list[BacktestResult], settings: Settings) -> dict[str, float]:
    adjusted_by_experiment: dict[str, float] = {}
    results_by_family: dict[str, list[BacktestResult]] = {}
    for result in results:
        results_by_family.setdefault(result.hypothesis_family, []).append(result)

    for family_results in results_by_family.values():
        pvalues = [
            combined_ic_tstat_pvalue(result.ic_tstat_nw, result.rankic_tstat_nw)
            for result in family_results
        ]
        adjusted = benjamini_hochberg(
            pvalues,
            q=settings.gatecheck.fdr_q,
            n_tests=max(len(pvalues), _FDR_EFFECTIVE_FAMILY_MIN),
        )
        adjusted_by_experiment.update(
            {
                result.experiment_id: adjusted_value
                for result, adjusted_value in zip(family_results, adjusted, strict=False)
            }
        )
    return adjusted_by_experiment


def run_gatecheck(
    result: BacktestResult,
    settings: Settings,
    *,
    method: MethodSpec,
    fdr_adjusted_pvalue: float | None = None,
) -> GateCheckResult:
    """Run the full 16-rule gatecheck on a single backtest result.

    *fdr_adjusted_pvalue* controls G3 semantics:
      - When passed (pipeline path): the value comes from family-stratified
        Benjamini-Hochberg FDR and G3 checks ``FDR_p ≤ 0.05``.
      - When ``None`` (standalone / test path): falls back to the raw
        ``combined_ic_tstat_pvalue``, i.e. G3 checks the uncorrected
        Newey-West IC t-stat p-value against 0.05.
    """
    items: list[GateCheckItem] = []
    fdr_adjusted_pvalue = combined_ic_tstat_pvalue(result.ic_tstat_nw, result.rankic_tstat_nw) if fdr_adjusted_pvalue is None else fdr_adjusted_pvalue
    _item(items, "G1", result.deflated_sharpe > 0.0, "Deflated Sharpe is positive after trial adjustment", result.deflated_sharpe, 0.0)
    _warn_item(items, "G3", fdr_adjusted_pvalue <= settings.gatecheck.fdr_q, f"Family FDR-adjusted NW IC p-value <= {settings.gatecheck.fdr_q}", fdr_adjusted_pvalue, settings.gatecheck.fdr_q)
    _warn_item(items, "G4", result.ic_tstat_nw > settings.gatecheck.ic_tstat_nw_min, "Newey-West IC t-stat contributes strength", result.ic_tstat_nw, settings.gatecheck.ic_tstat_nw_min)
    _warn_item(items, "G4R", result.rankic_tstat_nw > settings.gatecheck.rankic_tstat_nw_min, "Newey-West RankIC t-stat contributes strength", result.rankic_tstat_nw, settings.gatecheck.rankic_tstat_nw_min)
    _item(items, "G5", result.sharpe_ci_5_95[0] > settings.gatecheck.sharpe_ci_5_min, "Bootstrap Sharpe 5th percentile is positive", result.sharpe_ci_5_95[0], settings.gatecheck.sharpe_ci_5_min)
    pbo_threshold = settings.cpcv.ml_pbo_threshold if method.is_ml else settings.cpcv.pbo_threshold
    _item(items, "G2", (result.pbo if result.pbo is not None else 1.0) < pbo_threshold, "PBO below method threshold", result.pbo, pbo_threshold)
    _warn_item(items, "G7", result.oos_trade_count >= settings.gatecheck.min_oos_trades, "OOS trade count supports high statistical reliability", result.oos_trade_count, settings.gatecheck.min_oos_trades)
    _item(
        items,
        "G8",
        result.break_even_cost_bps > settings.gatecheck.break_even_cost_multiple * max(result.actual_cost_bps, 1e-12),
        "Break-even cost has safety margin",
        result.break_even_cost_bps,
        settings.gatecheck.break_even_cost_multiple * result.actual_cost_bps,
    )
    _warn_item(items, "G9", result.prior_posterior_ic_ratio < settings.gatecheck.prior_posterior_ic_max_ratio, "Prior/posterior IC calibration contributes quality", result.prior_posterior_ic_ratio, settings.gatecheck.prior_posterior_ic_max_ratio)
    _item(items, "G10", result.estimated_capacity_usd >= settings.capacity.min_capacity_usd, "Capacity exceeds minimum tradable capital", result.estimated_capacity_usd, settings.capacity.min_capacity_usd)
    _item(items, "G11", result.leakage_checks_passed and not result.split_overlap_detected, "Lookahead and split leakage checks passed", str(result.leakage_checks_passed), "true")
    concentration = _max_regime_pnl_concentration(result)
    _warn_item(items, "G6", concentration <= settings.gatecheck.max_regime_pnl_concentration, "Regime PnL is diversified enough for bonus", concentration, settings.gatecheck.max_regime_pnl_concentration)
    _warn_item(
        items,
        "G13",
        abs(result.return_autocorr_lag1) <= settings.gatecheck.return_autocorr_warn_abs,
        "Return lag-1 autocorrelation is suspicious",
        result.return_autocorr_lag1,
        settings.gatecheck.return_autocorr_warn_abs,
    )
    _data_quality_item(items, result, settings)
    _item(items, "G15", method.v1_schedulable, "Method is schedulable in BTC/ETH v1 scope", method.status, "implemented")
    raw_passed = _raw_gate_passed(items)
    return GateCheckResult(
        experiment_id=result.experiment_id,
        passed=raw_passed,
        raw_passed=raw_passed,
        risk_tier="full_pass" if raw_passed else "fail",
        factor_evidence_level="unknown",
        allocation_multiplier=_FULL_ALLOCATION if raw_passed else 0.0,
        review_after_days=None,
        tier_reasons=["raw_gate_passed" if raw_passed else "raw_gate_failed"],
        items=items,
    )


def apply_risk_stratified_gatechecks(
    results: list[BacktestResult],
    gatechecks: list[GateCheckResult],
    evidence_reports: list[FactorEvidenceReport],
    settings: Settings,
) -> list[GateCheckResult]:
    evidence_by_experiment = {report.experiment_id: report for report in evidence_reports}
    for result, gate in zip(results, gatechecks, strict=False):
        stratify_gatecheck(
            result=result,
            gatecheck=gate,
            evidence=evidence_by_experiment.get(result.experiment_id),
            settings=settings,
        )
    return gatechecks


def stratify_gatecheck(
    *,
    result: BacktestResult,
    gatecheck: GateCheckResult,
    evidence: FactorEvidenceReport | None,
    settings: Settings,
) -> GateCheckResult:
    raw_passed = _raw_gate_passed(gatecheck.items)
    evidence_level, evidence_reasons = factor_evidence_level(evidence)
    pbo = float(result.pbo if result.pbo is not None else 1.0)
    blocking_failures = sorted(
        item.rule_id
        for item in gatecheck.failures
        if item.rule_id in _BLOCKING_RULES
    )
    reasons = [f"evidence_{evidence_level}", f"pbo={pbo:.3f}"]
    reasons.extend(evidence_reasons)

    if blocking_failures:
        tier = "fail"
        allocation = 0.0
        review_after_days = None
        passed = False
        reasons.extend(f"blocking_fail:{rule_id}" for rule_id in blocking_failures)
    elif evidence_level == "weak":
        tier = "fail"
        allocation = 0.0
        review_after_days = None
        passed = False
        reasons.append("weak_factor_evidence")
    elif evidence_level == "strong":
        tier = "full_pass"
        allocation = _FULL_ALLOCATION
        review_after_days = None
        passed = True
        reasons.append("full_allocation")
    else:
        tier = "conditional_pass"
        allocation = _CONDITIONAL_ALLOCATION
        review_after_days = _CONDITIONAL_REVIEW_DAYS
        passed = True
        reasons.append("reduced_allocation")

    gatecheck.raw_passed = raw_passed
    gatecheck.passed = passed
    gatecheck.risk_tier = tier
    gatecheck.factor_evidence_level = evidence_level
    gatecheck.allocation_multiplier = allocation
    gatecheck.review_after_days = review_after_days
    gatecheck.tier_reasons = reasons
    _upsert_risk_tier_item(
        gatecheck.items,
        tier=tier,
        allocation=allocation,
        review_after_days=review_after_days,
        reasons=reasons,
    )
    return gatecheck


def factor_evidence_level(evidence: FactorEvidenceReport | None) -> tuple[str, list[str]]:
    if evidence is None:
        return "weak", ["missing_factor_evidence"]

    flags = evidence.evidence_flags
    dimensions = 0
    reasons: list[str] = []
    max_abs_ic = _max_abs(evidence.ic_by_horizon.values())
    max_abs_rankic = _max_abs(evidence.rankic_by_horizon.values())
    max_abs_spread = _max_abs(evidence.quantile_spread_by_horizon.values())

    if bool(flags.get("ic_ci_excludes_zero")) or max_abs_ic >= 0.015:
        dimensions += 1
        reasons.append("ic_supported")
    elif max_abs_ic >= 0.01:
        dimensions += 1
        reasons.append("ic_moderate")
    if max_abs_rankic >= 0.01:
        dimensions += 1
        reasons.append("rankic_supported")
    if bool(flags.get("positive_turnover_adjusted_return")) or evidence.turnover_adjusted_return > 0.0:
        dimensions += 1
        reasons.append("turnover_adjusted_return")
    if bool(flags.get("decay_curve_supported")) or evidence.decay_quality >= 0.25:
        dimensions += 1
        reasons.append("decay_supported")
    if bool(flags.get("long_short_spread")) or abs(evidence.long_short_spread_sharpe) >= 0.4:
        dimensions += 1
        reasons.append("long_short_spread")
    if max_abs_spread >= 1.0:
        dimensions += 1
        reasons.append("quantile_spread")

    reasons.append(f"evidence_dimensions={dimensions}")

    _level_map: dict[str, str] = {"strong": "moderate", "moderate": "weak", "weak": "weak"}

    if dimensions >= 4 and max_abs_ic >= 0.01:
        base = "strong"
    elif dimensions >= 2 and (max_abs_ic >= 0.01 or max_abs_rankic >= 0.01 or max_abs_spread >= 1.0):
        base = "moderate"
    else:
        base = "weak"

    if evidence.regime_conflict:
        downgraded = _level_map[base]
        reasons.append(f"regime_conflict:{base}->{downgraded}")
        return downgraded, reasons
    return base, reasons


def _upsert_risk_tier_item(
    items: list[GateCheckItem],
    *,
    tier: str,
    allocation: float,
    review_after_days: int | None,
    reasons: list[str],
) -> None:
    status = "pass" if tier == "full_pass" else "warn" if tier == "conditional_pass" else "fail"
    message = "Risk-stratified GateCheck tier: " + "; ".join(reasons[:8])
    value = tier if review_after_days is None else f"{tier}:review_{review_after_days}d"
    item = GateCheckItem(
        rule_id="G16",
        status=status,
        message=message,
        value=value,
        threshold=f"allocation={allocation:.2f}",
    )
    for idx, existing in enumerate(items):
        if existing.rule_id == "G16":
            items[idx] = item
            return
    items.append(item)


def _max_abs(values) -> float:
    return max((abs(float(value)) for value in values if value is not None), default=0.0)


def _item(items: list[GateCheckItem], rule_id: str, passed: bool, message: str, value, threshold) -> None:
    items.append(
        GateCheckItem(
            rule_id=rule_id,
            status="pass" if passed else "fail",
            message=message,
            value=value,
            threshold=threshold,
        )
    )


def _warn_item(items: list[GateCheckItem], rule_id: str, passed: bool, message: str, value, threshold) -> None:
    items.append(
        GateCheckItem(
            rule_id=rule_id,
            status="pass" if passed else "warn",
            message=message,
            value=value,
            threshold=threshold,
        )
    )


def _data_quality_item(items: list[GateCheckItem], result: BacktestResult, settings: Settings) -> None:
    degraded_ratio = result.max_data_quality_degraded_ratio
    warn_threshold = settings.gatecheck.data_quality_degraded_warn_ratio
    block_threshold = settings.gatecheck.data_quality_degraded_block_ratio
    if degraded_ratio > block_threshold:
        status = "fail"
        message = "Data quality degraded ratio exceeded blocking threshold"
    elif degraded_ratio > warn_threshold:
        status = "warn"
        message = "Data quality degraded ratio exceeded warning threshold"
    else:
        status = "pass"
        message = "Data quality degraded ratio is acceptable"
    items.append(
        GateCheckItem(
            rule_id="G14",
            status=status,
            message=message,
            value=degraded_ratio,
            threshold=f"warn>{warn_threshold:.2f};block>{block_threshold:.2f}",
        )
    )


def _raw_gate_passed(items: list[GateCheckItem]) -> bool:
    return all(item.status != "fail" for item in items if item.rule_id in _BLOCKING_RULES)


def _max_regime_pnl_concentration(result: BacktestResult) -> float:
    """Regime PnL concentration, excluding regimes the strategy barely traded.

    Strategies that intentionally avoid a regime (e.g. long-only in bull,
    regime-filtered repairs) will have negligible trade counts in those regimes
    and won't be penalised for concentration."""
    blocks = list(result.regime_conditional_metrics.values())
    total_trades = sum(block.trade_count for block in blocks)
    if total_trades == 0:
        return 0.0
    active_pnls = [
        max(0.0, block.pnl)
        for block in blocks
        if block.trade_count / max(total_trades, 1) >= 0.05
    ]
    if not active_pnls:
        return 0.0
    total = sum(active_pnls)
    if total <= 0:
        return 1.0
    return max(active_pnls) / total
