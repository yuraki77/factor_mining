from __future__ import annotations

import math

from factor_mining.config import Settings
from factor_mining.models import BacktestResult, GateCheckResult, HardScoreReport
from factor_mining.stats.metrics import annualization_factor, haircut_sharpe


def hardscore(
    result: BacktestResult,
    gatecheck: GateCheckResult,
    *,
    fdr_adjusted_pvalue: float,
    settings: Settings,
    blocked_method_reason: str | None = None,
) -> HardScoreReport:
    haircut = haircut_sharpe(
        result.metrics_primary.sharpe,
        trials_count=max(result.effective_trials_at_eval, 1),
        observations=max(result.metrics_primary.trade_count, 1),
        periods_per_year=annualization_factor(result.interval),
    )
    failures = [item.rule_id for item in gatecheck.items if item.status == "fail"]
    allocation = gatecheck.allocation_multiplier
    if allocation is None:
        allocation = 1.0 if gatecheck.passed else 0.0
    score = _score(
        result,
        gatecheck_passed=gatecheck.passed,
        fdr_adjusted_pvalue=fdr_adjusted_pvalue,
        settings=settings,
    ) * allocation
    return HardScoreReport(
        experiment_id=result.experiment_id,
        score=float(round(score, 4)),
        haircut_sharpe=haircut,
        fdr_adjusted_pvalue=fdr_adjusted_pvalue,
        prior_posterior_ic_ratio=result.prior_posterior_ic_ratio,
        effective_trials_count=result.effective_trials_at_eval,
        global_cumulative_trials_count=result.global_trials_at_eval,
        allocation_multiplier=allocation,
        blocked_method_reason=blocked_method_reason,
        gatecheck_failures=failures,
    )


def _score(
    result: BacktestResult,
    *,
    gatecheck_passed: bool,
    fdr_adjusted_pvalue: float,
    settings: Settings,
) -> float:
    if not gatecheck_passed:
        return 0.0
    pbo = 1.0 if result.pbo is None else float(result.pbo)
    ratio = float(result.prior_posterior_ic_ratio)
    autocorr_scale = max(settings.gatecheck.return_autocorr_warn_abs * 2.0, 1e-12)
    ic_strength = max(float(result.ic_tstat_nw), float(result.rankic_tstat_nw))
    raw = (
        25.0 * _smooth_positive(result.deflated_sharpe, scale=2.0)
        + 20.0 * _clip(1.0 - pbo)
        + 15.0 * _smooth_positive(ic_strength, scale=4.0)
        + 10.0 * _clip(1.0 - float(fdr_adjusted_pvalue) / max(settings.gatecheck.fdr_q, 1e-12))
        + 10.0 * math.sqrt(_clip(result.oos_trade_count / max(settings.gatecheck.min_oos_trades, 1)))
        + 8.0 * _regime_diversity_bonus(result)
        + 7.0 * _calibration_score(ratio)
        + 5.0 * _clip(1.0 - abs(result.return_autocorr_lag1) / autocorr_scale)
    )
    return float(round(raw, 4))


def _clip(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _smooth_positive(value: float, *, scale: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return float(math.tanh(max(0.0, value) / max(scale, 1e-12)))


def _calibration_score(ratio: float) -> float:
    if not math.isfinite(ratio) or ratio <= 0.0:
        return 0.0
    return _clip(1.0 - abs(math.log(ratio)) / math.log(5.0))


def _regime_diversity_bonus(result: BacktestResult) -> float:
    """Normalised 0-1 score: 1 = perfectly balanced across active regimes, 0 = single regime.

    Regimes where the strategy barely traded (<5% of total trades) are excluded
    from the calculation, so intentionally directional strategies aren't penalised
    for avoiding regimes they don't target."""
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
    n = len(active_pnls)
    if total <= 0.0 or n <= 1:
        return 0.0
    concentration = float(max(active_pnls) / total)
    min_possible = 1.0 / n
    return _clip((1.0 - concentration) / (1.0 - min_possible))
