from __future__ import annotations

import math

from factor_mining.config import Settings
from factor_mining.models import BacktestResult, GateCheckResult, HardScoreReport
from factor_mining.stats.metrics import haircut_sharpe


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
    )
    failures = [item.rule_id for item in gatecheck.items if item.status == "fail"]
    allocation = gatecheck.allocation_multiplier
    if allocation is None:
        allocation = 1.0 if gatecheck.passed else 0.0
    score = _score(
        result,
        gatecheck_passed=gatecheck.passed,
        haircut_sharpe_value=haircut,
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
    haircut_sharpe_value: float,
    fdr_adjusted_pvalue: float,
    settings: Settings,
) -> float:
    if not gatecheck_passed:
        return 0.0
    pbo = 1.0 if result.pbo is None else float(result.pbo)
    ratio = float(result.prior_posterior_ic_ratio)
    autocorr_scale = max(settings.gatecheck.return_autocorr_warn_abs * 2.0, 1e-12)
    raw = (
        20.0 * _clip(result.deflated_sharpe)
        + 15.0 * _clip(haircut_sharpe_value / 3.0)
        + 15.0 * _clip(1.0 - pbo)
        + 15.0 * _clip(result.probabilistic_sharpe)
        + 10.0 * math.tanh(max(0.0, result.ic_tstat_nw, result.rankic_tstat_nw) / 4.0)
        + 7.0 * _clip(1.0 - float(fdr_adjusted_pvalue) / max(settings.gatecheck.fdr_q, 1e-12))
        + 6.0 * math.sqrt(_clip(result.oos_trade_count / max(settings.gatecheck.min_oos_trades, 1)))
        + 5.0 * _clip(1.0 - _positive_regime_pnl_concentration(result))
        + 4.0 * _calibration_score(ratio)
        + 3.0 * _clip(1.0 - abs(result.return_autocorr_lag1) / autocorr_scale)
    )
    return float(round(raw, 4))


def _clip(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _calibration_score(ratio: float) -> float:
    if not math.isfinite(ratio) or ratio <= 0.0:
        return 0.0
    return _clip(1.0 - abs(math.log(ratio)) / math.log(5.0))


def _positive_regime_pnl_concentration(result: BacktestResult) -> float:
    positive_pnls = [max(0.0, block.pnl) for block in result.regime_conditional_metrics.values()]
    total = sum(positive_pnls)
    if total <= 0.0:
        return 1.0
    return float(max(positive_pnls) / total)
