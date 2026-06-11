"""Cross-sectional long-short factor backtest engine.

This is the cross-sectional counterpart to ``engine.evaluate_strategy_path``.
Whereas the single-asset engine produces a position/PnL trajectory for one
symbol, this engine produces *one* dollar-neutral long-short PnL series per
factor by ranking the entire universe at each rebalance.

Design choices (deliberate, surgical):

* **Construction:** rank-weighted, dollar-neutral by default — for small
  crypto universes (often 5–30 names) hard quintile cutoffs throw away too
  much information and produce thin extreme buckets. Quantile mode is
  available for the diagnostic that *that's* what the literature uses.
* **Standardization:** purely cross-sectional at each t.  No time-series
  z-scoring of factor scores — that's the responsibility of the factor
  itself, and doing it here would impose lookahead via the rolling mean.
* **Universe:** point-in-time, derived from the panel's ``universe_mask``.
  No survivorship correction beyond what the panel encodes — that lives in
  ``data.panel.build_panel``.
* **Costs:** turnover × bps (same parameters as the single-asset engine).
  Funding is not yet modeled at the panel level; per-asset funding curves
  could be plumbed in but are deferred to keep this change surgical.
* **Output:** a ``CrossSectionalBacktestResult`` that mirrors the fields the
  existing FDR/gatecheck/hardscore consume, so the rest of the pipeline
  works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from factor_mining.config import Settings
from factor_mining.data.panel import FactorPanel
from factor_mining.stats.cross_sectional import (
    cross_sectional_ic,
    cross_sectional_t_pvalue,
    ic_summary,
)
from factor_mining.stats.metrics import (
    _block_bootstrap_sharpes,
    annualization_factor,
    deflated_sharpe_ratio,
    max_drawdown,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    return_autocorrelation_lag1,
    sharpe_ratio,
)


WeightingMode = Literal["rank", "quantile"]


@dataclass(frozen=True)
class CrossSectionalBacktestResult:
    """All diagnostics for one cross-sectional factor specification.

    Mirrors the fields ``apply_fdr``/``run_gatecheck`` read from
    ``BacktestResult`` so the existing validation pipeline can consume this
    via a thin adapter (``to_legacy_result``).
    """

    factor_id: str
    hypothesis_family: str
    interval: str
    weighting: WeightingMode
    rebalance_bars: int

    # Per-rebalance series — kept around for diagnostics
    weights: pd.DataFrame
    portfolio_returns: pd.Series
    gross_returns: pd.Series
    ic_series: pd.Series
    turnover_series: pd.Series

    # Aggregates
    mean_ic: float
    ic_vol: float
    ic_tstat_nw: float
    rankic_tstat_nw: float
    hit_rate: float
    n_rebalances: int

    sharpe: float
    annualized_return: float
    annualized_vol: float
    max_drawdown: float
    sharpe_ci_5_95: tuple[float, float]
    probabilistic_sharpe: float
    deflated_sharpe: float
    return_autocorr_lag1: float
    avg_turnover: float
    avg_universe_size: float
    one_sided_pvalue: float

    def __post_init__(self) -> None:  # pragma: no cover — light guard
        if len(self.portfolio_returns) != len(self.ic_series):
            raise ValueError("portfolio_returns and ic_series must be aligned")


def _cross_sectional_zscore(factor_row: np.ndarray, mask_row: np.ndarray) -> np.ndarray:
    """Row-wise z-score over the tradeable universe; non-universe = 0."""
    out = np.zeros_like(factor_row, dtype=float)
    if not mask_row.any():
        return out
    vals = factor_row[mask_row]
    mean = vals.mean()
    std = vals.std(ddof=0)
    if std == 0.0:
        return out
    centered = (vals - mean) / std
    out[mask_row] = centered
    return out


def _rank_weights(factor_row: np.ndarray, mask_row: np.ndarray) -> np.ndarray:
    """Dollar-neutral rank-weighted weights.

    Demeans cross-sectional ranks and normalizes so ``sum(|w|) == 1`` on the
    long and short sides each, i.e. ``sum(w_long) = +0.5``, ``sum(w_short) =
    -0.5`` — total gross exposure 1.  This is the standard convention for
    factor PnL series so Sharpes are comparable across factors and across
    universes of different size.
    """
    weights = np.zeros_like(factor_row, dtype=float)
    n = int(mask_row.sum())
    if n < 2:
        return weights
    vals = factor_row[mask_row]
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return weights
    vals = np.where(finite_mask, vals, np.nan)
    order = np.argsort(np.where(np.isnan(vals), -np.inf, vals))
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    ranks[~finite_mask] = np.nan
    centered = ranks - np.nanmean(ranks)
    abs_sum = np.nansum(np.abs(centered))
    if abs_sum == 0.0:
        return weights
    normalized = centered / abs_sum
    normalized = np.where(np.isnan(normalized), 0.0, normalized)
    weights[mask_row] = normalized
    return weights


def _quantile_weights(
    factor_row: np.ndarray, mask_row: np.ndarray, *, top_frac: float
) -> np.ndarray:
    """Equal-weighted long top / short bottom quantile, dollar-neutral.

    ``top_frac`` is the fraction in the long bucket (and symmetrically the
    short bucket).  e.g. 0.2 = top/bottom quintile.  Falls back to a single
    long / single short when ``floor(N * top_frac) < 1``.
    """
    weights = np.zeros_like(factor_row, dtype=float)
    n = int(mask_row.sum())
    if n < 2:
        return weights
    vals = np.where(mask_row & np.isfinite(factor_row), factor_row, np.nan)
    if np.isnan(vals).sum() == n:
        return weights
    bucket_size = max(1, int(np.floor(n * top_frac)))
    order = np.argsort(np.where(np.isnan(vals), -np.inf, vals))
    long_idx = order[-bucket_size:]
    short_idx = order[:bucket_size]
    long_idx = long_idx[~np.isnan(vals[long_idx])]
    short_idx = short_idx[~np.isnan(vals[short_idx])]
    if len(long_idx) == 0 or len(short_idx) == 0:
        return weights
    weights[long_idx] = 0.5 / len(long_idx)
    weights[short_idx] = -0.5 / len(short_idx)
    return weights


def _compute_weights(
    panel: FactorPanel, *, mode: WeightingMode, top_frac: float, rebalance_bars: int
) -> pd.DataFrame:
    """Build the (T × N) weight matrix, holding weights between rebalances."""
    factor = panel.factor.to_numpy(dtype=float)
    mask = panel.universe_mask.to_numpy()
    T, _ = factor.shape
    out = np.zeros_like(factor)
    last = None
    for t in range(T):
        if t % rebalance_bars == 0 or last is None:
            if mode == "rank":
                last = _rank_weights(factor[t], mask[t])
            else:
                last = _quantile_weights(factor[t], mask[t], top_frac=top_frac)
        else:
            # Carry stale weights but zero out names that left the universe
            last = np.where(mask[t], last, 0.0)
        out[t] = last
    return pd.DataFrame(out, index=panel.factor.index, columns=panel.factor.columns)


def run_cross_sectional_backtest(
    panel: FactorPanel,
    *,
    settings: Settings,
    factor_id: str,
    hypothesis_family: str = "cross_sectional_factor",
    interval: str = "5m",
    weighting: WeightingMode = "rank",
    top_frac: float = 0.2,
    rebalance_bars: int = 1,
    ic_method: str = "spearman",
    trials_count: int = 1,
) -> CrossSectionalBacktestResult:
    """Run the cross-sectional long-short backtest for a single factor.

    Args:
        panel: aligned factor + forward-returns panel.  ``forward_returns[t]``
            is the return realized between t and t+horizon and must be
            *aligned* with ``factor[t]`` — the panel builder takes care of
            that.  We do not need to ``shift(1)`` here because the panel is
            already (decision_at_t, fwd_return_starting_at_t).
        settings: project settings (for costs, bootstrap, position sizing).
        factor_id: identifier for this factor specification.  Combined with
            ``hypothesis_family`` for the FDR grouping downstream.
        weighting: ``"rank"`` (default, dollar-neutral rank-weighted) or
            ``"quantile"`` (equal-weight top/bottom bucket).
        top_frac: fraction in each bucket when ``weighting="quantile"``.
        rebalance_bars: how often to refresh weights.  1 = every bar.
        ic_method: ``"spearman"`` (Rank-IC) or ``"pearson"``.
        trials_count: number of factor specifications tried in this run; used
            for deflated Sharpe.  Pass the *honest* count including
            hyperparameter sweeps.

    Returns:
        ``CrossSectionalBacktestResult`` with the single PnL series + all the
        diagnostics needed by the FDR/gatecheck layer.
    """
    if rebalance_bars < 1:
        raise ValueError("rebalance_bars must be >= 1")
    if not 0.0 < top_frac <= 0.5:
        raise ValueError("top_frac must be in (0, 0.5]")

    weights = _compute_weights(
        panel, mode=weighting, top_frac=top_frac, rebalance_bars=rebalance_bars
    )
    returns_matrix = panel.forward_returns.fillna(0.0).to_numpy(dtype=float)
    weights_arr = weights.to_numpy(dtype=float)
    gross = (weights_arr * returns_matrix).sum(axis=1)
    turnover = np.abs(np.diff(weights_arr, axis=0, prepend=0.0)).sum(axis=1)
    # Cost in fractional return units: total turnover × (taker + base slippage) / 1e4.
    # Slippage's nonlinear participation term is per-asset and not modelled
    # in the panel layer — taker + base slippage is a conservative floor.
    cost_bps = float(settings.costs.taker_bps + settings.costs.slippage_base_bps)
    cost_returns = turnover * cost_bps / 10_000.0
    net = gross - cost_returns

    gross_series = pd.Series(gross, index=panel.index, name="gross_return")
    net_series = pd.Series(net, index=panel.index, name="net_return")
    turnover_series = pd.Series(turnover, index=panel.index, name="turnover")

    periods = annualization_factor(interval)
    ic_series = cross_sectional_ic(panel, method=ic_method)
    rank_ic_series = (
        ic_series if ic_method == "spearman"
        else cross_sectional_ic(panel, method="spearman")
    )
    ic_stats = ic_summary(ic_series)
    rank_ic_stats = ic_summary(rank_ic_series)
    universe_sizes = panel.universe_size()

    sharpe = sharpe_ratio(net_series.dropna(), periods_per_year=periods)
    ann_return = float((1.0 + net_series.mean()) ** periods - 1.0) if not net_series.empty else 0.0
    ann_vol = float(net_series.std(ddof=1) * np.sqrt(periods)) if len(net_series) > 1 else 0.0
    equity = (1.0 + net_series).cumprod()
    mdd = max_drawdown(equity) if not equity.empty else 0.0

    # Block bootstrap CI: block length ~ 2× rebalance interval so blocks span
    # at least one full holding period and capture serial correlation.
    block_len = max(settings.bootstrap.min_block_length_bars, 2 * rebalance_bars)
    arr = net_series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if arr.size >= 3:
        boot = _block_bootstrap_sharpes(arr, periods, settings.bootstrap.n_resamples, block_len, seed=42)
        ci = (
            float(np.quantile(boot, settings.bootstrap.ci_levels[0])),
            float(np.quantile(boot, settings.bootstrap.ci_levels[1])),
        )
    else:
        ci = (0.0, 0.0)

    psr = probabilistic_sharpe_ratio(net_series, observed_sr=sharpe, periods_per_year=periods)
    dsr = deflated_sharpe_ratio(net_series, observed_sr=sharpe, trials_count=trials_count, periods_per_year=periods)
    one_sided_p = cross_sectional_t_pvalue(net_series.to_numpy(dtype=float))

    return CrossSectionalBacktestResult(
        factor_id=factor_id,
        hypothesis_family=hypothesis_family,
        interval=interval,
        weighting=weighting,
        rebalance_bars=rebalance_bars,
        weights=weights,
        portfolio_returns=net_series,
        gross_returns=gross_series,
        ic_series=ic_series,
        turnover_series=turnover_series,
        mean_ic=ic_stats["mean_ic"],
        ic_vol=ic_stats["ic_vol"],
        ic_tstat_nw=ic_stats["ic_tstat_nw"],
        rankic_tstat_nw=rank_ic_stats["ic_tstat_nw"],
        hit_rate=ic_stats["hit_rate"],
        n_rebalances=int(ic_stats["n_rebalances"]),
        sharpe=sharpe,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        max_drawdown=mdd,
        sharpe_ci_5_95=ci,
        probabilistic_sharpe=psr,
        deflated_sharpe=dsr,
        return_autocorr_lag1=return_autocorrelation_lag1(net_series),
        avg_turnover=float(turnover_series.mean()),
        avg_universe_size=float(universe_sizes.mean()),
        one_sided_pvalue=one_sided_p,
    )


def to_legacy_backtest_result(
    result: CrossSectionalBacktestResult,
    *,
    candidate_id: str,
    method_id: str = "cross_sectional_long_short",
    market: str = "um_futures",
    experiment_id: str | None = None,
    effective_trials_count: int = 1,
    global_trials_count: int = 1,
):
    """Adapt a cross-sectional result into the legacy ``BacktestResult`` shape.

    The downstream gatecheck/hardscore pipeline reads ``BacktestResult``;
    rather than fork that pipeline we expose a thin adapter.  Per-asset
    fields that don't apply cross-sectionally are set to neutral defaults
    that won't trigger spurious gate failures:

    * ``symbol`` → ``"XSEC:<factor_id>"`` (sentinel that gatechecks treat as
      universe-level).
    * ``regime_conditional_metrics`` → empty dict (regime gating happens at
      the panel level if at all).
    * ``factor_turnover`` → mean per-bar turnover from the long-short
      portfolio.
    """
    import uuid

    from factor_mining.models import BacktestResult, MetricsBlock

    metrics = MetricsBlock(
        total_return=float((1.0 + result.portfolio_returns).prod() - 1.0)
        if len(result.portfolio_returns)
        else 0.0,
        annualized_return=result.annualized_return,
        annualized_vol=result.annualized_vol,
        sharpe=result.sharpe,
        max_drawdown=result.max_drawdown,
        calmar=(
            result.annualized_return / abs(result.max_drawdown)
            if result.max_drawdown < 0
            else 0.0
        ),
        trade_count=int((result.turnover_series > 0).sum()),
        pnl=float(result.portfolio_returns.sum()),
    )
    return BacktestResult(
        experiment_id=experiment_id or str(uuid.uuid4()),
        candidate_id=candidate_id,
        hypothesis_family=result.hypothesis_family,
        method_id=method_id,
        symbol=f"XSEC:{result.factor_id}",
        market=market,
        interval=result.interval,
        metrics_primary=metrics,
        metrics_secondary=metrics,
        metrics_gross=metrics,
        ic_tstat_nw=result.ic_tstat_nw,
        rankic_tstat_nw=result.rankic_tstat_nw,
        sharpe_ci_5_95=result.sharpe_ci_5_95,
        probabilistic_sharpe=result.probabilistic_sharpe,
        deflated_sharpe=result.deflated_sharpe,
        effective_trials_at_eval=effective_trials_count,
        global_trials_at_eval=global_trials_count,
        permutation_test_pvalue=result.one_sided_pvalue,
        regime_conditional_metrics={},
        factor_turnover=result.avg_turnover,
        return_autocorr_lag1=result.return_autocorr_lag1,
    )


def returns_matrix(
    results: list[CrossSectionalBacktestResult],
) -> tuple[pd.DataFrame, list[str]]:
    """Stack per-factor PnL series into a (T × K) matrix for Romano–Wolf.

    Joins on the outer time index and fills missing observations with 0
    (factor not active at that time → contributes nothing to either the mean
    or the bootstrap variance).  Returns the matrix and the factor_id order.
    """
    if not results:
        return pd.DataFrame(), []
    series = {r.factor_id: r.portfolio_returns for r in results}
    matrix = pd.concat(series, axis=1).sort_index().fillna(0.0)
    return matrix, list(matrix.columns)
