"""Cross-sectional IC and FWE-controlled multiple-testing procedures.

The single-asset ``stats.metrics`` module computes rolling **time-series** IC
(correlation of a single asset's signal with its own forward return over a
window).  Cross-sectional factor tests need a different object: at each
rebalance t, IC_t is the correlation *across assets* between factor score and
forward return.  The aggregate test statistic is then a Newey–West t-stat on
the IC_t series.

For multiple-factor testing we use Romano–Wolf step-down on a stationary
block bootstrap of the per-rebalance return matrix.  Vanilla BH on per-factor
p-values assumes PRDS, which factor returns generally violate (momentum and
reversal share variance, value and carry overlap).  Romano–Wolf controls
FWER under arbitrary dependence.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from factor_mining.data.panel import FactorPanel
from factor_mining.stats.metrics import newey_west_tstat, normal_cdf


def cross_sectional_ic(
    panel: FactorPanel, *, method: str = "spearman", min_universe: int = 3
) -> pd.Series:
    """Per-rebalance cross-sectional IC.

    Args:
        panel: aligned factor + forward-returns panel.
        method: ``"spearman"`` (default, Rank-IC — robust to factor outliers)
            or ``"pearson"`` (raw IC).
        min_universe: rebalances with fewer than this many tradeable assets
            produce NaN (a 2-asset correlation is degenerate).

    Returns:
        Series indexed by ``open_time`` of IC_t values.  NaN where the
        universe was too small or the factor / returns had zero variance
        cross-sectionally.
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError(f"Unknown IC method {method!r}")
    factor = panel.factor.where(panel.universe_mask)
    returns = panel.forward_returns.where(panel.universe_mask)
    out = np.full(len(factor), np.nan)
    factor_arr = factor.to_numpy(dtype=float)
    returns_arr = returns.to_numpy(dtype=float)
    for i in range(factor_arr.shape[0]):
        row_f = factor_arr[i]
        row_r = returns_arr[i]
        mask = np.isfinite(row_f) & np.isfinite(row_r)
        if mask.sum() < min_universe:
            continue
        f = row_f[mask]
        r = row_r[mask]
        if method == "spearman":
            f = _ranks(f)
            r = _ranks(r)
        f_std = f.std(ddof=1)
        r_std = r.std(ddof=1)
        if f_std == 0.0 or r_std == 0.0:
            continue
        out[i] = float(np.corrcoef(f, r)[0, 1])
    return pd.Series(out, index=factor.index, name="ic")


def _ranks(arr: np.ndarray) -> np.ndarray:
    """Average ranks (ties get the mean rank)."""
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # tie handling: assign mean rank to equal values
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts), dtype=float)
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def ic_summary(ic_series: pd.Series) -> dict[str, float]:
    """Mean IC, IC volatility, and NW t-stat — the standard factor diagnostics."""
    arr = ic_series.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {"mean_ic": 0.0, "ic_vol": 0.0, "ic_tstat_nw": 0.0, "hit_rate": 0.0, "n_rebalances": 0}
    return {
        "mean_ic": float(arr.mean()),
        "ic_vol": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ic_tstat_nw": newey_west_tstat(arr),
        "hit_rate": float((arr > 0).mean()),
        "n_rebalances": int(arr.size),
    }


# ── Romano–Wolf step-down on stationary block bootstrap ──────────────


def romano_wolf_step_down(
    returns_matrix: np.ndarray | pd.DataFrame,
    *,
    n_resamples: int = 1000,
    block_length: int = 20,
    seed: int = 42,
    alternative: str = "greater",
) -> np.ndarray:
    """Family-wise-error-controlled adjusted p-values via Romano–Wolf.

    Tests the joint null ``mean(returns_matrix[:, k]) <= 0`` for each column
    k against the one-sided alternative ``> 0``.  The test statistic is the
    Studentized mean.  Critical values come from the stationary block
    bootstrap of the centered return matrix, which preserves the
    cross-factor dependence structure and the within-factor serial
    correlation.

    Args:
        returns_matrix: (T × K) matrix of per-rebalance returns, one column
            per factor.  Rows are time-aligned across factors so the
            bootstrap can preserve their joint distribution.
        n_resamples: number of bootstrap samples.  1000+ for production.
        block_length: expected block length in bars.  Should be ~2× the
            average holding period to capture serial correlation in
            individual factor PnLs.
        seed: deterministic RNG seed.
        alternative: ``"greater"`` (default, one-sided; we want factors with
            positive mean return) or ``"two-sided"``.

    Returns:
        Array of length K of FWER-adjusted p-values, in original column
        order.  Reject H_0,k iff adjusted_p[k] <= alpha.

    Reference:
        Romano & Wolf (2005), "Stepwise multiple testing as formalized data
        snooping," Econometrica 73(4): 1237–1282.  We use the studentized
        version because raw means can have very different scales across
        factors (rank-weighted long-short vs quintile).
    """
    if alternative not in {"greater", "two-sided"}:
        raise ValueError(f"Unknown alternative {alternative!r}")
    if isinstance(returns_matrix, pd.DataFrame):
        returns_matrix = returns_matrix.to_numpy(dtype=float)
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2D (T × K)")
    T, K = R.shape
    if T < 3:
        return np.ones(K, dtype=float)
    if K == 0:
        return np.zeros(0, dtype=float)
    n_resamples = max(100, int(n_resamples))
    block_length = max(1, min(int(block_length), T))

    # Studentized observed statistics
    means = np.nanmean(R, axis=0)
    stds = np.nanstd(R, axis=0, ddof=1)
    stds = np.where(stds == 0.0, np.nan, stds)
    t_obs = means * math.sqrt(T) / stds
    if alternative == "two-sided":
        t_obs_used = np.abs(t_obs)
    else:
        t_obs_used = t_obs

    # Bootstrap under H_0 (mean zero) by centering the columns
    centered = R - np.nanmean(R, axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    p_new_block = 1.0 / block_length
    null_stats = np.empty((n_resamples, K), dtype=float)
    for b in range(n_resamples):
        idx = np.empty(T, dtype=np.int64)
        idx[0] = rng.integers(0, T)
        for i in range(1, T):
            if rng.random() < p_new_block:
                idx[i] = rng.integers(0, T)
            else:
                idx[i] = (idx[i - 1] + 1) % T
        boot = centered[idx]
        boot_means = np.nanmean(boot, axis=0)
        boot_stds = np.nanstd(boot, axis=0, ddof=1)
        boot_stds = np.where(boot_stds == 0.0, np.nan, boot_stds)
        boot_t = boot_means * math.sqrt(T) / boot_stds
        null_stats[b] = np.abs(boot_t) if alternative == "two-sided" else boot_t

    # Step-down procedure: at each step, take the max null-statistic over the
    # currently-active hypotheses; this preserves FWER under arbitrary
    # dependence because we compare to the SAME bootstrap draws each step.
    order = np.argsort(-t_obs_used)  # most significant first
    adj_p = np.ones(K, dtype=float)
    prev_p = 0.0
    for step, k in enumerate(order):
        active = order[step:]
        max_null = np.nanmax(null_stats[:, active], axis=1)
        # P(max over active hypotheses >= observed_k under H_0)
        p_k = float((np.sum(max_null >= t_obs_used[k]) + 1) / (n_resamples + 1))
        p_k = max(p_k, prev_p)  # monotonicity in step-down order
        adj_p[k] = min(1.0, p_k)
        prev_p = p_k
    return adj_p


def cross_sectional_t_pvalue(returns: Sequence[float], *, alternative: str = "greater") -> float:
    """One-sided NW-adjusted t-test p-value for a single factor's mean return.

    Used as the unadjusted per-factor p-value.  Romano–Wolf above returns
    FWE-adjusted p-values; this is the building block when you want a single
    factor's marginal significance.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 1.0
    t = newey_west_tstat(arr)
    if alternative == "greater":
        return float(max(0.0, min(1.0, 1.0 - normal_cdf(t))))
    if alternative == "two-sided":
        return float(max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(abs(t))))))
    raise ValueError(f"Unknown alternative {alternative!r}")
