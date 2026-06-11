from __future__ import annotations

import math
from collections.abc import Sequence

import numba
import numpy as np
import pandas as pd


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def annualization_factor(interval: str) -> int:
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return int(365 * 24 * 60 / minutes)
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return int(365 * 24 / hours)
    return 365


def sharpe_ratio(returns: Sequence[float], *, periods_per_year: int) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(equity: Sequence[float]) -> float:
    arr = np.asarray(equity, dtype=float)
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    drawdowns = arr / np.where(peaks == 0, np.nan, peaks) - 1.0
    return float(np.nanmin(drawdowns))


def newey_west_tstat(values: Sequence[float], *, lag: int | None = None) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return 0.0
    lag = int(n ** (1 / 3)) if lag is None else lag
    centered = arr - arr.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        variance += 2.0 * weight * gamma
    if variance <= 0:
        return 0.0
    standard_error = math.sqrt(variance / n)
    if standard_error == 0:
        return 0.0
    return float(arr.mean() / standard_error)


def stationary_block_bootstrap_sharpe_ci(
    returns: Sequence[float],
    *,
    periods_per_year: int,
    n_resamples: int,
    block_length_bars: int,
    ci: tuple[float, float] = (0.05, 0.95),
    seed: int = 42,
) -> tuple[float, float]:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    block_length_bars = max(1, min(block_length_bars, n))
    p_new_block = 1.0 / block_length_bars
    samples = []
    for _ in range(n_resamples):
        idx = np.empty(n, dtype=int)
        idx[0] = rng.integers(0, n)
        for i in range(1, n):
            if rng.random() < p_new_block:
                idx[i] = rng.integers(0, n)
            else:
                idx[i] = (idx[i - 1] + 1) % n
        samples.append(sharpe_ratio(arr[idx], periods_per_year=periods_per_year))
    return (float(np.quantile(samples, ci[0])), float(np.quantile(samples, ci[1])))


def probabilistic_sharpe_ratio(
    returns: Sequence[float],
    *,
    observed_sr: float,
    periods_per_year: int,
    benchmark_sr: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    ``observed_sr`` and ``benchmark_sr`` are the *annualized* Sharpe ratios the
    rest of the system reports; they are converted back to **per-period units**
    here because the skewness/kurtosis adjustment and the ``sqrt(n - 1)``
    scaling are defined on the per-period return distribution. Passing the
    annualized Sharpe straight into the formula (the prior bug) scaled the
    ``skew * SR`` term by ``sqrt(periods_per_year)`` and the ``SR**2`` term by
    ``periods_per_year``, which collapsed the denominator (clamped at 1e-12) and
    drove the estimate to a meaningless 0/1.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return 0.0
    centered = arr - arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    scale = math.sqrt(max(int(periods_per_year), 1))
    sr = observed_sr / scale
    benchmark = benchmark_sr / scale
    skew = float(np.mean((centered / std) ** 3))
    kurtosis = float(np.mean((centered / std) ** 4))
    denominator = math.sqrt(max(1e-12, 1 - skew * sr + ((kurtosis - 1) / 4) * sr**2))
    z_score = (sr - benchmark) * math.sqrt(n - 1) / denominator
    return normal_cdf(z_score)


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    observed_sr: float,
    trials_count: int,
    periods_per_year: int,
    benchmark_sr: float = 0.0,
) -> float:
    """Deflated Sharpe haircut: ``observed_sr`` minus the expected maximum
    Sharpe under the null over ``trials_count`` independent trials.

    ``observed_sr``/``benchmark_sr`` are *annualized* Sharpe ratios. The
    expected-maximum penalty ``sqrt(2 ln N / n)`` is a *per-period* Sharpe
    quantity, so it is annualized (``× sqrt(periods_per_year)``) before being
    subtracted. The prior code subtracted the per-period penalty straight from
    the annualized Sharpe, under-stating the haircut by ``sqrt(periods_per_year)``
    (≈ 324× on 5m bars) so almost every candidate cleared a positive DSR.
    """
    arr = np.asarray(returns, dtype=float)
    n = max(len(arr), 1)
    per_period_penalty = math.sqrt(2.0 * math.log(max(trials_count, 1)) / n)
    annualized_penalty = per_period_penalty * math.sqrt(max(int(periods_per_year), 1))
    return float(observed_sr - benchmark_sr - annualized_penalty)


def haircut_sharpe(observed_sr: float, *, trials_count: int, observations: int, periods_per_year: int) -> float:
    """Haircut Sharpe: ``observed_sr`` minus the multiple-testing penalty.

    ``observed_sr`` is the *annualized* Sharpe the rest of the system reports,
    while the penalty ``sqrt(2 ln N / n)`` is a *per-period* Sharpe quantity, so
    it is annualized (``× sqrt(periods_per_year)``) before being subtracted —
    matching :func:`deflated_sharpe_ratio`. The prior code subtracted the raw
    per-period penalty from the annualized Sharpe, under-stating the haircut by
    ``sqrt(periods_per_year)`` (≈ 324× on 5m bars).
    """
    observations = max(observations, 1)
    per_period_penalty = math.sqrt(2.0 * math.log(max(trials_count, 1)) / observations)
    annualized_penalty = per_period_penalty * math.sqrt(max(int(periods_per_year), 1))
    return float(observed_sr - annualized_penalty)


def benjamini_hochberg(pvalues: Sequence[float], *, q: float = 0.05, n_tests: int | None = None) -> list[float]:
    pvalues_arr = np.asarray(pvalues, dtype=float)
    n = len(pvalues_arr)
    if n == 0:
        return []
    total_tests = max(n, int(n_tests or n))
    order = np.argsort(pvalues_arr)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        value = min(prev, pvalues_arr[idx] * total_tests / rank)
        adjusted[idx] = value
        prev = value
    return [float(min(1.0, value)) for value in adjusted]


def one_sided_tstat_pvalue(tstat: float) -> float:
    value = float(tstat)
    if not math.isfinite(value):
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - normal_cdf(value))))


def combined_ic_tstat_pvalue(ic_tstat: float, rankic_tstat: float) -> float:
    best = min(one_sided_tstat_pvalue(ic_tstat), one_sided_tstat_pvalue(rankic_tstat))
    return float(min(1.0, 2.0 * best))


def permutation_test_mean_ic(
    factor_values: Sequence[float],
    forward_returns: Sequence[float],
    *,
    n_permutations: int,
    seed: int = 42,
) -> float:
    factor = np.asarray(factor_values, dtype=float)
    returns = np.asarray(forward_returns, dtype=float)
    mask = np.isfinite(factor) & np.isfinite(returns)
    factor = factor[mask]
    returns = returns[mask]
    if factor.size < 3 or np.std(factor) == 0 or np.std(returns) == 0:
        return 1.0
    observed = abs(float(np.corrcoef(factor, returns)[0, 1]))
    rng = np.random.default_rng(seed)
    exceed = 0
    null_stats = np.empty(max(0, int(n_permutations)), dtype=float)
    for _ in range(n_permutations):
        permuted = rng.permutation(factor)
        stat = abs(float(np.corrcoef(permuted, returns)[0, 1]))
        null_stats[_] = stat
        if stat >= observed:
            exceed += 1
    empirical_p = float((exceed + 1) / (n_permutations + 1))
    null_std = float(null_stats.std(ddof=1)) if null_stats.size > 1 else 0.0
    if null_std <= 0.0:
        return empirical_p
    z_score = (observed - float(null_stats.mean())) / null_std
    return float(max(0.0, min(1.0, 1.0 - normal_cdf(z_score))))


def rank_ic(factor_values: Sequence[float], forward_returns: Sequence[float]) -> float:
    factor = pd.Series(factor_values, dtype=float).rank()
    returns = pd.Series(forward_returns, dtype=float).rank()
    if factor.std() == 0 or returns.std() == 0:
        return 0.0
    return float(factor.corr(returns))


def pbo_from_oos_scores(scores: Sequence[float]) -> float:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    median = float(np.median(arr))
    return float(np.mean(arr < median))


def return_autocorrelation_lag1(returns: Sequence[float]) -> float:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0
    return float(np.corrcoef(arr[1:], arr[:-1])[0, 1])


# ── numba-accelerated rolling IC & bootstrap ──────────────────────


@numba.njit(cache=True)
def rolling_rank_ic(signals: np.ndarray, forward_returns: np.ndarray, window: int = 288, min_periods: int = 10) -> np.ndarray:
    """Rolling Spearman (rank) IC — numba compiled, ~50-100x faster than pure Python."""
    n = len(signals)
    out = np.zeros(n)
    for i in range(min_periods, n + 1):
        start = max(0, i - window)
        s_win = signals[start:i]
        f_win = forward_returns[start:i]
        m = i - start
        # rank via argsort
        s_rank = np.empty(m)
        f_rank = np.empty(m)
        s_order = np.argsort(s_win)
        f_order = np.argsort(f_win)
        for j in range(m):
            s_rank[s_order[j]] = j + 1
            f_rank[f_order[j]] = j + 1
        # Pearson correlation on ranks
        s_mean = 0.0
        f_mean = 0.0
        for j in range(m):
            s_mean += s_rank[j]
            f_mean += f_rank[j]
        s_mean /= m
        f_mean /= m
        num = 0.0
        den1 = 0.0
        den2 = 0.0
        for j in range(m):
            ds = s_rank[j] - s_mean
            df = f_rank[j] - f_mean
            num += ds * df
            den1 += ds * ds
            den2 += df * df
        den = np.sqrt(den1 * den2)
        out[i - 1] = num / den if den > 0.0 else 0.0
    return out


@numba.njit(cache=True)
def rolling_pearson_ic(signals: np.ndarray, forward_returns: np.ndarray, window: int = 288, min_periods: int = 10) -> np.ndarray:
    """Rolling Pearson IC — numba compiled."""
    n = len(signals)
    out = np.zeros(n)
    for i in range(min_periods, n + 1):
        start = max(0, i - window)
        s_win = signals[start:i]
        f_win = forward_returns[start:i]
        m = i - start
        s_mean = 0.0
        f_mean = 0.0
        for j in range(m):
            s_mean += s_win[j]
            f_mean += f_win[j]
        s_mean /= m
        f_mean /= m
        num = 0.0
        den1 = 0.0
        den2 = 0.0
        for j in range(m):
            ds = s_win[j] - s_mean
            df = f_win[j] - f_mean
            num += ds * df
            den1 += ds * ds
            den2 += df * df
        den = np.sqrt(den1 * den2)
        out[i - 1] = num / den if den > 0.0 else 0.0
    return out


@numba.njit(cache=True)
def _block_bootstrap_sharpes(
    returns: np.ndarray, periods_per_year: int, n_resamples: int, block_length_bars: int, seed: int
) -> np.ndarray:
    """Generate n_resamples block-bootstrap Sharpe ratios in compiled code."""
    np.random.seed(seed)
    n = len(returns)
    p_new_block = 1.0 / block_length_bars
    samples = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = np.empty(n, dtype=np.int32)
        idx[0] = np.random.randint(0, n)
        for i in range(1, n):
            if np.random.random() < p_new_block:
                idx[i] = np.random.randint(0, n)
            else:
                idx[i] = (idx[i - 1] + 1) % n
        # compute Sharpe from bootstrapped returns
        mean_r = 0.0
        for j in range(n):
            mean_r += returns[idx[j]]
        mean_r /= n
        var = 0.0
        for j in range(n):
            diff = returns[idx[j]] - mean_r
            var += diff * diff
        var /= (n - 1)
        std = math.sqrt(var)
        if std > 0.0:
            samples[b] = mean_r / std * math.sqrt(periods_per_year)
        else:
            samples[b] = 0.0
    return samples
