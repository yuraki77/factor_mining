import math
import warnings

import numpy as np
import pytest

from factor_mining.stats.metrics import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    haircut_sharpe,
    newey_west_tstat,
    permutation_test_mean_ic,
    probabilistic_sharpe_ratio,
    return_autocorrelation_lag1,
)


def test_bh_fdr_adjusts_pvalues_monotonically() -> None:
    adjusted = benjamini_hochberg([0.001, 0.02, 0.20])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert adjusted[0] < 0.01


def test_noise_factor_fails_permutation_more_often_than_true_signal() -> None:
    rng = np.random.default_rng(7)
    factor = rng.normal(size=300)
    returns = factor * 0.02 + rng.normal(scale=0.1, size=300)
    noise = rng.normal(size=300)
    assert permutation_test_mean_ic(factor, returns, n_permutations=100) < 0.05
    assert permutation_test_mean_ic(noise, returns, n_permutations=100) > 0.05


def test_deflated_sharpe_gets_tighter_as_trials_grow() -> None:
    returns = np.repeat(0.001, 500) + np.random.default_rng(1).normal(scale=0.01, size=500)
    low_trials = deflated_sharpe_ratio(returns, observed_sr=1.5, trials_count=10, periods_per_year=365)
    high_trials = deflated_sharpe_ratio(returns, observed_sr=1.5, trials_count=100_000, periods_per_year=365)
    assert high_trials < low_trials
    assert newey_west_tstat(returns) != 0
    assert abs(return_autocorrelation_lag1(returns)) < 0.5


def test_return_autocorrelation_is_zero_and_warning_free_for_constant_series() -> None:
    """A constant return series has zero variance, so its lag-1 autocorrelation is
    undefined (0/0). It must short-circuit to 0.0 rather than let ``np.corrcoef``
    divide by a zero stddev — which returns NaN and emits a RuntimeWarning. WHY it
    matters: a degenerate backtest (e.g. a single-trade signal) produces a near-flat
    return series, and a NaN autocorrelation propagates into the reported metric.
    0.0 ("no autocorrelation") matches rank_ic's convention for an undefined corr."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a divide-by-zero warning fails the test
        result = return_autocorrelation_lag1(np.zeros(50))
    assert result == 0.0  # exactly 0.0, not NaN


def test_newey_west_bandwidth_accounts_for_rolling_ic_overlap() -> None:
    """Q4 regression: a rolling-IC series overlaps — adjacent observations share
    all but one bar of the window — so its autocovariance is non-zero out to
    ~window lags. The generic ``n**(1/3)`` Newey-West bandwidth (~12 here) ignores
    that overlap, under-counts the variance, and over-states the IC t-stat — which
    inflates the G3/G4 gate passes that decide whether a factor is 'real'. WHY it
    matters: an over-stated t-stat promotes noise to a tradable signal. The
    overlap-aware bandwidth (the rolling window) must yield a strictly smaller
    |t-stat| on an overlapping, weakly-trending series."""
    rng = np.random.default_rng(0)
    window = 288
    # Rolling mean of near-zero-mean noise reproduces the overlap structure of a
    # rolling IC: np.convolve(..., "valid") is the trailing mean over `window`.
    raw = rng.standard_normal(2000) + 0.03
    overlapping = np.convolve(raw, np.ones(window) / window, mode="valid")
    naive = newey_west_tstat(overlapping)                 # auto lag ~ n**(1/3)
    overlap_aware = newey_west_tstat(overlapping, lag=window)
    assert abs(naive) > 0.0
    assert abs(overlap_aware) < abs(naive)


def test_newey_west_explicit_lag_is_capped_to_sample() -> None:
    """The overlap bandwidth (e.g. 288) can exceed a short OOS IC slice; the t-stat
    must stay finite and defined (lag < n) rather than reaching past the sample."""
    arr = np.linspace(-1.0, 1.0, 20) + 0.1
    result = newey_west_tstat(arr, lag=288)
    assert math.isfinite(result)


def test_deflated_sharpe_penalty_is_annualized_to_match_sharpe_units() -> None:
    """Q1 regression: the multiple-testing penalty is a per-period Sharpe quantity
    and must be annualized to match the annualized ``observed_sr`` it is subtracted
    from. The prior code subtracted the raw per-period penalty from an annualized
    Sharpe, so intraday candidates were barely deflated. WHY it matters: two
    strategies with the same annualized Sharpe over the same number of bars do NOT
    deserve the same haircut — the intraday one packs the bars into far less
    calendar time, so its expected-max-over-N-trials penalty is larger."""
    n = 2000
    returns = np.zeros(n)  # only the length feeds the penalty term
    observed_sr, trials = 3.0, 100
    per_period_penalty = math.sqrt(2.0 * math.log(trials) / n)
    daily = deflated_sharpe_ratio(returns, observed_sr=observed_sr, trials_count=trials, periods_per_year=365)
    intraday = deflated_sharpe_ratio(returns, observed_sr=observed_sr, trials_count=trials, periods_per_year=105_120)
    # Closed form: observed_sr - per_period_penalty * sqrt(periods_per_year).
    assert math.isclose(daily, observed_sr - per_period_penalty * math.sqrt(365), rel_tol=1e-9)
    assert math.isclose(intraday, observed_sr - per_period_penalty * math.sqrt(105_120), rel_tol=1e-9)
    # The intraday haircut is materially larger than the daily one ...
    assert intraday < daily
    # ... and far larger than the old per-period-penalty bug, which never annualized.
    assert intraday < observed_sr - per_period_penalty


def test_haircut_sharpe_penalty_is_annualized_to_match_sharpe_units() -> None:
    """Same SR-unit fix as the deflated Sharpe, applied to the reported haircut:
    callers pass the *annualized* Sharpe as ``observed_sr`` but the multiple-testing
    penalty ``sqrt(2 ln N / n)`` is a per-period Sharpe quantity, so it must be
    annualized to match. WHY it matters: two strategies with the same annualized
    Sharpe over the same number of bars do NOT deserve the same haircut — the
    intraday one packs the bars into far less calendar time, so its
    expected-max-over-N-trials penalty is larger. The prior code subtracted the raw
    per-period penalty from an annualized Sharpe, leaving the reported haircut
    barely below the headline Sharpe for intraday candidates."""
    observations = 2000
    observed_sr, trials = 3.0, 100
    per_period_penalty = math.sqrt(2.0 * math.log(trials) / observations)
    daily = haircut_sharpe(observed_sr, trials_count=trials, observations=observations, periods_per_year=365)
    intraday = haircut_sharpe(observed_sr, trials_count=trials, observations=observations, periods_per_year=105_120)
    # Closed form: observed_sr - per_period_penalty * sqrt(periods_per_year).
    assert math.isclose(daily, observed_sr - per_period_penalty * math.sqrt(365), rel_tol=1e-9)
    assert math.isclose(intraday, observed_sr - per_period_penalty * math.sqrt(105_120), rel_tol=1e-9)
    # The intraday haircut is materially larger than the daily one ...
    assert intraday < daily
    # ... and far larger than the old per-period-penalty bug, which never annualized.
    assert intraday < observed_sr - per_period_penalty


def test_annualized_return_is_geometric_not_arithmetic() -> None:
    """Q12: annualized return must annualize realized *compound* growth, not the
    arithmetic per-period mean. WHY: compounding the arithmetic mean overstates the
    figure for any volatile series (arithmetic mean ≥ geometric mean), inflating
    Calmar and any return-based gate."""
    import pandas as pd

    from factor_mining.backtest.engine import _metrics_from_returns

    returns = pd.Series([0.10, -0.05] * 50)  # 100 obs, volatile
    metrics = _metrics_from_returns(returns, interval="1d", trade_count=0, pnl=0.0)
    n = len(returns)
    periods = 365  # annualization_factor("1d")
    final_equity = (1.10 * 0.95) ** (n // 2)
    expected_geometric = final_equity ** (periods / n) - 1.0
    assert math.isclose(metrics.annualized_return, expected_geometric, rel_tol=1e-9)
    arithmetic = (1.0 + returns.mean()) ** periods - 1.0
    assert metrics.annualized_return < arithmetic  # geometric strictly lower here


def test_probabilistic_sharpe_is_invariant_to_annualization() -> None:
    """Q2 regression: PSR is a function of the per-period return distribution, so the
    SAME returns must yield the SAME PSR whether ``observed_sr`` arrives in per-period
    units (periods_per_year=1) or annualized units. Passing the annualized Sharpe
    straight into the formula (the prior bug) inflated the skew/kurtosis denominator
    until it clamped at 1e-12 and the estimate degenerated."""
    rng = np.random.default_rng(11)
    returns = 0.0001 + rng.normal(scale=0.01, size=4000)
    sr_pp = returns.mean() / returns.std(ddof=1)  # per-period Sharpe
    periods = 105_120  # 5m bars/year
    sr_annualized = sr_pp * math.sqrt(periods)  # what the engine reports as .sharpe
    psr_from_annualized = probabilistic_sharpe_ratio(returns, observed_sr=sr_annualized, periods_per_year=periods)
    psr_from_per_period = probabilistic_sharpe_ratio(returns, observed_sr=sr_pp, periods_per_year=1)
    assert math.isclose(psr_from_annualized, psr_from_per_period, rel_tol=1e-9, abs_tol=1e-12)
    # A sane probability, not a degenerate 0/1 from a collapsed denominator.
    assert 0.0 < psr_from_annualized < 1.0



def test_deflated_sharpe_observations_matches_series_length() -> None:
    """Callers that know only the sample size (merge-pool re-penalty) must get
    the identical haircut as callers passing a series of that length — the
    penalty depends on n alone, and passing observations avoids fabricating
    placeholder arrays whose contents look load-bearing."""
    returns = np.random.default_rng(7).normal(0.0, 0.01, 500)
    from_series = deflated_sharpe_ratio(returns, observed_sr=1.5, trials_count=64, periods_per_year=365)
    from_count = deflated_sharpe_ratio(None, observed_sr=1.5, trials_count=64, periods_per_year=365, observations=500)
    assert from_count == from_series

    with pytest.raises(ValueError):
        deflated_sharpe_ratio(None, observed_sr=1.5, trials_count=64, periods_per_year=365)


def test_permutation_null_is_calibrated_for_autocorrelated_series() -> None:
    """WHY: an i.i.d. shuffle destroys the factor's autocorrelation, so a
    persistent (AR) signal tested against autocorrelated returns was rejected
    at a multiple of the nominal rate — the advertised robustness check was
    anti-conservative exactly for the signals this system mines. The
    circular-shift null must keep false rejections near nominal, and the
    returned value must be the finite-sample (k+1)/(n+1) estimator, not a
    thin-tailed normal approximation of the null."""
    rng = np.random.default_rng(123)
    n, reps, n_perm = 400, 120, 99
    rejections = 0
    for rep in range(reps):
        eps = rng.normal(size=n)
        factor = np.empty(n)
        factor[0] = eps[0]
        for t in range(1, n):
            factor[t] = 0.9 * factor[t - 1] + eps[t]
        shocks = rng.normal(size=n)
        vol = 0.5 + 0.5 * np.abs(np.sin(np.arange(n) / 25.0))
        ret = np.empty(n)
        ret[0] = shocks[0] * vol[0]
        for t in range(1, n):
            ret[t] = 0.35 * ret[t - 1] + shocks[t] * vol[t]

        p = permutation_test_mean_ic(factor, ret, n_permutations=n_perm, seed=rep)
        grid = p * (n_perm + 1)
        assert abs(grid - round(grid)) < 1e-9, "p must be the empirical (k+1)/(n+1) estimator"
        if p <= 0.05:
            rejections += 1
    assert rejections / reps <= 0.125, f"null rejection rate {rejections / reps:.3f} is anti-conservative"
