import numpy as np

from factor_mining.stats.metrics import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    newey_west_tstat,
    permutation_test_mean_ic,
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
    low_trials = deflated_sharpe_ratio(returns, observed_sr=1.5, trials_count=10)
    high_trials = deflated_sharpe_ratio(returns, observed_sr=1.5, trials_count=100_000)
    assert high_trials < low_trials
    assert newey_west_tstat(returns) != 0
    assert abs(return_autocorrelation_lag1(returns)) < 0.5

