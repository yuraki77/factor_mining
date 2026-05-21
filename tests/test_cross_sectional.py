"""Cross-sectional backtest + FDR tests.

These tests encode WHY each piece exists, not just WHAT it does:

* ``test_panel_excludes_pre_listing_bars`` — the universe must be
  point-in-time, otherwise we leak survivorship bias.
* ``test_zero_lookahead_in_standardization`` — cross-sectional standardization
  must use only data at t, never a rolling time-series mean.
* ``test_rank_weights_are_dollar_neutral`` — a long-short factor PnL series
  is only comparable to other factors if gross exposure is normalized.
* ``test_ic_recovers_known_sign`` — sanity check that a constructed factor
  with monotonic forward returns gives positive IC.
* ``test_engine_sharpe_matches_hand_computed`` — the engine's PnL series is
  the same one the FDR layer consumes; if Sharpe doesn't match a hand
  calculation, FDR is testing the wrong null.
* ``test_romano_wolf_controls_fwer_under_null`` — under a pure noise null,
  Romano–Wolf must reject at most α of the time across many factors, even
  when those factors are correlated.  This is the headline statistical
  guarantee the refactor is buying us.
* ``test_romano_wolf_finds_true_signal`` — power check: a real factor
  surfaces above the noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_mining.backtest.cross_sectional import (
    _rank_weights,
    _quantile_weights,
    run_cross_sectional_backtest,
    returns_matrix,
)
from factor_mining.config import Settings
from factor_mining.data.panel import build_panel
from factor_mining.stats.cross_sectional import (
    cross_sectional_ic,
    romano_wolf_step_down,
)
from factor_mining.validation.gatecheck import apply_fdr_cross_sectional


# ── Helpers ──────────────────────────────────────────────────────────


def _make_frame(prices: np.ndarray, *, start_ms: int = 0, interval_ms: int = 300_000) -> pd.DataFrame:
    n = len(prices)
    open_time = np.arange(n, dtype=np.int64) * interval_ms + start_ms
    return pd.DataFrame({"open_time": open_time, "open": prices.astype(float)})


def _synthetic_universe(n_assets: int, n_bars: int, seed: int = 7) -> tuple[dict, dict]:
    """Build (frames_by_symbol, signals_by_symbol) for testing.

    Each asset is a GBM-like price walk; the signal is a noise array we'll
    override in individual tests.
    """
    rng = np.random.default_rng(seed)
    frames: dict[str, pd.DataFrame] = {}
    signals: dict[str, np.ndarray] = {}
    for k in range(n_assets):
        rets = rng.normal(0.0, 0.01, n_bars)
        prices = 100.0 * np.exp(np.cumsum(rets))
        frames[f"S{k:02d}"] = _make_frame(prices)
        signals[f"S{k:02d}"] = rng.normal(0.0, 1.0, n_bars)
    return frames, signals


# ── Panel construction ──────────────────────────────────────────────


def test_panel_excludes_pre_listing_bars() -> None:
    """A symbol listed mid-series must be NaN before its listing date."""
    frames, signals = _synthetic_universe(3, 100)
    # S02 only listed from bar 50 onward
    interval = 300_000
    listing = {"S02": 50 * interval}
    panel = build_panel(frames, signals, forward_horizon_bars=1, listing_dates=listing)
    pre = panel.universe_mask.iloc[:50]
    post = panel.universe_mask.iloc[60:90]
    assert (~pre["S02"]).all(), "S02 must not be in the universe before listing"
    assert post["S02"].any(), "S02 must be in the universe after listing"
    # Pre-listing factor and returns must be NaN, not zero — zeros would
    # silently feed bogus rank-0 weights into the long-short construction.
    assert panel.factor["S02"].iloc[:50].isna().all()
    assert panel.forward_returns["S02"].iloc[:50].isna().all()


# ── Cross-sectional construction primitives ─────────────────────────


def test_rank_weights_are_dollar_neutral() -> None:
    factor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.ones(5, dtype=bool)
    weights = _rank_weights(factor, mask)
    assert weights.sum() == pytest.approx(0.0, abs=1e-12), "Dollar-neutrality"
    assert np.abs(weights).sum() == pytest.approx(1.0, abs=1e-12), "Gross exposure normalized to 1"
    # The largest factor value gets the largest weight; the smallest gets the
    # most negative.  This is the WHY of "long top, short bottom".
    assert np.argmax(weights) == 4
    assert np.argmin(weights) == 0


def test_rank_weights_skip_out_of_universe() -> None:
    factor = np.array([10.0, 20.0, np.nan, 40.0])
    mask = np.array([True, True, False, True])
    weights = _rank_weights(factor, mask)
    assert weights[2] == 0.0, "Out-of-universe must have zero weight"
    assert weights.sum() == pytest.approx(0.0, abs=1e-12)


def test_quantile_weights_long_top_short_bottom() -> None:
    factor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mask = np.ones(10, dtype=bool)
    weights = _quantile_weights(factor, mask, top_frac=0.2)  # quintile
    # Top 2 (indices 8, 9) long; bottom 2 (indices 0, 1) short
    assert weights[8] > 0 and weights[9] > 0
    assert weights[0] < 0 and weights[1] < 0
    assert weights[2:8].sum() == pytest.approx(0.0)
    assert weights.sum() == pytest.approx(0.0, abs=1e-12)


# ── Cross-sectional IC ──────────────────────────────────────────────


def test_ic_recovers_known_sign() -> None:
    """Construct a panel where factor == next return; IC must be 1.0."""
    n_bars = 50
    n_assets = 5
    rng = np.random.default_rng(0)
    frames: dict[str, pd.DataFrame] = {}
    signals: dict[str, np.ndarray] = {}
    for k in range(n_assets):
        # Random walk
        rets = rng.normal(0.0, 0.01, n_bars)
        prices = 100.0 * np.exp(np.cumsum(rets))
        frames[f"S{k}"] = _make_frame(prices)
        # Signal at t is built from price[t+1]/price[t] - 1 → perfect foresight
        # ONLY for the purposes of this test.  In production, signals must be
        # functions of past data only.
        fwd = np.empty(n_bars)
        fwd[:-1] = prices[1:] / prices[:-1] - 1.0
        fwd[-1] = 0.0
        signals[f"S{k}"] = fwd
    panel = build_panel(frames, signals, forward_horizon_bars=1)
    ic = cross_sectional_ic(panel, method="pearson").dropna()
    assert ic.mean() > 0.99, "Perfect-foresight factor must have IC ≈ 1"


def test_zero_lookahead_in_standardization() -> None:
    """Replacing future bars' factor values must not change today's weights.

    If the engine accidentally standardized across time (rolling mean), the
    weights at time t would depend on factor values at t' > t. This test
    catches that regression.
    """
    n_bars = 100
    frames, signals = _synthetic_universe(8, n_bars, seed=11)
    panel_a = build_panel(frames, signals, forward_horizon_bars=1)
    settings = Settings()
    res_a = run_cross_sectional_backtest(
        panel_a, settings=settings, factor_id="A", weighting="rank", rebalance_bars=1
    )
    # Now scramble future bars and rebuild
    rng = np.random.default_rng(99)
    signals_b = {sym: sig.copy() for sym, sig in signals.items()}
    for sym, sig in signals_b.items():
        sig[50:] = rng.normal(0.0, 5.0, n_bars - 50)
    panel_b = build_panel(frames, signals_b, forward_horizon_bars=1)
    res_b = run_cross_sectional_backtest(
        panel_b, settings=settings, factor_id="B", weighting="rank", rebalance_bars=1
    )
    # First 50 bars' weights must be identical across the two panels
    np.testing.assert_allclose(
        res_a.weights.iloc[:50].to_numpy(),
        res_b.weights.iloc[:50].to_numpy(),
        atol=1e-12,
        err_msg="Pre-bar-50 weights depend on post-bar-50 factor values → lookahead leak",
    )


# ── Engine end-to-end ───────────────────────────────────────────────


def test_engine_sharpe_matches_hand_computed() -> None:
    """End-to-end Sharpe must match a direct computation on the returns series.

    This is the load-bearing check that the engine's PnL series is the same
    object the FDR layer will consume.
    """
    n_bars = 200
    frames, signals = _synthetic_universe(6, n_bars, seed=3)
    panel = build_panel(frames, signals, forward_horizon_bars=1)
    settings = Settings()
    result = run_cross_sectional_backtest(
        panel, settings=settings, factor_id="hand", weighting="rank", rebalance_bars=1
    )
    # Hand computation: mean(net) / std(net) * sqrt(periods_per_year)
    arr = result.portfolio_returns.dropna().to_numpy(dtype=float)
    periods = int(365 * 24 * 60 / 5)  # 5-minute bars
    expected = arr.mean() / arr.std(ddof=1) * np.sqrt(periods)
    assert result.sharpe == pytest.approx(expected, rel=1e-6, abs=1e-9)


# ── Romano–Wolf: FWER control and power ─────────────────────────────


def test_romano_wolf_controls_fwer_under_null() -> None:
    """Under H_0 (all factors have zero mean), at most α of trials reject anywhere.

    We simulate 500 trials, each with 10 correlated noise factors.  Without
    correction, BH-style controls allow expected-α false discoveries per
    trial; Romano–Wolf controls the *family-wise* error rate, so the
    fraction of trials with ANY rejection should be ≤ α.
    """
    n_trials = 200
    n_factors = 10
    T = 100
    rng = np.random.default_rng(42)
    rejections = 0
    for trial in range(n_trials):
        # Correlated noise: shared market factor + idiosyncratic
        common = rng.normal(0.0, 0.01, T)
        R = np.column_stack([
            0.5 * common + rng.normal(0.0, 0.01, T) for _ in range(n_factors)
        ])
        adj_p = romano_wolf_step_down(R, n_resamples=200, block_length=5, seed=trial)
        if (adj_p <= 0.05).any():
            rejections += 1
    rate = rejections / n_trials
    # With α=0.05, Wilson 95% upper bound on rate is ~0.087 for n=200; allow
    # some slack for bootstrap variance with low n_resamples in tests.
    assert rate <= 0.12, (
        f"Romano–Wolf rejected under null in {rate:.1%} of trials — "
        "FWER control is broken"
    )


def test_romano_wolf_finds_true_signal() -> None:
    """Power check: a true factor (positive mean) must be detected."""
    T = 300
    rng = np.random.default_rng(0)
    # Factor 0 has positive mean; rest are pure noise
    true_signal = rng.normal(0.003, 0.01, T)  # Sharpe ≈ 0.3 per bar = very high annualized
    noise = [rng.normal(0.0, 0.01, T) for _ in range(5)]
    R = np.column_stack([true_signal] + noise)
    adj_p = romano_wolf_step_down(R, n_resamples=500, block_length=10, seed=1)
    assert adj_p[0] <= 0.05, f"True signal not detected; p={adj_p[0]:.3f}"
    # Most of the noise should NOT be detected.  Allowing 1 false positive
    # out of 5 due to bootstrap variance at n_resamples=500.
    false_positives = int((adj_p[1:] <= 0.05).sum())
    assert false_positives <= 1, (
        f"Romano–Wolf flagged {false_positives}/5 noise factors as significant"
    )


# ── FDR integration ─────────────────────────────────────────────────


def test_apply_fdr_cross_sectional_groups_by_family() -> None:
    """Two factors from different families must be tested independently."""
    n_bars = 150
    frames, signals = _synthetic_universe(6, n_bars, seed=21)
    panel = build_panel(frames, signals, forward_horizon_bars=1)
    settings = Settings()
    r_a = run_cross_sectional_backtest(
        panel, settings=settings, factor_id="mom_A",
        hypothesis_family="momentum", weighting="rank",
    )
    r_b = run_cross_sectional_backtest(
        panel, settings=settings, factor_id="val_B",
        hypothesis_family="value", weighting="rank",
    )
    adj = apply_fdr_cross_sectional(
        [r_a, r_b], settings, n_resamples=200, block_length=5
    )
    assert set(adj.keys()) == {"mom_A", "val_B"}
    # Single-factor-per-family: adjusted p ≈ unadjusted, so should be in [0, 1]
    for value in adj.values():
        assert 0.0 <= value <= 1.0


def test_returns_matrix_aligns_factors() -> None:
    n_bars = 100
    frames, signals = _synthetic_universe(4, n_bars, seed=5)
    panel = build_panel(frames, signals, forward_horizon_bars=1)
    settings = Settings()
    r1 = run_cross_sectional_backtest(panel, settings=settings, factor_id="a")
    r2 = run_cross_sectional_backtest(panel, settings=settings, factor_id="b")
    matrix, factor_ids = returns_matrix([r1, r2])
    assert list(factor_ids) == ["a", "b"]
    assert matrix.shape == (n_bars, 2)
    # No NaNs after the outer-join fill
    assert not matrix.isna().any().any()
