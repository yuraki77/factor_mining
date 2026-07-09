"""The power harness answers "how strong must a real alpha be" — so it must itself be
trustworthy: planted strength must raise achieved Sharpe monotonically (or the alpha
axis means nothing), MDA extraction and binding-gate attribution must be deterministic,
and the G2/PBO standalone fail-close must be surfaced, not hidden inside a composite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_mining.calibration import (
    PowerTrial,
    default_candidates,
    minimum_detectable_alpha,
    plant_signal,
    plant_signal_with_horizon,
    power_sweep,
    summarize_power,
)
from factor_mining.config import DataConfig, Settings


def _random_walk_frame(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.004, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.003, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.003, size=n))
    return pd.DataFrame({
        "open_time": [1_700_000_000_000 + i * 300_000 for i in range(n)],
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(80.0, 120.0, size=n),
        "quote_volume": rng.uniform(8e5, 1.2e6, size=n),
    })


def test_horizon_one_plant_is_the_far_power_arm() -> None:
    """The FAR harness's positive control and the power sweep's horizon-1 cell must be
    the same distribution, or the two harnesses' power numbers can't be compared."""
    frame = _random_walk_frame(400)
    a = plant_signal(frame, np.random.default_rng(5), alpha=0.5, noise=1.0)
    b = plant_signal_with_horizon(frame, np.random.default_rng(5), alpha=0.5, noise=1.0, horizon=1)
    np.testing.assert_allclose(a, b)


def test_longer_horizon_means_persistence_and_lower_turnover() -> None:
    """The horizon knob exists to give the cost gates a fair test: a 48-bar edge must be
    smoother than a per-bar edge IN BOTH components. The noise term matters most — with
    per-bar iid noise the position churns every bar regardless of horizon and turnover,
    not evidence, decides every gate (first smoke run: netSR ≈ −160 at alpha=0)."""
    frame = _random_walk_frame(1200)
    fast = plant_signal_with_horizon(frame, np.random.default_rng(3), alpha=1.0, noise=1.0, horizon=1)
    slow = plant_signal_with_horizon(frame, np.random.default_rng(3), alpha=1.0, noise=1.0, horizon=48)
    assert np.nanmean(np.abs(np.diff(slow))) < 0.25 * np.nanmean(np.abs(np.diff(fast)))
    # noise smoothing is variance-preserving: strength calibration stays comparable
    assert 0.5 < np.nanstd(slow) / np.nanstd(fast) < 2.0


def _sweep_rows():
    frame = _random_walk_frame()
    settings = Settings(data=DataConfig(symbols=["BTCUSDT"], markets=["spot"]))
    candidate = default_candidates(families=("momentum",), lookbacks=(12,))[0]
    trials = power_sweep(
        frame, candidate, settings,
        alphas=[0.0, 4.0], horizon=1, draws=5,
        effective_trials=50, resamples=40, seed=2,
    )
    rows = summarize_power(trials, ("G1", "G3", "ALL_STAT", "ALL_ECON", "PROD_X_PBO"))
    return trials, rows


def test_achieved_sharpe_and_gate_power_rise_with_alpha() -> None:
    trials, rows = _sweep_rows()
    weak, strong = rows[0], rows[1]
    assert strong["gross_sharpe"] > weak["gross_sharpe"] + 1.0
    assert strong["rates"]["G1"] >= weak["rates"]["G1"]
    assert strong["rates"]["G1"] > 0.0, "a clairvoyant alpha=4 edge must clear the DSR gate"

    mda = minimum_detectable_alpha(rows, "G1", level=0.5)
    assert mda is not None and mda["alpha"] == 4.0
    assert minimum_detectable_alpha(rows, "G1", level=1.01) is None


def test_g2_fail_close_is_visible_not_hidden() -> None:
    """PBO is pool-relative; a standalone trial can never pass G2. The harness must show
    that (G2 never passes, always attributed as a blocking failure) while PROD_X_PBO
    ignores exactly and only G2 — otherwise the production-proxy composite is fiction."""
    trials, _ = _sweep_rows()
    assert all(not t.passes.get("G2", False) for t in trials)
    assert all("G2" in t.failed_blocking for t in trials)
    for t in trials:
        assert t.passes["PROD_X_PBO"] == (set(t.failed_blocking) <= {"G2"})


def test_summarize_and_binding_attribution_are_deterministic() -> None:
    """Binding-constraint reporting must be exact bookkeeping, not sampling: hand-built
    trials with known failures must yield exact failure shares and pass rates."""
    def _trial(alpha: float, failed: tuple[str, ...], g1: bool) -> PowerTrial:
        return PowerTrial(
            alpha=alpha, horizon=1, gross_sharpe=2.0, net_sharpe=1.0,
            break_even_cost_bps=10.0, actual_cost_bps=6.0, factor_turnover=0.01,
            avg_holding_period_bars=50.0, oos_trade_count=120,
            passes={"G1": g1, "ALL_STAT": g1, "ALL_ECON": False, "PROD_X_PBO": False},
            failed_blocking=failed,
        )

    trials = [
        _trial(0.1, ("G2", "G8"), g1=False),
        _trial(0.1, ("G2", "G8"), g1=True),
        _trial(0.1, ("G1", "G2"), g1=False),
        _trial(0.1, ("G2",), g1=True),
    ]
    (row,) = summarize_power(trials, ("G1", "ALL_STAT"))
    assert row["n"] == 4
    assert row["rates"]["G1"] == 0.5
    assert row["blocking_failure_share"]["G2"] == 1.0
    assert row["blocking_failure_share"]["G8"] == 0.5
    assert row["blocking_failure_share"]["G1"] == 0.25
