"""The FAR harness must itself be trustworthy before its numbers mean anything:
a null signal must be rejected, a planted real edge must be accepted (a harness
that rejects everything is broken, not calibrated), and raising the trial count
must make the FAR-controlling gates strictly harder — the exact mechanism by
which lineage-deduped N (bias-audit finding A) could inflate the false-acceptance
rate.
"""

from __future__ import annotations

import numpy as np

from factor_mining.calibration import (
    calibrate,
    default_candidates,
    plant_signal,
    rotate_signal,
)
from factor_mining.config import DataConfig, Settings


def _random_walk_frame(n: int = 1500, seed: int = 7) -> "pd.DataFrame":
    import pandas as pd

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


def test_rotate_signal_is_a_length_preserving_rotation() -> None:
    rng = np.random.default_rng(0)
    sig = np.arange(1000, dtype=float)
    rot = rotate_signal(sig, rng, min_gap=100)
    assert rot.shape == sig.shape
    assert np.isclose(rot.sum(), sig.sum())  # rotation, not resample
    assert not np.array_equal(rot, sig)      # actually shifted


def test_plant_signal_is_bounded_and_aligned() -> None:
    frame = _random_walk_frame(400)
    rng = np.random.default_rng(1)
    planted = plant_signal(frame, rng, alpha=6.0, noise=0.0)
    assert planted.shape[0] == len(frame)
    assert np.all(np.abs(planted) <= 1.0)  # tanh-bounded


def _report():
    frame = _random_walk_frame()
    settings = Settings(data=DataConfig(symbols=["BTCUSDT"], markets=["spot"]))
    candidates = default_candidates(families=("momentum", "mean_reversion"), lookbacks=(12, 24))
    return calibrate(
        frame, candidates, settings,
        n_surrogates=12, power_draws=6, resamples=40,
        n_dedup=len(candidates), n_raw=len(candidates) * 40, seed=3,
    )


def test_larger_N_makes_the_far_gates_strictly_harder() -> None:
    """The finding-A mechanism, as a test: for the *same* null signals, raising the
    trial count can only raise the DSR expected-max penalty (G1) and the BH-FDR
    multiplicity (G3), so acceptances under raw-N never exceed those under deduped-N.
    If this monotonicity broke, the FAR-isolation arms would be meaningless."""
    report = _report()
    for gate in ("G1", "G3", "ALL_STAT"):
        assert report.null_raw.passes[gate] <= report.null_dedup.passes[gate], gate


def test_null_is_mostly_rejected_and_planted_edge_is_accepted() -> None:
    """Two-sided calibration: pure-noise (rotated) signals must not sail through the
    statistical gates, and a planted real edge must clear G1 more often than noise —
    otherwise a gate that simply rejects everything would masquerade as calibrated."""
    report = _report()
    # noise does not mostly pass the full statistical AND
    assert report.null_dedup.rate("ALL_STAT") < 0.5
    # planted clairvoyant edge clears the DSR gate strictly more than noise does
    assert report.power.rate("G1") > report.null_dedup.rate("G1")
    assert report.power.passes["G1"] > 0
