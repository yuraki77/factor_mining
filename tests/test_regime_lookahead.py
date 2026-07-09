"""Regime look-ahead pins (2026-07-09 audit).

Two regime systems feed the pipeline and each gets a causality pin here:

1. The HMM ``forward_regimes`` that gate signals (``_fit_regime_model``): prefix-fit
   model, FILTERED probabilities (never the smoother/Viterbi on live bars),
   transition-power forecast, prefix masked "unknown", then shift(1). The detector-level
   invariance is already pinned in test_regime_hmm.py; this pins the REAL end-to-end
   path including the mask and lag — if anyone swaps ``predict_proba_filtered`` for
   ``predict_proba`` (the smoother) or drops the shift, a future-data perturbation
   changes past labels and this fails.

2. ``label_btc_regime`` (engine.py:611) that buckets returns for the regime-conditional
   metrics (G6 concentration, regime IC, the regime_mixing near-miss diagnostics). Its
   Q6 fix replaced a full-sample ``.rank(pct=True)`` — which leaked future vol into
   every "high_vol" label — with a trailing expanding rank, but that fix never got a
   test. These diagnostics drove real decisions (the regime-conditioning brief), so the
   label at bar t must depend only on bars ≤ t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_mining.config import RegimeConfig
from factor_mining.pipeline import _fit_regime_model
from factor_mining.stats.regime import label_btc_regime


def _regime_frame(n: int = 900) -> pd.DataFrame:
    idx = np.arange(n)
    returns = (
        0.0002 * np.sin(idx / 17.0)
        + 0.0008 * np.where((idx // 120) % 2 == 0, 1.0, -1.0)
        + 0.0004 * np.sin(idx / 5.0)
    )
    close = 100.0 * np.exp(np.cumsum(returns))
    volume = 1_000.0 + 50.0 * np.sin(idx / 13.0)
    return pd.DataFrame({
        "close": close,
        "high": close * 1.002,
        "low": close * 0.998,
        "volume": volume,
    })


def _shock_future(frame: pd.DataFrame, start: int) -> pd.DataFrame:
    """A violent future regime change: if any future information reaches past labels,
    this is loud enough to move them."""
    perturbed = frame.copy()
    future = perturbed.index >= start
    n_future = int(future.sum())
    perturbed.loc[future, "close"] = perturbed.loc[future, "close"].to_numpy() * np.linspace(1.0, 4.0, n_future)
    perturbed.loc[future, "high"] = perturbed.loc[future, "close"] * 1.002
    perturbed.loc[future, "low"] = perturbed.loc[future, "close"] * 0.998
    perturbed.loc[future, "volume"] = perturbed.loc[future, "volume"] * 10.0
    return perturbed


def test_fit_regime_model_labels_are_causal_end_to_end() -> None:
    frame = _regime_frame(900)
    perturbed = _shock_future(frame, start=650)
    logs: list[str] = []

    base = _fit_regime_model(frame, tail=None, log_fn=logs.append)
    shocked = _fit_regime_model(perturbed, tail=None, log_fn=logs.append)

    # fit prefix (n//3 = 300 rows) must be masked: unavailable for OOS decisions
    assert (base.iloc[:301] == "unknown").all()  # +1 for the shift(1)
    # labels strictly before the shock (with margin for the shift) must be identical
    pd.testing.assert_series_equal(base.iloc[:640], shocked.iloc[:640])
    # sanity: the shock genuinely changes labels somewhere after it lands,
    # otherwise this test would pass vacuously against a constant series
    assert (base.iloc[660:] != shocked.iloc[660:]).any()


def test_label_btc_regime_is_causal_under_future_vol_shock() -> None:
    n = 4200
    rng = np.random.default_rng(9)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n))))
    frame = pd.DataFrame({"close": close})
    config = RegimeConfig()

    shocked = frame.copy()
    # violent future vol explosion: alternating ±5% bars in the last quarter
    tail_idx = np.arange(int(n * 0.75), n)
    shocked.loc[tail_idx, "close"] = shocked.loc[tail_idx, "close"].to_numpy() * np.where(
        (tail_idx % 2) == 0, 1.05, 0.95
    )

    base = label_btc_regime(frame, config)
    after = label_btc_regime(shocked, config)

    cut = int(n * 0.75)
    # the pre-Q6 full-sample rank would re-rank EVERY bar's vol against the future
    # explosion, flipping historical "high_vol" labels to calmer ones — the trailing
    # expanding rank must leave every pre-shock label untouched
    pd.testing.assert_series_equal(base.iloc[:cut], after.iloc[:cut])
    # sanity against vacuous passing: the shock genuinely moves labels — but only
    # at/after the cut, never before (probed: ~560 post-cut flips on this seed)
    assert (after.iloc[cut:] != base.iloc[cut:]).any()
