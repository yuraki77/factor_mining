"""Q6: the high_vol regime label must be point-in-time (no lookahead).

The defining property of a non-lookahead labeler: truncating the series to a
prefix must not change the labels of the bars in that prefix — bar t's label
never depends on bars after t. The previous full-sample ``vol_60d.rank(pct=True)``
ranked each bar against the whole series (future included) and fails this; the
trailing ``expanding().rank`` passes it exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_mining.config import RegimeConfig
from factor_mining.stats.regime import label_btc_regime


def _close_with_vol_shift(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rets = np.concatenate(
        [rng.normal(0.0, 0.001, n // 2), rng.normal(0.0, 0.01, n - n // 2)]
    )
    return pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(rets))})


def test_high_vol_label_is_prefix_invariant() -> None:
    config = RegimeConfig()
    n = 10_000
    frame = _close_with_vol_shift(n)
    full = label_btc_regime(frame, config)
    # Not vacuous: the trailing rank still flags some high_vol bars.
    assert (full == "high_vol").any()
    k = 7_000
    prefix = label_btc_regime(frame.iloc[:k].reset_index(drop=True), config)
    # Labels for the prefix region are identical whether or not the future exists.
    assert (prefix.to_numpy() == full.iloc[:k].to_numpy()).all()
