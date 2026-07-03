"""The volatility factor_signal must stay off pandas' per-window Python
callback path.

WHY: .rolling(...).apply(lambda x: x.quantile(...), raw=False) constructs a
Series and calls two quantiles per window — ~238x slower than the Cython
rolling quantiles with bit-identical output. On a full 666k-bar run this
stalled signal building for ~50 minutes (GIL-bound, so the two symbol-group
threads serialized and total CPU looked idle), and the 2026-07-03 CLI run
was killed by a watchdog as "stuck" while sitting in exactly this code
(run cli_20260703_150751_05a995b7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_mining.mining import factor_signal


def _frame(n: int = 3_000) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.001, n))
    return pd.DataFrame({"close": close, "volume": np.full(n, 10.0)})


def test_volatility_signal_matches_old_per_window_formula() -> None:
    frame = _frame()
    lookback = 24
    signal = factor_signal(frame, family="volatility", lookback=lookback)

    close = pd.Series(frame["close"].to_numpy(dtype=float))
    vol = close.pct_change().rolling(lookback).std()
    vol_median = vol.rolling(lookback * 4, min_periods=lookback).median()
    vol_iqr = vol.rolling(lookback * 4, min_periods=lookback).apply(
        lambda x: x.quantile(0.75) - x.quantile(0.25) if len(x) > 1 else 1.0,
        raw=False,
    ).replace(0, np.nan)
    z = ((vol - vol_median) / vol_iqr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    expected = np.tanh(z / 2.0)

    assert np.allclose(signal.to_numpy(), expected.to_numpy(), atol=1e-12)
    assert float(signal.abs().max()) > 0.0  # the fixture actually exercises the branch


def test_volatility_signal_never_enters_rolling_apply(monkeypatch) -> None:
    def _forbidden(self, *args, **kwargs):
        raise AssertionError(
            "Rolling.apply is a per-window Python callback — the volatility "
            "signal hot path must use Cython rolling quantiles instead"
        )

    monkeypatch.setattr(pd.core.window.rolling.Rolling, "apply", _forbidden)

    signal = factor_signal(_frame(500), family="volatility", lookback=12)

    assert len(signal) == 500
