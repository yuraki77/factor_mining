"""Q7: the final out-of-sample window must always be the chronologically last
segment of the frame. A regime-aware chooser used to shift it earlier toward a
"less alien" regime mix, but picking the test window by its regime composition
leaks information — the OOS is then no longer the genuine future tail.
"""

from __future__ import annotations

import pandas as pd

from factor_mining.pipeline import _FINAL_OOS_FRACTION, _build_data_split_plan


def _final_indices(plan) -> list[int]:
    mask = plan.final_oos_mask
    return [int(i) for i in mask[mask].index]


def test_final_oos_is_chronological_last_segment_with_regimes() -> None:
    n = 1000
    frame = pd.DataFrame({"open_time": range(n)})
    # Regimes arranged so a regime-mix chooser would have preferred an earlier
    # window; chronological placement must ignore that.
    regimes = pd.Series(["bull"] * 800 + ["bear"] * 200)
    plan = _build_data_split_plan(frame, regimes=regimes)
    final_count = max(1, round(n * _FINAL_OOS_FRACTION))
    assert plan.final_oos_start_idx == n - final_count
    assert _final_indices(plan) == list(range(n - final_count, n))


def test_final_oos_matches_with_and_without_regimes() -> None:
    n = 500
    frame = pd.DataFrame({"open_time": range(n)})
    regimes = pd.Series(["bull", "bear"] * (n // 2))
    with_regimes = _build_data_split_plan(frame, regimes=regimes)
    without = _build_data_split_plan(frame, regimes=None)
    # OOS placement is identical: regimes no longer move the test window.
    assert with_regimes.final_oos_start_idx == without.final_oos_start_idx
    assert _final_indices(with_regimes) == _final_indices(without)
