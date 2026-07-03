"""Q16: funding rate must never be backfilled into the warm-up window.

Bars before the first funding event have no known rate; filling them with the
first *future* rate (the old ``.bfill()``) is lookahead bias. They must be 0.0.
"""

from __future__ import annotations

import pandas as pd

from factor_mining.data.loader import merge_funding_to_frame


def test_funding_warmup_is_zero_not_backfilled() -> None:
    frame = pd.DataFrame({"open_time": [0, 1, 2, 3, 4, 5]})
    # First funding event is at open_time 3.
    funding = pd.DataFrame({"calc_time": [3], "last_funding_rate": [0.01]})
    aligned = merge_funding_to_frame(frame, funding)
    # Bars before the first event stay 0.0 (no backfilled future rate) ...
    assert list(aligned.iloc[:3]) == [0.0, 0.0, 0.0]
    # ... and forward-fill applies from the first event onward.
    assert list(aligned.iloc[3:]) == [0.01, 0.01, 0.01]


def _funding_frame(open_times: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"open_time": open_times})


def test_funding_settles_on_next_bar_when_settlement_bar_is_missing() -> None:
    """WHY: the engine matched calc_time == open_time exactly, so a funding
    settlement falling inside a data gap (or off the bar grid) silently
    dropped the cash flow — costs vanished exactly where data quality was
    worst. The flow must land on the first available bar at-or-after the
    settlement instead."""
    from factor_mining.backtest.engine import _apply_funding

    # 5-minutely grid with the 300..600 region missing (gap).
    frame = _funding_frame([0, 300, 900, 1200])
    position = pd.Series([1.0, 1.0, 0.5, 1.0], index=frame.index)
    funding = pd.DataFrame({"calc_time": [600], "last_funding_rate": [0.01]})

    impact = _apply_funding(position, frame, funding)

    # Settlement at 600 falls in the gap → lands on the 900 bar with that bar's position.
    assert impact.tolist() == [0.0, 0.0, -0.5 * 0.01, 0.0]


def test_multiple_gap_funding_events_accumulate_on_reopen_bar() -> None:
    from factor_mining.backtest.engine import _apply_funding

    frame = _funding_frame([0, 300, 2100])
    position = pd.Series([1.0, 1.0, 1.0], index=frame.index)
    # Two 8h-style settlements both inside the long gap.
    funding = pd.DataFrame({"calc_time": [600, 1500], "last_funding_rate": [0.01, 0.02]})

    impact = _apply_funding(position, frame, funding)

    assert impact.tolist() == [0.0, 0.0, -(0.01 + 0.02)]


def test_funding_outside_window_is_not_charged_and_aligned_match_unchanged() -> None:
    from factor_mining.backtest.engine import _apply_funding

    frame = _funding_frame([300, 600, 900])
    position = pd.Series([1.0, -1.0, 1.0], index=frame.index)
    funding = pd.DataFrame(
        {"calc_time": [0, 600, 1200], "last_funding_rate": [0.05, 0.01, 0.05]}
    )

    impact = _apply_funding(position, frame, funding)

    # Pre-window (0) and post-window (1200) events are outside the evaluation;
    # the aligned event at 600 charges exactly as before: -position * rate.
    assert impact.tolist() == [0.0, -(-1.0) * 0.01, 0.0]
