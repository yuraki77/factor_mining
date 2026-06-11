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
