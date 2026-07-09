"""The cost meters feed G8, and G8 is the measured bottleneck of the whole factory —
these are the first direct unit tests of the cost arithmetic (the coverage gap that let
two definitional drifts go unnoticed: break-even computed from NET returns, and "actual
cost" as an idle-bar-diluted per-bar rate mean).

Anchor identity encoded below: a strategy whose gross PnL exactly equals the cost it
paid has break_even == its realized cost rate. Under the old net-based formula it read 0,
which made G8 (break_even > 2× actual) demand ~3× instead of its stated 2×.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_mining.backtest.engine import _break_even_cost_bps, _strategy_returns
from factor_mining.config import CostConfig, Settings


def _frame(opens: list[float], quote_volumes: list[float]) -> pd.DataFrame:
    n = len(opens)
    return pd.DataFrame({
        "open_time": [1_700_000_000_000 + i * 300_000 for i in range(n)],
        "open": opens,
        "high": [o * 1.001 for o in opens],
        "low": [o * 0.999 for o in opens],
        "close": opens,
        "volume": [1.0] * n,
        "quote_volume": quote_volumes,
    })


def _settings(**cost_overrides) -> Settings:
    return Settings().model_copy(update={"costs": CostConfig(**cost_overrides)})


def test_cost_charged_only_on_position_changes_and_weighted_rate_exact() -> None:
    """k=0 makes the one-way rate a constant 6bps: two unit trades must cost exactly
    2×6bps of notional, idle bars nothing, and the realized rate must be exactly 6."""
    frame = _frame([100.0] * 4, [1e9] * 4)
    open_returns = frame["open"].shift(-1) / frame["open"] - 1.0
    position = pd.Series([0.0, 1.0, 1.0, 0.0])
    settings = _settings(taker_bps=5.0, slippage_base_bps=1.0, slippage_k=0.0)

    net, realized_bps, _, total_cost = _strategy_returns(
        frame, open_returns, position, settings=settings, funding=None
    )
    assert total_cost == pytest.approx(2 * 6.0 / 10_000.0)
    assert realized_bps == pytest.approx(6.0)
    assert float(net.sum()) == pytest.approx(-2 * 6.0 / 10_000.0)  # flat prices: pure cost drag
    assert float(net.iloc[2]) == 0.0, "holding an unchanged position costs nothing"


def test_break_even_anchor_identity_gross_equals_cost_paid() -> None:
    """THE meter fix: price the frame so gross PnL exactly equals the cost paid
    (net = 0). Gross break-even must equal the realized cost rate (6bps); the old
    net-based formula returned 0 here and G8 punished the strategy for its own costs."""
    # position earns open_returns[1] = 0.0012 on one bar; round trip pays 2×6bps = 0.0012
    frame = _frame([100.0, 100.0, 100.12, 100.12], [1e9] * 4)
    open_returns = frame["open"].shift(-1) / frame["open"] - 1.0
    position = pd.Series([0.0, 1.0, 1.0, 0.0])
    settings = _settings(taker_bps=5.0, slippage_base_bps=1.0, slippage_k=0.0)

    net, realized_bps, _, total_cost = _strategy_returns(
        frame, open_returns, position, settings=settings, funding=None
    )
    assert float(net.sum()) == pytest.approx(0.0, abs=1e-12)

    break_even = _break_even_cost_bps(net, position, total_cost_return=total_cost)
    assert break_even == pytest.approx(realized_bps)
    # regression pin on the old semantics: net-based break-even would be ~0
    assert _break_even_cost_bps(net, position, total_cost_return=0.0) == pytest.approx(0.0, abs=1e-9)


def test_realized_cost_is_turnover_weighted_not_idle_bar_mean() -> None:
    """gamma=1, k=25: trade bars pay 18.5 and 12.25 bps (participation term), idle bars
    pay nothing. Weighted realized = 15.375; the old per-bar rate mean (10.6875, diluted
    by idle bars at the 6bps base rate) understated concentrated trading costs."""
    frame = _frame([100.0] * 4, [1e6, 20_000.0, 1e6, 40_000.0])
    open_returns = frame["open"].shift(-1) / frame["open"] - 1.0
    position = pd.Series([0.0, 1.0, 1.0, 0.0])
    settings = _settings(taker_bps=5.0, slippage_base_bps=1.0, slippage_k=25.0, slippage_gamma=1.0)

    _, realized_bps, _, _ = _strategy_returns(frame, open_returns, position, settings=settings, funding=None)
    assert realized_bps == pytest.approx(15.375)
    assert not np.isclose(realized_bps, 10.6875), "must not be the idle-bar-diluted mean"


def test_zero_turnover_falls_back_to_first_trade_rate() -> None:
    frame = _frame([100.0] * 4, [1e9] * 4)
    open_returns = frame["open"].shift(-1) / frame["open"] - 1.0
    position = pd.Series([0.0, 0.0, 0.0, 0.0])
    settings = _settings(taker_bps=5.0, slippage_base_bps=1.0, slippage_k=0.0)

    net, realized_bps, _, total_cost = _strategy_returns(
        frame, open_returns, position, settings=settings, funding=None
    )
    assert total_cost == 0.0
    assert realized_bps == pytest.approx(6.0), "rate a first trade would pay, not 0"
    assert _break_even_cost_bps(net, position, total_cost_return=total_cost) == 0.0
