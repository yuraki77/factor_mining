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

from factor_mining.backtest.engine import _break_even_cost_bps, _strategy_returns, run_backtest
from factor_mining.config import CostConfig, Settings, apply_trade_overrides, costs_for_market
from factor_mining.models import CandidateStrategySpec


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


# ── WS1: per-market costs ───────────────────────────────────────────────────

def _market_settings() -> Settings:
    return Settings().model_copy(update={
        "costs": CostConfig(
            taker_bps=5.0, slippage_base_bps=1.0, slippage_k=0.0,
            per_market={"spot": {"taker_bps": 10.0}, "um_futures": {"taker_bps": 5.0}},
        )
    })


def test_costs_for_market_layers_only_listed_fields() -> None:
    settings = _market_settings()
    spot = costs_for_market(settings, "spot")
    assert spot.taker_bps == 10.0            # overridden
    assert spot.slippage_base_bps == 1.0     # inherited from base
    assert costs_for_market(settings, "um_futures").taker_bps == 5.0
    # an unlisted market falls back to the base object unchanged
    assert costs_for_market(settings, "unknown") is settings.costs


def test_engine_prices_spot_higher_than_futures() -> None:
    """Same signal, same data — only the market differs — must cost more on spot
    (10bps taker) than um_futures (5bps), so break-even/realized diverge by market."""
    frame = _frame([100.0, 100.0, 100.5, 100.5], [1e9] * 4)
    settings = _market_settings()
    sig = pd.Series([1.0, 1.0, 1.0, 0.0])

    def _cand(market: str) -> CandidateStrategySpec:
        return CandidateStrategySpec(
            candidate_id=f"c_{market}", hypothesis_id="h", method_id="factor_scoring",
            hypothesis_family="momentum", symbol="BTCUSDT", market=market, interval="5m",
            params={"position_buffer": 0.0},
        )

    spot = run_backtest(frame, sig, _cand("spot"), settings)
    fut = run_backtest(frame, sig, _cand("um_futures"), settings)
    assert spot.actual_cost_bps == pytest.approx(11.0)   # 10 taker + 1 base slippage
    assert fut.actual_cost_bps == pytest.approx(6.0)      # 5 + 1
    assert spot.actual_cost_bps > fut.actual_cost_bps


def test_global_override_composes_and_wins_over_per_market() -> None:
    """A run-scoped scenario knob must reach every market: overriding taker globally
    propagates into per-market entries (wins on collision), while a non-overlapping
    slippage override still layers on both."""
    settings = _market_settings()
    scenario = apply_trade_overrides(settings, taker_bps=3.0, slippage_base_bps=2.0)
    spot = costs_for_market(scenario, "spot")
    assert spot.taker_bps == 3.0             # global scenario wins over per-market 10
    assert spot.slippage_base_bps == 2.0     # non-overlapping override applies too
    assert costs_for_market(scenario, "um_futures").taker_bps == 3.0


# ── WS2: hysteresis band ────────────────────────────────────────────────────

def test_hysteresis_band_cuts_trades_and_keeps_direction() -> None:
    """A signal that oscillates through zero must trade less under a band, because the
    band holds through conviction and releases only on decay — but the trades it does
    take must keep the signal's sign (the band gates magnitude, not direction)."""
    from factor_mining.backtest.engine import _apply_hysteresis_band

    # strong entries punctuating runs of sub-threshold wobble: raw tracks every wobble
    # (a trade each), the band holds flat through them (enter only on the 0.5/-0.55 bars)
    raw = pd.Series([0.5, 0.05, 0.09, 0.03, 0.07, 0.5, 0.06, -0.55, 0.04, 0.08])
    banded = _apply_hysteresis_band(raw, entry_band=0.3, exit_band=0.1)

    def _trades(s: pd.Series) -> int:
        return int((s.diff().fillna(s).abs() > 1e-9).sum())

    assert _trades(banded) < _trades(raw)          # 6 < 10
    # every nonzero banded value keeps the raw sign at that bar (band gates magnitude)
    for r, b in zip(raw, banded):
        assert b == 0.0 or (b > 0) == (r > 0)


def test_flat_band_is_identity() -> None:
    """entry=exit=0 must be a pass-through so the whole feature is off by default and
    every existing backtest is byte-identical."""
    from factor_mining.backtest.engine import _apply_hysteresis_band

    raw = pd.Series([0.0, 0.5, -0.3, 0.1, 0.0, -0.7])
    out = _apply_hysteresis_band(raw, entry_band=0.0, exit_band=0.0)
    assert out is raw  # untouched object, not just equal


def test_band_reduces_backtest_turnover_end_to_end() -> None:
    """Through the full engine: the same signal with an active band must trade less —
    lower turnover — which is the whole point (fewer, larger trades that can clear
    costs). Turnover is the continuous target; a slow oscillator spends real stretches
    below the exit band where the band holds flat instead of tracking every wobble."""
    n = 600
    frame = _frame([100.0 + 5.0 * np.sin(i / 40.0) for i in range(n)], [1e9] * n)
    sig = pd.Series(0.6 * np.sin(np.arange(n) / 20.0))  # slow: long sub-exit stretches
    settings = _market_settings()

    def _cand(bands: dict) -> CandidateStrategySpec:
        return CandidateStrategySpec(
            candidate_id="c", hypothesis_id="h", method_id="factor_scoring",
            hypothesis_family="momentum", symbol="BTCUSDT", market="um_futures", interval="5m",
            params={"position_buffer": 0.1, **bands},
        )

    plain = run_backtest(frame, sig, _cand({}), settings)
    banded = run_backtest(frame, sig, _cand({"entry_band": 0.4, "exit_band": 0.15}), settings)
    assert banded.factor_turnover < plain.factor_turnover
    assert banded.oos_trade_count <= plain.oos_trade_count


def test_combo_turnover_controls_emit_bands_only_when_churny() -> None:
    """The optimizer's combo path must add band params when (and only when) the combo's
    components are turnover-heavy — otherwise a fine low-turnover combo gets no band."""
    from factor_mining.optimizers.traditional_optimizer import _combo_turnover_controls
    from factor_mining.models import BacktestResult, MetricsBlock

    def _res(cid: str, turnover: float) -> BacktestResult:
        return BacktestResult(
            experiment_id=f"e{cid}", candidate_id=cid, hypothesis_family="momentum",
            method_id="factor_scoring", symbol="BTCUSDT", market="um_futures", interval="5m",
            metrics_primary=MetricsBlock(sharpe=1.0), metrics_gross=MetricsBlock(sharpe=1.2),
            factor_turnover=turnover,
        )

    components = [{"candidate_id": "a"}]
    churny = _combo_turnover_controls({}, components, {"a": _res("a", 0.9)})
    calm = _combo_turnover_controls({}, components, {"a": _res("a", 0.01)})

    assert churny["entry_band"] >= 0.30 and 0.0 < churny["exit_band"] <= churny["entry_band"]
    assert "entry_band" not in calm and "exit_band" not in calm
