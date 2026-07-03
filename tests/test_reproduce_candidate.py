"""A1/I1: factor_mining-side guardrail for ``pipeline.reproduce_candidate``.

This is the producer half of the cross-repo seam the backend's reproduce_bridge
imports (``from factor_mining.pipeline import reproduce_candidate``). It asserts
the function actually re-runs a candidate and is deterministic — the plan's §7
reproduce guarantee — not just that the symbol exists (the backend seam contract
test covers existence).

Data-gated: skipped when no local parquet is present (e.g. base CI), since a
faithful reproduce needs the same market data the mining run used.
"""

from __future__ import annotations

import math

import pytest

from factor_mining.config import load_settings
from factor_mining.data.loader import data_extent
from factor_mining.models import CandidateStrategySpec
from factor_mining.pipeline import reproduce_candidate

_COMPARABLE = ("total_return", "sharpe", "max_drawdown")


def _spec() -> CandidateStrategySpec:
    return CandidateStrategySpec(
        candidate_id="repro-test",
        hypothesis_id="h",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        params={"signal_source": "factor_signal", "factor_family": "momentum", "factor_lookback": 24},
    )


def test_reproduce_candidate_is_deterministic() -> None:
    settings = load_settings()
    if not data_extent(settings, symbol="BTCUSDT", market="um_futures"):
        pytest.skip("no local BTCUSDT parquet for reproduce integration test")
    spec = _spec()
    first = reproduce_candidate(spec, settings)
    second = reproduce_candidate(spec, settings)
    for field in _COMPARABLE:
        value = getattr(first.metrics_primary, field)
        assert math.isfinite(value), f"{field} is not finite"
        # Plan §7: same spec + data → identical metrics within 1e-9.
        assert value == pytest.approx(getattr(second.metrics_primary, field), abs=1e-9)


def test_reproduce_candidate_data_end_ms_pins_extent() -> None:
    settings = load_settings()
    extent = data_extent(settings, symbol="BTCUSDT", market="um_futures")
    if not extent:
        pytest.skip("no local BTCUSDT parquet for reproduce integration test")
    spec = _spec()
    full = reproduce_candidate(spec, settings)
    # Pin to roughly the midpoint of the available extent → a different OOS window.
    midpoint = (extent["data_start_ms"] + extent["data_end_ms"]) // 2
    pinned = reproduce_candidate(spec, settings, data_end_ms=midpoint)
    # Truncating the data before feature/regime build must change the final-OOS
    # metrics — otherwise data_end_ms is silently ignored and "faithful" is a lie.
    assert pinned.metrics_primary.trade_count != full.metrics_primary.trade_count


def test_reproduce_candidate_rederives_trial_context_dsr() -> None:
    """Archived results carry effective_trials_at_eval; feeding those counts
    back must re-derive the deflated-Sharpe haircut (deterministic in counts
    and sample size) without perturbing the reproducible metrics_primary —
    otherwise an archived headline DSR can never be independently rechecked."""
    settings = load_settings()
    if not data_extent(settings, symbol="BTCUSDT", market="um_futures"):
        pytest.skip("no local BTCUSDT parquet for reproduce integration test")
    spec = _spec()
    base = reproduce_candidate(spec, settings)
    penalized = reproduce_candidate(
        spec,
        settings,
        trial_counts={"effective_trials_count": 64, "global_cumulative_trials_count": 64},
    )
    for field in _COMPARABLE:
        assert getattr(penalized.metrics_primary, field) == pytest.approx(
            getattr(base.metrics_primary, field), abs=1e-9
        )
    assert penalized.effective_trials_at_eval == 64
    assert penalized.deflated_sharpe < base.deflated_sharpe
