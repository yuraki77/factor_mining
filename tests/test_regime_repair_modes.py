"""Regime repair shapes beyond the hard filter (G7×G8 joint-frontier fix).

WHY: of 1,820 regime_mixing near-misses, positive-cost-margin (358) and
enough-OOS-trades (749) populations were completely disjoint — the hard filter
trades one gate against the other by construction. entry_only and soft must
preserve trades while cutting noise-regime exposure; signed must preserve trades
for edges that INVERT by regime. All of it must stay causal, and none of it may
change default behavior (hard stays byte-identical).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_mining.models import (
    BacktestResult,
    CandidateStrategySpec,
    FactorEvidenceReport,
    MetricsBlock,
)
from factor_mining.pipeline import (
    _apply_candidate_filters,
    _apply_regime_filter_mode,
    _apply_regime_signs,
    _pre_gate_repairs_for_parent,
    _regime_signed_repair_params,
    _repair_signature_params,
)


def test_hard_mode_is_default_and_byte_identical() -> None:
    sig = pd.Series([0.5, -0.4, 0.3, -0.2])
    mask = np.array([True, False, True, False])
    out = _apply_regime_filter_mode(sig, mask, {})
    pd.testing.assert_series_equal(out, sig.where(mask, 0.0))


def test_soft_mode_scales_outside_regime_by_exact_weight() -> None:
    sig = pd.Series([0.5, -0.4, 0.3])
    mask = np.array([True, False, False])
    out = _apply_regime_filter_mode(sig, mask, {"regime_filter_mode": "soft", "regime_soft_weight": 0.25})
    assert out.tolist() == [0.5, -0.1, 0.075]


def test_entry_only_holds_through_flip_blocks_entries_and_sign_flips() -> None:
    #            enter  hold   hold   flat   blocked  allowed-enter
    sig = pd.Series([0.5, 0.5, 0.4, 0.0, 0.4, -0.4])
    mask = np.array([True, False, False, False, False, True])
    out = _apply_regime_filter_mode(sig, mask, {"regime_filter_mode": "entry_only"})
    assert out.tolist() == [0.5, 0.5, 0.4, 0.0, 0.0, -0.4]

    # a sign flip outside allowed regimes is an ENTRY and must be blocked
    sig2 = pd.Series([0.5, -0.5])
    mask2 = np.array([True, False])
    out2 = _apply_regime_filter_mode(sig2, mask2, {"regime_filter_mode": "entry_only"})
    assert out2.tolist() == [0.5, 0.0]


def test_entry_only_is_causal_truncation_invariant() -> None:
    """Stateful forward pass must never look ahead: the output on a prefix is
    identical whether or not the future exists."""
    rng = np.random.default_rng(3)
    sig = pd.Series(np.round(rng.normal(0, 0.5, 400), 3))
    mask = rng.random(400) > 0.6
    full = _apply_regime_filter_mode(sig, mask, {"regime_filter_mode": "entry_only"})
    trunc = _apply_regime_filter_mode(sig.iloc[:250], mask[:250], {"regime_filter_mode": "entry_only"})
    pd.testing.assert_series_equal(full.iloc[:250], trunc)


def test_signed_flips_negative_zeroes_zero_and_defaults_plus_one() -> None:
    sig = pd.Series([0.5, 0.5, 0.5, 0.5])
    regimes = pd.Series(["bull", "bear", "high_vol", "sideways"])
    out = _apply_regime_signs(sig, {"bear": -1, "high_vol": 0}, regimes, None)
    assert out.tolist() == [0.5, -0.5, 0.0, 0.5]  # absent labels (bull/sideways) → +1


def test_signal_controls_compose_signs_then_filter_mode() -> None:
    """Integration through the real entry point: signs apply first, then the
    soft filter scales what is outside the kept regimes."""
    sig = pd.Series([0.8, 0.8, 0.8])
    regimes = pd.Series(["bull", "bear", "sideways"])
    out = _apply_candidate_filters(
        sig,
        {
            "regime_filter": ["bull"],
            "regime_filter_mode": "soft",
            "regime_soft_weight": 0.5,
            "regime_signs": {"bear": -1},
        },
        regimes,
        None,
    )
    assert out.tolist() == [0.8, -0.4, 0.4]  # bear: flipped then halved; sideways: halved


def test_trade_preservation_ordering_soft_entryonly_hard() -> None:
    """The point of the whole change: on a regime-alternating frame, soft keeps
    every trading bar, entry_only keeps strictly more than hard, and hard keeps
    the fewest — while both new modes still cut outside-regime exposure vs no
    filter."""
    rng = np.random.default_rng(7)
    n = 600
    sig = pd.Series(np.sign(rng.normal(size=n)) * rng.uniform(0.2, 1.0, n))
    mask = (np.arange(n) // 50) % 2 == 0  # alternating 50-bar regime blocks

    hard = _apply_regime_filter_mode(sig, mask, {})
    entry = _apply_regime_filter_mode(sig, mask, {"regime_filter_mode": "entry_only"})
    soft = _apply_regime_filter_mode(sig, mask, {"regime_filter_mode": "soft", "regime_soft_weight": 0.25})

    def nonzero(s: pd.Series) -> int:
        return int((s.abs() > 1e-12).sum())

    assert nonzero(soft) == nonzero(sig)
    assert nonzero(hard) < nonzero(entry) <= nonzero(soft)
    # exposure outside the allowed regime: hard 0 < soft < unfiltered
    outside = ~mask
    assert float(hard[outside].abs().sum()) == 0.0
    assert 0.0 < float(soft[outside].abs().sum()) < float(sig[outside].abs().sum())


def _evidence(regime_ic: dict[str, dict[str, float]]) -> FactorEvidenceReport:
    return FactorEvidenceReport(
        experiment_id="e1", candidate_id="c1", hypothesis_family="momentum",
        method_id="factor_scoring", symbol="BTCUSDT", market="um_futures", interval="5m",
        ic_by_horizon={"12": 0.001},  # keeps the regime threshold at the 0.015 floor
        rankic_by_horizon={"12": 0.001},
        quantile_spread_by_horizon={},
        regime_conditional_ic=regime_ic,
    )


def test_signed_params_require_genuine_sign_inversion() -> None:
    mixed = _regime_signed_repair_params(_evidence({"bull": {"12": 0.05}, "bear": {"12": -0.05}}))
    assert mixed == {"regime_signs": {"bull": 1, "bear": -1}}

    same_sign = _regime_signed_repair_params(_evidence({"bull": {"12": 0.05}, "sideways": {"12": 0.03}}))
    assert same_sign == {}

    below_floor = _regime_signed_repair_params(_evidence({"bull": {"12": 0.05}, "bear": {"12": -0.001}}))
    assert below_floor == {}  # bear IC under the 0.015 floor → no inversion evidence

    unknown_skipped = _regime_signed_repair_params(_evidence({"bull": {"12": 0.05}, "unknown": {"12": -0.5}}))
    assert unknown_skipped == {}


def test_pre_gate_repairs_spawn_four_regime_shapes_with_distinct_signatures() -> None:
    parent = CandidateStrategySpec(
        candidate_id="parent-1", hypothesis_id="h", method_id="factor_scoring",
        hypothesis_family="momentum", symbol="BTCUSDT", market="um_futures", interval="5m",
    )
    result = BacktestResult(
        experiment_id="e1", candidate_id="parent-1", hypothesis_family="momentum",
        method_id="factor_scoring", symbol="BTCUSDT", market="um_futures", interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.5), metrics_gross=MetricsBlock(sharpe=0.8),
    )
    evidence = _evidence({"bull": {"12": 0.05}, "bear": {"12": -0.05}})

    repairs = _pre_gate_repairs_for_parent(parent, result, evidence)
    variants = {str(r.params.get("search_variant")) for r in repairs}
    assert {
        "pre_gate_regime_filter",
        "pre_gate_regime_entry_only",
        "pre_gate_regime_soft",
        "pre_gate_regime_signed",
    } <= variants

    regime_repairs = [r for r in repairs if "regime" in str(r.params.get("search_variant"))]
    # lineage inheritance: repairs must not mint new DSR trials
    assert all(r.lineage_id == "parent-1" for r in regime_repairs)
    # distinct dedupe signatures — the four shapes must not collapse into one
    signatures = {str(sorted(_repair_signature_params(r.params).items())) for r in regime_repairs}
    assert len(signatures) == len(regime_repairs)
    signed = next(r for r in regime_repairs if r.params["search_variant"] == "pre_gate_regime_signed")
    assert signed.params["regime_signs"] == {"bull": 1, "bear": -1}
