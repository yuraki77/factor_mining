"""Tests for explicit indicator candidate expansion and signal dispatch."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_mining.factors.engineering import generate_features, INDICATOR_META
from factor_mining.mining import (
    build_indicator_candidates,
    default_hypotheses,
    factor_signal,
    normalize_family,
    _direction_sign,
)
from factor_mining.models import CandidateStrategySpec, HypothesisSpec


# ── fixtures ────────────────────────────────────────────────────────

def _make_frame(n: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    close = 50_000 * np.exp(np.cumsum(np.random.randn(n) * 0.001))
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.0005),
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": np.random.lognormal(10, 1, n),
    })


def _make_microstructure_frame(n: int = 2000) -> pd.DataFrame:
    frame = _make_frame(n)
    quote_volume = frame["volume"] * frame["close"]
    buy_share = pd.Series(0.5 + 0.25 * np.sin(np.arange(n) / 17.0)).clip(0.05, 0.95)
    frame["quote_volume"] = quote_volume
    frame["trade_count"] = (100 + 20 * np.cos(np.arange(n) / 11.0)).astype(int)
    frame["taker_buy_volume"] = frame["volume"] * buy_share
    frame["taker_buy_quote_volume"] = quote_volume * buy_share
    return frame


def _make_hypotheses() -> list[HypothesisSpec]:
    return [
        HypothesisSpec(
            hypothesis_id="h_test_momentum",
            hypothesis_family="momentum",
            economic_mechanism="test",
            testable_prediction="test",
            null_hypothesis="test",
            expected_ic_range=(0.005, 0.03),
            expected_decay_halflife_bars=24,
        ),
        HypothesisSpec(
            hypothesis_id="h_test_mr",
            hypothesis_family="mean_reversion",
            economic_mechanism="test",
            testable_prediction="test",
            null_hypothesis="test",
            expected_ic_range=(0.005, 0.025),
            expected_decay_halflife_bars=48,
        ),
    ]


# ── direction conversion ───────────────────────────────────────────

def test_direction_sign_positive():
    assert _direction_sign("positive") == 1


def test_direction_sign_negative():
    assert _direction_sign("negative") == -1


def test_direction_sign_negative_when_high():
    assert _direction_sign("negative_when_high") == -1


def test_direction_sign_neutral_returns_none():
    assert _direction_sign("neutral") is None


# ── family normalization ────────────────────────────────────────────

def test_normalize_family_standard():
    assert normalize_family("momentum") == "momentum"
    assert normalize_family("mean_reversion") == "mean_reversion"


def test_normalize_family_llm_names():
    assert normalize_family("Time-Series Momentum") == "momentum"
    assert normalize_family("Volatility Feedback") == "volatility"
    assert normalize_family("Basis Carry") == "funding_basis"
    assert normalize_family("Volume-Price Divergence") == "volume_confirmation"
    assert normalize_family("Basis Arbitrage") == "funding_basis"
    assert normalize_family("Basis") is None
    assert normalize_family("Funding Rate Momentum") is None
    assert normalize_family("Basis Volatility") is None


def test_normalize_family_unknown():
    assert normalize_family("exotic_quant_alpha") is None


# ── candidate expansion ────────────────────────────────────────────

def test_build_indicator_candidates_produces_explicit_params():
    frame = _make_frame()
    _, feature_meta = generate_features(frame)
    hypotheses = _make_hypotheses()

    candidates = build_indicator_candidates(
        hypotheses,
        symbols=["BTCUSDT"],
        feature_meta=feature_meta,
    )
    assert len(candidates) > 0

    # Every candidate must have signal_source
    for c in candidates:
        assert "signal_source" in c.params, f"{c.candidate_id} missing signal_source"
        source = c.params["signal_source"]
        assert source in ("feature", "factor_signal"), f"Unknown source: {source}"

        if source == "feature":
            assert "indicator_name" in c.params
            assert "direction" in c.params
            assert c.params["direction"] in (1, -1)
            assert c.params["indicator_name"] in feature_meta
            assert c.params["search_variant"] in {"base", "inverse", "low_turnover"}
        elif source == "factor_signal":
            assert "factor_family" in c.params
            assert "factor_lookback" in c.params
            assert c.params["search_variant"] in {"base", "inverse", "low_turnover"}


def test_generate_features_uses_existing_kline_microstructure_columns():
    frame = _make_microstructure_frame()

    features_df, feature_meta = generate_features(frame)

    expected = {
        "taker_buy_pressure",
        "taker_quote_pressure",
        "order_flow_imbalance",
        "order_flow_imbalance_z_288",
        "aggressive_buy_volume_z_288",
        "quote_volume_z_288",
        "quote_volume_chg_12",
        "trade_density_z_288",
    }
    assert expected <= set(features_df.columns)
    assert all(feature_meta[name]["family"] == "volume_confirmation" for name in expected)
    assert features_df["order_flow_imbalance"].between(-1.0, 1.0).all()


def test_build_indicator_candidates_includes_inverse_and_low_turnover_variants():
    frame = _make_frame()
    _, feature_meta = generate_features(frame)

    candidates = build_indicator_candidates(
        [_make_hypotheses()[0]],
        symbols=["BTCUSDT"],
        feature_meta=feature_meta,
        max_indicators_per_family=1,
    )

    variants = {c.params["search_variant"] for c in candidates}
    assert {"base", "inverse", "low_turnover"} <= variants
    inverse = [c for c in candidates if c.params["search_variant"] == "inverse"]
    assert inverse
    assert all(c.params["direction"] in (1, -1) for c in inverse)
    low_turnover = [c for c in candidates if c.params["search_variant"] == "low_turnover"]
    assert low_turnover
    assert all(c.params["signal_threshold"] > 0 for c in low_turnover)
    assert all(c.params["position_buffer"] > 0.05 for c in low_turnover)


def test_funding_basis_hypotheses_can_use_supplemental_features():
    feature_meta = {
        "premium_index_z_288": {"family": "funding_basis", "direction": "negative_when_high", "source": "binance_supplemental"}
    }
    hypothesis = HypothesisSpec(
        hypothesis_id="h_funding_test",
        hypothesis_family="funding_basis",
        economic_mechanism="Rich perp premium can indicate crowded longs.",
        testable_prediction="High premium predicts weaker forward perp returns.",
        null_hypothesis="Premium has zero predictive IC.",
        expected_ic_range=(0.005, 0.02),
        expected_decay_halflife_bars=24,
    )

    candidates = build_indicator_candidates([hypothesis], symbols=["BTCUSDT"], feature_meta=feature_meta)

    feature_candidates = [c for c in candidates if c.params.get("signal_source") == "feature"]
    assert feature_candidates
    assert {c.params["indicator_name"] for c in feature_candidates} == {"premium_index_z_288"}
    assert {c.params["direction"] for c in feature_candidates} == {-1, 1}


def test_close_position_base_candidate_is_mean_reversion_direction():
    hypothesis = HypothesisSpec(
        hypothesis_id="h_close_position_test",
        hypothesis_family="mean_reversion",
        economic_mechanism="Closes near candle extremes can exhaust short-horizon flow.",
        testable_prediction="High close position predicts lower next-bar returns.",
        null_hypothesis="Close position has zero predictive IC.",
        expected_ic_range=(0.003, 0.02),
        expected_decay_halflife_bars=8,
    )
    candidates = build_indicator_candidates(
        [hypothesis],
        symbols=["BTCUSDT"],
        feature_meta={"close_position": INDICATOR_META["close_position"]},
    )

    by_variant = {c.params["search_variant"]: c.params["direction"] for c in candidates if c.params.get("signal_source") == "feature"}

    assert by_variant["base"] == -1
    assert by_variant["low_turnover"] == -1
    assert by_variant["inverse"] == 1


def test_funding_basis_factor_signal_requires_nonzero_funding_rate():
    frame = _make_frame(120)

    with pytest.raises(ValueError, match="funding_basis factor_signal requires non-zero funding_rate"):
        factor_signal(frame, family="funding_basis", lookback=12, funding_rate=None)

    with pytest.raises(ValueError, match="funding_basis factor_signal requires non-zero funding_rate"):
        factor_signal(frame, family="funding_basis", lookback=12, funding_rate=pd.Series(0.0, index=frame.index))


def test_default_hypotheses_are_twenty_and_mappable():
    hypotheses = default_hypotheses()

    assert len(hypotheses) == 20
    assert len({hyp.hypothesis_id for hyp in hypotheses}) == 20
    mappable = [hyp for hyp in hypotheses if normalize_family(hyp.hypothesis_family) is not None]
    assert len(mappable) >= 19


def test_momentum_candidates_use_trend_following_indicators():
    """Momentum hypothesis should map to trend_following indicator family."""
    frame = _make_frame()
    _, feature_meta = generate_features(frame)

    candidates = build_indicator_candidates(
        [_make_hypotheses()[0]],  # momentum only
        symbols=["BTCUSDT"],
        feature_meta=feature_meta,
    )
    feature_candidates = [c for c in candidates if c.params.get("signal_source") == "feature"]
    assert len(feature_candidates) > 0

    for c in feature_candidates:
        indicator_family = c.params["indicator_family"]
        assert indicator_family == "trend_following", (
            f"Momentum hypothesis produced indicator from '{indicator_family}' family"
        )


def test_mean_reversion_candidates_negative_when_high_direction():
    """RSI/stoch/cci are negative_when_high → direction should be -1."""
    frame = _make_frame()
    _, feature_meta = generate_features(frame)

    candidates = build_indicator_candidates(
        [_make_hypotheses()[1]],  # mean_reversion only
        symbols=["BTCUSDT"],
        feature_meta=feature_meta,
    )
    feature_candidates = [c for c in candidates if c.params.get("signal_source") == "feature"]

    rsi_candidates = [
        c for c in feature_candidates
        if c.params.get("indicator_name", "").startswith("rsi")
    ]
    assert len(rsi_candidates) > 0
    for c in rsi_candidates:
        expected_direction = 1 if c.params["search_variant"] == "inverse" else -1
        assert c.params["direction"] == expected_direction, (
            f"RSI {c.params['search_variant']} should have direction={expected_direction}, "
            f"got {c.params['direction']}"
        )


def test_neutral_indicators_excluded():
    """Neutral-direction indicators (ATR, bb_width, etc.) should not appear."""
    frame = _make_frame()
    _, feature_meta = generate_features(frame)

    candidates = build_indicator_candidates(
        _make_hypotheses(),
        symbols=["BTCUSDT"],
        feature_meta=feature_meta,
    )
    feature_candidates = [c for c in candidates if c.params.get("signal_source") == "feature"]
    indicator_names = {c.params["indicator_name"] for c in feature_candidates}

    # These are all neutral-direction volatility indicators
    for neutral_ind in ["atr_14", "atr_20", "bb_width_20", "bb_width_50", "hist_vol_10"]:
        assert neutral_ind not in indicator_names, f"Neutral indicator {neutral_ind} should be excluded"


def test_candidates_cover_multiple_features():
    """Generated candidates should use many distinct feature columns, not just one."""
    frame = _make_frame()
    _, feature_meta = generate_features(frame)

    candidates = build_indicator_candidates(
        _make_hypotheses(),
        symbols=["BTCUSDT"],
        feature_meta=feature_meta,
    )
    feature_candidates = [c for c in candidates if c.params.get("signal_source") == "feature"]
    unique_indicators = {c.params["indicator_name"] for c in feature_candidates}
    assert len(unique_indicators) >= 10, (
        f"Expected >=10 unique indicators, got {len(unique_indicators)}: {unique_indicators}"
    )


def test_max_indicators_per_family_cap():
    frame = _make_frame()
    _, feature_meta = generate_features(frame)

    candidates = build_indicator_candidates(
        _make_hypotheses(),
        symbols=["BTCUSDT"],
        feature_meta=feature_meta,
        max_indicators_per_family=3,
    )
    # momentum feature candidates cap unique indicators; variants intentionally multiply candidates.
    mom_features = [
        c for c in candidates
        if c.hypothesis_family == "momentum" and c.params.get("signal_source") == "feature"
    ]
    assert len({c.params["indicator_name"] for c in mom_features}) <= 3


# ── signal dispatch ─────────────────────────────────────────────────

def test_feature_signal_source_reads_correct_column():
    """signal_source='feature' should use the named indicator column."""
    frame = _make_frame()
    features_df, feature_meta = generate_features(frame)

    # Import the signal builder (need pipeline internals)
    from factor_mining.pipeline import _build_signal_for

    candidate = CandidateStrategySpec(
        candidate_id="c_test",
        hypothesis_id="h_test",
        method_id="factor_scoring",
        hypothesis_family="mean_reversion",
        symbol="BTCUSDT",
        params={
            "signal_source": "feature",
            "indicator_name": "rsi_14",
            "direction": -1,
            "transform": "tanh_zscore",
            "zscore_window": 288,
            "tanh_scale": 2.0,
        },
    )
    regimes = pd.Series("unknown", index=frame.index)
    signal = _build_signal_for(candidate, frame, features_df, feature_meta, 0, regimes)

    assert signal.shape == (len(frame),)
    assert np.all(np.isfinite(signal))
    # Should have continuous values, not just {-1, 0, 1}
    assert len(set(signal)) > 10


def test_missing_indicator_fails_loud():
    """Missing indicator_name should raise ValueError, not silently fallback."""
    frame = _make_frame()
    features_df, feature_meta = generate_features(frame)
    from factor_mining.pipeline import _build_signal_for

    candidate = CandidateStrategySpec(
        candidate_id="c_test_missing",
        hypothesis_id="h_test",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={
            "signal_source": "feature",
            "indicator_name": "nonexistent_indicator_999",
            "direction": 1,
            "transform": "tanh_zscore",
        },
    )
    regimes = pd.Series("unknown", index=frame.index)
    with pytest.raises(ValueError, match="nonexistent_indicator_999"):
        _build_signal_for(candidate, frame, features_df, feature_meta, 0, regimes)


def test_negative_when_high_produces_inverted_signal():
    """direction=-1 should invert the signal relative to direction=+1."""
    frame = _make_frame()
    features_df, feature_meta = generate_features(frame)
    from factor_mining.pipeline import _build_signal_for

    regimes = pd.Series("unknown", index=frame.index)

    base_params = {
        "signal_source": "feature",
        "indicator_name": "rsi_14",
        "transform": "tanh_zscore",
        "zscore_window": 288,
        "tanh_scale": 2.0,
    }

    c_pos = CandidateStrategySpec(
        candidate_id="c_pos", hypothesis_id="h", method_id="factor_scoring",
        hypothesis_family="mean_reversion", symbol="BTCUSDT",
        params={**base_params, "direction": 1},
    )
    c_neg = CandidateStrategySpec(
        candidate_id="c_neg", hypothesis_id="h", method_id="factor_scoring",
        hypothesis_family="mean_reversion", symbol="BTCUSDT",
        params={**base_params, "direction": -1},
    )

    sig_pos = _build_signal_for(c_pos, frame, features_df, feature_meta, 0, regimes)
    sig_neg = _build_signal_for(c_neg, frame, features_df, feature_meta, 0, regimes)

    # After regime modulation (which is symmetric), the signs should be opposite
    # where the raw signal is non-zero
    nonzero = np.abs(sig_pos) > 1e-6
    if nonzero.any():
        np.testing.assert_allclose(sig_pos[nonzero], -sig_neg[nonzero], atol=1e-10)


def test_low_turnover_controls_zero_weak_signal():
    frame = _make_frame()
    features_df, feature_meta = generate_features(frame)
    from factor_mining.pipeline import _build_signal_for

    candidate = CandidateStrategySpec(
        candidate_id="c_low_turnover",
        hypothesis_id="h",
        method_id="factor_scoring",
        hypothesis_family="mean_reversion",
        symbol="BTCUSDT",
        params={
            "signal_source": "feature",
            "indicator_name": "rsi_14",
            "direction": -1,
            "transform": "tanh_zscore",
            "zscore_window": 288,
            "tanh_scale": 2.0,
            "signal_threshold": 2.0,
            "smooth_span": 24,
        },
    )
    regimes = pd.Series("unknown", index=frame.index)

    signal = _build_signal_for(candidate, frame, features_df, feature_meta, 0, regimes)

    assert np.all(signal == 0.0)
