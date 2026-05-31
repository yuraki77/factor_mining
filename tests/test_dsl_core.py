"""Boundary tests for the phase-2 factor DSL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_mining.dsl import (
    canonicalize,
    extract_features,
    extract_lookbacks,
    max_depth,
    operator_count,
    parse,
    render,
    structural_fingerprint,
)
from factor_mining.dsl.complexity import free_constant_count
from factor_mining.dsl.evaluator import evaluate, supported_operators
from factor_mining.dsl.fingerprint_store import FingerprintStore


def _frame(n: int = 100) -> pd.DataFrame:
    index = pd.RangeIndex(n)
    close = np.linspace(100.0, 120.0, n)
    return pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.linspace(1000.0, 2000.0, n),
        "vwap": close + 0.1,
        "funding_rate": np.linspace(-0.001, 0.001, n),
        "open_interest": np.linspace(10_000.0, 11_000.0, n),
        "regime_state": np.where(np.arange(n) % 2 == 0, 1.0, 2.0),
    }, index=index)


def _features(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({"rsi_14": np.linspace(20.0, 80.0, n)}, index=pd.RangeIndex(n))


class TestParserCatalog:
    def test_requires_dollar_leaf_syntax(self) -> None:
        assert parse("$close") == {"type": "factor", "value": "$close"}
        with pytest.raises(ValueError, match="bare identifiers"):
            parse("close")

    def test_rejects_non_catalog_leaf_and_indicator_shortcut(self) -> None:
        with pytest.raises(ValueError, match="unknown leaf"):
            parse("$rsi_14")
        with pytest.raises(ValueError, match="bare identifiers"):
            parse("rsi_14")

    def test_accepts_phase_two_catalog_case_insensitively(self) -> None:
        ast = parse("rank(ts_corr($returns, delta($volume, 1) / $volume, 20))")
        assert ast["name"] == "RANK"
        assert ast["args"][0]["name"] == "TS_CORR"

    def test_rejects_explicitly_excluded_operators(self) -> None:
        for expr in ("TS_SUM($close, 5)", "SIGNEDPOWER($close, 2)", "DECAYLINEAR($close, 10)"):
            with pytest.raises(ValueError, match="unknown operator"):
                parse(expr)

    def test_window_arguments_are_discrete(self) -> None:
        parse("TS_MEAN($close, 20)")
        with pytest.raises(ValueError, match="allowed grid|constant"):
            parse("TS_MEAN($close, 23)")
        with pytest.raises(ValueError, match="window constant"):
            parse("TS_MEAN($close, 5.1)")

    def test_free_constants_snap_to_grid_and_cap_at_three(self) -> None:
        ast = parse("WHERE(GT($returns, 0.4998), $returns * 2, 0)")
        assert ast["args"][0]["args"][1] == {"type": "constant", "value": 0.5}
        assert ast["args"][1]["right"] == {"type": "constant", "value": 2.0}
        with pytest.raises(ValueError, match="too many free constants"):
            parse("$close + 0.01 + 0.05 + 0.1 + 0.2")

    def test_regime_category_does_not_count_as_free_constant(self) -> None:
        ast = parse("WHERE(EQ($regime_state, 2), NEG($returns), $returns)")
        assert free_constant_count(ast) == 0

    def test_unsupported_comparison_surface_is_rejected(self) -> None:
        for expr in ("$close >= $open", "$close <= $open", "$close != $open"):
            with pytest.raises(ValueError):
                parse(expr)


class TestCanonicalization:
    def test_commutative_and_symmetric_ops_share_fingerprint(self) -> None:
        assert structural_fingerprint(parse("$close + $open")) == structural_fingerprint(parse("$open + $close"))
        assert structural_fingerprint(parse("EQ($close, $open)")) == structural_fingerprint(parse("EQ($open, $close)"))
        assert structural_fingerprint(parse("TS_CORR($returns, $volume, 20)")) == structural_fingerprint(parse("TS_CORR($volume, $returns, 20)"))

    def test_subtraction_rewrites_to_negated_addition(self) -> None:
        assert structural_fingerprint(parse("$close - $open")) == structural_fingerprint(parse("$close + NEG($open)"))

    def test_does_not_fold_nan_unsafe_multiply_by_zero(self) -> None:
        canonical = canonicalize(parse("$close * 0"))
        assert canonical["type"] == "binary_op"
        assert canonical["op"] == "*"

    def test_where_not_condition_flips_branches(self) -> None:
        left = canonicalize(parse("WHERE(NOT(GT($close, $open)), $low, $high)"))
        right = canonicalize(parse("WHERE(GT($close, $open), $high, $low)"))
        assert left == right

    def test_cross_sectional_idempotence(self) -> None:
        assert canonicalize(parse("RANK(RANK($returns))")) == canonicalize(parse("RANK($returns)"))
        assert canonicalize(parse("ZSCORE(ZSCORE($returns))")) == canonicalize(parse("ZSCORE($returns)"))

    def test_roundtrip_render_parse_fingerprint(self) -> None:
        expr = "RANK(TS_CORR($returns, DELTA($volume, 1) / $volume, 20) * TS_MEAN(($close - $open) / $close, 5))"
        canonical = canonicalize(parse(expr))
        rendered = render(canonical)
        assert structural_fingerprint(parse(rendered)) == structural_fingerprint(canonical)


class TestComplexity:
    def test_extracts_features_lookbacks_and_free_constants(self) -> None:
        ast = parse("WHERE(GT(TS_STD($returns, 20), 0.5), RANK(TS_MEAN($returns, 5)), 0)")
        assert extract_features(ast) == {"$returns"}
        assert extract_lookbacks(ast) == {5, 20}
        assert free_constant_count(ast) == 1
        assert operator_count(ast)["WHERE"] == 1
        assert max_depth(ast) > 0


class TestEvaluator:
    def test_evaluates_raw_leaf_and_returns_leaf(self) -> None:
        frame = _frame(20)
        close = evaluate(parse("$close"), frame, _features(20))
        returns = evaluate(parse("$returns"), frame, _features(20))
        np.testing.assert_allclose(close.to_numpy(), frame["close"].to_numpy())
        assert np.isnan(returns.iloc[0])
        assert returns.iloc[1:].notna().all()

    def test_does_not_read_indicator_features_as_dsl_leaves(self) -> None:
        with pytest.raises(ValueError):
            parse("rsi_14")

    def test_math_invalid_values_remain_nan_not_zero(self) -> None:
        frame = _frame(3)
        frame["close"] = [-1.0, 0.0, 1.0]
        log_result = evaluate(parse("LOG($close)"), frame, _features(3))
        div_result = evaluate(parse("$close / 0"), frame, _features(3))
        assert log_result.isna().iloc[:2].all()
        assert log_result.iloc[2] == 0.0
        assert div_result.isna().all()

    def test_rolling_warmup_is_nan(self) -> None:
        result = evaluate(parse("TS_MEAN($close, 5)"), _frame(10), _features(10))
        assert result.iloc[:4].isna().all()
        assert result.iloc[4:].notna().all()

    def test_cross_sectional_ops_parse_but_need_panel_adapter(self) -> None:
        with pytest.raises(ValueError, match="panel adapter"):
            evaluate(parse("RANK($returns)"), _frame(10), _features(10))

    def test_supported_operator_catalog_matches_phase_two_names(self) -> None:
        ops = supported_operators()
        assert "TS_SUM" not in ops
        assert "TS_MEAN" in ops
        assert ops["RANK"]["cross_sectional"] is True


class TestFingerprintStore:
    def test_internal_batch_dedup_does_not_need_passed_sets(self) -> None:
        store = FingerprintStore()
        store.register("fp1", "c1")
        assert store.is_novel("fp1") is False
        assert store.candidate_ids_for("fp1") == {"c1"}

    def test_parent_exempts_archive_but_not_batch_duplicates(self) -> None:
        store = FingerprintStore()
        store.register_parent("parent_fp")
        assert store.is_novel("parent_fp", archive_fingerprints={"parent_fp"}) is True
        store.register("parent_fp", "child")
        assert store.is_novel("parent_fp", archive_fingerprints={"parent_fp"}) is False

    def test_clear_caches_resets_transient_sets(self) -> None:
        store = FingerprintStore()
        store.register("fp1", "c1")
        store.register_parent("fp_parent")
        store.clear_caches()
        assert store.is_novel("fp1") is True


def test_curated_equivalence_and_non_equivalence_pairs() -> None:
    equiv_pairs = [
        ("$close + $open", "$open + $close"),
        ("$close * $open", "$open * $close"),
        ("$close + 0", "$close"),
        ("0 + $close", "$close"),
        ("$close * 1", "$close"),
        ("1 * $close", "$close"),
        ("$close / 1", "$close"),
        ("$close - $open", "$close + NEG($open)"),
        ("NEG(NEG($returns))", "$returns"),
        ("NOT(NOT(GT($close, $open)))", "GT($close, $open)"),
        ("ABS(ABS($returns))", "ABS($returns)"),
        ("ABS(NEG($returns))", "ABS($returns)"),
        ("SIGN(SIGN($returns))", "SIGN($returns)"),
        ("EQ($close, $open)", "EQ($open, $close)"),
        ("AND(GT($close, $open), GT($volume, 1))", "AND(GT($volume, 1), GT($close, $open))"),
        ("OR(GT($close, $open), GT($volume, 1))", "OR(GT($volume, 1), GT($close, $open))"),
        ("TS_CORR($returns, $volume, 20)", "TS_CORR($volume, $returns, 20)"),
        ("LT($close, $open)", "GT($open, $close)"),
        ("$close < $open", "GT($open, $close)"),
        ("WHERE(GT($close, $open), $returns, $returns)", "$returns"),
        ("WHERE(NOT(GT($close, $open)), $low, $high)", "WHERE(GT($close, $open), $high, $low)"),
        ("RANK(RANK($returns))", "RANK($returns)"),
        ("ZSCORE(ZSCORE($returns))", "ZSCORE($returns)"),
    ]
    for leaf in ("$open", "$high", "$low", "$close", "$volume", "$vwap", "$returns", "$funding_rate", "$open_interest", "$regime_state"):
        equiv_pairs.append((f"{leaf} + 0", leaf))
        equiv_pairs.append((f"{leaf} * 1", leaf))
        equiv_pairs.append((f"NEG(NEG({leaf}))", leaf))

    non_equiv_pairs = [
        ("$close + $open", "$close - $open"),
        ("$close * 2", "$close + $close"),
        ("TS_MEAN($returns, 5)", "TS_MEAN($returns, 20)"),
        ("TS_STD($returns, 20)", "TS_ZSCORE($returns, 20)"),
        ("TS_CORR($returns, $volume, 20)", "TS_CORR($returns, $volume, 60)"),
        ("$returns / $volume", "$volume / $returns"),
        ("GT($close, $open)", "GT($open, $close)"),
        ("WHERE(GT($close, $open), $high, $low)", "WHERE(GT($close, $open), $low, $high)"),
        ("LOG($close)", "SQRT($close)"),
        ("ABS($returns)", "SIGN($returns)"),
    ]
    for window in (1, 3, 5, 10, 20, 60, 120, 250):
        other = 20 if window != 20 else 60
        non_equiv_pairs.append((f"DELAY($returns, {window})", f"DELAY($returns, {other})"))
        non_equiv_pairs.append((f"TS_MEAN($close, {window})", f"TS_STD($close, {window})"))
        non_equiv_pairs.append((f"TS_MAX($close, {window})", f"TS_MIN($close, {window})"))
    for leaf in ("$open", "$high", "$low", "$close", "$volume", "$vwap", "$returns", "$funding_rate", "$open_interest", "$regime_state"):
        non_equiv_pairs.append((leaf, f"NEG({leaf})"))
        non_equiv_pairs.append((f"{leaf} + 0.5", leaf))

    assert len(equiv_pairs) >= 50
    assert len(non_equiv_pairs) >= 50
    for left, right in equiv_pairs:
        assert structural_fingerprint(parse(left)) == structural_fingerprint(parse(right)), (left, right)
    for left, right in non_equiv_pairs:
        assert structural_fingerprint(parse(left)) != structural_fingerprint(parse(right)), (left, right)


def test_idempotence_samples_cover_catalog() -> None:
    expressions = [
        "$close",
        "$returns + 0",
        "$close - $open",
        "$high * $low",
        "$close / $volume",
        "DELTA($close, 1)",
        "DELAY($close, 5)",
        "TS_MEAN($returns, 20)",
        "TS_STD($returns, 20)",
        "TS_MAX($close, 60)",
        "TS_MIN($close, 60)",
        "TS_RANK($returns, 20)",
        "TS_CORR($returns, $volume, 20)",
        "TS_ZSCORE($returns, 20)",
        "TS_PCTCHANGE($close, 5)",
        "RANK($returns)",
        "ZSCORE($returns)",
        "ABS(NEG($returns))",
        "SIGN($returns)",
        "LOG($close)",
        "SQRT($close)",
        "WHERE(EQ($regime_state, 2), NEG($returns), $returns)",
        "AND(GT($close, $open), GT($volume, 1))",
        "OR(GT($close, $open), GT($volume, 1))",
        "NOT(GT($close, $open))",
    ]
    for i in range(10_000):
        expr = expressions[i % len(expressions)]
        c1 = canonicalize(parse(expr))
        c2 = canonicalize(c1)
        assert c1 == c2
