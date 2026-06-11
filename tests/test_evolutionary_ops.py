"""Tests for Phase 3 evolutionary selector, operations, and LLM mutation."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from factor_mining.config import EvolutionaryConfig, Settings
from factor_mining.dsl import parse, structural_fingerprint, LEAVES
from factor_mining.models import (
    CandidateStrategySpec,
    FactorEvidenceReport,
    GateCheckResult,
    MetricsBlock,
    NearMissAnalysis,
    ResearchGateResult,
)
from factor_mining.optimizers.evolutionary_operations import (
    crossover_dsl_composite,
    extract_freeze_point,
)
from factor_mining.optimizers.evolutionary_selector import (
    allocate_evolutionary_budget,
    select_evolutionary_parents,
)


# ── helpers ──────────────────────────────────────────────────────────


def _settings(**kwargs: object) -> Settings:
    s = Settings()
    if kwargs:
        s = s.model_copy(update={"evolutionary": EvolutionaryConfig(**kwargs)})  # type: ignore[dict-item]
    return s


def _candidate(cid: str, **params: object) -> CandidateStrategySpec:
    return CandidateStrategySpec(
        candidate_id=cid,
        hypothesis_id=f"h_{cid}",
        method_id="factor_scoring",
        hypothesis_family=params.pop("family", "momentum"),  # type: ignore[arg-type]
        symbol="BTCUSDT",
        params=dict(params),
    )


def _candidate_dsl(cid: str, expr: str, *, allow_cross_sectional: bool = False, **params: object) -> CandidateStrategySpec:
    ast = parse(expr, allow_cross_sectional=allow_cross_sectional)
    fp = structural_fingerprint(ast)
    return CandidateStrategySpec(
        candidate_id=cid,
        hypothesis_id=f"h_{cid}",
        method_id="factor_scoring",
        hypothesis_family=params.pop("family", "momentum"),  # type: ignore[arg-type]
        symbol="BTCUSDT",
        dsl_expression=expr,
        dsl_ast=ast,
        dsl_fingerprint=fp,
        dsl_version="0.1.0",
        params=dict(params),
    )


def _survivor_summary(cid: str, score: float = 3.0, sharpe: float = 1.0) -> dict:
    return {
        "candidate_id": cid,
        "research_score": score,
        "sharpe": sharpe,
        "gatecheck_passed": True,
        "params": {"complexity_score": 2, "optimizer_variant_key": ""},
    }


def _near_miss_dict(cid: str, reason: str = "cost_destroyed_edge", actionable: bool = True) -> dict:
    return {
        "candidate_id": cid,
        "primary_reason": reason,
        "actionable": actionable,
        "diagnostics": {"cost_margin_bps": -3.0, "sharpe": 0.5},
    }


def _context(**kwargs: object) -> dict:
    ctx: dict = {
        "research_survivors": [],
        "near_misses": [],
        "factors": [],
        "optimizer_outcomes": [],
    }
    ctx.update(kwargs)
    return ctx


def _market_frame(n: int = 40) -> pd.DataFrame:
    index = pd.RangeIndex(n)
    close = np.linspace(100.0, 120.0, n)
    return pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.linspace(1000.0, 1100.0, n),
        "vwap": close + 0.1,
    }, index=index)


# ── budget allocator tests ──────────────────────────────────────────


class TestBudgetAllocator:
    def test_returns_all_categories(self) -> None:
        ctx = _context(
            research_survivors=[_survivor_summary(f"c{i}") for i in range(20)],
            near_misses=[_near_miss_dict(f"c{i}") for i in range(10)],
        )
        budget = allocate_evolutionary_budget(ctx, total_budget=20)
        assert "exploit" in budget
        assert "variation" in budget
        assert "targeted_repair" in budget
        assert "seed" in budget
        assert sum(budget.values()) <= 20

    def test_exploit_gets_largest_share(self) -> None:
        ctx = _context(
            research_survivors=[_survivor_summary(f"c{i}") for i in range(20)],
            near_misses=[_near_miss_dict(f"c{i}") for i in range(10)],
        )
        budget = allocate_evolutionary_budget(ctx, total_budget=20)
        assert budget["exploit"] >= budget.get("variation", 0)

    def test_surplus_redistributes_to_exploit_when_variation_starved(self) -> None:
        ctx = _context(
            research_survivors=[_survivor_summary("c0")],
            near_misses=[],
        )
        budget = allocate_evolutionary_budget(ctx, total_budget=20)
        assert budget["exploit"] > 0

    def test_zero_budget_all_categories_zero(self) -> None:
        ctx = _context()
        budget = allocate_evolutionary_budget(ctx, total_budget=0)
        assert all(v == 0 for v in budget.values())

    def test_no_survivors_no_near_misses_returns_zeros(self) -> None:
        ctx = _context()
        budget = allocate_evolutionary_budget(ctx, total_budget=10)
        assert budget["exploit"] == 0
        assert budget["variation"] == 0
        assert budget["targeted_repair"] == 0


# ── parent selector tests ───────────────────────────────────────────


class TestParentSelector:
    def test_exploit_selects_top_survivors(self) -> None:
        ctx = _context(
            research_survivors=[
                _survivor_summary("c0", score=1.0),
                _survivor_summary("c1", score=5.0),
                _survivor_summary("c2", score=3.0),
            ],
        )
        selected = select_evolutionary_parents(ctx, operator="exploit", count=2)
        assert len(selected) == 2
        assert selected[0]["candidate_id"] == "c1"  # highest score

    def test_variation_selects_less_explored(self) -> None:
        ctx = _context(
            research_survivors=[
                {**_survivor_summary("c0"), "params": {"complexity_score": 5, "optimizer_variant_key": "a"}},
                {**_survivor_summary("c1"), "params": {"complexity_score": 1, "optimizer_variant_key": ""}},
            ],
        )
        selected = select_evolutionary_parents(ctx, operator="variation", count=1)
        # c1 has lower complexity + empty variant key → lower variation score → preferred
        assert selected[0]["candidate_id"] == "c1"

    def test_targeted_repair_returns_actionable_near_misses(self) -> None:
        ctx = _context(
            near_misses=[
                _near_miss_dict("c0", actionable=True),
                _near_miss_dict("c1", actionable=False),
                _near_miss_dict("c2", actionable=True),
            ],
        )
        selected = select_evolutionary_parents(ctx, operator="targeted_repair", count=5)
        assert len(selected) == 2  # only actionable

    def test_seed_returns_unique_families(self) -> None:
        ctx = _context(
            factors=[
                {"hypothesis_family": "momentum", "hypothesis_id": "h1"},
                {"hypothesis_family": "momentum", "hypothesis_id": "h2"},
                {"hypothesis_family": "mean_reversion", "hypothesis_id": "h3"},
            ],
        )
        selected = select_evolutionary_parents(ctx, operator="seed", count=10)
        families = {s["hypothesis_family"] for s in selected}
        assert families == {"momentum", "mean_reversion"}

    def test_unknown_operator_returns_empty(self) -> None:
        selected = select_evolutionary_parents(_context(), operator="nonexistent", count=5)
        assert selected == []

    def test_zero_count_returns_empty(self) -> None:
        selected = select_evolutionary_parents(
            _context(research_survivors=[_survivor_summary("c0")]),
            operator="exploit", count=0,
        )
        assert selected == []


# ── freeze point tests ──────────────────────────────────────────────


class TestFreezePoint:
    def test_extracts_hypothesis_metadata(self) -> None:
        parent = _candidate("c_parent", signal_source="feature")
        fp = extract_freeze_point(parent, freeze_depth=3)
        assert fp["parent_candidate_id"] == "c_parent"
        assert fp["hypothesis_family"] == "momentum"
        assert fp["freeze_depth"] == 3
        assert fp["params_prefix"]["signal_source"] == "feature"

    def test_includes_dsl_prefix_when_dsl_present(self) -> None:
        parent = _candidate_dsl("c_dsl", "$close + $open")
        fp = extract_freeze_point(parent, freeze_depth=1)
        assert fp["dsl_expression"] == "$close + $open"
        assert fp["dsl_fingerprint"] is not None
        assert fp["dsl_ast_prefix"] is not None

    def test_freeze_depth_zero_returns_none_prefix(self) -> None:
        parent = _candidate_dsl("c_dsl", "$close + $open")
        fp = extract_freeze_point(parent, freeze_depth=0)
        assert fp["dsl_ast_prefix"] is None

    def test_fixed_params_excludes_dynamic_keys(self) -> None:
        parent = _candidate(
            "c_p", signal_source="feature", direction=1,
            smooth_span=48, signal_threshold=0.25, position_buffer=0.20,
        )
        fp = extract_freeze_point(parent)
        prefix = fp["params_prefix"]
        assert "signal_source" in prefix
        assert "direction" in prefix
        assert "smooth_span" not in prefix  # dynamic
        assert "signal_threshold" not in prefix


# ── crossover tests ─────────────────────────────────────────────────


class TestCrossover:
    def test_no_dsl_parents_returns_empty(self) -> None:
        a = _candidate("ca")
        b = _candidate("cb")
        children = crossover_dsl_composite(a, b, _settings())
        assert children == []

    def test_dsl_crossover_creates_children(self) -> None:
        a = _candidate_dsl("ca", "$close - ts_mean($close, 20)")
        b = _candidate_dsl("cb", "$volume / ts_mean($volume, 20)", family="liquidity")
        children = crossover_dsl_composite(a, b, _settings())
        assert len(children) >= 1
        child = children[0]
        assert child.candidate_type == "optimizer"
        assert child.params["optimizer_proposal_kind"] == "crossover"
        assert child.dsl_fingerprint is not None
        assert child.params["generated_by"] == "crossover_dsl_composite"

    def test_crossover_children_have_unique_fingerprints(self) -> None:
        a = _candidate_dsl("ca", "$close - ts_mean($close, 20)")
        b = _candidate_dsl("cb", "$volume / ts_mean($volume, 20)", family="liquidity")
        children = crossover_dsl_composite(a, b, _settings())
        assert len(children) >= 1
        fps = {c.dsl_fingerprint for c in children}
        assert len(fps) == len(children)  # all unique

    def test_crossover_skips_same_hypothesis_family(self) -> None:
        a = _candidate_dsl("ca", "$close")
        b = _candidate_dsl("cb", "$volume")
        assert crossover_dsl_composite(a, b, _settings()) == []

    def test_crossover_respects_child_budget(self) -> None:
        a = _candidate_dsl("ca", "$close - TS_MEAN($close, 20)")
        b = _candidate_dsl("cb", "$volume / TS_MEAN($volume, 20)", family="liquidity")
        assert crossover_dsl_composite(a, b, _settings(), max_children=0) == []
        assert len(crossover_dsl_composite(a, b, _settings(), max_children=1)) == 1

    def test_cross_sectional_crossover_is_rejected_until_panel_adapter(self) -> None:
        a = _candidate_dsl("ca", "RANK($returns)", allow_cross_sectional=True)
        b = _candidate_dsl("cb", "$volume", family="liquidity")
        assert crossover_dsl_composite(a, b, _settings()) == []

    def test_dsl_crossover_signal_uses_dsl_expression(self) -> None:
        from factor_mining.pipeline import _build_signal_for

        frame = _market_frame(40)
        regimes = pd.Series("sideways", index=frame.index)
        a = _candidate_dsl("ca", "$close")
        b = _candidate_dsl("cb", "$volume", family="liquidity")
        child = crossover_dsl_composite(a, b, _settings())[0]
        signal = _build_signal_for(child, frame, pd.DataFrame(index=frame.index), {}, 0, regimes)
        assert float(np.abs(signal).sum()) > 0.0

    def test_output_correlation_gate_rejects_parent_duplicate_signal(self) -> None:
        from factor_mining.pipeline import _filter_evolutionary_output_correlation

        frame = _market_frame(40)
        regimes = pd.Series("sideways", index=frame.index)
        a = _candidate_dsl("ca", "$close")
        b = _candidate_dsl("cb", "$close * 2", family="liquidity")
        child = crossover_dsl_composite(a, b, _settings())[0]
        accepted, rejected, warned = _filter_evolutionary_output_correlation(
            [child],
            {"ca": a, "cb": b},
            frame,
            pd.DataFrame(index=frame.index),
            {},
            regimes,
            None,
            _settings(),
        )
        assert accepted == []
        assert rejected == 1
        assert warned == 0

    def test_output_correlation_helpers_reject_multidimensional_signals(self) -> None:
        from factor_mining.pipeline import _absolute_signal_correlation, _is_empty_signal

        panel_signal = np.zeros((40, 2))
        assert _is_empty_signal(panel_signal)
        assert _absolute_signal_correlation(panel_signal, np.arange(40)) is None


# ── LLM mutation tests (offline — no actual API call) ───────────────


class TestLLMMutation:
    def test_mutate_returns_none_when_llm_not_configured(self) -> None:
        from factor_mining.llm.mutation import mutate_with_mechanism

        parent = _candidate_dsl("cp", "$close - ts_mean($close, 20)")
        fp = extract_freeze_point(parent, freeze_depth=2)
        child = mutate_with_mechanism(parent, fp, _settings())
        # Without API key, should return None
        assert child is None

    def test_mutate_returns_none_for_invalid_llm_output(self) -> None:
        """When LLM returns something that isn't parseable DSL, return None."""
        from factor_mining.llm.mutation import mutate_with_mechanism, _call_llm_for_dsl

        with mock.patch("factor_mining.llm.mutation._call_llm_for_dsl", return_value="not a valid dsl!!!!"):
            parent = _candidate_dsl("cp", "$close")
            fp = extract_freeze_point(parent)
            child = mutate_with_mechanism(parent, fp, _settings())
            assert child is None

    def test_mutate_returns_child_for_valid_output(self) -> None:
        """When LLM returns valid DSL, a child candidate is created."""
        from factor_mining.llm.mutation import mutate_with_mechanism
        from factor_mining.trajectory_ledger import TrajectoryLedger

        valid_expr = "WHERE(GT($returns, 0), $returns * 2, $returns)"
        with mock.patch("factor_mining.llm.mutation._call_llm_for_dsl", return_value=valid_expr):
            parent = _candidate_dsl("cp", "$close")
            fp = extract_freeze_point(parent, freeze_depth=0)
            child = mutate_with_mechanism(parent, fp, _settings())
            assert child is not None
            assert child.candidate_type == "optimizer"
            assert child.dsl_expression == valid_expr
            assert child.dsl_fingerprint is not None
            assert child.params["generated_by"] == "llm_mutation_at_mechanism"
            assert TrajectoryLedger(None, _settings()).classify_operator(child, {}) == "MUTATION_AT_MECHANISM"

    def test_mutate_rejects_exact_parent_fingerprint(self) -> None:
        from factor_mining.llm.mutation import mutate_with_mechanism

        with mock.patch("factor_mining.llm.mutation._call_llm_for_dsl", return_value="$close"):
            parent = _candidate_dsl("cp", "$close")
            fp = extract_freeze_point(parent, freeze_depth=0)
            assert mutate_with_mechanism(parent, fp, _settings()) is None

    def test_mutate_enforces_freeze_prefix(self) -> None:
        from factor_mining.llm.mutation import mutate_with_mechanism

        with mock.patch("factor_mining.llm.mutation._call_llm_for_dsl", return_value="TS_STD($close, 20)"):
            parent = _candidate_dsl("cp", "TS_MEAN($close, 20)")
            fp = extract_freeze_point(parent, freeze_depth=1)
            assert mutate_with_mechanism(parent, fp, _settings()) is None

    def test_mutate_accepts_json_mode_output(self) -> None:
        from factor_mining.llm.mutation import _extract_dsl_expression

        assert _extract_dsl_expression('{"dsl_expression":"$returns"}') == "$returns"
        assert _extract_dsl_expression("$returns") == "$returns"

    def test_mutate_rejects_over_complex_expression(self) -> None:
        """Expression exceeding max_complexity should be rejected."""
        from factor_mining.llm.mutation import mutate_with_mechanism

        # A complex nested expression
        complex_expr = (
            "WHERE(GT($returns, 0), TS_ZSCORE($close, 20) * TS_RANK($volume, 20), "
            "WHERE(GT($close, TS_MEAN($close, 50)), TS_STD($close, 20), "
            "TS_CORR($returns, $volume, 20)))"
        )
        with mock.patch("factor_mining.llm.mutation._call_llm_for_dsl", return_value=complex_expr):
            parent = _candidate_dsl("cp", "$close")
            fp = extract_freeze_point(parent)
            child = mutate_with_mechanism(parent, fp, _settings(max_complexity=5))
            # Should be rejected because complexity > 5
            assert child is None

    def test_mutate_rejects_too_many_free_constants(self) -> None:
        """Expression with more than max_constants should be rejected."""
        from factor_mining.llm.mutation import mutate_with_mechanism

        expr_with_constants = "$close + 0.01 + 0.05 + 0.1 + 0.2"
        with mock.patch("factor_mining.llm.mutation._call_llm_for_dsl", return_value=expr_with_constants):
            parent = _candidate_dsl("cp", "$close")
            fp = extract_freeze_point(parent)
            child = mutate_with_mechanism(parent, fp, _settings(max_constants_per_expression=3))
            assert child is None

    def test_mutation_prompt_includes_available_leaves(self) -> None:
        from factor_mining.llm.mutation import _build_mechanism_mutation_prompt

        fp = {
            "parent_candidate_id": "cp",
            "hypothesis_family": "momentum",
            "economic_mechanism": "Trend continuation after volume confirmation.",
            "symbol": "BTCUSDT",
            "freeze_depth": 2,
            "params_prefix": {"signal_source": "feature"},
        }
        prompt = _build_mechanism_mutation_prompt(fp, _settings())
        assert "$close" in prompt
        assert "$returns" in prompt
        assert "momentum" in prompt
        assert "TS_MEAN" in prompt
        assert "RANK" not in prompt or "cross-sectional" in prompt.lower()


# ── config defaults test ────────────────────────────────────────────


class TestEvolutionaryConfig:
    def test_disabled_by_default(self) -> None:
        s = Settings()
        assert s.evolutionary.enabled is False

    def test_budget_defaults(self) -> None:
        s = Settings()
        assert s.evolutionary.budget_per_round == 20
        assert s.evolutionary.freeze_depth == 3
        assert s.evolutionary.max_complexity == 30

    def test_thresholds(self) -> None:
        s = Settings()
        assert 0 < s.evolutionary.output_correlation_warn_threshold < s.evolutionary.output_correlation_reject_threshold < 1.0

    def test_max_categorical_comparisons(self) -> None:
        s = Settings()
        assert s.evolutionary.max_categorical_comparisons == 4
