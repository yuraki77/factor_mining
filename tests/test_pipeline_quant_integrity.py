import numpy as np
import pandas as pd

from factor_mining.config import BootstrapConfig, CPCVConfig, DataConfig, PermutationTestConfig, Settings
from factor_mining.models import BacktestResult, CandidateStrategySpec, FactorEvidenceReport, GateCheckItem, GateCheckResult, MetricsBlock
from factor_mining.optimizers.traditional_optimizer import apply_exit_adjustments, apply_optimization_result, build_optimization_context, optimize_traditionally
from factor_mining.pipeline import (
    _apply_batch_pbo,
    _apply_merge_pool_trial_penalty,
    _build_data_split_plan,
    _build_pre_gate_repair_candidates,
    _build_signal_for,
    _build_tasks,
    _check_mining_boundaries,
    _filter_unfunded_factor_signal_candidates,
    _cscv_splits,
    _filter_candidates_by_mining_boundaries,
    _gatecheck_diagnostics,
    _run_backtests_parallel,
    _sample_frame_blocks,
    _select_repair_merge_pool,
    _checkpoint_fingerprint,
    _save_stage_checkpoint,
    _load_stage_checkpoint,
)
from factor_mining.registry import get_method
from factor_mining.storage import MetadataStore


def _frame(n: int = 160) -> pd.DataFrame:
    opens = [100.0]
    for idx in range(1, n):
        opens.append(opens[-1] * (1.0 + 0.002 * np.sin(idx / 5.0)))
    return pd.DataFrame(
        {
            "open_time": [1_700_000_000_000 + idx * 300_000 for idx in range(n)],
            "open": opens,
            "high": [price * 1.01 for price in opens],
            "low": [price * 0.99 for price in opens],
            "close": [price * 1.001 for price in opens],
            "volume": [100.0] * n,
            "quote_volume": [1_000_000.0] * n,
        }
    )


def _result(experiment_id: str, candidate_id: str, *, pbo: float, sharpe: float) -> BacktestResult:
    return BacktestResult(
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=sharpe),
        metrics_gross=MetricsBlock(sharpe=sharpe),
        pbo=pbo,
    )


def test_block_sampling_is_deterministic_chronological_and_sized() -> None:
    frame = _frame(500)

    first = _sample_frame_blocks(frame, sample_bars=120, interval_ms=300_000, seed=7)
    second = _sample_frame_blocks(frame, sample_bars=120, interval_ms=300_000, seed=7)

    assert len(first) == 120
    assert first["open_time"].is_monotonic_increasing
    assert first["open_time"].is_unique
    assert first["open_time"].tolist() == second["open_time"].tolist()


def test_stage_checkpoint_round_trips_and_rejects_mismatch(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    frame = _frame(40)
    run_args = {"tail": None, "sample_bars": 40, "sample_mode": "block", "seed": 42}
    fingerprint = _checkpoint_fingerprint(
        settings,
        run_args=run_args,
        symbol="BTCUSDT",
        market="um_futures",
        frame=frame,
    )

    _save_stage_checkpoint(
        store,
        "run-a",
        round_num=1,
        symbol="BTCUSDT",
        market="um_futures",
        stage="discovery_backtests",
        fingerprint=fingerprint,
        payload={"items": [{"candidate_id": "c1"}]},
    )

    loaded = _load_stage_checkpoint(
        store,
        "run-a",
        round_num=1,
        symbol="BTCUSDT",
        market="um_futures",
        stage="discovery_backtests",
        fingerprint=fingerprint,
    )
    assert loaded == {"items": [{"candidate_id": "c1"}]}

    mismatched = dict(fingerprint)
    mismatched["row_count"] = 999
    try:
        _load_stage_checkpoint(
            store,
            "run-a",
            round_num=1,
            symbol="BTCUSDT",
            market="um_futures",
            stage="discovery_backtests",
            fingerprint=mismatched,
        )
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("checkpoint mismatch should fail loud")


def test_composite_signal_is_weighted_component_signal() -> None:
    frame = _frame()
    regimes = pd.Series(["unknown"] * len(frame), index=frame.index)
    component_a = CandidateStrategySpec(
        candidate_id="c_a",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
    )
    component_b = CandidateStrategySpec(
        candidate_id="c_b",
        hypothesis_id="h2",
        method_id="factor_scoring",
        hypothesis_family="mean_reversion",
        symbol="BTCUSDT",
    )
    composite = CandidateStrategySpec(
        candidate_id="c_combo",
        hypothesis_id="h_combo",
        method_id="bayesian_optimization",
        hypothesis_family="composite",
        symbol="BTCUSDT",
        params={
            "weights": [0.75, 0.25],
            "components": [component_a.model_dump(mode="json"), component_b.model_dump(mode="json")],
        },
    )

    signal_a = _build_signal_for(component_a, frame, pd.DataFrame(index=frame.index), {}, 0, regimes)
    signal_b = _build_signal_for(component_b, frame, pd.DataFrame(index=frame.index), {}, 1, regimes)
    signal_combo = _build_signal_for(composite, frame, pd.DataFrame(index=frame.index), {}, 0, regimes)

    assert np.allclose(signal_combo, np.clip(0.75 * signal_a + 0.25 * signal_b, -3.0, 3.0))


def test_optimization_result_creates_schedulable_next_candidates() -> None:
    base_a = CandidateStrategySpec(
        candidate_id="c_a",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    base_b = CandidateStrategySpec(
        candidate_id="c_b",
        hypothesis_id="h2",
        method_id="factor_scoring",
        hypothesis_family="mean_reversion",
        symbol="BTCUSDT",
        market="um_futures",
    )
    optimization = {
        "combinations": [{
            "factor_ids": ["c_a", "c_b"],
            "weights": [0.6, 0.4],
            "horizon": "5m",
        }],
        "next_hypotheses": [{
            "family": "Basis Arbitrage",
            "mechanism": "basis dislocations converge",
            "expected_ic": 0.03,
        }],
    }

    new_candidates, summary = apply_optimization_result(optimization, [base_a, base_b], [])

    assert summary["combinations_created"] == 1
    assert summary["hypotheses_suggested"] == 1
    assert any(c.hypothesis_family == "composite" for c in new_candidates)
    assert all(get_method(c.method_id).v1_schedulable for c in new_candidates)
    assert next(c for c in new_candidates if c.hypothesis_family == "composite").candidate_type == "composite"
    next_hypothesis_candidates = [
        c for c in new_candidates
        if c.params.get("generated_by") == "traditional_next_hypothesis"
    ]
    assert next_hypothesis_candidates
    assert {c.candidate_type for c in next_hypothesis_candidates} == {"optimizer"}
    assert {c.params["factor_family"] for c in next_hypothesis_candidates} == {"funding_basis"}


def test_optimization_result_accepts_compact_schema_variants() -> None:
    base_a = CandidateStrategySpec(
        candidate_id="c_a",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    base_b = CandidateStrategySpec(
        candidate_id="c_b",
        hypothesis_id="h2",
        method_id="factor_scoring",
        hypothesis_family="mean_reversion",
        symbol="BTCUSDT",
        market="um_futures",
    )
    optimization = {
        "combinations": [{
            "components": ["c_a", "c_b"],
            "weights": [0.55, 0.45],
            "turnover_control": {"smooth_span": 48, "signal_threshold": 0.12, "position_buffer": 0.08},
        }],
        "adjustments": [
            {"candidate_id": "c_a", "suggested_param": "smooth_span: 48", "rationale": "compact shorthand"},
            {"candidate_id": "c_b", "suggested": {"regime_filter": ["sideways", "unknown"]}, "rationale": "bulk repair"},
        ],
    }

    new_candidates, summary = apply_optimization_result(optimization, [base_a, base_b], [])

    combo = next(c for c in new_candidates if c.hypothesis_family == "composite")
    repaired = next(c for c in new_candidates if c.params.get("generated_by") == "optimizer_repair")
    adjusted = next(c for c in new_candidates if c.params.get("parent_id") == "c_a")
    assert summary["combinations_created"] == 1
    assert summary["adjustments_applied"] == 2
    assert combo.params["factor_ids"] == ["c_a", "c_b"]
    assert combo.params["smooth_span"] == 48
    assert combo.params["signal_threshold"] == 0.12
    assert combo.params["position_buffer"] == 0.08
    assert adjusted.params["smooth_span"] == 48
    assert adjusted.candidate_type == "optimizer"
    assert adjusted.parent_candidate_id == "c_a"
    assert repaired.candidate_type == "repair"
    assert repaired.parent_candidate_id == "c_b"
    assert repaired.params["search_variant"] == "repair_optimizer_repair"
    assert repaired.params["regime_filter"] == ["sideways", "unknown"]


def test_status_adjustment_does_not_mutate_source_candidate() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_failed",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )

    new_candidates, summary = apply_optimization_result(
        {
            "adjustments": [{
                "candidate_id": "c_failed",
                "param": "status",
                "suggested": "paused",
            }]
        },
        [candidate],
        [],
    )

    assert new_candidates == []
    assert "status" not in candidate.params
    assert summary["status_adjustments_recorded"] == 1
    assert summary["adjustments_applied"] == 0


def test_build_tasks_skips_unknown_legacy_family_without_fallback_features(capsys) -> None:
    frame = _frame(40)
    candidate = CandidateStrategySpec(
        candidate_id="c_unknown_family",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="unknown_llm_family",
        symbol="BTCUSDT",
        market="um_futures",
    )

    tasks = _build_tasks(
        [candidate],
        frame,
        pd.DataFrame(index=frame.index),
        {},
        pd.Series("unknown", index=frame.index),
    )

    assert tasks == []
    assert "SKIP signal" in capsys.readouterr().out


def test_exit_adjustments_can_disable_exit_rules() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_exit_base",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
        params={
            "stop_loss_pct": -0.05,
            "max_hold_bars": 100,
            "tp_tiers": [[0.02, 0.50]],
            "trailing_stop_pct": 0.02,
        },
    )

    candidates = apply_exit_adjustments(
        {
            "exit_adjustments": [{
                "candidate_id": "c_exit_base",
                "stop_loss_pct": 0,
                "max_hold_bars": 0,
                "tp_tiers": [],
                "trailing_stop_pct": 0,
                "trailing_after_first_tp": False,
            }]
        },
        [candidate],
        Settings(),
    )

    assert len(candidates) == 2
    adjusted = candidates[-1]
    assert adjusted.params["stop_loss_pct"] == 0.0
    assert adjusted.params["max_hold_bars"] == 0
    assert adjusted.params["tp_tiers"] == []
    assert adjusted.params["trailing_stop_pct"] == 0.0
    assert adjusted.params["trailing_after_first_tp"] is False


def test_exit_adjustments_skip_effective_duplicate_defaults() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_default_exit",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )

    candidates = apply_exit_adjustments(
        {"exit_adjustments": [{"candidate_id": "c_default_exit", "stop_loss_pct": -0.05}]},
        [candidate],
        Settings(),
    )

    assert candidates == [candidate]


def test_exit_adjustments_dedupe_within_parent_not_across_candidates() -> None:
    first = CandidateStrategySpec(
        candidate_id="c_first",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    second = first.model_copy(update={"candidate_id": "c_second"})

    candidates = apply_exit_adjustments(
        {
            "exit_adjustments": [
                {"candidate_id": "c_first", "stop_loss_pct": -0.03},
                {"candidate_id": "c_second", "stop_loss_pct": -0.03},
            ]
        },
        [first, second],
        Settings(),
    )

    generated = [c for c in candidates if c.params.get("generated_by") == "traditional_exit_adjustment"]
    assert len(generated) == 2
    assert {c.params["parent_id"] for c in generated} == {"c_first", "c_second"}


def test_optimizer_uses_research_survivors_when_gatecheck_has_no_passes() -> None:
    base_a = CandidateStrategySpec(
        candidate_id="c_a",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    base_b = CandidateStrategySpec(
        candidate_id="c_b",
        hypothesis_id="h2",
        method_id="factor_scoring",
        hypothesis_family="volume_confirmation",
        symbol="BTCUSDT",
        market="um_futures",
    )
    results = [
        BacktestResult(
            experiment_id="exp-a",
            candidate_id="c_a",
            hypothesis_family="momentum",
            method_id="factor_scoring",
            symbol="BTCUSDT",
            market="um_futures",
            interval="5m",
            metrics_primary=MetricsBlock(sharpe=0.2),
            metrics_gross=MetricsBlock(sharpe=1.0),
            break_even_cost_bps=10.0,
            actual_cost_bps=2.0,
            factor_turnover=0.08,
            oos_trade_count=200,
        ),
        BacktestResult(
            experiment_id="exp-b",
            candidate_id="c_b",
            hypothesis_family="volume_confirmation",
            method_id="factor_scoring",
            symbol="BTCUSDT",
            market="um_futures",
            interval="5m",
            metrics_primary=MetricsBlock(sharpe=0.1),
            metrics_gross=MetricsBlock(sharpe=0.8),
            ic_tstat_nw=2.1,
            break_even_cost_bps=8.0,
            actual_cost_bps=2.0,
            factor_turnover=0.07,
            oos_trade_count=200,
        ),
    ]
    gates = [
        GateCheckResult(
            experiment_id="exp-a",
            passed=False,
            items=[GateCheckItem(rule_id="G5", status="fail", message="final gate")],
        ),
        GateCheckResult(
            experiment_id="exp-b",
            passed=False,
            items=[GateCheckItem(rule_id="G2", status="fail", message="final gate")],
        ),
    ]

    ctx = build_optimization_context([base_a, base_b], results, gates, iteration=0)
    optimization = optimize_traditionally(ctx, "full")
    new_candidates, summary = apply_optimization_result(optimization, [base_a, base_b], results)

    assert ctx["num_gatecheck_passed"] == 0
    assert ctx["num_research_survivors"] == 2
    assert set(optimization["combinations"][0]["factor_ids"]) == {"c_a", "c_b"}
    assert summary["combinations_created"] == 1
    combo = next(c for c in new_candidates if c.hypothesis_family == "composite")
    assert combo.params["search_variant"] == "survivor_combo_low_turnover"
    assert combo.params["smooth_span"] >= 24
    assert combo.params["signal_threshold"] >= 0.20
    assert combo.params["position_buffer"] >= 0.15


def test_single_research_survivor_generates_low_turnover_variant() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_single",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    result = BacktestResult(
        experiment_id="exp-single",
        candidate_id="c_single",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.1),
        metrics_gross=MetricsBlock(sharpe=0.7),
        oos_trade_count=200,
    )
    gate = GateCheckResult(
        experiment_id="exp-single",
        passed=False,
        items=[GateCheckItem(rule_id="G5", status="fail", message="final gate")],
    )

    ctx = build_optimization_context([candidate], [result], [gate], iteration=0)
    optimization = optimize_traditionally(ctx, "full")
    new_candidates, summary = apply_optimization_result(optimization, [candidate], [result])

    assert ctx["num_research_survivors"] == 1
    assert summary["combinations_created"] == 0
    assert summary["adjustments_applied"] == 1
    assert new_candidates[0].params["search_variant"] == "survivor_low_turnover"
    assert new_candidates[0].params["smooth_span"] == 48
    assert new_candidates[0].params["signal_threshold"] == 0.25
    assert new_candidates[0].params["position_buffer"] == 0.20


def test_boundary_conditions_do_not_block_repair_candidates() -> None:
    repair = CandidateStrategySpec(
        candidate_id="c_repair",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"generated_by": "near_miss_repair", "search_variant": "repair_cost_destroyed_edge"},
    )

    should_stop = _check_mining_boundaries(
        [repair],
        trial_counts={"momentum": 10_000},
        round_backtests=[],
        hypotheses=[],
    )

    assert should_stop is False


def test_mining_boundaries_prune_exhausted_new_hypotheses_but_keep_survivor_repairs() -> None:
    exhausted = CandidateStrategySpec(
        candidate_id="c_hyp",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="funding_basis",
        symbol="BTCUSDT",
        params={"generated_by": "traditional_next_hypothesis"},
    )
    repair = CandidateStrategySpec(
        candidate_id="c_rep",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="funding_basis",
        symbol="BTCUSDT",
        params={"generated_by": "near_miss_repair", "search_variant": "repair_regime_mixing"},
    )

    allowed = _filter_candidates_by_mining_boundaries(
        [exhausted, repair],
        trial_counts={"funding_basis": 1_275},
        round_backtests=[],
        hypotheses=[],
        log_blocks=False,
    )

    assert allowed == [repair]


def test_pre_gate_repair_generates_bounded_production_candidates() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_raw",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
        params={
            "signal_source": "factor_signal",
            "factor_family": "momentum",
            "factor_lookback": 12,
        },
    )
    result = BacktestResult(
        experiment_id="exp-raw",
        candidate_id="c_raw",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=-0.2),
        metrics_gross=MetricsBlock(sharpe=1.0),
        factor_turnover=0.24,
        avg_holding_period_bars=3.0,
        break_even_cost_bps=1.0,
        actual_cost_bps=3.0,
    )
    evidence = FactorEvidenceReport(
        experiment_id="exp-raw",
        candidate_id="c_raw",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        best_horizon_bars=48,
        ic_by_horizon={"48": 0.02},
        rankic_by_horizon={"48": 0.018},
        regime_conditional_ic={"bull": {"48": 0.04}},
        funding_conditional_ic={"state:positive": {"48": 0.04}},
        long_only_metrics=MetricsBlock(sharpe=0.7),
        short_only_metrics=MetricsBlock(sharpe=0.1),
    )

    repairs = _build_pre_gate_repair_candidates([candidate], [result], [evidence])

    assert 1 <= len(repairs) <= 4
    assert all(repair.params["generated_by"] == "pre_gate_repair" for repair in repairs)
    assert all(repair.params["complexity_score"] <= 4 for repair in repairs)
    low_turnover = next(repair for repair in repairs if repair.params["search_variant"] == "pre_gate_low_turnover")
    assert low_turnover.params["parent_id"] == "c_raw"
    assert low_turnover.params["smooth_span"] == 48
    assert low_turnover.params["signal_threshold"] == 0.30
    assert low_turnover.params["position_buffer"] == 0.25
    assert any(repair.params.get("factor_lookback") == 48 for repair in repairs)


def test_pre_gate_repair_allows_regime_conflict_when_filter_can_address_it() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_conflict",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
        params={
            "signal_source": "factor_signal",
            "factor_family": "momentum",
            "factor_lookback": 12,
        },
    )
    result = BacktestResult(
        experiment_id="exp-conflict",
        candidate_id="c_conflict",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.3),
        metrics_gross=MetricsBlock(sharpe=0.8),
        factor_turnover=0.03,
        break_even_cost_bps=10.0,
        actual_cost_bps=2.0,
    )
    evidence = FactorEvidenceReport(
        experiment_id="exp-conflict",
        candidate_id="c_conflict",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        ic_by_horizon={"12": 0.011},
        regime_conditional_ic={"bull": {"12": 0.04}, "bear": {"12": -0.03}},
        regime_conflict=True,
    )

    repairs = _build_pre_gate_repair_candidates([candidate], [result], [evidence])

    assert any(repair.params.get("regime_filter") == ["bull"] for repair in repairs)


def test_data_split_plan_keeps_final_oos_out_of_repair_window() -> None:
    frame = _frame(100)

    plan = _build_data_split_plan(frame)

    assert plan.discovery_mask.sum() == 60
    assert plan.repair_validation_mask.sum() == 20
    assert plan.repair_mask.sum() == 80
    assert plan.final_oos_mask.sum() == 20
    assert not bool((plan.discovery_mask & plan.repair_validation_mask).any())
    assert not bool((plan.repair_mask & plan.final_oos_mask).any())
    assert plan.repair_validation_start_idx == 60
    assert plan.final_oos_start_idx == 80


def test_data_split_plan_prefers_regime_covered_contiguous_oos_window() -> None:
    frame = _frame(100)
    regimes = pd.Series(
        ["bull", "bear"] * 30
        + ["bull", "bear"] * 10
        + ["high_vol"] * 20,
        index=frame.index,
    )

    plan = _build_data_split_plan(frame, regimes=regimes)

    assert plan.final_oos_start_idx < 80
    assert plan.final_oos_mask.sum() == 20
    assert plan.repair_validation_mask.sum() == 20
    assert not bool((plan.repair_mask & plan.final_oos_mask).any())
    final_regimes = set(regimes.loc[plan.final_oos_mask].unique())
    assert final_regimes == {"bull", "bear"}


def test_repair_merge_pool_prunes_high_pbo_and_redundant_repairs() -> None:
    parent = CandidateStrategySpec(
        candidate_id="c_parent",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"factor_family": "momentum"},
    )
    high_corr = parent.model_copy(deep=True)
    high_corr.candidate_id = "c_high_corr"
    high_corr.params.update({"parent_id": "c_parent", "generated_by": "pre_gate_repair"})
    high_pbo = parent.model_copy(deep=True)
    high_pbo.candidate_id = "c_high_pbo"
    high_pbo.params.update({"parent_id": "c_parent", "generated_by": "pre_gate_repair"})
    valid = parent.model_copy(deep=True)
    valid.candidate_id = "c_valid"
    valid.params.update({"parent_id": "c_parent", "generated_by": "pre_gate_repair"})
    parent_signal = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    tasks = [
        (parent_signal, parent.model_dump(mode="json"), 0, {}, []),
        (parent_signal.copy(), high_corr.model_dump(mode="json"), 1, {}, []),
        (np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]), high_pbo.model_dump(mode="json"), 2, {}, []),
        (np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0]), valid.model_dump(mode="json"), 3, {}, []),
    ]
    results = [
        _result("exp-parent", "c_parent", pbo=0.2, sharpe=0.4),
        _result("exp-high-corr", "c_high_corr", pbo=0.1, sharpe=0.8),
        _result("exp-high-pbo", "c_high_pbo", pbo=0.9, sharpe=1.0),
        _result("exp-valid", "c_valid", pbo=0.2, sharpe=0.9),
    ]

    plan = _select_repair_merge_pool(
        original_candidates=[parent],
        repair_candidates=[high_corr, high_pbo, valid],
        validation_candidates=[parent, high_corr, high_pbo, valid],
        validation_full_tasks=tasks,
        validation_tasks=tasks,
        validation_results=results,
    )

    kept_ids = {candidate.candidate_id for candidate in plan.candidates}
    assert kept_ids == {"c_parent", "c_valid"}
    assert plan.merged_repairs == 1
    assert plan.rejected_repairs == 2
    assert valid.params["repair_validation_pbo"] == 0.2
    assert abs(valid.params["parent_signal_correlation"]) < 0.98
    assert "low_incremental_orthogonality" in high_corr.params["merge_pool_reasons"]
    assert "high_validation_pbo" in high_pbo.params["merge_pool_reasons"]


def test_merge_pool_trial_penalty_counts_original_and_repair_trials() -> None:
    result = _result("exp-final", "c_final", pbo=0.2, sharpe=2.0)
    result.effective_trials_at_eval = 1
    result.global_trials_at_eval = 1

    _apply_merge_pool_trial_penalty([result], effective_trials_count=50, observations=100)

    assert result.effective_trials_at_eval == 50
    assert result.global_trials_at_eval == 50
    assert result.deflated_sharpe < result.metrics_primary.sharpe


def test_unfunded_filter_skips_only_funding_factor_signal_candidates() -> None:
    candidates = [
        CandidateStrategySpec(
            candidate_id="c_feature",
            hypothesis_id="h1",
            method_id="factor_scoring",
            hypothesis_family="funding_basis",
            symbol="BTCUSDT",
            params={"signal_source": "feature", "indicator_name": "premium_index_z_288"},
        ),
        CandidateStrategySpec(
            candidate_id="c_factor",
            hypothesis_id="h1",
            method_id="factor_scoring",
            hypothesis_family="funding_basis",
            symbol="BTCUSDT",
            params={"signal_source": "factor_signal", "factor_family": "funding_basis"},
        ),
        CandidateStrategySpec(
            candidate_id="c_legacy",
            hypothesis_id="h1",
            method_id="factor_scoring",
            hypothesis_family="funding_basis",
            symbol="BTCUSDT",
        ),
        CandidateStrategySpec(
            candidate_id="c_momentum",
            hypothesis_id="h2",
            method_id="factor_scoring",
            hypothesis_family="momentum",
            symbol="BTCUSDT",
            params={"signal_source": "factor_signal", "factor_family": "momentum"},
        ),
    ]

    filtered, skipped = _filter_unfunded_factor_signal_candidates(candidates, pd.Series([0.0, 0.0]))

    assert skipped == 2
    assert {candidate.candidate_id for candidate in filtered} == {"c_feature", "c_momentum"}

    funded, funded_skipped = _filter_unfunded_factor_signal_candidates(candidates, pd.Series([0.0, 1.0]))
    assert funded_skipped == 0
    assert funded == candidates


def test_batch_pbo_is_computed_from_candidate_returns(tmp_path) -> None:
    settings = Settings(
        data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"),
        cpcv=CPCVConfig(n_groups=4, test_groups=1),
    )
    frame = _frame(220)
    candidate_a = CandidateStrategySpec(
        candidate_id="c_a",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"pbo": 0.0},
    )
    candidate_b = CandidateStrategySpec(
        candidate_id="c_b",
        hypothesis_id="h2",
        method_id="factor_scoring",
        hypothesis_family="mean_reversion",
        symbol="BTCUSDT",
        params={"pbo": 0.0},
    )
    signal_a = pd.Series([1.0] * len(frame)).to_numpy(dtype=float)
    signal_b = pd.Series([-1.0] * len(frame)).to_numpy(dtype=float)
    tasks = [
        (signal_a, candidate_a.model_dump(mode="json"), 0, {}, []),
        (signal_b, candidate_b.model_dump(mode="json"), 1, {}, []),
    ]
    results = [
        BacktestResult(
            experiment_id="exp-a",
            candidate_id="c_a",
            hypothesis_family="momentum",
            method_id="factor_scoring",
            symbol="BTCUSDT",
            market="um_futures",
            interval="5m",
            metrics_primary=MetricsBlock(),
            pbo=None,
        ),
        BacktestResult(
            experiment_id="exp-b",
            candidate_id="c_b",
            hypothesis_family="mean_reversion",
            method_id="factor_scoring",
            symbol="BTCUSDT",
            market="um_futures",
            interval="5m",
            metrics_primary=MetricsBlock(),
            pbo=None,
        ),
    ]

    _apply_batch_pbo(frame, tasks, results, settings, funding_df=None)

    assert all(result.pbo is not None for result in results)
    assert all(0.0 <= result.pbo <= 1.0 for result in results)
    assert _cscv_splits(len(frame), settings)


def test_parallel_backtest_passes_funding_to_worker(tmp_path) -> None:
    settings = Settings(
        data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"),
        bootstrap=BootstrapConfig(n_resamples=20),
        permutation_test=PermutationTestConfig(n_permutations=20),
    )
    frame = _frame(220)
    candidate = CandidateStrategySpec(
        candidate_id="c_funding",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="funding_basis",
        symbol="BTCUSDT",
        market="um_futures",
    )
    signal = pd.Series([1.0] * len(frame)).to_numpy(dtype=float)
    tasks = [(signal, candidate.model_dump(mode="json"), 0, {}, [])]
    funding_df = pd.DataFrame({
        "calc_time": frame["open_time"],
        "last_funding_rate": [-0.001] * len(frame),
    })

    unfunded = _run_backtests_parallel(tasks, frame, settings, max_workers=1, funding_df=None)[0]
    funded = _run_backtests_parallel(tasks, frame, settings, max_workers=1, funding_df=funding_df)[0]

    assert unfunded.metrics_gross is not None
    assert unfunded.metrics_gross.total_return > unfunded.metrics_primary.total_return
    assert funded.metrics_primary.total_return > unfunded.metrics_primary.total_return


def test_gatecheck_diagnostics_summarizes_failures_and_costs() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_diag",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"search_variant": "low_turnover", "signal_source": "factor_signal"},
    )
    result = BacktestResult(
        experiment_id="exp-diag",
        candidate_id="c_diag",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=-1.0, annualized_return=-0.2, max_drawdown=-0.1),
        metrics_gross=MetricsBlock(sharpe=0.5),
        break_even_cost_bps=4.0,
        actual_cost_bps=3.0,
        factor_turnover=0.2,
        oos_trade_count=120,
    )
    gate = GateCheckResult(
        experiment_id="exp-diag",
        passed=False,
        items=[
            GateCheckItem(rule_id="G1", status="fail", message="bad", value=-1.0, threshold=0.0),
            GateCheckItem(rule_id="G8", status="fail", message="cost", value=4.0, threshold=6.0),
        ],
    )

    diagnostics = _gatecheck_diagnostics([candidate], [result], [gate], Settings())

    assert diagnostics["passed"] == 0
    assert diagnostics["failure_counts"][0] == {"rule_id": "G1", "count": 1}
    top = diagnostics["top_by_net_sharpe"][0]
    assert top["search_variant"] == "low_turnover"
    assert top["cost_drag_sharpe"] == 1.5
    assert top["cost_margin_bps"] == -2.0
