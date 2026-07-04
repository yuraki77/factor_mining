import numpy as np
import pandas as pd

from factor_mining.backtest.engine import evaluate_strategy_path
from factor_mining.config import Settings
from factor_mining.models import (
    BacktestResult,
    CandidateStrategySpec,
    FactorEvidenceReport,
    GateCheckItem,
    GateCheckResult,
    MetricsBlock,
    ResearchGateResult,
)
from factor_mining.near_miss import analyze_near_miss, repair_adjustments_from_near_misses
from factor_mining.optimizers.traditional_optimizer import apply_optimization_result, build_optimization_context, optimize_traditionally
from factor_mining.pipeline import _build_signal_for


def _result() -> BacktestResult:
    return BacktestResult(
        experiment_id="exp-near",
        candidate_id="cand-near",
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
        oos_trade_count=200,
    )


def _gate() -> GateCheckResult:
    return GateCheckResult(
        experiment_id="exp-near",
        passed=False,
        items=[
            GateCheckItem(rule_id="G5", status="fail", message="final gate"),
            GateCheckItem(rule_id="G8", status="fail", message="cost"),
        ],
    )


def test_near_miss_classifies_cost_and_turnover_repairs() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="cand-near",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"signal_source": "factor_signal", "factor_family": "momentum", "factor_lookback": 12},
    )
    evidence = FactorEvidenceReport(
        experiment_id="exp-near",
        candidate_id="cand-near",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        ic_by_horizon={"12": 0.01},
    )
    research = ResearchGateResult(
        experiment_id="exp-near",
        candidate_id="cand-near",
        status="research_survivor",
        production_gate_failures=["G5", "G8"],
        research_score=2.0,
        reasons=["gross_edge"],
    )

    miss = analyze_near_miss(
        candidate=candidate,
        result=_result(),
        gatecheck=_gate(),
        evidence=evidence,
        research_gate=research,
    )

    assert miss.primary_reason == "cost_destroyed_edge"
    assert "excess_turnover" in miss.reasons
    assert miss.actionable is True
    assert miss.suggested_params["smooth_span"] == 48
    assert miss.suggested_params["signal_threshold"] == 0.30
    assert len(miss.suggested_param_variants) == 4
    assert {variant["smooth_span"] for variant in miss.suggested_param_variants} == {12, 24, 48, 96}


def test_near_miss_ladder_advances_instead_of_repeating_same_turnover_repair() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="cand-near",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={
            "signal_source": "factor_signal",
            "factor_family": "momentum",
            "factor_lookback": 12,
            "smooth_span": 48,
            "signal_threshold": 0.30,
            "position_buffer": 0.25,
        },
    )
    gate = GateCheckResult(
        experiment_id="exp-near",
        passed=False,
        items=[GateCheckItem(rule_id="G8", status="fail", message="cost")],
    )

    miss = analyze_near_miss(
        candidate=candidate,
        result=_result(),
        gatecheck=gate,
        evidence=None,
        research_gate=None,
    )

    assert miss.actionable is True
    assert miss.suggested_params["smooth_span"] == 96
    assert miss.suggested_params["signal_threshold"] == 0.40


def test_near_miss_marks_turnover_repair_saturated_as_not_actionable() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="cand-near",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={
            "signal_source": "factor_signal",
            "factor_family": "momentum",
            "factor_lookback": 12,
            "smooth_span": 96,
            "signal_threshold": 0.40,
            "position_buffer": 0.30,
        },
    )
    gate = GateCheckResult(
        experiment_id="exp-near",
        passed=False,
        items=[GateCheckItem(rule_id="G8", status="fail", message="cost")],
    )

    miss = analyze_near_miss(
        candidate=candidate,
        result=_result(),
        gatecheck=gate,
        evidence=None,
        research_gate=None,
    )

    assert "turnover_repair_saturated" in miss.reasons
    assert miss.actionable is False
    assert miss.suggested_params == {}


def test_near_miss_uses_raw_gatecheck_when_research_gate_is_missing() -> None:
    miss = analyze_near_miss(
        candidate=None,
        result=_result().model_copy(update={
            "metrics_gross": MetricsBlock(sharpe=0.1),
            "metrics_primary": MetricsBlock(sharpe=-0.1),
            "factor_turnover": 0.01,
            "break_even_cost_bps": 2.0,
            "actual_cost_bps": 1.0,
        }),
        gatecheck=GateCheckResult(
            experiment_id="exp-near",
            passed=False,
            items=[GateCheckItem(rule_id="G5", status="fail", message="bootstrap")],
        ),
        evidence=None,
        research_gate=None,
    )

    assert miss.primary_reason == "overfit_or_unstable"
    assert miss.diagnostics["gate_failures"] == "G5"
    assert miss.diagnostics["gate_failure_count"] == 1
    assert miss.suggested_params["repair_complexity"] == "simplify"


def test_near_miss_marks_statistically_underpowered_survivor() -> None:
    result = BacktestResult(
        experiment_id="exp-underpowered",
        candidate_id="cand-underpowered",
        hypothesis_family="composite",
        method_id="condition_combination_search",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=3.93, trade_count=40),
        metrics_gross=MetricsBlock(sharpe=4.58),
        deflated_sharpe=3.897,
        deflated_sharpe_prob=0.62,
        oos_trade_count=40,
        break_even_cost_bps=35.77,
        actual_cost_bps=5.0,
    )
    gate = GateCheckResult(
        experiment_id=result.experiment_id,
        passed=False,
        items=[
            GateCheckItem(rule_id="G3", status="fail", message="FDR", value=0.6284, threshold=0.05),
            GateCheckItem(rule_id="G7", status="fail", message="trades", value=40, threshold=100),
        ],
    )

    miss = analyze_near_miss(
        candidate=None,
        result=result,
        gatecheck=gate,
        evidence=None,
        research_gate=ResearchGateResult(
            experiment_id=result.experiment_id,
            candidate_id=result.candidate_id,
            status="research_survivor",
            production_gate_failures=["G3", "G7"],
            reasons=["statistically_underpowered_survivor"],
        ),
    )

    assert miss.primary_reason == "statistically_underpowered_survivor"
    assert miss.repair_actions == ["accumulate_oos_evidence"]
    assert miss.actionable is False
    assert miss.diagnostics["fdr_adjusted_pvalue"] == 0.6284


def test_optimizer_turns_near_miss_into_repair_candidate() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="cand-near",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"signal_source": "factor_signal", "factor_family": "momentum", "factor_lookback": 12},
    )
    result = _result()
    gate = _gate()
    research = ResearchGateResult(
        experiment_id="exp-near",
        candidate_id="cand-near",
        status="research_survivor",
        production_gate_failures=["G5", "G8"],
        research_score=2.0,
        reasons=["gross_edge"],
    )
    miss = analyze_near_miss(
        candidate=candidate,
        result=result,
        gatecheck=gate,
        evidence=None,
        research_gate=research,
    )

    ctx = build_optimization_context([candidate], [result], [gate], 0, research_gates=[research], near_misses=[miss])
    optimization = optimize_traditionally(ctx, "full")
    new_candidates, summary = apply_optimization_result(optimization, [candidate], [result])

    repair_candidates = [item for item in new_candidates if item.params.get("generated_by") == "near_miss_repair"]
    assert ctx["repair_adjustments"]
    assert summary["repairs_created"] == 4
    assert repair_candidates
    assert repair_candidates[0].params["near_miss_reason"] == "cost_destroyed_edge"
    assert repair_candidates[0].params["search_variant"] == "repair_cost_destroyed_edge"
    assert repair_candidates[0].params["optimizer_proposal_id"]
    assert repair_candidates[0].params["optimizer_root_parent_id"] == "cand-near"
    assert repair_candidates[0].params["optimizer_parent_metrics"]["sharpe"] == -0.2


def test_optimizer_dedupes_equivalent_near_miss_repair_adjustments() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="cand-near",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"signal_source": "factor_signal", "factor_family": "momentum", "factor_lookback": 12},
    )
    result = _result()
    suggested = {
        "smooth_span": 48,
        "signal_threshold": 0.30,
        "position_buffer": 0.25,
        "near_miss_reason": "cost_destroyed_edge",
    }
    optimization = {
        "adjustments": [
            {"candidate_id": "cand-near", "param": "repair_params", "suggested": suggested},
            {"candidate_id": "cand-near", "param": "repair_params", "suggested": dict(suggested)},
        ]
    }

    new_candidates, summary = apply_optimization_result(optimization, [candidate], [result])

    repair_candidates = [item for item in new_candidates if item.params.get("generated_by") == "near_miss_repair"]
    assert len(repair_candidates) == 1
    assert summary["repairs_created"] == 1


def test_side_mode_and_filters_affect_generated_signals() -> None:
    frame = pd.DataFrame({
        "open_time": [1_700_000_000_000 + idx * 300_000 for idx in range(80)],
        "open": np.linspace(100, 105, 80),
        "high": np.linspace(101, 106, 80),
        "low": np.linspace(99, 104, 80),
        "close": np.linspace(100, 105, 80),
        "volume": [100.0] * 80,
        "quote_volume": [1_000_000.0] * 80,
    })
    features = pd.DataFrame({"alpha": [-1.0, 1.0] * 40}, index=frame.index)
    regimes = pd.Series(["bull"] * 40 + ["bear"] * 40, index=frame.index)
    candidate = CandidateStrategySpec(
        candidate_id="cand-filter",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={
            "signal_source": "feature",
            "indicator_name": "alpha",
            "transform": "raw_clip",
            "regime_filter": ["bull"],
            "side_mode": "long_only",
        },
    )

    signal = pd.Series(_build_signal_for(candidate, frame, features, {"alpha": {"family": "momentum"}}, 0, regimes), index=frame.index)
    path = evaluate_strategy_path(frame, signal, candidate, Settings(), funding=None)

    assert signal.iloc[40:].abs().sum() == 0.0
    assert path.signals.min() >= 0.0
    assert path.position.min() >= 0.0


def test_repair_adjustments_from_near_misses_limits_to_actionable() -> None:
    actionable = analyze_near_miss(
        candidate=None,
        result=_result(),
        gatecheck=_gate(),
        evidence=None,
        research_gate=ResearchGateResult(
            experiment_id="exp-near",
            candidate_id="cand-near",
            status="research_survivor",
            production_gate_failures=["G5"],
        ),
    )
    passed = analyze_near_miss(
        candidate=None,
        result=_result().model_copy(update={"experiment_id": "exp-pass"}),
        gatecheck=GateCheckResult(experiment_id="exp-pass", passed=True, items=[]),
        evidence=None,
        research_gate=None,
    )

    adjustments = repair_adjustments_from_near_misses([actionable, actionable, passed])

    assert len(adjustments) == 4
    assert {item["param"] for item in adjustments} == {"repair_params"}
    assert {item["variant_key"] for item in adjustments}
