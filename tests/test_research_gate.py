from factor_mining.models import (
    BacktestResult,
    FactorEvidenceReport,
    GateCheckItem,
    GateCheckResult,
    MetricsBlock,
    ResearchGateResult,
)
from factor_mining.optimizers.traditional_optimizer import build_optimization_context
from factor_mining.models import CandidateStrategySpec
from factor_mining.config import Settings
from factor_mining.research_gate import apply_research_gate, build_research_survivor_records, evaluate_research_gate, research_survivor_payloads
from factor_mining.storage import MetadataStore


def _result(candidate_id: str = "cand-1") -> BacktestResult:
    return BacktestResult(
        experiment_id=f"exp-{candidate_id}",
        candidate_id=candidate_id,
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=-0.1),
        metrics_gross=MetricsBlock(sharpe=0.2),
        factor_turnover=0.1,
        break_even_cost_bps=1.0,
        actual_cost_bps=2.0,
    )


def _failed_gate(experiment_id: str) -> GateCheckResult:
    return GateCheckResult(
        experiment_id=experiment_id,
        passed=False,
        items=[GateCheckItem(rule_id="G5", status="fail", message="final gate")],
    )


def test_research_gate_keeps_production_passed_candidates() -> None:
    result = _result()
    gate = GateCheckResult(experiment_id=result.experiment_id, passed=True, items=[])

    research = evaluate_research_gate(result=result, gatecheck=gate, evidence=None)

    assert research.status == "production_passed"
    assert research.production_gate_passed is True
    assert "production_gate_passed" in research.reasons


def test_research_gate_keeps_failed_candidate_with_factor_evidence() -> None:
    result = _result()
    evidence = FactorEvidenceReport(
        experiment_id=result.experiment_id,
        candidate_id=result.candidate_id,
        hypothesis_family=result.hypothesis_family,
        method_id=result.method_id,
        symbol=result.symbol,
        market=result.market,
        interval=result.interval,
        ic_by_horizon={"12": 0.02},
        rankic_by_horizon={"12": 0.015},
        quantile_spread_by_horizon={"12": 2.5},
        regime_conditional_ic={"high_vol": {"12": 0.03}},
    )

    research = evaluate_research_gate(result=result, gatecheck=_failed_gate(result.experiment_id), evidence=evidence)

    assert research.status == "research_survivor"
    assert research.production_gate_failures == ["G5"]
    assert {"ic_signal", "rankic_signal", "quantile_spread", "regime_conditional_signal"}.issubset(
        set(research.reasons)
    )


def test_research_gate_rejects_candidate_without_evidence() -> None:
    result = _result()

    research = evaluate_research_gate(result=result, gatecheck=_failed_gate(result.experiment_id), evidence=None)

    assert research.status == "rejected"
    assert research.research_score < 1.5


def test_research_gate_uses_backtest_metrics_and_gatecheck_flags_without_evidence() -> None:
    result = _result().model_copy(update={
        "metrics_primary": MetricsBlock(sharpe=0.2),
        "metrics_gross": MetricsBlock(sharpe=0.8),
        "break_even_cost_bps": 10.0,
        "actual_cost_bps": 2.0,
        "oos_trade_count": 150,
    })
    gate = GateCheckResult(
        experiment_id=result.experiment_id,
        passed=False,
        items=[
            GateCheckItem(rule_id="G2", status="fail", message="PBO"),
            GateCheckItem(rule_id="G8", status="fail", message="cost"),
        ],
    )

    research = evaluate_research_gate(result=result, gatecheck=gate, evidence=None)

    assert research.status == "research_survivor"
    assert {"gross_edge", "positive_net_sharpe", "cost_margin"}.issubset(set(research.reasons))
    assert research.evidence_flags["production_gate_failures"] == "G2,G8"
    assert research.evidence_flags["failed_g2_pbo"] is True
    assert research.evidence_flags["failed_g8_cost"] is True
    assert research.evidence_flags["gross_sharpe"] == 0.8


def test_formal_research_gate_feeds_optimizer_survivors() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="cand-1",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    result = _result("cand-1")
    gate = _failed_gate(result.experiment_id)
    research_gate = ResearchGateResult(
        experiment_id=result.experiment_id,
        candidate_id=result.candidate_id,
        status="research_survivor",
        production_gate_failures=["G5"],
        research_score=2.0,
        reasons=["ic_signal", "gross_edge"],
    )

    ctx = build_optimization_context([candidate], [result], [gate], iteration=0, research_gates=[research_gate])
    payloads = research_survivor_payloads({candidate.candidate_id: candidate}, [result], [research_gate])

    assert ctx["num_research_survivors"] == 1
    assert ctx["research_survivors"][0]["research_gate_status"] == "research_survivor"
    assert payloads[0]["status"] == "research_survivor"
    assert payloads[0]["survivor_reason"] == "ic_signal,gross_edge"


def test_apply_research_gate_aligns_by_experiment_id() -> None:
    result = _result("cand-1")
    evidence = FactorEvidenceReport(
        experiment_id=result.experiment_id,
        candidate_id=result.candidate_id,
        hypothesis_family=result.hypothesis_family,
        method_id=result.method_id,
        symbol=result.symbol,
        market=result.market,
        interval=result.interval,
        funding_conditional_ic={"state:positive": {"12": 0.04}},
    )

    gates = apply_research_gate([result], [_failed_gate(result.experiment_id)], [evidence])

    assert gates[0].status == "research_survivor"
    assert "funding_conditional_signal" in gates[0].reasons


def test_research_survivor_record_tracks_underpowered_high_edge_candidate() -> None:
    settings = Settings()
    candidate = CandidateStrategySpec(
        candidate_id="cand-underpowered",
        hypothesis_id="hyp-1",
        method_id="condition_combination_search",
        hypothesis_family="composite",
        symbol="BTCUSDT",
        market="um_futures",
    )
    result = BacktestResult(
        experiment_id="exp-underpowered",
        candidate_id=candidate.candidate_id,
        hypothesis_family=candidate.hypothesis_family,
        method_id=candidate.method_id,
        symbol=candidate.symbol,
        market=candidate.market,
        interval=candidate.interval,
        metrics_primary=MetricsBlock(sharpe=3.93, trade_count=40),
        metrics_gross=MetricsBlock(sharpe=4.58),
        deflated_sharpe=3.897,
        oos_trade_count=40,
        break_even_cost_bps=35.77,
        actual_cost_bps=5.0,
    )
    gate = GateCheckResult(
        experiment_id=result.experiment_id,
        passed=False,
        items=[
            GateCheckItem(rule_id="G3", status="fail", message="FDR"),
            GateCheckItem(rule_id="G7", status="fail", message="trades"),
        ],
    )
    research = evaluate_research_gate(result=result, gatecheck=gate, evidence=None)

    records = build_research_survivor_records(
        candidates_by_id={candidate.candidate_id: candidate},
        results=[result],
        research_gates=[research],
        fdr_map={result.experiment_id: 0.6284},
        settings=settings,
    )

    assert research.status == "research_survivor"
    assert "statistically_underpowered_survivor" in research.reasons
    assert research.evidence_flags["failed_g3_fdr"] is True
    assert research.evidence_flags["failed_g7_trades"] is True
    assert len(records) == 1
    assert records[0].status == "active"
    assert records[0].current_trades == 40
    assert records[0].required_additional_trades == 60
    assert records[0].required_oos_days == 90
    assert records[0].fdr_pvalue == 0.6284
    assert records[0].sharpe == 3.93
    assert records[0].dsr == 3.897
    assert round(records[0].cost_margin_bps or 0.0, 2) == 25.77
    assert records[0].promotion_criteria == "NW FDR P < 0.10 AND trades >= 100"
    assert records[0].recheck_trigger == "on_next_round_if_new_trades >= 60"
    assert "statistically_underpowered_survivor" in records[0].survivor_reason


def test_research_survivor_store_preserves_paper_trade_start_and_updates_status(tmp_path) -> None:
    settings = Settings()
    candidate = CandidateStrategySpec(
        candidate_id="cand-store",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    result = _result("cand-store").model_copy(update={
        "metrics_primary": MetricsBlock(sharpe=1.2, trade_count=40),
        "metrics_gross": MetricsBlock(sharpe=1.6),
        "break_even_cost_bps": 12.0,
        "actual_cost_bps": 2.0,
        "oos_trade_count": 40,
    })
    research = evaluate_research_gate(result=result, gatecheck=_failed_gate(result.experiment_id), evidence=None)
    first = build_research_survivor_records(
        candidates_by_id={candidate.candidate_id: candidate},
        results=[result],
        research_gates=[research],
        fdr_map={result.experiment_id: 0.4},
        settings=settings,
    )[0]
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.upsert_research_survivors([first])

    later = first.model_copy(update={
        "current_trades": 80,
        "required_additional_trades": 20,
        "paper_trade_start_date": first.paper_trade_start_date.replace(year=first.paper_trade_start_date.year + 1),
    })
    store.upsert_research_survivors([later])
    loaded = store.list_research_survivors(status="active")

    assert len(loaded) == 1
    assert loaded[0].paper_trade_start_date == first.paper_trade_start_date
    assert loaded[0].current_trades == 80

    store.update_research_survivor_status("cand-store", "promoted", "promotion_criteria_met")
    promoted = store.list_research_survivors(status=None)

    assert promoted[0].status == "promoted"
    assert promoted[0].status_reason == "promotion_criteria_met"
