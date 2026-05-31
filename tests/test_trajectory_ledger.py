"""Tests for TrajectoryLedger, TrajectoryRecord storage round-trip,
and success contract report generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from factor_mining.config import Settings
from factor_mining.models import (
    BacktestResult,
    CandidateStrategySpec,
    FactorEvidenceReport,
    GateCheckItem,
    GateCheckResult,
    MetricsBlock,
    NearMissAnalysis,
    ResearchGateResult,
    TrajectoryRecord,
)
from factor_mining.storage import MetadataStore
from factor_mining.success_contract import build_success_contract_report
from factor_mining.trajectory_ledger import TrajectoryLedger


# ── helpers ─────────────────────────────────────────────────────────


def _store() -> MetadataStore:
    path = Path(tempfile.mkdtemp()) / "test.sqlite3"
    return MetadataStore(path)


def _settings() -> Settings:
    return Settings()


def _candidate(cid: str, ctype: str = "original", parent_id: str | None = None, **params: object) -> CandidateStrategySpec:
    return CandidateStrategySpec(
        candidate_id=cid,
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={**params},
        candidate_type=ctype,  # type: ignore[arg-type]
        parent_candidate_id=parent_id,
    )


def _result(cid: str, sharpe: float = 1.0) -> BacktestResult:
    return BacktestResult(
        experiment_id=f"exp-{cid}",
        candidate_id=cid,
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=sharpe),
        deflated_sharpe=0.5,
        factor_turnover=0.10,
        break_even_cost_bps=5.0,
        actual_cost_bps=2.0,
    )


def _evidence(cid: str) -> FactorEvidenceReport:
    return FactorEvidenceReport(
        experiment_id=f"exp-{cid}",
        candidate_id=cid,
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
    )


def _research_gate(cid: str, status: str = "research_survivor") -> ResearchGateResult:
    return ResearchGateResult(
        experiment_id=f"exp-{cid}",
        candidate_id=cid,
        status=status,  # type: ignore[arg-type]
    )


def _near_miss(cid: str, reason: str = "production_passed") -> NearMissAnalysis:
    return NearMissAnalysis(
        experiment_id=f"exp-{cid}",
        candidate_id=cid,
        primary_reason=reason,  # type: ignore[arg-type]
    )


# ── TrajectoryRecord model tests ────────────────────────────────────


def test_trajectory_record_round_trips_through_json() -> None:
    record = TrajectoryRecord(
        trajectory_id="traj_001",
        candidate_id="c_001",
        parent_ids=["c_root"],
        operator="SEED",
        hypothesis_family="momentum",
        classification="research_survivor",
    )
    payload = record.model_dump(mode="json")
    restored = TrajectoryRecord.model_validate(payload)
    assert restored.trajectory_id == "traj_001"
    assert restored.operator == "SEED"
    assert restored.parent_ids == ["c_root"]


def test_trajectory_record_defaults() -> None:
    record = TrajectoryRecord(trajectory_id="traj_def", candidate_id="c_def")
    assert record.operator == "SEED"
    assert record.parent_ids == []
    assert record.classification == "rejected"
    assert record.freeze_point == {}
    assert record.trial_ids == []
    assert record.trial_refs == []
    assert record.parent_trajectory_ids == []
    assert record.artifact_references == []


# ── Storage round-trip tests ────────────────────────────────────────


def test_save_and_load_trajectory() -> None:
    store = _store()
    record = TrajectoryRecord(
        trajectory_id="traj_save",
        candidate_id="c_save",
        operator="MUTATION_AT_DSL",
        operator_detail="GRID_TUNING",
        hypothesis_family="mean_reversion",
        candidate_snapshot={"key": "value"},
        backtest_result={"sharpe": 1.5},
    )
    store.save_trajectory(record)
    loaded = store.load_trajectory("traj_save")
    assert loaded is not None
    assert loaded.trajectory_id == "traj_save"
    assert loaded.operator == "MUTATION_AT_DSL"
    assert loaded.operator_detail == "GRID_TUNING"
    assert loaded.candidate_snapshot == {"key": "value"}
    assert loaded.backtest_result == {"sharpe": 1.5}


def test_legacy_operator_normalizes_to_semantic_operator() -> None:
    record = TrajectoryRecord(
        trajectory_id="traj_legacy",
        candidate_id="c_legacy",
        operator="GRID_TUNING",  # type: ignore[arg-type]
    )
    assert record.operator == "MUTATION_AT_DSL"
    assert record.operator_detail == "GRID_TUNING"


def test_list_trajectories_filters_by_candidate_id() -> None:
    store = _store()
    for i in range(3):
        store.save_trajectory(
            TrajectoryRecord(
                trajectory_id=f"traj_{i}",
                candidate_id=f"c_{i}",
            )
        )
    results = store.list_trajectories(candidate_id="c_1")
    assert len(results) == 1
    assert results[0].candidate_id == "c_1"


def test_list_trajectories_filters_by_operator() -> None:
    store = _store()
    store.save_trajectory(TrajectoryRecord(trajectory_id="t1", candidate_id="c1", operator="SEED"))
    store.save_trajectory(TrajectoryRecord(trajectory_id="t2", candidate_id="c2", operator="MUTATION_AT_DSL"))
    store.save_trajectory(TrajectoryRecord(trajectory_id="t3", candidate_id="c3", operator="SEED"))

    seeds = store.list_trajectories(operator="SEED")
    assert len(seeds) == 2


def test_save_trajectory_is_idempotent_for_same_deterministic_id() -> None:
    store = _store()
    record = TrajectoryRecord(
        trajectory_id="traj_same",
        candidate_id="c_same",
        operator="SEED",
        classification="rejected",
    )
    store.save_trajectory(record)
    store.save_trajectory(record.model_copy(update={"classification": "research_survivor"}))
    results = store.list_trajectories(candidate_id="c_same")
    assert len(results) == 1
    assert results[0].classification == "research_survivor"


def test_prune_trajectories_keeps_recent() -> None:
    store = _store()
    for i in range(10):
        store.save_trajectory(
            TrajectoryRecord(trajectory_id=f"traj_{i}", candidate_id=f"c_{i}")
        )
    assert len(store.list_trajectories()) == 10
    pruned = store.prune_trajectories(max_unprotected_rows=5)
    assert pruned == 5
    assert len(store.list_trajectories()) == 5


def test_load_missing_trajectory_returns_none() -> None:
    store = _store()
    assert store.load_trajectory("nonexistent") is None


# ── TrajectoryLedger tests ──────────────────────────────────────────


def test_classify_seed() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="original")
    assert ledger.classify_operator(c, {}) == "SEED"


def test_classify_seed_informed_with_lab_direction() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="original", lab_direction_factor_ids=["rsi14"])
    assert ledger.classify_operator(c, {}) == "SEED_INFORMED"


def test_classify_grid_tuning() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="grid_tuning", parent_id="p1")
    assert ledger.classify_operator(c, {}) == "MUTATION_AT_DSL"
    assert ledger.classify_operator_with_detail(c, {}) == ("MUTATION_AT_DSL", "GRID_TUNING")


def test_classify_pre_gate_repair() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="repair", parent_id="p1")
    assert ledger.classify_operator(c, {}) == "MUTATION_AT_DSL"
    assert ledger.classify_operator_with_detail(c, {}) == ("MUTATION_AT_DSL", "PRE_GATE_REPAIR")


def test_classify_composite() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="composite", parent_id="p1")
    assert ledger.classify_operator(c, {}) == "CROSSOVER"
    assert ledger.classify_operator_with_detail(c, {}) == ("CROSSOVER", "COMPOSITE_EQUI_WEIGHT")


def test_classify_optimizer_hill_climb() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="optimizer", parent_id="p1", optimizer_proposal_kind="hill_climb")
    assert ledger.classify_operator(c, {}) == "MUTATION_AT_DSL"
    assert ledger.classify_operator_with_detail(c, {}) == ("MUTATION_AT_DSL", "OPTIMIZER_HILL_CLIMB")


def test_classify_optimizer_evolution() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="optimizer", parent_id="p1", optimizer_proposal_kind="evolution")
    assert ledger.classify_operator(c, {}) == "MUTATION_AT_DSL"
    assert ledger.classify_operator_with_detail(c, {}) == ("MUTATION_AT_DSL", "OPTIMIZER_EVOLUTION")


def test_classify_optimizer_turnover_control() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="optimizer", parent_id="p1", optimizer_proposal_kind="turnover_control")
    assert ledger.classify_operator(c, {}) == "MUTATION_AT_DSL"
    assert ledger.classify_operator_with_detail(c, {}) == ("MUTATION_AT_DSL", "OPTIMIZER_EVOLUTION")


def test_classify_optimizer_crossover() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="optimizer", parent_id="p1", optimizer_proposal_kind="crossover")
    assert ledger.classify_operator(c, {}) == "CROSSOVER"
    assert ledger.classify_operator_with_detail(c, {}) == ("CROSSOVER", "DSL_COMPOSITE")


def test_classify_legacy_parent() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="original", parent_id="p1")
    assert ledger.classify_operator(c, {}) == "MUTATION_AT_DSL"
    assert ledger.classify_operator_with_detail(c, {}) == ("MUTATION_AT_DSL", "OPTIMIZER_EVOLUTION")


def test_classify_grid_tuning_from_local_grid_kind() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c1", ctype="optimizer", parent_id="p1", optimizer_proposal_kind="local_grid_tuning")
    assert ledger.classify_operator(c, {}) == "MUTATION_AT_DSL"
    assert ledger.classify_operator_with_detail(c, {}) == ("MUTATION_AT_DSL", "GRID_TUNING")


def test_optimizer_crossover_record_keeps_all_parent_ids() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate(
        "c_xo",
        ctype="optimizer",
        parent_id="p1",
        optimizer_proposal_kind="crossover",
        parent_ids=["p1", "p2"],
    )
    record = ledger.create_record(
        c, _result("c_xo"), _evidence("c_xo"),
        _research_gate("c_xo", "research_survivor"),
        _near_miss("c_xo"),
        {},
        artifact_scope="round1",
    )
    assert record.operator == "CROSSOVER"
    assert record.operator_detail == "DSL_COMPOSITE"
    assert record.parent_ids == ["p1", "p2"]


# ── create_record tests ─────────────────────────────────────────────


def test_create_record_seed() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c_seed", ctype="original")
    record = ledger.create_record(
        c, _result("c_seed"), _evidence("c_seed"),
        _research_gate("c_seed", "research_survivor"),
        _near_miss("c_seed"),
        {},
        artifact_scope="round1",
    )
    assert record.operator == "SEED"
    assert record.classification == "research_survivor"
    assert record.parent_ids == []
    assert record.candidate_snapshot["candidate_id"] == "c_seed"
    assert record.backtest_result is not None
    assert record.experiment_id == "exp-c_seed"
    assert record.artifact_scope == "round1"
    assert record.trial_refs == [{
        "experiment_id": "exp-c_seed",
        "effective_trials_at_eval": 0,
        "global_trials_at_eval": 0,
    }]
    assert record.hypothesis_family == "momentum"


def test_create_record_grid_tuning() -> None:
    ledger = TrajectoryLedger(None, _settings())
    parent = _candidate("c_parent", ctype="original")
    child = _candidate(
        "c_grid_child",
        ctype="grid_tuning",
        parent_id="c_parent",
        generated_by="local_grid_tuning",
        optimizer_root_parent_id="c_parent",
        optimizer_proposal_kind="local_grid_tuning",
    )
    record = ledger.create_record(
        child, _result("c_grid_child"), _evidence("c_grid_child"),
        _research_gate("c_grid_child", "research_survivor"),
        _near_miss("c_grid_child"),
        {"c_parent": parent},
        artifact_scope="round1_BTCUSDT_um_futures",
    )
    assert record.operator == "MUTATION_AT_DSL"
    assert record.operator_detail == "GRID_TUNING"
    assert record.source_candidate_type == "grid_tuning"
    assert record.parent_ids == ["c_parent"]
    assert record.candidate_snapshot["params"]["generated_by"] == "local_grid_tuning"
    assert "gatechecks_round1_BTCUSDT_um_futures" in record.artifact_references


def test_create_record_uses_deterministic_trajectory_id() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c_seed", ctype="original")
    first = ledger.create_record(
        c, _result("c_seed"), _evidence("c_seed"),
        _research_gate("c_seed", "research_survivor"),
        _near_miss("c_seed"),
        {},
        artifact_scope="round1",
    )
    second = ledger.create_record(
        c, _result("c_seed"), _evidence("c_seed"),
        _research_gate("c_seed", "research_survivor"),
        _near_miss("c_seed"),
        {},
        artifact_scope="round1",
    )
    changed_scope = ledger.create_record(
        c, _result("c_seed"), _evidence("c_seed"),
        _research_gate("c_seed", "research_survivor"),
        _near_miss("c_seed"),
        {},
        artifact_scope="round2",
    )
    assert first.trajectory_id == second.trajectory_id
    assert first.trajectory_id != changed_scope.trajectory_id


def test_create_record_best_effort_parent_trajectory_ids() -> None:
    store = _store()
    parent_record = TrajectoryRecord(
        trajectory_id="traj_parent",
        candidate_id="c_parent",
        operator="SEED",
    )
    store.save_trajectory(parent_record)
    ledger = TrajectoryLedger(store, _settings())
    child = _candidate("c_child", ctype="grid_tuning", parent_id="c_parent")
    record = ledger.create_record(
        child, _result("c_child"), _evidence("c_child"),
        _research_gate("c_child", "research_survivor"),
        _near_miss("c_child"),
        {},
    )
    assert record.parent_ids == ["c_parent"]
    assert record.parent_trajectory_ids == ["traj_parent"]


def test_create_record_rejected_near_miss() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c_fail", ctype="original")
    record = ledger.create_record(
        c, _result("c_fail", sharpe=0.1), _evidence("c_fail"),
        _research_gate("c_fail", "rejected"),
        _near_miss("c_fail", "cost_destroyed_edge"),
        {},
    )
    assert record.classification == "rejected"
    assert record.diagnosis_text is not None
    assert "near_miss=cost_destroyed_edge" in record.diagnosis_text


def test_create_record_handles_none_backtest() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c_nobt", ctype="original")
    record = ledger.create_record(
        c, None, None, None, None, {},
    )
    assert record.backtest_result is None
    assert record.evidence_snapshot is None
    assert record.classification == "rejected"


def test_create_record_production_passed() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c_prod", ctype="original")
    record = ledger.create_record(
        c, _result("c_prod"), _evidence("c_prod"),
        _research_gate("c_prod", "production_passed"),
        _near_miss("c_prod"),
        {},
    )
    assert record.classification == "production_passed"
    assert record.promotion_reason == "production_gate_passed"


# ── TrajectoryLedger no-store safety ────────────────────────────────


def test_ledger_without_store_does_not_crash() -> None:
    ledger = TrajectoryLedger(None, _settings())
    c = _candidate("c_noop", ctype="original")
    record = ledger.create_record(c, _result("c_noop"), _evidence("c_noop"), None, None, {})
    # save should be a no-op when store is None
    ledger.save(record)  # does not raise


# ── success_contract tests ──────────────────────────────────────────


def test_success_contract_empty_store() -> None:
    store = _store()
    report = build_success_contract_report(store=store, settings=_settings())
    assert report["total_candidates_evaluated"] == 0


def test_success_contract_with_mixed_operators() -> None:
    store = _store()
    for i in range(3):
        store.save_trajectory(
            TrajectoryRecord(
                trajectory_id=f"ts{i}",
                candidate_id=f"cs{i}",
                operator="SEED",
                classification="research_survivor",
                backtest_result={"metrics_primary": {"sharpe": 1.0 + i * 0.5}},
            )
        )
    for i in range(3, 5):
        store.save_trajectory(
            TrajectoryRecord(
                trajectory_id=f"ts{i}",
                candidate_id=f"cs{i}",
                operator="MUTATION_AT_DSL",
                operator_detail="GRID_TUNING",
                classification="production_passed",
                parent_ids=["cs0"],
                backtest_result={"metrics_primary": {"sharpe": 2.0}},
            )
        )

    report = build_success_contract_report(store=store, settings=_settings())
    assert report["total_candidates_evaluated"] == 5
    assert report["trajectory_counts_by_operator"]["SEED"] == 3
    assert report["trajectory_counts_by_operator"]["MUTATION_AT_DSL"] == 2
    assert report["production_rate_by_operator"]["MUTATION_AT_DSL"] == 1.0
    assert report["research_survivor_rate_by_operator"]["SEED"] == 1.0
    assert report["gatecheck_pass_rate"] == 0.4
    assert report["research_survivor_rate"] == 0.6
    assert report["promoted_rate"] == 1.0
    assert report["lineage_depth_stats"]["max"] == 1.0
    assert report["lineage_depth_stats"]["min"] == 0.0


def test_success_contract_lineage_depths() -> None:
    store = _store()
    store.save_trajectory(
        TrajectoryRecord(
            trajectory_id="t0", candidate_id="c0", operator="SEED",
            parent_ids=[], backtest_result={"metrics_primary": {"sharpe": 0.5}},
        )
    )
    store.save_trajectory(
        TrajectoryRecord(
            trajectory_id="t1", candidate_id="c1", operator="MUTATION_AT_DSL",
            parent_ids=["c0"], backtest_result={"metrics_primary": {"sharpe": 0.8}},
        )
    )
    store.save_trajectory(
        TrajectoryRecord(
            trajectory_id="t2", candidate_id="c2", operator="MUTATION_AT_DSL",
            operator_detail="OPTIMIZER_EVOLUTION",
            parent_ids=["c0", "c1"], backtest_result={"metrics_primary": {"sharpe": 1.2}},
        )
    )
    report = build_success_contract_report(store=store, settings=_settings())
    assert report["lineage_depth_stats"]["min"] == 0.0
    assert report["lineage_depth_stats"]["max"] == 2.0
    # parent_ids lengths are 0, 1, 2 → mean = 1.0
    assert report["lineage_depth_stats"]["mean"] == 1.0


def test_success_contract_scoped_report_excludes_records_without_matching_experiment() -> None:
    store = _store()
    store.save_trajectory(
        TrajectoryRecord(
            trajectory_id="t_match",
            candidate_id="c_match",
            experiment_id="exp-match",
            operator="SEED",
            classification="production_passed",
            backtest_result={"experiment_id": "exp-match", "metrics_primary": {"sharpe": 1.0}},
        )
    )
    store.save_trajectory(
        TrajectoryRecord(
            trajectory_id="t_no_exp",
            candidate_id="c_no_exp",
            operator="SEED",
            classification="production_passed",
            backtest_result=None,
        )
    )
    report = build_success_contract_report(
        store=store,
        settings=_settings(),
        experiment_ids=["exp-match"],
    )
    assert report["total_candidates_evaluated"] == 1
    assert report["gatecheck_pass_rate"] == 1.0


def test_create_records_for_candidates_joins_by_candidate_id_not_position() -> None:
    ledger = TrajectoryLedger(None, _settings())
    first = _candidate("c_first")
    second = _candidate("c_second")

    records, skipped = ledger.create_records_for_candidates(
        [first, second],
        [_result("c_first"), _result("c_second")],
        [_evidence("c_second")],
        [_research_gate("c_second", "research_survivor"), _research_gate("c_first", "production_passed")],
        [_near_miss("c_second", "cost_destroyed_edge")],
        {},
        artifact_scope="round1",
    )

    assert skipped == []
    by_candidate = {record.candidate_id: record for record in records}
    assert by_candidate["c_first"].evidence_snapshot is None
    assert by_candidate["c_first"].near_miss_snapshot is None
    assert by_candidate["c_first"].classification == "production_passed"
    assert by_candidate["c_second"].evidence_snapshot is not None
    assert by_candidate["c_second"].near_miss_snapshot is not None
    assert by_candidate["c_second"].classification == "research_survivor"


# ── DSL fields backward-compatibility ───────────────────────────────


def test_candidate_without_dsl_fields_serializes_fine() -> None:
    c = CandidateStrategySpec(
        candidate_id="c_nodsl",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
    )
    payload = c.model_dump(mode="json")
    restored = CandidateStrategySpec.model_validate(payload)
    assert restored.dsl_expression is None
    assert restored.dsl_fingerprint is None
    assert restored.dsl_version is None


def test_candidate_with_dsl_fields_round_trips() -> None:
    c = CandidateStrategySpec(
        candidate_id="c_dsl",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        dsl_expression="TS_MEAN($close, 20)",
        dsl_canonical_expression="TS_MEAN($close, 20)",
        dsl_fingerprint="abc123",
        dsl_version="0.1.0",
    )
    payload = c.model_dump(mode="json")
    restored = CandidateStrategySpec.model_validate(payload)
    assert restored.dsl_expression == "TS_MEAN($close, 20)"
    assert restored.dsl_fingerprint == "abc123"


def test_trajectory_json_serialization_is_deterministic() -> None:
    """Verify that trajectory JSON serialization produces the same output
    for equivalent records (important for checksum-based reproducibility)."""
    from datetime import datetime, timezone
    frozen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    r1 = TrajectoryRecord(
        trajectory_id="t1", candidate_id="c1", operator="SEED",
        parent_ids=["p1", "p2"], created_at=frozen,
    )
    r2 = TrajectoryRecord(
        trajectory_id="t1", candidate_id="c1", operator="SEED",
        parent_ids=["p1", "p2"], created_at=frozen,
    )
    p1 = json.dumps(r1.model_dump(mode="json"), sort_keys=True, default=str)
    p2 = json.dumps(r2.model_dump(mode="json"), sort_keys=True, default=str)
    assert p1 == p2
