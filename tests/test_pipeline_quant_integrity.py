import threading

import numpy as np
import pandas as pd

from factor_mining.config import BootstrapConfig, CPCVConfig, DataConfig, PermutationTestConfig, Settings
from factor_mining.models import (
    BacktestResult,
    CandidateStrategySpec,
    FactorEvidenceReport,
    GateCheckItem,
    GateCheckResult,
    HardScoreReport,
    HypothesisSpec,
    MetricsBlock,
    ResearchGateResult,
)
from factor_mining.optimizers.traditional_optimizer import (
    apply_exit_adjustments,
    apply_optimization_result,
    build_optimization_context,
    optimize_exits_traditionally,
    optimize_traditionally,
)
import factor_mining.pipeline as pipeline
from factor_mining.pipeline import (
    _apply_batch_pbo,
    _apply_merge_pool_trial_penalty,
    _build_data_split_plan,
    _build_local_grid_tuning_candidates,
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
    _run_mining_round,
    _symbol_round_parallelism,
    MarketDataContext,
    run_pipeline,
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


def _hypothesis() -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="h1",
        hypothesis_family="momentum",
        economic_mechanism="Trend persistence",
        testable_prediction="Positive follow-through after momentum confirmation",
        null_hypothesis="No follow-through",
        expected_ic_range=(0.005, 0.02),
        expected_decay_halflife_bars=24,
    )


def _candidate(candidate_id: str, symbol: str) -> CandidateStrategySpec:
    return CandidateStrategySpec(
        candidate_id=candidate_id,
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol=symbol,
        market="um_futures",
        params={"signal_source": "factor_signal", "factor_family": "momentum"},
    )


def _context_for(candidate: CandidateStrategySpec) -> MarketDataContext:
    frame = _frame(100)
    return MarketDataContext(
        symbol=candidate.symbol,
        market=candidate.market,
        frame=frame,
        features_df=pd.DataFrame(index=frame.index),
        feature_meta={},
        forward_regimes=pd.Series(["unknown"] * len(frame), index=frame.index),
        funding_df=None,
        funding_rate=pd.Series([0.0] * len(frame), index=frame.index),
        data_quality_notes=[],
    )


def test_symbol_round_parallelism_splits_total_backtest_worker_budget() -> None:
    assert _symbol_round_parallelism(1, 8) == (1, 8)
    assert _symbol_round_parallelism(2, 8) == (2, 4)
    assert _symbol_round_parallelism(10, 4) == (4, 1)


def test_run_pipeline_executes_symbol_rounds_in_parallel(monkeypatch) -> None:
    candidates = [_candidate("c_btc", "BTCUSDT"), _candidate("c_eth", "ETHUSDT")]
    contexts = {_data_key: _context_for(candidate) for _data_key, candidate in ((pipeline._data_key(c), c) for c in candidates)}
    barrier = threading.Barrier(2)
    max_workers_seen: list[int | None] = []

    monkeypatch.setattr(pipeline, "build_v1_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(pipeline, "build_indicator_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(pipeline, "_load_data_contexts", lambda *args, **kwargs: contexts)

    def fake_round(*, current_candidates, max_workers, **kwargs):
        max_workers_seen.append(max_workers)
        barrier.wait(timeout=2.0)
        candidate = current_candidates[0]
        result = BacktestResult(
            experiment_id=f"exp-{candidate.symbol}",
            candidate_id=candidate.candidate_id,
            hypothesis_family=candidate.hypothesis_family,
            method_id=candidate.method_id,
            symbol=candidate.symbol,
            market=candidate.market,
            interval=candidate.interval,
            metrics_primary=MetricsBlock(sharpe=0.0),
            pbo=0.2,
        )
        return {
            "candidates": [candidate],
            "backtests": [result],
            "gatechecks": [],
            "hardscores": [],
            "factor_evidence": [],
            "research_gates": [],
            "near_misses": [],
            "new_candidates": [],
            "research_survivors": [],
            "detail_artifact_ids": [],
            "history_entry": {"symbol": candidate.symbol},
        }

    monkeypatch.setattr(pipeline, "_run_mining_round", fake_round)

    result = run_pipeline(
        Settings(data=DataConfig(symbols=["BTCUSDT", "ETHUSDT"])),
        use_llm=False,
        seed_hypotheses=[_hypothesis()],
        max_workers=2,
        archive_top=0,
    )

    assert {backtest.symbol for backtest in result.backtests} == {"BTCUSDT", "ETHUSDT"}
    assert max_workers_seen == [1, 1]


def test_split_round_controls_disable_repairs_and_stop_on_optimizer_convergence(monkeypatch) -> None:
    candidate = _candidate("c_btc", "BTCUSDT")
    contexts = {pipeline._data_key(candidate): _context_for(candidate)}
    calls: list[dict] = []

    monkeypatch.setattr(pipeline, "build_v1_candidates", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(pipeline, "build_indicator_candidates", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(pipeline, "_load_data_contexts", lambda *args, **kwargs: contexts)

    def fake_round(
        *,
        current_candidates,
        phase,
        allow_pre_gate_repair,
        allow_optimizer_repairs,
        allow_next_hypotheses,
        round_num,
        **kwargs,
    ):
        calls.append({
            "phase": phase,
            "allow_pre_gate_repair": allow_pre_gate_repair,
            "allow_optimizer_repairs": allow_optimizer_repairs,
            "allow_next_hypotheses": allow_next_hypotheses,
        })
        current = current_candidates[0]
        result = BacktestResult(
            experiment_id=f"exp-{round_num}",
            candidate_id=current.candidate_id,
            hypothesis_family=current.hypothesis_family,
            method_id=current.method_id,
            symbol=current.symbol,
            market=current.market,
            interval=current.interval,
            metrics_primary=MetricsBlock(sharpe=0.0),
            pbo=0.2,
        )
        next_candidate = current.model_copy(deep=True)
        next_candidate.candidate_id = f"c_next_{round_num}"
        next_candidate.parent_candidate_id = current.candidate_id
        next_candidate.params.update({
            "generated_by": "traditional_survivor_adjustment",
            "search_variant": "survivor_low_turnover",
            "parent_id": current.candidate_id,
            "smooth_span": 48,
            "signal_threshold": 0.25,
        })
        return {
            "candidates": [current],
            "backtests": [result],
            "gatechecks": [],
            "hardscores": [],
            "factor_evidence": [],
            "research_gates": [],
            "near_misses": [],
            "new_candidates": [next_candidate],
            "research_survivors": [],
            "detail_artifact_ids": [],
            "history_entry": {"phase": phase, "num_candidates": 1, "num_backtests": 1},
        }

    monkeypatch.setattr(pipeline, "_run_mining_round", fake_round)

    result = run_pipeline(
        Settings(data=DataConfig(symbols=["BTCUSDT"])),
        use_llm=False,
        seed_hypotheses=[_hypothesis()],
        archive_top=0,
        discovery_rounds=1,
        optimization_rounds=4,
    )

    assert [call["phase"] for call in calls] == ["discovery", "optimization", "optimization"]
    assert calls[0]["allow_pre_gate_repair"] is True
    assert calls[1]["allow_pre_gate_repair"] is False
    assert calls[1]["allow_optimizer_repairs"] is False
    assert calls[1]["allow_next_hypotheses"] is False
    assert result.optimization_history[-1]["converged"] is True
    assert result.total_rounds == 3


def test_mining_round_skips_discovery_pbo_and_uses_validation_pbo(monkeypatch) -> None:
    import factor_mining.hardscore as hardscore_module
    import factor_mining.optimizers.traditional_optimizer as optimizer_module
    import factor_mining.validation.gatecheck as gatecheck_module

    candidate = _candidate("c_btc", "BTCUSDT")
    frame = _frame(100)
    pbo_frame_lengths: list[int] = []

    def fake_build_tasks(candidates, frame_arg, *args, trial_counts_by_candidate=None, **kwargs):
        trial_counts_by_candidate = trial_counts_by_candidate or {}
        return [
            (
                np.ones(len(frame_arg)),
                item.model_dump(mode="json"),
                idx,
                trial_counts_by_candidate.get(item.candidate_id, {}),
                [],
            )
            for idx, item in enumerate(candidates)
        ]

    def fake_backtests(tasks, frame_arg, settings, max_workers, funding_df=None):
        results = []
        for _signal, candidate_dict, _idx, _trial_counts, _notes in tasks:
            item = CandidateStrategySpec.model_validate(candidate_dict)
            results.append(
                BacktestResult(
                    experiment_id=f"exp-{len(frame_arg)}-{item.candidate_id}",
                    candidate_id=item.candidate_id,
                    hypothesis_family=item.hypothesis_family,
                    method_id=item.method_id,
                    symbol=item.symbol,
                    market=item.market,
                    interval=item.interval,
                    metrics_primary=MetricsBlock(sharpe=0.1),
                    metrics_gross=MetricsBlock(sharpe=0.2),
                    pbo=None,
                )
            )
        return results

    def fake_apply_batch_pbo(frame_arg, tasks, results, settings, funding_df):
        pbo_frame_lengths.append(len(frame_arg))
        for result in results:
            result.pbo = 0.2

    def fake_evidence(*, results, **kwargs):
        return [
            FactorEvidenceReport(
                experiment_id=result.experiment_id,
                candidate_id=result.candidate_id,
                hypothesis_family=result.hypothesis_family,
                method_id=result.method_id,
                symbol=result.symbol,
                market=result.market,
                interval=result.interval,
                ic_by_horizon={"12": 0.02},
            )
            for result in results
        ]

    monkeypatch.setattr(pipeline, "_build_tasks", fake_build_tasks)
    monkeypatch.setattr(pipeline, "_run_backtests_parallel", fake_backtests)
    monkeypatch.setattr(pipeline, "_apply_batch_pbo", fake_apply_batch_pbo)
    monkeypatch.setattr(pipeline, "build_factor_evidence_reports", fake_evidence)
    monkeypatch.setattr(pipeline, "_build_pre_gate_repair_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "_build_local_grid_tuning_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        gatecheck_module,
        "apply_fdr",
        lambda results, settings: {result.experiment_id: 0.5 for result in results},
    )
    monkeypatch.setattr(
        gatecheck_module,
        "run_gatecheck",
        lambda result, *args, **kwargs: GateCheckResult(
            experiment_id=result.experiment_id,
            candidate_id=result.candidate_id,
            passed=True,
            items=[],
            risk_tier="full_pass",
        ),
    )
    monkeypatch.setattr(gatecheck_module, "apply_risk_stratified_gatechecks", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "apply_research_gate",
        lambda results, gatechecks, evidence: [
            ResearchGateResult(experiment_id=result.experiment_id, candidate_id=result.candidate_id, status="rejected")
            for result in results
        ],
    )
    monkeypatch.setattr(pipeline, "research_survivor_payloads", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "build_research_survivor_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "analyze_near_misses", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        hardscore_module,
        "hardscore",
        lambda result, gatecheck, **kwargs: HardScoreReport(
            experiment_id=result.experiment_id,
            score=0.0,
            haircut_sharpe=0.0,
            fdr_adjusted_pvalue=0.5,
            prior_posterior_ic_ratio=1.0,
            effective_trials_count=1,
            global_cumulative_trials_count=1,
        ),
    )
    monkeypatch.setattr(optimizer_module, "build_optimization_context", lambda *args, **kwargs: {"research_survivors": []})
    monkeypatch.setattr(optimizer_module, "optimize_traditionally", lambda *args, **kwargs: {"action": "hold"})
    monkeypatch.setattr(
        optimizer_module,
        "apply_optimization_result",
        lambda _optimization, candidates, _results, **_kwargs: (
            list(candidates),
            {
                "combinations_created": 0,
                "adjustments_applied": 0,
                "repairs_created": 0,
                "hypotheses_suggested": 0,
            },
        ),
    )
    monkeypatch.setattr(optimizer_module, "optimize_exits_traditionally", lambda *args, **kwargs: {"exit_adjustments": []})
    monkeypatch.setattr(optimizer_module, "apply_exit_adjustments", lambda _exit_opt, candidates, _settings: candidates)

    round_data = _run_mining_round(
        current_candidates=[candidate],
        frame=frame,
        features_df=pd.DataFrame(index=frame.index),
        feature_meta={},
        forward_regimes=pd.Series(["unknown"] * len(frame), index=frame.index),
        funding_rate=pd.Series([0.0] * len(frame), index=frame.index),
        funding_df=None,
        data_quality_notes=[],
        settings=Settings(data=DataConfig(symbols=["BTCUSDT"])),
        max_workers=1,
        store=None,
        iteration=0,
        round_num=1,
        cumulative_trial_counts={},
        run_args={},
    )

    assert pbo_frame_lengths == [20]
    assert round_data["backtests"][0].pbo == 0.2


def _signal_build_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
    pd.Series,
    pd.Series,
    list[CandidateStrategySpec],
]:
    frame = _frame(220)
    x = np.linspace(-2.0, 2.0, len(frame))
    features_df = pd.DataFrame(
        {
            "feature_alpha": np.sin(x),
            "feature_beta": np.cos(x),
        },
        index=frame.index,
    )
    feature_meta = {
        "feature_alpha": {"family": "momentum"},
        "feature_beta": {"family": "mean_reversion"},
    }
    regime_values = (["bull", "bear", "sideways", "high_vol"] * (len(frame) // 4 + 1))[:len(frame)]
    regimes = pd.Series(regime_values, index=frame.index)
    funding_rate = pd.Series(np.linspace(-2.0, 2.0, len(frame)), index=frame.index)
    factor_candidate = CandidateStrategySpec(
        candidate_id="signal-factor",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={
            "signal_source": "factor_signal",
            "factor_family": "momentum",
            "factor_lookback": 12,
            "direction": 1,
            "regime_filter": ["bull", "sideways"],
            "funding_state_filter": ["negative", "neutral"],
        },
    )
    factor_inverse = factor_candidate.model_copy(
        update={
            "candidate_id": "signal-factor-inverse",
            "params": {**factor_candidate.params, "direction": -1},
        }
    )
    feature_candidate = CandidateStrategySpec(
        candidate_id="signal-feature",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={
            "signal_source": "feature",
            "indicator_name": "feature_alpha",
            "direction": 1,
            "transform": "tanh_zscore",
            "zscore_window": 24,
            "tanh_scale": 1.5,
            "smooth_span": 2,
            "signal_threshold": 0.05,
        },
    )
    legacy_candidate = CandidateStrategySpec(
        candidate_id="signal-legacy",
        hypothesis_id="h1",
        method_id="parameter_sweep",
        hypothesis_family="mean_reversion",
        symbol="BTCUSDT",
        params={"regime_filter": ["bear", "sideways"]},
    )
    composite = CandidateStrategySpec(
        candidate_id="signal-composite",
        hypothesis_id="h1",
        method_id="composite",
        hypothesis_family="composite",
        symbol="BTCUSDT",
        params={
            "components": [
                factor_candidate.model_dump(mode="json"),
                feature_candidate.model_dump(mode="json"),
            ],
            "weights": [0.6, 0.4],
            "smooth_span": 2,
            "funding_trend_filter": ["rising", "stable"],
        },
    )
    return frame, features_df, feature_meta, regimes, funding_rate, [
        factor_candidate,
        factor_inverse,
        feature_candidate,
        legacy_candidate,
        composite,
    ]


def test_build_tasks_parallel_matches_serial_outputs() -> None:
    frame, features_df, feature_meta, regimes, funding_rate, candidates = _signal_build_inputs()
    trial_counts = {
        "signal-factor": {
            "effective_trials_count": 7,
            "global_cumulative_trials_count": 11,
        }
    }

    serial = _build_tasks(
        candidates,
        frame,
        features_df,
        feature_meta,
        regimes,
        funding_rate,
        trial_counts_by_candidate=trial_counts,
        max_workers=1,
    )
    parallel = _build_tasks(
        candidates,
        frame,
        features_df,
        feature_meta,
        regimes,
        funding_rate,
        trial_counts_by_candidate=trial_counts,
        max_workers=4,
    )

    assert [task[1]["candidate_id"] for task in parallel] == [task[1]["candidate_id"] for task in serial]
    for serial_task, parallel_task in zip(serial, parallel, strict=True):
        np.testing.assert_allclose(parallel_task[0], serial_task[0])
        assert parallel_task[1:] == serial_task[1:]


def test_build_tasks_caches_shared_factor_signal(monkeypatch) -> None:
    frame, features_df, feature_meta, regimes, funding_rate, _candidates = _signal_build_inputs()
    candidates = [
        _candidate(f"shared-factor-{idx}", "BTCUSDT").model_copy(
            update={
                "params": {
                    "signal_source": "factor_signal",
                    "factor_family": "momentum",
                    "factor_lookback": 12,
                    "direction": 1 if idx % 2 == 0 else -1,
                }
            }
        )
        for idx in range(8)
    ]
    calls: list[tuple[str, int]] = []
    original_factor_signal = pipeline.factor_signal

    def counted_factor_signal(frame_arg, *, family, lookback=12, funding_rate=None):
        calls.append((family, int(lookback)))
        return original_factor_signal(frame_arg, family=family, lookback=lookback, funding_rate=funding_rate)

    monkeypatch.setattr(pipeline, "factor_signal", counted_factor_signal)

    tasks = _build_tasks(
        candidates,
        frame,
        features_df,
        feature_meta,
        regimes,
        funding_rate,
        max_workers=4,
    )

    assert len(tasks) == len(candidates)
    assert calls == [("momentum", 12)]


def test_build_tasks_composite_reuses_component_factor_cache(monkeypatch) -> None:
    frame, features_df, feature_meta, regimes, funding_rate, _candidates = _signal_build_inputs()
    component_long = _candidate("component-long", "BTCUSDT").model_copy(
        update={
            "params": {
                "signal_source": "factor_signal",
                "factor_family": "momentum",
                "factor_lookback": 12,
                "direction": 1,
            }
        }
    )
    component_short = component_long.model_copy(
        update={
            "candidate_id": "component-short",
            "params": {**component_long.params, "direction": -1},
        }
    )
    composite = CandidateStrategySpec(
        candidate_id="composite-shared-components",
        hypothesis_id="h1",
        method_id="composite",
        hypothesis_family="composite",
        symbol="BTCUSDT",
        params={
            "components": [
                component_long.model_dump(mode="json"),
                component_short.model_dump(mode="json"),
            ],
            "weights": [0.5, -0.5],
        },
    )
    calls: list[tuple[str, int]] = []
    original_factor_signal = pipeline.factor_signal

    def counted_factor_signal(frame_arg, *, family, lookback=12, funding_rate=None):
        calls.append((family, int(lookback)))
        return original_factor_signal(frame_arg, family=family, lookback=lookback, funding_rate=funding_rate)

    monkeypatch.setattr(pipeline, "factor_signal", counted_factor_signal)

    tasks = _build_tasks(
        [composite],
        frame,
        features_df,
        feature_meta,
        regimes,
        funding_rate,
        max_workers=1,
    )

    assert len(tasks) == 1
    assert calls == [("momentum", 12)]
    assert float(np.abs(tasks[0][0]).sum()) > 0.0


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


def test_optimization_result_can_suppress_repairs_and_next_hypotheses() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_a",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    optimization = {
        "adjustments": [
            {"candidate_id": "c_a", "suggested": {"regime_filter": ["bull"]}},
        ],
        "next_hypotheses": [
            {"family": "Momentum", "mechanism": "trend persistence"},
        ],
    }

    new_candidates, summary = apply_optimization_result(
        optimization,
        [candidate],
        [],
        allow_repairs=False,
        allow_next_hypotheses=False,
    )

    assert new_candidates == []
    assert summary["repairs_suppressed"] == 1
    assert summary["hypotheses_suppressed"] == 1
    assert summary["repairs_created"] == 0
    assert summary["hypotheses_suggested"] == 0


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


def test_exit_optimizer_generates_bounded_variants_with_metadata() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_exit_opt",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    result = BacktestResult(
        experiment_id="exp-exit-opt",
        candidate_id="c_exit_opt",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.3, max_drawdown=-0.15, trade_count=120),
        metrics_gross=MetricsBlock(sharpe=1.0),
        avg_holding_period_bars=800,
    )
    gate = GateCheckResult(experiment_id="exp-exit-opt", passed=True, items=[])
    ctx = build_optimization_context([candidate], [result], [gate], iteration=0)

    optimization = optimize_exits_traditionally(ctx)
    candidates = apply_exit_adjustments(optimization, [candidate], Settings())
    generated = [item for item in candidates if item.params.get("generated_by") == "traditional_exit_adjustment"]

    assert len(optimization["exit_adjustments"]) == 3
    assert len(generated) == 3
    assert {item.params.get("exit_proposal_kind") for item in generated} == {"exit"}
    assert {item.params.get("exit_variant_key") for item in generated}


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
    assert summary["combinations_created"] >= 2
    combo = next(c for c in new_candidates if c.hypothesis_family == "composite")
    assert combo.params["search_variant"] == "survivor_combo_low_turnover"
    assert combo.params["smooth_span"] >= 24
    assert combo.params["signal_threshold"] >= 0.20
    assert combo.params["position_buffer"] >= 0.15
    assert any(c.params.get("weighting_scheme") == "inverse_turnover" for c in new_candidates if c.hypothesis_family == "composite")


def test_optimizer_generates_multiple_composite_subset_and_weight_variants() -> None:
    candidates = [
        CandidateStrategySpec(
            candidate_id=f"c_{idx}",
            hypothesis_id=f"h{idx}",
            method_id="factor_scoring",
            hypothesis_family=family,
            symbol="BTCUSDT",
            market="um_futures",
        )
        for idx, family in enumerate(["momentum", "mean_reversion", "volatility", "funding_basis", "volume_confirmation"], start=1)
    ]
    results = [
        BacktestResult(
            experiment_id=f"exp-{idx}",
            candidate_id=candidate.candidate_id,
            hypothesis_family=candidate.hypothesis_family,
            method_id=candidate.method_id,
            symbol=candidate.symbol,
            market=candidate.market,
            interval="5m",
            metrics_primary=MetricsBlock(sharpe=0.2 + idx * 0.05),
            metrics_gross=MetricsBlock(sharpe=0.7 + idx * 0.05),
            ic_tstat_nw=1.5 + idx * 0.2,
            factor_turnover=0.04 + idx * 0.03,
            break_even_cost_bps=10.0,
            actual_cost_bps=2.0,
            oos_trade_count=200,
        )
        for idx, candidate in enumerate(candidates, start=1)
    ]
    gates = [
        GateCheckResult(experiment_id=result.experiment_id, passed=True, items=[])
        for result in results
    ]

    ctx = build_optimization_context(candidates, results, gates, iteration=0)
    optimization = optimize_traditionally(ctx, "full")
    new_candidates, summary = apply_optimization_result(optimization, candidates, results)

    composites = [item for item in new_candidates if item.hypothesis_family == "composite"]
    assert len(optimization["combinations"]) > 1
    assert summary["combinations_created"] == len(composites)
    assert "top4" in {item.params.get("subset_strategy") for item in composites}
    assert "low_turnover8" in {item.params.get("subset_strategy") for item in composites}
    assert "inverse_turnover" in {item.params.get("weighting_scheme") for item in composites}
    assert any(item.params.get("subset_strategy") == "pair_grid" for item in composites)


def test_survivor_evolution_generates_bounded_variants() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_survivor",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
        params={"signal_source": "factor_signal", "factor_family": "momentum", "factor_lookback": 12},
    )
    result = BacktestResult(
        experiment_id="exp-survivor",
        candidate_id="c_survivor",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.5),
        metrics_gross=MetricsBlock(sharpe=0.7),
        factor_turnover=0.16,
        break_even_cost_bps=8.0,
        actual_cost_bps=2.0,
        regime_conditional_metrics={
            "bull": MetricsBlock(sharpe=1.2),
            "bear": MetricsBlock(sharpe=-0.1),
        },
    )
    gate = GateCheckResult(experiment_id="exp-survivor", passed=True, items=[])

    ctx = build_optimization_context([candidate], [result], [gate], iteration=0)
    optimization = optimize_traditionally(ctx, "full")
    new_candidates, summary = apply_optimization_result(optimization, [candidate], [result])

    evolved = [item for item in new_candidates if item.params.get("generated_by") == "traditional_survivor_evolution"]
    assert summary["evolutions_created"] == 3
    assert {item.params["search_variant"] for item in evolved} == {
        "survivor_evolve_low_turnover",
        "survivor_evolve_lookback",
        "survivor_evolve_regime_filter",
    }
    assert all(item.params["optimizer_proposal_id"] for item in evolved)
    assert all(item.params["optimizer_root_parent_id"] == "c_survivor" for item in evolved)
    assert next(item for item in evolved if item.params["search_variant"] == "survivor_evolve_lookback").params["factor_lookback"] == 24
    assert next(item for item in evolved if item.params["search_variant"] == "survivor_evolve_regime_filter").params["regime_filter"] == ["bull"]


def test_optimizer_lineage_context_computes_outcome_delta() -> None:
    parent = CandidateStrategySpec(
        candidate_id="c_parent",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    parent_result = BacktestResult(
        experiment_id="exp-parent",
        candidate_id="c_parent",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.6, max_drawdown=-0.08),
        metrics_gross=MetricsBlock(sharpe=0.7),
        factor_turnover=0.20,
        break_even_cost_bps=8.0,
        actual_cost_bps=2.0,
    )
    optimization = {
        "adjustments": [{
            "candidate_id": "c_parent",
            "param": "evolution_params",
            "suggested": {"smooth_span": 48, "signal_threshold": 0.25, "position_buffer": 0.20},
            "proposal_kind": "evolution",
            "variant_key": "survivor_evolve_low_turnover",
            "rationale": "test lineage",
        }]
    }
    generated, _summary = apply_optimization_result(optimization, [parent], [parent_result])
    child = generated[0]
    child_result = BacktestResult(
        experiment_id="exp-child",
        candidate_id=child.candidate_id,
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.4, max_drawdown=-0.10),
        metrics_gross=MetricsBlock(sharpe=0.5),
        factor_turnover=0.22,
        break_even_cost_bps=6.0,
        actual_cost_bps=2.0,
    )
    ctx = build_optimization_context(
        [child],
        [child_result],
        [GateCheckResult(experiment_id="exp-child", passed=False, items=[])],
        iteration=1,
    )

    outcome = ctx["optimizer_outcomes"][0]
    assert outcome["status"] == "failed"
    assert round(outcome["delta_sharpe"], 6) == -0.2
    assert round(outcome["delta_turnover"], 6) == 0.02
    assert ctx["optimizer_outcome_counts"] == {"failed": 1}


def test_optimizer_memory_skips_failed_duplicate_proposal() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_memory",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    result = BacktestResult(
        experiment_id="exp-memory",
        candidate_id="c_memory",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.5),
        metrics_gross=MetricsBlock(sharpe=0.6),
        factor_turnover=0.18,
    )
    gate = GateCheckResult(experiment_id="exp-memory", passed=True, items=[])
    first_ctx = build_optimization_context([candidate], [result], [gate], iteration=0)
    first = optimize_traditionally(first_ctx, "full")
    failed_signature = next(
        item["proposal_signature"]
        for item in first["adjustments"]
        if item.get("param") == "evolution_params"
    )

    second_ctx = build_optimization_context(
        [candidate],
        [result],
        [gate],
        iteration=1,
        previous_actions=[{
            "optimizer_outcomes": [{
                "status": "failed",
                "proposal_signature": failed_signature,
            }]
        }],
    )
    second = optimize_traditionally(second_ctx, "full")

    assert not any(item.get("param") == "evolution_params" for item in second["adjustments"])
    assert second["proposal_counts"]["memory_skipped"] >= 1


def test_optimizer_hill_climbs_from_improved_turnover_proposal() -> None:
    parent = CandidateStrategySpec(
        candidate_id="c_hill_parent",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    parent_result = BacktestResult(
        experiment_id="exp-hill-parent",
        candidate_id="c_hill_parent",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.5),
        metrics_gross=MetricsBlock(sharpe=0.7),
        factor_turnover=0.20,
    )
    generated, _summary = apply_optimization_result(
        {
            "adjustments": [{
                "candidate_id": "c_hill_parent",
                "param": "evolution_params",
                "suggested": {"smooth_span": 24, "signal_threshold": 0.20, "position_buffer": 0.15},
                "proposal_kind": "evolution",
                "variant_key": "survivor_evolve_low_turnover",
            }]
        },
        [parent],
        [parent_result],
    )
    child = generated[0]
    child_result = BacktestResult(
        experiment_id="exp-hill-child",
        candidate_id=child.candidate_id,
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.5),
        metrics_gross=MetricsBlock(sharpe=0.7),
        factor_turnover=0.12,
    )
    ctx = build_optimization_context(
        [child],
        [child_result],
        [GateCheckResult(experiment_id="exp-hill-child", passed=True, items=[])],
        iteration=1,
    )

    optimization = optimize_traditionally(ctx, "full")
    hill = next(item for item in optimization["adjustments"] if item.get("proposal_kind") == "hill_climb")

    assert hill["suggested"]["smooth_span"] == 48
    assert hill["suggested"]["signal_threshold"] == 0.30
    assert hill["suggested"]["position_buffer"] == 0.25
    assert optimization["proposal_counts"]["hill_climb"] == 1


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
    assert summary["adjustments_applied"] == 2
    survivor_low_turnover = next(item for item in new_candidates if item.params["search_variant"] == "survivor_low_turnover")
    assert survivor_low_turnover.params["smooth_span"] == 48
    assert survivor_low_turnover.params["signal_threshold"] == 0.25
    assert survivor_low_turnover.params["position_buffer"] == 0.20
    assert summary["evolutions_created"] == 1


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


def test_local_grid_tuning_generates_bounded_validation_candidates() -> None:
    candidate = CandidateStrategySpec(
        candidate_id="c_grid_parent",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
        params={
            "signal_source": "factor_signal",
            "factor_family": "momentum",
            "factor_lookback": 12,
            "smooth_span": 1,
            "signal_threshold": 0.0,
            "position_buffer": 0.05,
        },
    )
    result = BacktestResult(
        experiment_id="exp-grid-parent",
        candidate_id="c_grid_parent",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=-0.1),
        metrics_gross=MetricsBlock(sharpe=0.9),
        factor_turnover=0.22,
        break_even_cost_bps=1.0,
        actual_cost_bps=3.0,
    )
    evidence = FactorEvidenceReport(
        experiment_id="exp-grid-parent",
        candidate_id="c_grid_parent",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        best_horizon_bars=48,
        ic_by_horizon={"12": 0.014, "48": 0.025},
        rankic_by_horizon={"48": 0.018},
    )

    candidates = _build_local_grid_tuning_candidates(
        [candidate],
        [result],
        [evidence],
        parent_limit=1,
        max_per_parent=64,
        total_limit=64,
    )

    assert 10 < len(candidates) <= 64
    assert all(child.candidate_type == "repair" for child in candidates)
    assert all(child.params["generated_by"] == "local_grid_tuning" for child in candidates)
    assert all(child.params["parent_id"] == "c_grid_parent" for child in candidates)
    assert all(child.params["optimizer_param_diff"] for child in candidates)
    assert all(child.params["parent_validation_baseline"]["evidence_best_horizon_bars"] == 48 for child in candidates)
    assert 48 in {child.params.get("factor_lookback") for child in candidates}
    assert len({child.params.get("smooth_span") for child in candidates}) > 1
    assert len({child.params.get("signal_threshold") for child in candidates}) > 1


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


def test_local_grid_tuning_merge_keeps_top_validation_variants_despite_high_signal_corr() -> None:
    parent = CandidateStrategySpec(
        candidate_id="c_parent_grid",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"factor_family": "momentum"},
    )
    variants = []
    for idx, sharpe in enumerate([0.5, 0.8, 0.7, 0.6], start=1):
        child = parent.model_copy(deep=True)
        child.candidate_id = f"c_grid_{idx}"
        child.params.update({
            "parent_id": "c_parent_grid",
            "generated_by": "local_grid_tuning",
            "search_variant": "local_grid",
            "smooth_span": 12 * idx,
        })
        variants.append((child, sharpe))

    parent_signal = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    tasks = [(parent_signal, parent.model_dump(mode="json"), 0, {}, [])]
    tasks.extend(
        (parent_signal.copy(), child.model_dump(mode="json"), idx, {}, [])
        for idx, (child, _sharpe) in enumerate(variants, start=1)
    )
    results = [_result("exp-parent-grid", "c_parent_grid", pbo=0.2, sharpe=0.4)]
    results.extend(
        _result(f"exp-{child.candidate_id}", child.candidate_id, pbo=0.2, sharpe=sharpe)
        for child, sharpe in variants
    )

    plan = _select_repair_merge_pool(
        original_candidates=[parent],
        repair_candidates=[child for child, _sharpe in variants],
        validation_candidates=[parent] + [child for child, _sharpe in variants],
        validation_full_tasks=tasks,
        validation_tasks=tasks,
        validation_results=results,
    )

    kept_ids = {candidate.candidate_id for candidate in plan.candidates}
    assert kept_ids == {"c_parent_grid", "c_grid_2", "c_grid_3", "c_grid_4"}
    assert plan.merged_repairs == 3
    assert plan.rejected_repairs == 1
    assert variants[1][0].params["parent_signal_correlation"] > 0.99
    assert variants[1][0].params["merge_pool_score"] == variants[1][0].params["validation_selection_score"]
    rejected = next(child for child, _sharpe in variants if child.candidate_id == "c_grid_1")
    assert rejected.params["merge_pool_reasons"] == ["local_tuning_parent_ratio_cap"]


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
