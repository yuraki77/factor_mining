"""Full pipeline orchestrator: DeepSeek → backtest → gatecheck → hardscore → optimize → archive.

Supports iterative optimization: MiniMax suggestions are backtested in subsequent rounds
until convergence or max_iterations.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable
import uuid

import numpy as np
import pandas as pd

from factor_mining.config import Settings
from factor_mining.backtest.engine import evaluate_strategy_path
from factor_mining.data.loader import funding_event_zscore_to_frame, load_frame, load_funding, load_supplemental_features
from factor_mining.evidence import build_factor_evidence_reports, funding_state_labels, funding_trend_labels
from factor_mining.data.quality import interval_to_ms, kline_quality_notes
from factor_mining.factors.engineering import generate_features
from factor_mining.factors.returns import forward_returns
from factor_mining.hypotheses.discovered import should_continue_mining
from factor_mining.mining import build_indicator_candidates, build_v1_candidates, default_hypotheses, factor_signal, generate_hypotheses_with_deepseek, normalize_family
from factor_mining.near_miss import analyze_near_misses
from factor_mining.research_gate import apply_research_gate, build_research_survivor_records, research_survivor_payloads
from factor_mining.models import (
    BacktestResult,
    CandidateStrategySpec,
    DataQualityNote,
    FactorEvidenceReport,
    GateCheckResult,
    HardScoreReport,
    HypothesisSpec,
    NearMissAnalysis,
    ResearchGateResult,
    ResearchSurvivorRecord,
    TrialRecord,
)
from factor_mining.registry import METHOD_REGISTRY
from factor_mining.stats.metrics import annualization_factor, combined_ic_tstat_pvalue, deflated_sharpe_ratio, sharpe_ratio
from factor_mining.storage import MetadataStore
from factor_mining.trial_ledger import TrialLedger


@dataclass
class PipelineResult:
    """Structured result from a pipeline run (one or more mining rounds)."""

    hypotheses: list[HypothesisSpec] = field(default_factory=list)
    candidates: list[CandidateStrategySpec] = field(default_factory=list)
    backtests: list[BacktestResult] = field(default_factory=list)
    gatechecks: list[GateCheckResult] = field(default_factory=list)
    hardscores: list[HardScoreReport] = field(default_factory=list)
    factor_evidence: list[FactorEvidenceReport] = field(default_factory=list)
    research_gates: list[ResearchGateResult] = field(default_factory=list)
    near_misses: list[NearMissAnalysis] = field(default_factory=list)
    optimization_history: list[dict] = field(default_factory=list)
    n_gatecheck_passed: int = 0
    total_rounds: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def top_candidates(self) -> list[tuple[CandidateStrategySpec, BacktestResult, HardScoreReport]]:
        """Return (candidate, result, score) triples sorted by score descending."""
        scored = []
        for c, r, s in zip(self.candidates, self.backtests, self.hardscores):
            if s.score > 0:
                scored.append((c, r, s))
        scored.sort(key=lambda x: x[2].score, reverse=True)
        return scored

    @property
    def last_optimization(self) -> dict:
        if self.optimization_history:
            return self.optimization_history[-1].get("optimization", {})
        return {}


@dataclass
class MarketDataContext:
    symbol: str
    market: str
    frame: pd.DataFrame
    features_df: pd.DataFrame
    feature_meta: dict
    forward_regimes: pd.Series
    funding_df: pd.DataFrame | None
    funding_rate: pd.Series
    data_quality_notes: list[DataQualityNote] = field(default_factory=list)


@dataclass(frozen=True)
class DataSplitPlan:
    discovery_mask: pd.Series
    repair_validation_mask: pd.Series
    repair_mask: pd.Series
    final_oos_mask: pd.Series
    repair_validation_start_idx: int
    final_oos_start_idx: int

    @property
    def final_oos_start_time(self) -> int | None:
        if not bool(self.final_oos_mask.any()):
            return None
        return int(self.final_oos_mask[self.final_oos_mask].index[0])


@dataclass
class RepairMergePlan:
    candidates: list[CandidateStrategySpec]
    full_tasks: list[tuple]
    validation_results: list[BacktestResult]
    merged_repairs: int = 0
    rejected_repairs: int = 0
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


# ── multiprocessing worker state ────────────────────────────────────

_worker_frame: pd.DataFrame | None = None
_worker_settings: Settings | None = None
_worker_funding: pd.DataFrame | None = None
_EVENT_SINK: Callable[[str, str, str, dict[str, Any] | None], None] | None = None

_PRE_GATE_REPAIR_LIMIT = 96
_PRE_GATE_REPAIR_MAX_PER_PARENT = 4
_PRE_GATE_MIN_ABS_IC = 0.01
_PRE_GATE_MIN_CONDITIONAL_IC = 0.015
_DISCOVERY_FRACTION = 0.60
_REPAIR_VALIDATION_FRACTION = 0.20
_FINAL_OOS_FRACTION = 0.20
_MAX_FINAL_COMPLEXITY = 4
_REPAIR_MAX_PBO = 0.60
_REPAIR_MAX_PARENT_CORR = 0.98
_DETAIL_ARTIFACT_LIMIT_PER_SCOPE = 96
_DETAIL_BUCKET_LIMIT = 24


def _init_worker(frame: pd.DataFrame, settings: Settings, funding_df: pd.DataFrame | None = None) -> None:
    global _worker_frame, _worker_settings, _worker_funding
    _worker_frame = frame
    _worker_settings = settings
    _worker_funding = funding_df


def _run_one_backtest(args: tuple) -> BacktestResult | Exception:
    global _worker_frame, _worker_settings
    signal_arr, candidate_dict, trial_counts, data_quality_note_dicts = args
    from factor_mining.backtest.engine import run_backtest
    from factor_mining.models import CandidateStrategySpec, DataQualityNote

    candidate = CandidateStrategySpec.model_validate(candidate_dict)
    signal = pd.Series(signal_arr, index=_worker_frame.index)
    data_quality_notes = [DataQualityNote.model_validate(item) for item in data_quality_note_dicts]
    try:
        return run_backtest(
            _worker_frame,
            signal,
            candidate,
            _worker_settings,
            trial_counts=trial_counts,
            data_quality_notes=data_quality_notes,
            funding=_worker_funding,
        )
    except Exception as exc:
        return exc


# ── main pipeline ───────────────────────────────────────────────────


def _normalize_direction_scope(direction_scope: dict[str, Any] | None) -> dict[str, Any] | None:
    if not direction_scope:
        return None

    raw_symbols = direction_scope.get("symbols") or direction_scope.get("universe_symbols") or []
    symbols = [str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()]

    raw_factor_ids = direction_scope.get("factor_ids") or []
    factor_ids = [str(factor_id).strip() for factor_id in raw_factor_ids if str(factor_id).strip()]

    brief = str(direction_scope.get("brief") or direction_scope.get("research_brief") or "").strip()
    objective = str(direction_scope.get("objective") or "").strip().upper()

    normalized: dict[str, Any] = {}
    if symbols:
        normalized["symbols"] = symbols
    if factor_ids:
        normalized["factor_ids"] = factor_ids
    if brief:
        normalized["brief"] = brief
    if objective:
        normalized["objective"] = objective
    return normalized or None


def _settings_for_direction_scope(settings: Settings, direction_scope: dict[str, Any] | None) -> Settings:
    if not direction_scope or not direction_scope.get("symbols"):
        return settings
    return settings.model_copy(
        update={
            "data": settings.data.model_copy(update={"symbols": list(direction_scope["symbols"])}),
        },
    )


def _research_brief_for_direction_scope(
    research_brief: str | None,
    direction_scope: dict[str, Any] | None,
) -> str | None:
    if research_brief:
        return research_brief
    if not direction_scope:
        return None
    pieces: list[str] = []
    brief = str(direction_scope.get("brief") or "").strip()
    if brief:
        pieces.append(brief)
    factor_ids = direction_scope.get("factor_ids") or []
    if factor_ids:
        pieces.append("Prioritize forkable Lab factors: " + ", ".join(str(item) for item in factor_ids))
    symbols = direction_scope.get("symbols") or []
    if symbols:
        pieces.append("Limit the research universe to: " + ", ".join(str(item) for item in symbols))
    objective = str(direction_scope.get("objective") or "").strip()
    if objective:
        pieces.append(f"Primary Lab objective: {objective}")
    return "\n".join(pieces) if pieces else None


def _annotate_candidates_for_direction_scope(
    candidates: list[CandidateStrategySpec],
    direction_scope: dict[str, Any] | None,
) -> list[CandidateStrategySpec]:
    if not direction_scope:
        return candidates

    annotation = {
        "lab_direction_factor_ids": list(direction_scope.get("factor_ids") or []),
        "lab_direction_symbols": list(direction_scope.get("symbols") or []),
        "lab_direction_objective": str(direction_scope.get("objective") or ""),
    }
    return [
        candidate.model_copy(update={"params": {**candidate.params, **annotation}})
        for candidate in candidates
    ]


def run_pipeline(
    settings: Settings,
    *,
    use_llm: bool = True,
    max_workers: int | None = None,
    tail: int | None = None,
    archive_top: int = 3,
    research_brief: str | None = None,
    hypothesis_count: int = 5,
    iterations: int = 1,
    store: MetadataStore | None = None,
    event_sink: Callable[[str, str, str, dict[str, Any] | None], None] | None = None,
    seed_hypotheses: list[HypothesisSpec] | None = None,
    direction_scope: dict[str, Any] | None = None,
    stop_event: threading.Event | None = None,
) -> PipelineResult:
    """Execute the full factor mining workflow with optional iterative optimization.

    Steps:
      1. Hypothesis generation (DeepSeek or defaults)
      2. Build initial candidates + load data + generate features
      3-6. Mining round(s): backtest → gatecheck → hardscore → optimize
         Each round backtests new candidates from the previous round's MiniMax output.

    Args:
        iterations: Maximum number of mining rounds (1 = single pass, >1 = iterative).
    """
    global _EVENT_SINK
    previous_sink = _EVENT_SINK
    _EVENT_SINK = event_sink
    t_start = time.perf_counter()
    result = PipelineResult()
    normalized_scope = _normalize_direction_scope(direction_scope)
    effective_settings = _settings_for_direction_scope(settings, normalized_scope)
    effective_research_brief = _research_brief_for_direction_scope(research_brief, normalized_scope)
    try:
        return _run_pipeline_impl(
            effective_settings,
            use_llm=use_llm,
            max_workers=max_workers,
            tail=tail,
            archive_top=archive_top,
            research_brief=effective_research_brief,
            hypothesis_count=hypothesis_count,
            iterations=iterations,
            store=store,
            result=result,
            t_start=t_start,
            seed_hypotheses=seed_hypotheses,
            direction_scope=normalized_scope,
            stop_event=stop_event,
        )
    finally:
        _EVENT_SINK = previous_sink


def _run_pipeline_impl(
    settings: Settings,
    *,
    use_llm: bool,
    max_workers: int | None,
    tail: int | None,
    archive_top: int,
    research_brief: str | None,
    hypothesis_count: int,
    iterations: int,
    store: MetadataStore | None,
    result: PipelineResult,
    t_start: float,
    seed_hypotheses: list[HypothesisSpec] | None,
    direction_scope: dict[str, Any] | None,
    stop_event: threading.Event | None = None,
) -> PipelineResult:
    active_survivor_records = store.list_research_survivors(status="active") if store else []
    survivor_seed_candidates = _survivor_seed_candidates(active_survivor_records, result.errors)
    survivor_candidate_ids = {candidate.candidate_id for candidate in survivor_seed_candidates}
    if survivor_seed_candidates:
        _step_header(0, "Loading active Research Survivors for recheck")
        _log(
            f"Queued {len(survivor_seed_candidates)} active survivors for first-round OOS re-evaluation "
            "before new optimizer mutations"
        )
        for record in active_survivor_records[:3]:
            _log(
                f"  survivor {record.candidate_id[:16]}... "
                f"trades={record.current_trades}, "
                f"need={record.required_additional_trades}, "
                f"trigger={record.recheck_trigger}"
            )

    # ── Step 1: Hypotheses ──────────────────────────────────────────
    if seed_hypotheses is not None:
        header = "Using seeded hypotheses"
    else:
        header = "Generating hypotheses via DeepSeek" if use_llm else "Loading default hypotheses"
    _step_header(1, header)
    t0 = time.perf_counter()

    if seed_hypotheses is not None:
        result.hypotheses = list(seed_hypotheses)
        _log(f"Using {len(result.hypotheses)} seeded hypotheses")
    elif use_llm:
        try:
            result.hypotheses = generate_hypotheses_with_deepseek(
                settings, count=hypothesis_count, research_brief=research_brief,
            )
            _log(f"DeepSeek generated {len(result.hypotheses)} hypotheses in {time.perf_counter() - t0:.0f}s")
        except Exception as exc:
            _log(f"DeepSeek failed: {exc}")
            _log("Falling back to default + discovered hypotheses")
            result.hypotheses = default_hypotheses()
            result.errors.append(f"step1_deepseek: {exc}")
    else:
        result.hypotheses = default_hypotheses()

    _log(f"Using {len(result.hypotheses)} hypotheses")
    for h in result.hypotheses:
        mechanism = h.economic_mechanism[:100] + "..." if len(h.economic_mechanism) > 100 else h.economic_mechanism
        _log(f"  {h.hypothesis_id}: [{h.hypothesis_family}] {mechanism}")

    if store:
        store.save_artifact("latest_hypotheses", "hypotheses", {
            "items": [h.model_dump(mode="json") for h in result.hypotheses],
        })

    # ── Step 2: Candidates + Data (once) ────────────────────────────
    _step_header(2, "Building candidates and loading data")
    t0 = time.perf_counter()

    initial_candidates = build_v1_candidates(
        result.hypotheses, symbols=settings.data.symbols, interval=settings.data.default_interval,
    )
    initial_candidates = _annotate_candidates_for_direction_scope(initial_candidates, direction_scope)
    if survivor_seed_candidates:
        initial_candidates = _dedupe_candidates(survivor_seed_candidates + initial_candidates)
    _log(f"{len(initial_candidates)} initial candidates ({len(result.hypotheses)} hypotheses × {len(settings.data.symbols)} symbols × methods)")

    data_contexts = _load_data_contexts(initial_candidates, settings, tail=tail)
    survivor_seed_candidates = [
        c for c in survivor_seed_candidates
        if _data_key(c) in data_contexts
    ]
    initial_candidates = [
        c for c in initial_candidates
        if _data_key(c) in data_contexts
    ]
    survivor_candidate_ids = {
        candidate.candidate_id
        for candidate in initial_candidates
        if candidate.candidate_id in survivor_candidate_ids
    }
    _log(f"Runnable legacy candidates: {len(initial_candidates)}")

    # Expand using explicit indicator parameterization (primary path)
    any_context = next(iter(data_contexts.values()))
    indicator_candidates = build_indicator_candidates(
        result.hypotheses,
        symbols=settings.data.symbols,
        feature_meta=any_context.feature_meta,
        interval=settings.data.default_interval,
    )
    indicator_candidates = _annotate_candidates_for_direction_scope(indicator_candidates, direction_scope)
    indicator_candidates = [
        c for c in indicator_candidates
        if _data_key(c) in data_contexts
    ]
    _log(f"Indicator candidates: {len(indicator_candidates)} (replaces {len(initial_candidates)} legacy)")
    initial_candidates = _dedupe_candidates(survivor_seed_candidates + indicator_candidates) if survivor_seed_candidates else indicator_candidates
    survivor_candidate_ids = {
        candidate.candidate_id
        for candidate in initial_candidates
        if candidate.candidate_id in survivor_candidate_ids
    }
    if survivor_candidate_ids:
        _log(f"Active Research Survivor rechecks in round 1: {len(survivor_candidate_ids)}")

    if store:
        store.save_artifact("initial_candidates", "candidates", {
            "items": [c.model_dump(mode="json") for c in initial_candidates],
        })

    _log(f"Step 2 done in {time.perf_counter() - t0:.0f}s")

    # ── Iterative mining rounds ─────────────────────────────────────
    current_candidates = initial_candidates
    all_backtests: list[BacktestResult] = []
    all_gatechecks: list[GateCheckResult] = []
    all_hardscores: list[HardScoreReport] = []
    all_factor_evidence: list[FactorEvidenceReport] = []
    all_research_gates: list[ResearchGateResult] = []
    all_near_misses: list[NearMissAnalysis] = []
    all_retained_candidates: list[CandidateStrategySpec] = []
    all_research_survivors: list[dict[str, Any]] = []
    all_detail_artifact_ids: list[str] = []
    cumulative_trial_counts: dict[str, int] = {}
    round_num = 0

    for round_num in range(1, iterations + 1):
        _step_header(2 + round_num, f"Mining round {round_num}/{iterations} — {len(current_candidates)} candidates")
        _check_stop(stop_event)

        round_backtests: list[BacktestResult] = []
        round_gatechecks: list[GateCheckResult] = []
        round_hardscores: list[HardScoreReport] = []
        round_candidates: list[CandidateStrategySpec] = []
        new_candidates: list[CandidateStrategySpec] = []
        child_history: list[dict[str, Any]] = []

        for key, symbol_candidates in _group_candidates_by_data(current_candidates).items():
            _check_stop(stop_event)
            context = data_contexts.get(key)
            if context is None:
                _log(f"  Skip {key[0]}/{key[1]}: no local parquet data")
                continue
            _log(f"  {context.symbol}/{context.market}: {len(symbol_candidates)} candidates")
            round_data = _run_mining_round(
                current_candidates=symbol_candidates,
                frame=context.frame,
                features_df=context.features_df,
                feature_meta=context.feature_meta,
                forward_regimes=context.forward_regimes,
                funding_df=context.funding_df,
                funding_rate=context.funding_rate,
                data_quality_notes=context.data_quality_notes,
                settings=settings,
                max_workers=max_workers,
                store=store,
                iteration=round_num - 1,
                round_num=round_num,
                artifact_scope=f"round{round_num}_{context.symbol}_{context.market}",
                cumulative_trial_counts=cumulative_trial_counts,
                survivor_candidate_ids=survivor_candidate_ids,
                previous_actions=list(result.optimization_history),
                stop_event=stop_event,
            )

            round_backtests.extend(round_data["backtests"])
            round_gatechecks.extend(round_data["gatechecks"])
            round_hardscores.extend(round_data["hardscores"])
            all_factor_evidence.extend(round_data["factor_evidence"])
            all_research_gates.extend(round_data["research_gates"])
            all_near_misses.extend(round_data["near_misses"])
            round_candidates.extend(round_data["candidates"])
            new_candidates.extend(round_data["new_candidates"])
            all_research_survivors.extend(round_data.get("research_survivors", []))
            all_detail_artifact_ids.extend(round_data.get("detail_artifact_ids", []))
            if round_data["history_entry"]:
                child_history.append(round_data["history_entry"])

        all_backtests.extend(round_backtests)
        all_gatechecks.extend(round_gatechecks)
        all_hardscores.extend(round_hardscores)
        all_retained_candidates.extend(round_candidates)
        result.optimization_history.append(_combine_round_history(round_num, round_num - 1, child_history, new_candidates))

        if not new_candidates and round_num < iterations:
            _log("No new candidates from optimization — stopping early.")
            break

        # Check boundary conditions for continued mining
        if iterations > 1:
            new_candidates = _filter_candidates_by_mining_boundaries(
                new_candidates, cumulative_trial_counts, round_backtests, result.hypotheses, log_blocks=True,
            )
            if not new_candidates:
                _log("Boundary conditions triggered — stopping early.")
                break

        current_candidates = new_candidates

    # ── Assemble final result ───────────────────────────────────────
    result.candidates = all_retained_candidates
    result.backtests = all_backtests
    result.gatechecks = all_gatechecks
    result.hardscores = all_hardscores
    result.factor_evidence = all_factor_evidence
    result.research_gates = all_research_gates
    result.near_misses = all_near_misses
    result.n_gatecheck_passed = sum(1 for g in all_gatechecks if g.passed)
    result.total_rounds = round_num

    if store:
        store.save_artifact("latest_candidates", "candidates", {
            "items": [c.model_dump(mode="json") for c in result.candidates],
        })
        store.save_artifact("latest_backtests", "backtests", {
            "items": [r.model_dump(mode="json") for r in result.backtests],
        })
        store.save_artifact("latest_gatechecks", "gatechecks", {
            "items": [g.model_dump(mode="json") for g in result.gatechecks],
        })
        store.save_artifact("latest_gatecheck_diagnostics", "gatecheck_diagnostics", _gatecheck_diagnostics(
            result.candidates,
            result.backtests,
            result.gatechecks,
            settings,
        ))
        store.save_artifact("latest_hardscores", "hardscores", {
            "items": [s.model_dump(mode="json") for s in result.hardscores],
        })
        store.save_artifact("latest_factor_evidence", "factor_evidence", {
            "items": [report.model_dump(mode="json") for report in result.factor_evidence],
        })
        store.save_artifact("latest_research_gate", "research_gate", {
            "items": [gate.model_dump(mode="json") for gate in result.research_gates],
        })
        store.save_artifact("latest_near_misses", "near_misses", {
            "items": [item.model_dump(mode="json") for item in result.near_misses],
        })
        store.save_artifact("latest_research_survivors", "research_survivors", {
            "items": all_research_survivors,
        })
        store.save_artifact("latest_research_survivor_store", "research_survivor_store", {
            "items": [record.model_dump(mode="json") for record in store.list_research_survivors(status=None)],
        })
        store.save_artifact("latest_optimization_history", "optimization_history", {
            "items": result.optimization_history,
        })
        store.save_artifact("latest_detail_index", "detail_index", {
            "artifact_ids": all_detail_artifact_ids,
        })
        if all_detail_artifact_ids:
            pruned = store.prune_artifacts(
                kind="experiment_detail",
                keep_artifact_ids=set(all_detail_artifact_ids),
                max_unprotected_rows=0,
            )
            if pruned:
                _log(f"Pruned {pruned} stale experiment detail artifacts")

    # Archive top experiments across all rounds
    archived = 0
    if archive_top > 0:
        archived = _archive_top(result, settings, archive_top)

    result.elapsed_s = time.perf_counter() - t_start
    _separator()
    _log(f"PIPELINE COMPLETE in {result.elapsed_s:.0f}s")
    _log(f"  Rounds:       {result.total_rounds}")
    _log(f"  Hypotheses:   {len(result.hypotheses)}")
    _log(f"  Backtests:    {len(result.backtests)}")
    _log(f"  GateCheck OK: {result.n_gatecheck_passed}/{len(result.gatechecks)}")
    _log(f"  HardScore >0: {sum(1 for s in result.hardscores if s.score > 0)}")
    _log(f"  Archived:     {archived}")
    if result.optimization_history:
        combos = result.last_optimization.get("combinations", [])
        _log(f"  MiniMax combos: {len(combos)}")
        for combo in combos[:3]:
            _log(f"    {combo.get('factor_ids', [])} weights={combo.get('weights', [])}")
    if result.errors:
        _log(f"  Errors:       {len(result.errors)}")
        for err in result.errors:
            _log(f"    - {err}")

    return result


def _check_stop(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise _PipelineCancelled("Stop requested from dashboard")


class _PipelineCancelled(BaseException):
    pass


def _run_mining_round(
    *,
    current_candidates: list[CandidateStrategySpec],
    frame: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_meta: dict,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None,
    funding_df: pd.DataFrame | None,
    data_quality_notes: list[DataQualityNote],
    settings: Settings,
    max_workers: int | None,
    store: MetadataStore | None,
    iteration: int,
    round_num: int,
    cumulative_trial_counts: dict[str, int],
    artifact_scope: str | None = None,
    survivor_candidate_ids: set[str] | None = None,
    previous_actions: list[dict] | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Execute one complete mining round: backtest → gatecheck → hardscore → optimize."""
    artifact_scope = artifact_scope or f"round{round_num}"
    _check_stop(stop_event)
    survivor_candidate_ids = survivor_candidate_ids or set()

    # ── Backtest ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    split_plan = _build_data_split_plan(frame, regimes=forward_regimes)
    discovery_frame = _masked_frame(frame, split_plan.discovery_mask)
    repair_validation_frame = _masked_frame(frame, split_plan.repair_validation_mask)
    final_frame = _masked_frame(frame, split_plan.final_oos_mask)
    discovery_regimes = _masked_series(forward_regimes, split_plan.discovery_mask)
    final_regimes = _masked_series(forward_regimes, split_plan.final_oos_mask)
    discovery_funding_rate = _masked_series(funding_rate, split_plan.discovery_mask) if funding_rate is not None else None
    final_funding_rate = _masked_series(funding_rate, split_plan.final_oos_mask) if funding_rate is not None else None
    _log(
        "  Split: "
        f"discovery={len(discovery_frame):,} bars, "
        f"repair_validation={len(repair_validation_frame):,} bars, "
        f"final_oos={len(final_frame):,} bars "
        f"(validation_start_idx={split_plan.repair_validation_start_idx}, "
        f"final_start_idx={split_plan.final_oos_start_idx})"
    )

    trial_counts_by_candidate = _record_candidate_trials(current_candidates, store, settings, cumulative_trial_counts)
    full_tasks = _build_tasks(
        current_candidates,
        frame,
        features_df,
        feature_meta,
        forward_regimes,
        funding_rate,
        trial_counts_by_candidate=trial_counts_by_candidate,
        data_quality_notes=data_quality_notes,
    )
    discovery_tasks = _slice_tasks(full_tasks, split_plan.discovery_mask)
    discovery_backtests = _run_backtests_parallel(discovery_tasks, discovery_frame, settings, max_workers, funding_df)

    # Align candidates with successful backtests
    discovery_backtest_ids = {r.candidate_id for r in discovery_backtests}
    discovery_candidates = [c for c in current_candidates if c.candidate_id in discovery_backtest_ids]

    _log(f"  Discovery backtests: {len(discovery_backtests)}/{len(current_candidates)} completed "
         f"({time.perf_counter() - t0:.0f}s)")
    _check_stop(stop_event)

    if discovery_backtests:
        _apply_batch_pbo(discovery_frame, discovery_tasks, discovery_backtests, settings, funding_df)
        _check_stop(stop_event)

    if not discovery_backtests:
        return {
            "candidates": [], "backtests": [], "gatechecks": [], "hardscores": [],
            "factor_evidence": [], "research_gates": [], "near_misses": [],
            "new_candidates": [], "research_survivors": [], "detail_artifact_ids": [],
            "history_entry": {},
        }

    # This lightweight evidence pass feeds deterministic repair candidates before
    # repair validation. Repaired candidates still count as trials and must pass a
    # held-aside merge-pool check before they can touch the final OOS window.
    initial_factor_evidence = build_factor_evidence_reports(
        frame=discovery_frame,
        tasks=discovery_tasks,
        candidates=discovery_candidates,
        results=discovery_backtests,
        settings=settings,
        forward_regimes=discovery_regimes,
        funding_rate=discovery_funding_rate,
        funding_df=funding_df,
    )

    pre_gate_candidates = _build_pre_gate_repair_candidates(
        discovery_candidates,
        discovery_backtests,
        initial_factor_evidence,
    )
    pre_gate_generated = len(pre_gate_candidates)
    pre_gate_completed = 0
    pre_gate_merged = 0
    pre_gate_rejected = 0
    repair_merge_diagnostics: list[dict[str, Any]] = []
    validation_full_tasks = [
        task for task in full_tasks
        if task[1]["candidate_id"] in discovery_backtest_ids
    ]
    validation_candidates = list(discovery_candidates)

    if pre_gate_candidates:
        t_repair = time.perf_counter()
        repair_counts = _record_candidate_trials(pre_gate_candidates, store, settings, cumulative_trial_counts)
        repair_full_tasks = _build_tasks(
            pre_gate_candidates,
            frame,
            features_df,
            feature_meta,
            forward_regimes,
            funding_rate,
            trial_counts_by_candidate=repair_counts,
            data_quality_notes=data_quality_notes,
        )
        validation_full_tasks.extend(repair_full_tasks)
        validation_candidates.extend(pre_gate_candidates)
        _log(
            f"  Pre-Gate repair generated: {len(pre_gate_candidates)} candidates "
            f"({time.perf_counter() - t_repair:.0f}s)"
        )

    validation_tasks = _slice_tasks(validation_full_tasks, split_plan.repair_validation_mask)
    validation_backtests = _run_backtests_parallel(
        validation_tasks,
        repair_validation_frame,
        settings,
        max_workers,
        funding_df,
    )
    pre_gate_ids = {candidate.candidate_id for candidate in pre_gate_candidates}
    pre_gate_completed = sum(1 for result in validation_backtests if result.candidate_id in pre_gate_ids)
    if validation_backtests:
        _apply_batch_pbo(repair_validation_frame, validation_tasks, validation_backtests, settings, funding_df)
    _check_stop(stop_event)

    if not validation_backtests:
        return {
            "candidates": [], "backtests": [], "gatechecks": [], "hardscores": [],
            "factor_evidence": [], "research_gates": [], "near_misses": [],
            "new_candidates": [], "research_survivors": [], "detail_artifact_ids": [],
            "history_entry": {},
        }

    merge_plan = _select_repair_merge_pool(
        original_candidates=discovery_candidates,
        repair_candidates=pre_gate_candidates,
        validation_candidates=validation_candidates,
        validation_full_tasks=validation_full_tasks,
        validation_tasks=validation_tasks,
        validation_results=validation_backtests,
    )
    pre_gate_merged = merge_plan.merged_repairs
    pre_gate_rejected = merge_plan.rejected_repairs
    repair_merge_diagnostics = merge_plan.diagnostics
    _log(
        f"  Repair validation: {pre_gate_completed}/{pre_gate_generated} repairs evaluated, "
        f"merged={pre_gate_merged}, rejected={pre_gate_rejected}"
    )
    if store and pre_gate_candidates:
        store.save_artifact(f"pre_gate_repairs_{artifact_scope}", "candidates", {
            "items": [candidate.model_dump(mode="json") for candidate in pre_gate_candidates],
            "merged_ids": [
                candidate.candidate_id
                for candidate in merge_plan.candidates
                if candidate.params.get("generated_by") == "pre_gate_repair"
            ],
            "diagnostics": repair_merge_diagnostics,
        })

    validation_result_by_candidate = {
        result.candidate_id: result
        for result in merge_plan.validation_results
    }

    final_tasks = _slice_tasks(merge_plan.full_tasks, split_plan.final_oos_mask)
    final_backtests = _run_backtests_parallel(final_tasks, final_frame, settings, max_workers, funding_df)
    for result in final_backtests:
        validation_result = validation_result_by_candidate.get(result.candidate_id)
        result.pbo = validation_result.pbo if validation_result is not None else 1.0
        if validation_result is not None:
            result.global_trials_at_eval = validation_result.global_trials_at_eval
            result.effective_trials_at_eval = validation_result.effective_trials_at_eval

    merge_pool_trials = _merge_pool_effective_trials(
        validation_backtests,
        cumulative_trial_counts,
        tested_candidates=len(validation_candidates),
    )
    _apply_merge_pool_trial_penalty(
        final_backtests,
        effective_trials_count=merge_pool_trials,
        observations=len(final_frame),
    )

    final_backtest_ids = {result.candidate_id for result in final_backtests}
    round_candidates = [
        candidate for candidate in merge_plan.candidates
        if candidate.candidate_id in final_backtest_ids
    ]
    round_backtests = final_backtests
    tasks = final_tasks
    _log(f"  Final OOS backtests: {len(round_backtests)} completed")
    _check_stop(stop_event)

    if store:
        store.save_artifact(f"backtests_{artifact_scope}", "backtests", {
            "items": [r.model_dump(mode="json") for r in round_backtests],
            "split": {
                "discovery_bars": len(discovery_frame),
                "repair_validation_bars": len(repair_validation_frame),
                "repair_bars": int(split_plan.repair_mask.sum()),
                "final_oos_bars": len(final_frame),
                "repair_validation_start_idx": split_plan.repair_validation_start_idx,
                "final_oos_start_idx": split_plan.final_oos_start_idx,
            },
        })

    # ── Factor evidence (PR1: diagnostics only, no GateCheck behavior changes) ──
    round_factor_evidence = build_factor_evidence_reports(
        frame=final_frame,
        tasks=tasks,
        candidates=round_candidates,
        results=round_backtests,
        settings=settings,
        forward_regimes=final_regimes,
        funding_rate=final_funding_rate,
        funding_df=funding_df,
    )
    _log(f"  Factor evidence: {len(round_factor_evidence)} reports")
    if store:
        store.save_artifact(f"factor_evidence_{artifact_scope}", "factor_evidence", {
            "items": [report.model_dump(mode="json") for report in round_factor_evidence],
        })

    # ── GateCheck ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    from factor_mining.validation.gatecheck import apply_fdr, apply_risk_stratified_gatechecks, run_gatecheck
    from factor_mining.registry import get_method

    fdr_map = apply_fdr(round_backtests, settings)
    methods_map = {m.method_id: m for m in METHOD_REGISTRY}
    round_gatechecks = []
    for r in round_backtests:
        method = methods_map.get(r.method_id) or get_method(r.method_id)
        fdr_p = fdr_map.get(r.experiment_id, combined_ic_tstat_pvalue(r.ic_tstat_nw, r.rankic_tstat_nw))
        gc = run_gatecheck(r, settings, method=method, fdr_adjusted_pvalue=fdr_p)
        round_gatechecks.append(gc)
    apply_risk_stratified_gatechecks(round_backtests, round_gatechecks, round_factor_evidence, settings)

    n_passed = sum(1 for g in round_gatechecks if g.passed)
    tier_counts = Counter(g.risk_tier for g in round_gatechecks)
    _log(
        f"  GateCheck: {n_passed}/{len(round_gatechecks)} accepted "
        f"(full={tier_counts.get('full_pass', 0)}, "
        f"conditional={tier_counts.get('conditional_pass', 0)}, "
        f"fail={tier_counts.get('fail', 0)}) "
        f"({time.perf_counter() - t0:.0f}s)"
    )
    diagnostics = _gatecheck_diagnostics(round_candidates, round_backtests, round_gatechecks, settings)
    top_failures = diagnostics["failure_counts"][:4]
    if top_failures:
        failure_str = ", ".join(f"{item['rule_id']}={item['count']}" for item in top_failures)
        _log(f"  Gate diagnostics: top failures {failure_str}")
    best = (diagnostics["top_by_net_sharpe"] or [None])[0]
    if best:
        gross_sharpe = best.get("gross_sharpe")
        gross_text = "n/a" if gross_sharpe is None else f"{gross_sharpe:+.2f}"
        _log(
            "  Gate diagnostics: best net SR "
            f"{best['net_sharpe']:+.2f}, gross SR {gross_text}, "
            f"cost margin {best['cost_margin_bps']:+.2f}bps, variant={best.get('search_variant', 'unknown')}"
        )

    for r, g in zip(round_backtests, round_gatechecks):
        if not g.passed:
            fail_str = " | ".join(item.rule_id for item in g.failures)
            _log(f"    FAIL {r.candidate_id[:16]}... [{fail_str}] SR={r.metrics_primary.sharpe:+.2f}")
        else:
            label = "COND" if g.risk_tier == "conditional_pass" else "PASS"
            allocation = g.allocation_multiplier if g.allocation_multiplier is not None else 0.0
            _log(
                f"    {label} {r.candidate_id[:16]}... "
                f"SR={r.metrics_primary.sharpe:+.2f} DSR={r.deflated_sharpe:+.3f} "
                f"alloc={allocation:.2f} evidence={g.factor_evidence_level}"
            )

    if store:
        store.save_artifact(f"gatechecks_{artifact_scope}", "gatechecks", {
            "items": [g.model_dump(mode="json") for g in round_gatechecks],
        })
        store.save_artifact(f"gatecheck_diagnostics_{artifact_scope}", "gatecheck_diagnostics", diagnostics)

    # ── Research Gate (PR2: soft discovery classification) ─────────
    round_research_gates = apply_research_gate(round_backtests, round_gatechecks, round_factor_evidence)
    status_counts = Counter(gate.status for gate in round_research_gates)
    _log(
        "  Research Gate: "
        f"production={status_counts.get('production_passed', 0)}, "
        f"survivor={status_counts.get('research_survivor', 0)}, "
        f"rejected={status_counts.get('rejected', 0)}"
    )
    formal_research_survivors = research_survivor_payloads(
        {candidate.candidate_id: candidate for candidate in round_candidates},
        round_backtests,
        round_research_gates,
    )
    persistent_survivors = build_research_survivor_records(
        candidates_by_id={candidate.candidate_id: candidate for candidate in round_candidates},
        results=round_backtests,
        research_gates=round_research_gates,
        fdr_map=fdr_map,
        settings=settings,
    )
    _augment_research_survivor_payloads(formal_research_survivors, persistent_survivors)
    if store:
        _update_research_survivor_store(
            store=store,
            records=persistent_survivors,
            rechecked_candidate_ids=survivor_candidate_ids,
            research_gates=round_research_gates,
            results=round_backtests,
            fdr_map=fdr_map,
            settings=settings,
        )
        store.save_artifact(f"research_gate_{artifact_scope}", "research_gate", {
            "items": [gate.model_dump(mode="json") for gate in round_research_gates],
        })
        store.save_artifact(f"research_survivors_{artifact_scope}", "research_survivors", {
            "items": formal_research_survivors,
            "criteria": "production_passed or soft research gate based on factor evidence and gross/net diagnostics",
        })
        store.save_artifact(f"research_survivor_store_{artifact_scope}", "research_survivor_store", {
            "items": [record.model_dump(mode="json") for record in persistent_survivors],
            "criteria": "active research survivors are paper-traded until promotion criteria or retirement criteria are met",
        })

    # ── Near Miss Analyzer (PR3: failure reason → repair params) ────
    round_near_misses = analyze_near_misses(
        candidates=round_candidates,
        results=round_backtests,
        gatechecks=round_gatechecks,
        evidence_reports=round_factor_evidence,
        research_gates=round_research_gates,
    )
    near_miss_counts = Counter(item.primary_reason for item in round_near_misses)
    top_near_miss = ", ".join(
        f"{reason}={count}"
        for reason, count in near_miss_counts.most_common(4)
    )
    actionable_near_misses = sum(1 for item in round_near_misses if item.actionable)
    _log(f"  Near Miss: actionable={actionable_near_misses}, top reasons {top_near_miss or 'none'}")
    if store:
        store.save_artifact(f"near_misses_{artifact_scope}", "near_misses", {
            "items": [item.model_dump(mode="json") for item in round_near_misses],
        })

    # ── HardScore ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    from factor_mining.hardscore import hardscore

    round_hardscores = []
    for r, g in zip(round_backtests, round_gatechecks):
        fdr_p = fdr_map.get(r.experiment_id, combined_ic_tstat_pvalue(r.ic_tstat_nw, r.rankic_tstat_nw))
        hs = hardscore(r, g, fdr_adjusted_pvalue=fdr_p, settings=settings)
        round_hardscores.append(hs)

    for hs in sorted(round_hardscores, key=lambda s: s.score, reverse=True):
        if hs.score > 0:
            _log(f"    score={hs.score:.1f} haircut={hs.haircut_sharpe:+.3f} fdr_p={hs.fdr_adjusted_pvalue:.4f}")

    detail_artifact_ids: list[str] = []
    if store:
        store.save_artifact(f"hardscores_{artifact_scope}", "hardscores", {
            "items": [s.model_dump(mode="json") for s in round_hardscores],
        })
        detail_artifact_ids = _save_experiment_details(
            store,
            frame=final_frame,
            tasks=tasks,
            candidates=round_candidates,
            results=round_backtests,
            gatechecks=round_gatechecks,
            hardscores=round_hardscores,
            settings=settings,
            funding_df=funding_df,
        )

    _log(f"  HardScore: {sum(1 for s in round_hardscores if s.score > 0)} positive ({time.perf_counter() - t0:.0f}s)")

    # ── Optimize: signal-side ───────────────────────────────────────
    _check_stop(stop_event)
    t0 = time.perf_counter()
    from factor_mining.optimizers.minimax_optimizer import (
        apply_exit_adjustments,
        apply_optimization_result,
        build_optimization_context,
        minimax_optimize,
        _fallback_optimization,
    )

    ctx = build_optimization_context(
        round_candidates, round_backtests, round_gatechecks, iteration,
        previous_actions=previous_actions,
        research_gates=round_research_gates, near_misses=round_near_misses,
    )
    research_survivors = ctx.get("research_survivors", [])
    _log(f"  Research survivors: {len(research_survivors)} selected for optimizer")
    for survivor in research_survivors[:3]:
        _log(
            "    survivor "
            f"{survivor.get('candidate_id', '')[:16]}... "
            f"score={survivor.get('research_score', 0):+.2f} "
            f"netSR={survivor.get('sharpe', 0):+.2f} "
            f"grossSR={_format_optional_float(survivor.get('gross_sharpe'))} "
            f"reason={survivor.get('survivor_reason') or 'ranked'}"
        )
    try:
        optimization = minimax_optimize(ctx, settings, mode="full")
        _log(f"  MiniMax signal: {optimization.get('action', 'unknown')}")
    except Exception as exc:
        _log(f"  MiniMax signal failed: {exc}, using fallback")
        optimization = _fallback_optimization(ctx, "full")

    signal_candidates, opt_summary = apply_optimization_result(optimization, round_candidates, round_backtests)
    new_candidates = list(signal_candidates)
    _log(f"  Signal optimization: {opt_summary['combinations_created']} combos, "
         f"{opt_summary['adjustments_applied']} adjustments, "
         f"{opt_summary.get('repairs_created', 0)} repairs, "
         f"{opt_summary['hypotheses_suggested']} new hypotheses")

    # ── Optimize: exit-side ──────────────────────────────────────────
    try:
        exit_opt = minimax_optimize(ctx, settings, mode="exit_params")
        _log(f"  MiniMax exit: {len(exit_opt.get('exit_adjustments', []))} adjustments")
        new_candidates = apply_exit_adjustments(exit_opt, new_candidates, settings)
    except Exception as exc:
        _log(f"  MiniMax exit failed: {exc}, skipping exit optimization")
        exit_opt = {"exit_adjustments": []}

    _log(f"  Total optimization: {len(new_candidates) - len(round_candidates)} new candidates "
         f"({time.perf_counter() - t0:.0f}s)")

    history_entry = {
        "round": round_num,
        "iteration": iteration,
        "symbol": round_candidates[0].symbol if round_candidates else None,
        "market": round_candidates[0].market if round_candidates else None,
        "num_candidates": len(round_candidates),
        "num_backtests": len(round_backtests),
        "num_pre_gate_repairs": pre_gate_completed,
        "num_pre_gate_repairs_generated": pre_gate_generated,
        "num_pre_gate_repairs_merged": pre_gate_merged,
        "num_pre_gate_repairs_rejected": pre_gate_rejected,
        "repair_merge_diagnostics": repair_merge_diagnostics,
        "split": {
            "discovery_bars": len(discovery_frame),
            "repair_validation_bars": len(repair_validation_frame),
            "final_oos_bars": len(final_frame),
            "repair_validation_start_idx": split_plan.repair_validation_start_idx,
            "final_oos_start_idx": split_plan.final_oos_start_idx,
        },
        "num_gatecheck_passed": n_passed,
        "gatecheck_tier_counts": dict(tier_counts),
        "num_research_survivors": len(research_survivors),
        "research_gate_counts": dict(status_counts),
        "near_miss_counts": dict(near_miss_counts),
        "actionable_near_misses": actionable_near_misses,
        "optimization": optimization,
        "summary": opt_summary,
        "new_candidates_count": len(new_candidates),
    }

    return {
        "candidates": round_candidates,
        "backtests": round_backtests,
        "gatechecks": round_gatechecks,
        "hardscores": round_hardscores,
        "factor_evidence": round_factor_evidence,
        "research_gates": round_research_gates,
        "near_misses": round_near_misses,
        "new_candidates": new_candidates,
        "research_survivors": formal_research_survivors,
        "detail_artifact_ids": detail_artifact_ids,
        "history_entry": history_entry,
    }


# ── helpers ─────────────────────────────────────────────────────────


def _build_data_split_plan(
    frame: pd.DataFrame,
    *,
    regimes: pd.Series | None = None,
    discovery_fraction: float = _DISCOVERY_FRACTION,
    repair_validation_fraction: float = _REPAIR_VALIDATION_FRACTION,
    final_oos_fraction: float = _FINAL_OOS_FRACTION,
) -> DataSplitPlan:
    n_rows = len(frame)
    if n_rows <= 1:
        mask = pd.Series([True] * n_rows, index=frame.index)
        empty = pd.Series([False] * n_rows, index=frame.index)
        return DataSplitPlan(
            discovery_mask=mask,
            repair_validation_mask=empty,
            repair_mask=mask,
            final_oos_mask=empty,
            repair_validation_start_idx=n_rows,
            final_oos_start_idx=n_rows,
        )

    final_count = max(1, int(round(n_rows * final_oos_fraction)))
    final_count = min(final_count, n_rows - 1)
    validation_count = 0
    if n_rows - final_count >= 2:
        validation_count = max(1, int(round(n_rows * repair_validation_fraction)))
        validation_count = min(validation_count, n_rows - final_count - 1)
    base_final_start = n_rows - final_count
    final_start = _choose_regime_aware_final_start(
        regimes,
        n_rows=n_rows,
        final_count=final_count,
        validation_count=validation_count,
        base_final_start=base_final_start,
    )
    final_end = min(n_rows, final_start + final_count)
    validation_start = final_start - validation_count

    # Without regime labels, keep the original 60/20/20 behavior. With regime
    # labels, final_start may move to a nearby contiguous OOS window whose
    # regime mix is less alien to discovery/validation while still avoiding
    # non-contiguous masks.
    if regimes is None:
        target_discovery_count = int(round(n_rows * discovery_fraction))
        if validation_count > 0 and target_discovery_count > 0:
            validation_start = min(validation_start, max(1, target_discovery_count))
    validation_start = max(1, min(validation_start, final_start))

    discovery_mask = pd.Series(False, index=frame.index)
    validation_mask = pd.Series(False, index=frame.index)
    repair_mask = pd.Series(False, index=frame.index)
    final_mask = pd.Series(False, index=frame.index)
    discovery_mask.iloc[:validation_start] = True
    validation_mask.iloc[validation_start:final_start] = True
    final_mask.iloc[final_start:final_end] = True
    repair_mask = discovery_mask | validation_mask
    return DataSplitPlan(
        discovery_mask=discovery_mask,
        repair_validation_mask=validation_mask,
        repair_mask=repair_mask,
        final_oos_mask=final_mask,
        repair_validation_start_idx=validation_start,
        final_oos_start_idx=final_start,
    )


def _choose_regime_aware_final_start(
    regimes: pd.Series | None,
    *,
    n_rows: int,
    final_count: int,
    validation_count: int,
    base_final_start: int,
) -> int:
    if regimes is None or len(regimes) != n_rows:
        return base_final_start

    labels = pd.Series(regimes.to_numpy(), index=range(n_rows)).astype(str).fillna("unknown")
    known = labels[labels != "unknown"]
    if known.nunique() < 2:
        return base_final_start

    min_start = max(validation_count + 1, int(n_rows * 0.50))
    max_start = n_rows - final_count
    if min_start >= max_start:
        return base_final_start

    radius = max(final_count, validation_count, n_rows // 10)
    lower = max(min_start, base_final_start - radius)
    upper = min(max_start, base_final_start + radius)
    step = max(1, final_count // 64)
    candidates = set(range(lower, upper + 1, step))
    candidates.update({base_final_start, min_start, max_start})

    best_start = base_final_start
    best_score = float("-inf")
    for start in sorted(item for item in candidates if min_start <= item <= max_start):
        discovery_end = start - validation_count
        if discovery_end <= 0:
            continue
        score = _regime_split_score(
            labels.iloc[:discovery_end],
            labels.iloc[discovery_end:start],
            labels.iloc[start:start + final_count],
            base_final_start=base_final_start,
            start=start,
            n_rows=n_rows,
        )
        if score > best_score:
            best_score = score
            best_start = start
    return best_start


def _regime_split_score(
    discovery: pd.Series,
    validation: pd.Series,
    final: pd.Series,
    *,
    base_final_start: int,
    start: int,
    n_rows: int,
) -> float:
    discovery_set = _known_regime_set(discovery)
    validation_set = _known_regime_set(validation)
    final_set = _known_regime_set(final)
    if not final_set:
        return -10.0

    discovery_overlap = len(final_set & discovery_set) / max(1, len(final_set))
    validation_overlap = len(final_set & validation_set) / max(1, len(final_set))
    final_diversity = min(len(final_set), 3) / 3.0
    validation_diversity = min(len(validation_set), 3) / 3.0
    unknown_ratio = float((final == "unknown").mean()) if len(final) else 1.0
    chronological_penalty = abs(start - base_final_start) / max(1, n_rows)
    return (
        2.0 * discovery_overlap
        + 1.5 * validation_overlap
        + final_diversity
        + 0.5 * validation_diversity
        - 1.0 * unknown_ratio
        - 0.25 * chronological_penalty
    )


def _known_regime_set(values: pd.Series) -> set[str]:
    return {str(value) for value in values.dropna().unique() if str(value) != "unknown"}


def _masked_frame(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return frame.loc[mask.to_numpy()].reset_index(drop=True)


def _masked_series(series: pd.Series, mask: pd.Series) -> pd.Series:
    return pd.Series(series.to_numpy(), index=mask.index).loc[mask.to_numpy()].reset_index(drop=True)


def _slice_tasks(tasks: list[tuple], mask: pd.Series) -> list[tuple]:
    mask_arr = mask.to_numpy(dtype=bool)
    sliced: list[tuple] = []
    for idx, (signal_arr, candidate_dict, _task_idx, trial_counts, notes) in enumerate(tasks):
        sliced.append((np.asarray(signal_arr, dtype=float)[mask_arr], candidate_dict, idx, trial_counts, notes))
    return sliced


def _record_candidate_trials(
    candidates: list[CandidateStrategySpec],
    store: MetadataStore | None,
    settings: Settings,
    cumulative_trial_counts: dict[str, int],
) -> dict[str, dict[str, int]]:
    ledger = TrialLedger(store, settings) if store is not None else None
    counts_by_candidate: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        family = candidate.hypothesis_family
        complexity_score = _candidate_complexity_score(candidate)
        candidate.params["complexity_score"] = complexity_score
        cumulative_trial_counts[family] = cumulative_trial_counts.get(family, 0) + 1
        if ledger is not None:
            ledger.record(
                TrialRecord(
                    trial_id=str(uuid.uuid4()),
                    candidate_id=candidate.candidate_id,
                    experiment_id=None,
                    hypothesis_family=family,
                    method_id=candidate.method_id,
                )
            )
            counts = ledger.counts_for(family)
            cumulative_trial_counts[family] = max(cumulative_trial_counts[family], counts["family_trials_count"])
        else:
            family_count = cumulative_trial_counts[family]
            global_count = sum(cumulative_trial_counts.values())
            counts = {
                "family_trials_count": family_count,
                "rolling_90d_trials_count": family_count,
                "effective_trials_count": family_count,
                "global_cumulative_trials_count": global_count,
            }
        counts["complexity_score"] = complexity_score
        counts["effective_trials_count"] = max(
            int(counts["effective_trials_count"]),
            int(counts["effective_trials_count"]) * max(1, complexity_score),
        )
        counts_by_candidate[candidate.candidate_id] = counts
    return counts_by_candidate


def _candidate_complexity_score(candidate: CandidateStrategySpec) -> int:
    params = candidate.params
    score = 1
    if params.get("factor_lookback") is not None or params.get("lookback") is not None:
        score += 1
    has_turnover_controls = (
        float(params.get("signal_threshold") or 0.0) > 0.0
        or int(params.get("smooth_span") or 1) > 1
        or abs(float(params.get("position_buffer") or 0.05) - 0.05) > 1e-12
    )
    if has_turnover_controls:
        score += 1
    for key in ("regime_filter", "funding_state_filter", "funding_trend_filter"):
        if params.get(key):
            score += 1
    if str(params.get("side_mode", "both")).lower() in {"long_only", "short_only"}:
        score += 1
    components = params.get("components")
    if isinstance(components, list) and components:
        score += len(components)
    return score


def _build_pre_gate_repair_candidates(
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
    evidence_reports: list[FactorEvidenceReport],
    *,
    limit: int = _PRE_GATE_REPAIR_LIMIT,
) -> list[CandidateStrategySpec]:
    """Create bounded repair/mutation candidates before the final GateCheck."""
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    evidence_by_exp = {report.experiment_id: report for report in evidence_reports}
    ordered = sorted(
        results,
        key=lambda result: _pre_gate_priority(result, evidence_by_exp.get(result.experiment_id)),
        reverse=True,
    )
    repairs: list[CandidateStrategySpec] = []
    signatures: set[tuple[str, str, str]] = set()

    for result in ordered:
        parent = candidate_by_id.get(result.candidate_id)
        if parent is None:
            continue
        evidence = evidence_by_exp.get(result.experiment_id)
        if not _passes_pre_gate_evidence(parent, result, evidence):
            continue
        parent_repairs = sorted(
            _pre_gate_repairs_for_parent(parent, result, evidence),
            key=lambda repair: _repair_acquisition_score(parent, repair, result, evidence),
            reverse=True,
        )
        kept_for_parent = 0
        for repair in parent_repairs:
            if not _repair_respects_family(parent, repair):
                continue
            repair_complexity = _candidate_complexity_score(repair)
            repair.params["complexity_score"] = repair_complexity
            if repair_complexity > _MAX_FINAL_COMPLEXITY:
                continue
            repair.params["repair_acquisition_score"] = _repair_acquisition_score(parent, repair, result, evidence)
            signature = (
                str(repair.params.get("parent_id")),
                str(repair.params.get("search_variant")),
                _json_dumps_sorted(_repair_signature_params(repair.params)),
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            repairs.append(repair)
            kept_for_parent += 1
            if kept_for_parent >= _PRE_GATE_REPAIR_MAX_PER_PARENT:
                break
            if len(repairs) >= limit:
                return repairs
    return repairs


def _passes_pre_gate_evidence(
    parent: CandidateStrategySpec,
    result: BacktestResult,
    evidence: FactorEvidenceReport | None,
) -> bool:
    if evidence is None:
        return False
    if _candidate_complexity_score(parent) > _MAX_FINAL_COMPLEXITY:
        return False

    flags = evidence.evidence_flags
    dimensions = 0
    if bool(flags.get("ic_ci_excludes_zero")) or _evidence_max_abs_ic(evidence) >= _PRE_GATE_MIN_ABS_IC:
        dimensions += 1
    if bool(flags.get("positive_turnover_adjusted_return")) or evidence.turnover_adjusted_return > 0.0:
        dimensions += 1
    if bool(flags.get("decay_curve_supported")) or evidence.decay_quality >= 0.25:
        dimensions += 1
    if bool(flags.get("long_short_spread")) or abs(evidence.long_short_spread_sharpe) >= 0.4:
        dimensions += 1
    if max((abs(float(value)) for value in evidence.quantile_spread_by_horizon.values()), default=0.0) >= 1.0:
        dimensions += 1
    if evidence.regime_conflict and (_regime_repair_params(evidence) or _funding_repair_params(evidence)):
        dimensions += 1

    gross_sharpe = result.metrics_gross.sharpe if result.metrics_gross is not None else result.metrics_primary.sharpe
    if gross_sharpe >= 0.4 and _has_cost_or_turnover_problem(result):
        dimensions += 1
    return dimensions >= 2


def _pre_gate_repairs_for_parent(
    parent: CandidateStrategySpec,
    result: BacktestResult,
    evidence: FactorEvidenceReport | None,
) -> list[CandidateStrategySpec]:
    repairs: list[CandidateStrategySpec] = []
    bundled_params: dict[str, Any] = {}
    bundled_reasons = 0

    if _has_cost_or_turnover_problem(result):
        params = _low_turnover_repair_params(result)
        repairs.append(_spawn_pre_gate_repair(parent, "pre_gate_low_turnover", params))
        bundled_params.update(params)
        bundled_reasons += 1

    horizon_params = _horizon_repair_params(parent, evidence)
    if horizon_params:
        repairs.append(_spawn_pre_gate_repair(parent, "pre_gate_horizon", horizon_params))
        bundled_params.update(horizon_params)
        bundled_reasons += 1

    regime_params = _regime_repair_params(evidence)
    if regime_params:
        repairs.append(_spawn_pre_gate_repair(parent, "pre_gate_regime_filter", regime_params))
        bundled_params.update(regime_params)
        bundled_reasons += 1

    funding_params = _funding_repair_params(evidence)
    if funding_params:
        repairs.append(_spawn_pre_gate_repair(parent, "pre_gate_funding_filter", funding_params))
        bundled_params.update(funding_params)
        bundled_reasons += 1

    side_params = _side_repair_params(result, evidence)
    if side_params:
        repairs.append(_spawn_pre_gate_repair(parent, "pre_gate_side_mode", side_params))
        bundled_params.update(side_params)
        bundled_reasons += 1

    if bundled_reasons >= 2:
        repairs.insert(0, _spawn_pre_gate_repair(parent, "pre_gate_bundle", bundled_params))

    return repairs


def _spawn_pre_gate_repair(
    parent: CandidateStrategySpec,
    variant: str,
    params: dict[str, Any],
) -> CandidateStrategySpec:
    repair = parent.model_copy(deep=True)
    repair.candidate_id = f"c_pre_{uuid.uuid4().hex[:12]}"
    repair.params.update(params)
    repair.params["parent_id"] = parent.candidate_id
    repair.params["generated_by"] = "pre_gate_repair"
    repair.params["search_variant"] = variant
    return repair


def _pre_gate_priority(result: BacktestResult, evidence: FactorEvidenceReport | None) -> float:
    gross_sharpe = result.metrics_gross.sharpe if result.metrics_gross is not None else result.metrics_primary.sharpe
    max_abs_ic = _evidence_max_abs_ic(evidence)
    max_abs_rankic = _max_abs_mapping(evidence.rankic_by_horizon if evidence else {})
    cost_bonus = 1.0 if _has_cost_or_turnover_problem(result) else 0.0
    return (
        100.0 * max_abs_ic
        + 80.0 * max_abs_rankic
        + max(0.0, gross_sharpe)
        + max(0.0, result.metrics_primary.sharpe)
        + cost_bonus
    )


def _repair_acquisition_score(
    parent: CandidateStrategySpec,
    repair: CandidateStrategySpec,
    result: BacktestResult,
    evidence: FactorEvidenceReport | None,
) -> float:
    """Rank repair trials by expected evidence lift under a small trial budget."""
    score = _pre_gate_priority(result, evidence)
    variant = str(repair.params.get("search_variant", ""))
    if variant == "pre_gate_low_turnover" and _has_cost_or_turnover_problem(result):
        score += 1.2
    if variant == "pre_gate_horizon" and evidence is not None:
        score += 0.8
    if variant in {"pre_gate_regime_filter", "pre_gate_funding_filter"}:
        score += 0.7
    if variant == "pre_gate_side_mode":
        score += 0.5
    if variant == "pre_gate_bundle":
        score += 0.4

    ci_width = 0.0
    if evidence is not None:
        ci_width = max(
            (
                abs(float(high) - float(low))
                for low, high in evidence.ic_ci_by_horizon.values()
                if low is not None and high is not None
            ),
            default=0.0,
        )
    score += min(1.0, 10.0 * ci_width)
    score -= 0.75 * max(0, _candidate_complexity_score(repair) - 2)
    score -= 0.5 * float(result.pbo if result.pbo is not None else 1.0)
    if not _repair_respects_family(parent, repair):
        score -= 100.0
    return float(score)


def _repair_respects_family(parent: CandidateStrategySpec, repair: CandidateStrategySpec) -> bool:
    parent_family = _candidate_theoretical_family(parent)
    repair_family = _candidate_theoretical_family(repair)
    if parent_family is None or repair_family is None:
        return True
    return parent_family == repair_family


def _candidate_theoretical_family(candidate: CandidateStrategySpec) -> str | None:
    family = candidate.params.get("factor_family") or candidate.hypothesis_family
    if family is None:
        return None
    canonical = _normalize_family(str(family))
    return canonical or str(family).strip().lower().replace(" ", "_").replace("-", "_")


def _has_cost_or_turnover_problem(result: BacktestResult) -> bool:
    gross_sharpe = result.metrics_gross.sharpe if result.metrics_gross is not None else None
    net_sharpe = result.metrics_primary.sharpe
    cost_drag = None if gross_sharpe is None else gross_sharpe - net_sharpe
    cost_margin = result.break_even_cost_bps - 2.0 * result.actual_cost_bps
    return (
        result.factor_turnover >= 0.12
        or cost_margin < 0.0
        or (
            gross_sharpe is not None
            and gross_sharpe >= 0.4
            and (net_sharpe <= 0.0 or (cost_drag is not None and cost_drag >= 0.5))
        )
    )


def _low_turnover_repair_params(result: BacktestResult) -> dict[str, float | int]:
    smooth_span = 48 if result.factor_turnover >= 0.20 else 24
    signal_threshold = 0.30 if result.factor_turnover >= 0.20 else 0.20
    position_buffer = 0.25 if result.factor_turnover >= 0.20 else 0.15
    holding_based_span = int(result.avg_holding_period_bars // 2) if result.avg_holding_period_bars else smooth_span
    return {
        "smooth_span": max(smooth_span, min(96, holding_based_span)),
        "signal_threshold": signal_threshold,
        "position_buffer": position_buffer,
    }


def _horizon_repair_params(
    parent: CandidateStrategySpec,
    evidence: FactorEvidenceReport | None,
) -> dict[str, int]:
    if evidence is None or evidence.best_horizon_bars is None:
        return {}
    if _evidence_max_abs_ic(evidence) < _PRE_GATE_MIN_ABS_IC:
        return {}
    current = parent.params.get("factor_lookback")
    if current is None or not _is_int_like(current):
        return {}
    best_horizon = int(evidence.best_horizon_bars)
    if int(current) == best_horizon:
        return {}
    return {"factor_lookback": best_horizon}


def _regime_repair_params(evidence: FactorEvidenceReport | None) -> dict[str, list[str]]:
    if evidence is None:
        return {}
    value, label = _best_nested_abs(evidence.regime_conditional_ic)
    if label is None:
        return {}
    threshold = max(_PRE_GATE_MIN_CONDITIONAL_IC, _evidence_max_abs_ic(evidence) * 1.5)
    if value < threshold:
        return {}
    return {"regime_filter": [label]}


def _funding_repair_params(evidence: FactorEvidenceReport | None) -> dict[str, list[str]]:
    if evidence is None:
        return {}
    value, label = _best_nested_abs(evidence.funding_conditional_ic)
    if label is None:
        return {}
    threshold = max(_PRE_GATE_MIN_CONDITIONAL_IC, _evidence_max_abs_ic(evidence) * 1.5)
    if value < threshold:
        return {}
    if label.startswith("state:"):
        return {"funding_state_filter": [label.split(":", 1)[1]]}
    if label.startswith("trend:"):
        return {"funding_trend_filter": [label.split(":", 1)[1]]}
    return {}


def _side_repair_params(
    result: BacktestResult,
    evidence: FactorEvidenceReport | None,
) -> dict[str, str]:
    if evidence is None:
        return {}
    long_sharpe = evidence.long_only_metrics.sharpe
    short_sharpe = evidence.short_only_metrics.sharpe
    best_side = "long_only" if long_sharpe >= short_sharpe else "short_only"
    best_sharpe = max(long_sharpe, short_sharpe)
    if best_sharpe >= 0.4 and best_sharpe - result.metrics_primary.sharpe >= 0.5:
        return {"side_mode": best_side}
    return {}


def _evidence_max_abs_ic(evidence: FactorEvidenceReport | None) -> float:
    return _max_abs_mapping(evidence.ic_by_horizon if evidence else {})


def _max_abs_mapping(values: dict[str, float]) -> float:
    return max((abs(float(value)) for value in values.values() if value is not None), default=0.0)


def _best_nested_abs(values: dict[str, dict[str, float]]) -> tuple[float, str | None]:
    best_value = 0.0
    best_label: str | None = None
    for label, nested in values.items():
        for value in nested.values():
            abs_value = abs(float(value))
            if abs_value > best_value:
                best_value = abs_value
                best_label = label
    return best_value, best_label


def _repair_signature_params(params: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "smooth_span",
        "signal_threshold",
        "position_buffer",
        "factor_lookback",
        "regime_filter",
        "funding_state_filter",
        "funding_trend_filter",
        "side_mode",
    }
    return {key: params[key] for key in keys if key in params}


def _select_repair_merge_pool(
    *,
    original_candidates: list[CandidateStrategySpec],
    repair_candidates: list[CandidateStrategySpec],
    validation_candidates: list[CandidateStrategySpec],
    validation_full_tasks: list[tuple],
    validation_tasks: list[tuple],
    validation_results: list[BacktestResult],
) -> RepairMergePlan:
    result_by_candidate = {result.candidate_id: result for result in validation_results}
    full_task_by_candidate = _task_by_candidate_id(validation_full_tasks)
    validation_task_by_candidate = _task_by_candidate_id(validation_tasks)
    original_by_id = {candidate.candidate_id: candidate for candidate in original_candidates}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in validation_candidates}

    kept_candidates: list[CandidateStrategySpec] = []
    kept_full_tasks: list[tuple] = []
    kept_results: list[BacktestResult] = []
    diagnostics: list[dict[str, Any]] = []

    for candidate in original_candidates:
        result = result_by_candidate.get(candidate.candidate_id)
        task = full_task_by_candidate.get(candidate.candidate_id)
        if result is None or task is None:
            continue
        candidate.params["merge_pool_status"] = "original"
        kept_candidates.append(candidate)
        kept_full_tasks.append(task)
        kept_results.append(result)

    repairs_with_results = [
        candidate for candidate in repair_candidates
        if candidate.candidate_id in result_by_candidate
    ]
    repair_rows: list[tuple[float, CandidateStrategySpec, float | None]] = []
    for repair in repairs_with_results:
        parent_id = str(repair.params.get("parent_id") or "")
        parent_task = validation_task_by_candidate.get(parent_id)
        repair_task = validation_task_by_candidate.get(repair.candidate_id)
        corr = None
        if parent_task is not None and repair_task is not None:
            corr = _signal_correlation(parent_task[0], repair_task[0])
        repair_rows.append((
            _repair_merge_score(result_by_candidate[repair.candidate_id], corr),
            repair,
            corr,
        ))

    per_parent_count: Counter[str] = Counter()
    merged = 0
    rejected = 0
    for _, repair, parent_corr in sorted(repair_rows, key=lambda item: item[0], reverse=True):
        parent_id = str(repair.params.get("parent_id") or "")
        result = result_by_candidate[repair.candidate_id]
        parent = original_by_id.get(parent_id) or candidate_by_id.get(parent_id)
        pbo = float(result.pbo if result.pbo is not None else 1.0)
        reasons: list[str] = []
        if parent is None or parent_id not in result_by_candidate:
            reasons.append("parent_missing_validation")
        elif not _repair_respects_family(parent, repair):
            reasons.append("cross_family_mutation")
        if _candidate_complexity_score(repair) > _MAX_FINAL_COMPLEXITY:
            reasons.append("complexity_cap")
        if pbo > _REPAIR_MAX_PBO:
            reasons.append("high_validation_pbo")
        if parent_corr is None:
            reasons.append("missing_parent_correlation")
        elif abs(parent_corr) >= _REPAIR_MAX_PARENT_CORR:
            reasons.append("low_incremental_orthogonality")
        if per_parent_count[parent_id] >= _PRE_GATE_REPAIR_MAX_PER_PARENT:
            reasons.append("repair_parent_ratio_cap")

        if reasons:
            rejected += 1
            repair.params["merge_pool_status"] = "rejected"
            repair.params["merge_pool_reasons"] = reasons
            diagnostics.append(_repair_merge_diagnostic(repair, result, parent_corr, "rejected", reasons))
            continue

        task = full_task_by_candidate.get(repair.candidate_id)
        if task is None:
            rejected += 1
            reasons = ["missing_full_task"]
            repair.params["merge_pool_status"] = "rejected"
            repair.params["merge_pool_reasons"] = reasons
            diagnostics.append(_repair_merge_diagnostic(repair, result, parent_corr, "rejected", reasons))
            continue

        per_parent_count[parent_id] += 1
        merged += 1
        repair.params["merge_pool_status"] = "merged"
        repair.params["repair_validation_pbo"] = pbo
        repair.params["parent_signal_correlation"] = parent_corr
        repair.params["merge_pool_score"] = _repair_merge_score(result, parent_corr)
        kept_candidates.append(repair)
        kept_full_tasks.append(task)
        kept_results.append(result)
        diagnostics.append(_repair_merge_diagnostic(repair, result, parent_corr, "merged", []))

    missing_repairs = len(repair_candidates) - len(repairs_with_results)
    rejected += max(0, missing_repairs)
    for repair in repair_candidates:
        if repair.candidate_id in result_by_candidate:
            continue
        repair.params["merge_pool_status"] = "rejected"
        repair.params["merge_pool_reasons"] = ["validation_backtest_failed"]
        diagnostics.append({
            "candidate_id": repair.candidate_id,
            "parent_id": repair.params.get("parent_id"),
            "status": "rejected",
            "reasons": ["validation_backtest_failed"],
        })

    return RepairMergePlan(
        candidates=kept_candidates,
        full_tasks=kept_full_tasks,
        validation_results=kept_results,
        merged_repairs=merged,
        rejected_repairs=rejected,
        diagnostics=diagnostics,
    )


def _task_by_candidate_id(tasks: list[tuple]) -> dict[str, tuple]:
    task_by_id: dict[str, tuple] = {}
    for task in tasks:
        _signal_arr, cdict, *_rest = task
        task_by_id[cdict["candidate_id"]] = task
    return task_by_id


def _repair_merge_score(result: BacktestResult, parent_corr: float | None) -> float:
    gross_sharpe = result.metrics_gross.sharpe if result.metrics_gross is not None else result.metrics_primary.sharpe
    pbo = float(result.pbo if result.pbo is not None else 1.0)
    cost_margin = result.break_even_cost_bps - 2.0 * result.actual_cost_bps
    corr_penalty = 0.0 if parent_corr is None else 0.25 * abs(parent_corr)
    return float(
        2.0 * result.metrics_primary.sharpe
        + max(0.0, gross_sharpe)
        + 0.01 * max(0.0, cost_margin)
        - pbo
        - corr_penalty
    )


def _repair_merge_diagnostic(
    repair: CandidateStrategySpec,
    result: BacktestResult,
    parent_corr: float | None,
    status: str,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "candidate_id": repair.candidate_id,
        "parent_id": repair.params.get("parent_id"),
        "status": status,
        "reasons": reasons,
        "validation_pbo": result.pbo,
        "parent_signal_correlation": parent_corr,
        "net_sharpe": result.metrics_primary.sharpe,
        "gross_sharpe": result.metrics_gross.sharpe if result.metrics_gross is not None else None,
        "complexity_score": repair.params.get("complexity_score"),
        "search_variant": repair.params.get("search_variant"),
    }


def _signal_correlation(left: Any, right: Any) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    n_rows = min(len(left_arr), len(right_arr))
    if n_rows < 3:
        return 1.0
    left_arr = left_arr[:n_rows]
    right_arr = right_arr[:n_rows]
    mask = np.isfinite(left_arr) & np.isfinite(right_arr)
    if int(mask.sum()) < 3:
        return 1.0
    left_arr = left_arr[mask]
    right_arr = right_arr[mask]
    if float(np.std(left_arr)) <= 1e-12 or float(np.std(right_arr)) <= 1e-12:
        return 1.0
    corr = float(np.corrcoef(left_arr, right_arr)[0, 1])
    return corr if np.isfinite(corr) else 1.0


def _merge_pool_effective_trials(
    validation_results: list[BacktestResult],
    cumulative_trial_counts: dict[str, int],
    *,
    tested_candidates: int,
) -> int:
    observed = [
        int(result.effective_trials_at_eval)
        for result in validation_results
        if result.effective_trials_at_eval is not None
    ]
    return max(
        1,
        int(tested_candidates),
        sum(int(value) for value in cumulative_trial_counts.values()),
        max(observed, default=1),
    )


def _apply_merge_pool_trial_penalty(
    results: list[BacktestResult],
    *,
    effective_trials_count: int,
    observations: int,
) -> None:
    returns_placeholder = np.empty(max(1, int(observations)), dtype=float)
    for result in results:
        result.effective_trials_at_eval = max(int(result.effective_trials_at_eval), effective_trials_count)
        result.global_trials_at_eval = max(int(result.global_trials_at_eval), effective_trials_count)
        result.deflated_sharpe = deflated_sharpe_ratio(
            returns_placeholder,
            observed_sr=result.metrics_primary.sharpe,
            trials_count=result.effective_trials_at_eval,
        )


def _json_dumps_sorted(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, default=str, sort_keys=True)


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _apply_batch_pbo(
    frame: pd.DataFrame,
    tasks: list[tuple],
    results: list[BacktestResult],
    settings: Settings,
    funding_df: pd.DataFrame | None,
) -> None:
    result_by_candidate = {result.candidate_id: result for result in results}
    returns_by_candidate: dict[str, pd.Series] = {}
    periods = annualization_factor(settings.data.default_interval)

    for signal_arr, cdict, *_ in tasks:
        candidate_id = cdict["candidate_id"]
        if candidate_id not in result_by_candidate:
            continue
        candidate = CandidateStrategySpec.model_validate(cdict)
        path = evaluate_strategy_path(
            frame,
            pd.Series(signal_arr, index=frame.index),
            candidate,
            settings,
            funding=funding_df,
        )
        returns_by_candidate[candidate_id] = path.strategy_returns.reset_index(drop=True)
        periods = annualization_factor(candidate.interval)

    if len(returns_by_candidate) < 2:
        for result in results:
            result.pbo = 1.0
        return

    n_rows = min(len(series) for series in returns_by_candidate.values())
    split_defs = _cscv_splits(n_rows, settings)
    if not split_defs:
        for result in results:
            result.pbo = 1.0
        return

    selected_count = {candidate_id: 0 for candidate_id in returns_by_candidate}
    poor_oos_count = {candidate_id: 0 for candidate_id in returns_by_candidate}
    for train_mask, test_mask in split_defs:
        train_scores = {
            candidate_id: sharpe_ratio(series.iloc[:n_rows].iloc[train_mask], periods_per_year=periods)
            for candidate_id, series in returns_by_candidate.items()
        }
        selected_id = max(train_scores, key=train_scores.get)
        test_scores = {
            candidate_id: sharpe_ratio(series.iloc[:n_rows].iloc[test_mask], periods_per_year=periods)
            for candidate_id, series in returns_by_candidate.items()
        }
        selected_test_score = test_scores[selected_id]
        percentile = sum(score <= selected_test_score for score in test_scores.values()) / len(test_scores)
        percentile = float(np.clip(percentile, 1e-6, 1.0 - 1e-6))
        logit = float(np.log(percentile / (1.0 - percentile)))
        selected_count[selected_id] += 1
        if logit < 0.0:
            poor_oos_count[selected_id] += 1

    for result in results:
        count = selected_count.get(result.candidate_id, 0)
        result.pbo = float(poor_oos_count.get(result.candidate_id, 0) / count) if count else 1.0


def _cscv_splits(n_rows: int, settings: Settings) -> list[tuple[np.ndarray, np.ndarray]]:
    n_groups = min(settings.cpcv.n_groups, n_rows)
    if n_groups % 2 == 1:
        n_groups -= 1
    if n_groups < 4:
        return []
    group_edges = np.linspace(0, n_rows, n_groups + 1, dtype=int)
    group_ids = np.empty(n_rows, dtype=int)
    for group_idx in range(n_groups):
        group_ids[group_edges[group_idx]:group_edges[group_idx + 1]] = group_idx

    split_defs: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    train_group_count = n_groups // 2
    all_groups = set(range(n_groups))
    for train_group_ids in combinations(range(n_groups), train_group_count):
        train_tuple = tuple(train_group_ids)
        test_tuple = tuple(sorted(all_groups.difference(train_tuple)))
        pair_key = min(train_tuple, test_tuple)
        if pair_key in seen:
            continue
        seen.add(pair_key)
        train_mask = np.isin(group_ids, train_tuple)
        test_mask = np.isin(group_ids, test_tuple)
        if bool(train_mask.any()) and bool(test_mask.any()):
            split_defs.append((train_mask, test_mask))
        if len(split_defs) >= 128:
            break
    return split_defs


def _cpcv_splits(n_rows: int, settings: Settings) -> list[tuple[np.ndarray, np.ndarray]]:
    return _cscv_splits(n_rows, settings)


def _data_key(candidate: CandidateStrategySpec) -> tuple[str, str]:
    return candidate.symbol, candidate.market


def _group_candidates_by_data(
    candidates: list[CandidateStrategySpec],
) -> dict[tuple[str, str], list[CandidateStrategySpec]]:
    grouped: dict[tuple[str, str], list[CandidateStrategySpec]] = {}
    for candidate in candidates:
        grouped.setdefault(_data_key(candidate), []).append(candidate)
    return grouped


def _load_data_contexts(
    candidates: list[CandidateStrategySpec],
    settings: Settings,
    *,
    tail: int | None,
) -> dict[tuple[str, str], MarketDataContext]:
    contexts: dict[tuple[str, str], MarketDataContext] = {}
    for symbol, market in sorted({(c.symbol, c.market) for c in candidates}):
        try:
            frame = load_frame(settings, symbol=symbol, market=market, tail=tail)
        except FileNotFoundError as exc:
            _log(f"Skip {symbol}/{market}: {exc}")
            continue

        n_rows = len(frame)
        _log(f"Data {symbol}/{market}: {n_rows:,} rows, {frame['open_time'].min()} → {frame['open_time'].max()}")
        quality_notes = kline_quality_notes(
            frame,
            interval_ms=interval_to_ms(settings.data.default_interval),
            scope=f"{market}:{symbol}:{settings.data.default_interval}",
        )
        for note in quality_notes:
            _log(f"  Data quality [{note.severity}] {note.scope}: {note.message} ({note.degraded_ratio:.2%})")
        features_df, feature_meta = generate_features(frame)
        supplemental_features, supplemental_meta = load_supplemental_features(
            settings,
            frame,
            symbol=symbol,
            market=market,
            interval=settings.data.default_interval,
        )
        if not supplemental_features.empty:
            features_df = pd.concat([features_df, supplemental_features], axis=1)
            feature_meta.update(supplemental_meta)
        _ = forward_returns(frame["close"], horizons=[1, 12, 48])
        _log(f"Features {symbol}/{market}: {len(features_df.columns)} cols")
        if supplemental_meta:
            _log(f"Supplemental features {symbol}/{market}: {len(supplemental_meta)} cols")

        funding_df = load_funding(settings, symbol=symbol) if market == "um_futures" else None
        funding_rate = funding_event_zscore_to_frame(frame, funding_df)
        if funding_df is not None and not funding_df.empty:
            _log(
                f"Funding {symbol}: {len(funding_df)} 8h snapshots, "
                f"event-z range [{funding_rate.min():+.3f}, {funding_rate.max():+.3f}]"
            )
        else:
            _log(f"Funding {symbol}: not available, using price-based proxy")

        forward_regimes = _fit_regime_model(frame, tail, _log)
        _log(f"Regime {symbol}/{market}: {dict(forward_regimes.value_counts())}")
        contexts[(symbol, market)] = MarketDataContext(
            symbol=symbol,
            market=market,
            frame=frame,
            features_df=features_df,
            feature_meta=feature_meta,
            forward_regimes=forward_regimes,
            funding_df=funding_df,
            funding_rate=funding_rate,
            data_quality_notes=quality_notes,
        )

    if not contexts:
        raise FileNotFoundError("No local parquet data found for any candidate symbol/market.")
    return contexts


def _combine_round_history(
    round_num: int,
    iteration: int,
    child_history: list[dict[str, Any]],
    new_candidates: list[CandidateStrategySpec],
) -> dict[str, Any]:
    optimization = child_history[-1].get("optimization", {}) if child_history else {}
    return {
        "round": round_num,
        "iteration": iteration,
        "children": child_history,
        "optimization": optimization,
        "new_candidates_count": len(new_candidates),
        "num_candidates": sum(child.get("num_candidates", 0) for child in child_history),
        "num_backtests": sum(child.get("num_backtests", 0) for child in child_history),
        "num_pre_gate_repairs": sum(child.get("num_pre_gate_repairs", 0) for child in child_history),
        "num_pre_gate_repairs_generated": sum(child.get("num_pre_gate_repairs_generated", 0) for child in child_history),
        "num_pre_gate_repairs_merged": sum(child.get("num_pre_gate_repairs_merged", 0) for child in child_history),
        "num_pre_gate_repairs_rejected": sum(child.get("num_pre_gate_repairs_rejected", 0) for child in child_history),
        "num_gatecheck_passed": sum(child.get("num_gatecheck_passed", 0) for child in child_history),
        "gatecheck_tier_counts": _sum_gatecheck_tier_counts(child_history),
        "num_research_survivors": sum(child.get("num_research_survivors", 0) for child in child_history),
        "research_gate_counts": _sum_research_gate_counts(child_history),
        "near_miss_counts": _sum_near_miss_counts(child_history),
        "actionable_near_misses": sum(child.get("actionable_near_misses", 0) for child in child_history),
    }


def _survivor_seed_candidates(
    records: list[ResearchSurvivorRecord],
    errors: list[str] | None = None,
) -> list[CandidateStrategySpec]:
    candidates: list[CandidateStrategySpec] = []
    for record in records:
        if record.status != "active" or not record.candidate_payload:
            continue
        try:
            candidates.append(CandidateStrategySpec.model_validate(record.candidate_payload))
        except Exception as exc:
            if errors is not None:
                errors.append(f"research_survivor_seed:{record.candidate_id}:{exc}")
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: list[CandidateStrategySpec]) -> list[CandidateStrategySpec]:
    out: list[CandidateStrategySpec] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        out.append(candidate)
        seen.add(candidate.candidate_id)
    return out


def _augment_research_survivor_payloads(
    payloads: list[dict[str, Any]],
    records: list[ResearchSurvivorRecord],
) -> None:
    by_candidate = {record.candidate_id: record for record in records}
    for payload in payloads:
        record = by_candidate.get(str(payload.get("candidate_id")))
        if record is None:
            continue
        payload.update({
            "survivor_store_status": record.status,
            "paper_trade_start_date": record.paper_trade_start_date.isoformat(),
            "required_additional_trades": record.required_additional_trades,
            "required_oos_days": record.required_oos_days,
            "recheck_trigger": record.recheck_trigger,
            "promotion_criteria": record.promotion_criteria,
            "promotion_ready": record.promotion_ready,
            "fdr_pvalue": record.fdr_pvalue,
            "dsr": record.dsr,
        })


def _update_research_survivor_store(
    *,
    store: MetadataStore,
    records: list[ResearchSurvivorRecord],
    rechecked_candidate_ids: set[str],
    research_gates: list[ResearchGateResult],
    results: list[BacktestResult],
    fdr_map: dict[str, float],
    settings: Settings,
) -> None:
    store.upsert_research_survivors(records)
    if not rechecked_candidate_ids:
        return

    gate_by_candidate = {gate.candidate_id: gate for gate in research_gates}
    result_by_candidate = {result.candidate_id: result for result in results}
    min_trades = int(settings.gatecheck.min_oos_trades)
    promotion_fdr = float(settings.gatecheck.research_survivor_promotion_fdr_p)
    for candidate_id in rechecked_candidate_ids:
        gate = gate_by_candidate.get(candidate_id)
        result = result_by_candidate.get(candidate_id)
        if gate is None or result is None:
            continue
        current_trades = int(result.oos_trade_count or result.metrics_primary.trade_count)
        fdr_pvalue = float(fdr_map.get(result.experiment_id, combined_ic_tstat_pvalue(result.ic_tstat_nw, result.rankic_tstat_nw)))
        if gate.status == "production_passed":
            store.update_research_survivor_status(candidate_id, "promoted", "production_gate_passed")
        elif fdr_pvalue < promotion_fdr and current_trades >= min_trades:
            store.update_research_survivor_status(candidate_id, "promoted", "promotion_criteria_met")
        elif gate.status == "rejected" and current_trades >= min_trades:
            store.update_research_survivor_status(candidate_id, "retired", "rejected_after_min_trades")


def _sum_research_gate_counts(child_history: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for child in child_history:
        counts.update(child.get("research_gate_counts", {}))
    return dict(counts)


def _sum_gatecheck_tier_counts(child_history: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for child in child_history:
        counts.update(child.get("gatecheck_tier_counts", {}))
    return dict(counts)


def _sum_near_miss_counts(child_history: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for child in child_history:
        counts.update(child.get("near_miss_counts", {}))
    return dict(counts)


def _save_experiment_details(
    store: MetadataStore,
    *,
    frame: pd.DataFrame,
    tasks: list[tuple],
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
    gatechecks: list[GateCheckResult],
    hardscores: list[HardScoreReport],
    settings: Settings,
    funding_df: pd.DataFrame | None,
) -> list[str]:
    from factor_mining.backtest.engine import build_backtest_detail

    task_by_candidate = {
        cdict["candidate_id"]: signal_arr
        for signal_arr, cdict, *_ in tasks
    }
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    gate_by_exp = {gate.experiment_id: gate for gate in gatechecks}
    score_by_exp = {score.experiment_id: score for score in hardscores}
    selected_ids = _selected_detail_experiment_ids(results, gatechecks, hardscores)
    saved_artifact_ids: list[str] = []

    for result in results:
        if result.experiment_id not in selected_ids:
            continue
        candidate = candidate_by_id.get(result.candidate_id)
        signal_arr = task_by_candidate.get(result.candidate_id)
        if candidate is None or signal_arr is None:
            continue
        detail = build_backtest_detail(
            frame,
            pd.Series(signal_arr, index=frame.index),
            candidate,
            settings,
            result,
            funding=funding_df,
        )
        gate = gate_by_exp.get(result.experiment_id)
        score = score_by_exp.get(result.experiment_id)
        if gate is not None:
            detail["gatecheck"] = gate.model_dump(mode="json")
        if score is not None:
            detail["hardscore"] = score.model_dump(mode="json")
        artifact_id = f"experiment_detail_{result.experiment_id}"
        store.save_artifact(artifact_id, "experiment_detail", detail)
        saved_artifact_ids.append(artifact_id)
    return saved_artifact_ids


def _selected_detail_experiment_ids(
    results: list[BacktestResult],
    gatechecks: list[GateCheckResult],
    hardscores: list[HardScoreReport],
    *,
    limit: int = _DETAIL_ARTIFACT_LIMIT_PER_SCOPE,
) -> set[str]:
    selected: list[str] = []
    gate_by_exp = {gate.experiment_id: gate for gate in gatechecks}
    score_by_exp = {score.experiment_id: score for score in hardscores}

    def add(experiment_id: str) -> None:
        if experiment_id not in selected:
            selected.append(experiment_id)

    for result in results:
        gate = gate_by_exp.get(result.experiment_id)
        score = score_by_exp.get(result.experiment_id)
        if (gate and gate.passed) or (score and score.score > 0):
            add(result.experiment_id)

    buckets = [
        lambda result: result.metrics_primary.sharpe,
        lambda result: result.metrics_gross.sharpe if result.metrics_gross is not None else -999.0,
        lambda result: result.break_even_cost_bps - 2.0 * result.actual_cost_bps,
        lambda result: abs(result.ic_tstat_nw),
        lambda result: abs(result.rankic_tstat_nw),
    ]
    for key_fn in buckets:
        for result in sorted(results, key=key_fn, reverse=True)[:_DETAIL_BUCKET_LIMIT]:
            add(result.experiment_id)
            if len(selected) >= limit:
                return set(selected[:limit])

    return set(selected[:limit])


def _gatecheck_diagnostics(
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
    gatechecks: list[GateCheckResult],
    settings: Settings,
) -> dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}

    for result, gate in zip(results, gatechecks, strict=False):
        candidate = candidate_by_id.get(result.candidate_id)
        failures = [item.rule_id for item in gate.failures]
        warnings = [item.rule_id for item in gate.warnings]
        failure_counts.update(failures)
        warning_counts.update(warnings)
        gross_sharpe = result.metrics_gross.sharpe if result.metrics_gross is not None else None
        net_sharpe = result.metrics_primary.sharpe
        cost_margin_bps = result.break_even_cost_bps - (
            settings.gatecheck.break_even_cost_multiple * result.actual_cost_bps
        )
        rows.append({
            "candidate_id": result.candidate_id,
            "experiment_id": result.experiment_id,
            "hypothesis_family": result.hypothesis_family,
            "method_id": result.method_id,
            "symbol": result.symbol,
            "search_variant": (candidate.params.get("search_variant") if candidate else None) or "unknown",
            "signal_source": (candidate.params.get("signal_source") if candidate else None),
            "net_sharpe": net_sharpe,
            "gross_sharpe": gross_sharpe,
            "cost_drag_sharpe": None if gross_sharpe is None else gross_sharpe - net_sharpe,
            "annualized_return": result.metrics_primary.annualized_return,
            "max_drawdown": result.metrics_primary.max_drawdown,
            "ic_tstat_nw": result.ic_tstat_nw,
            "rankic_tstat_nw": result.rankic_tstat_nw,
            "permutation_pvalue": result.permutation_test_pvalue,
            "pbo": result.pbo,
            "gate_raw_passed": gate.raw_passed if gate else None,
            "risk_tier": gate.risk_tier if gate else None,
            "factor_evidence_level": gate.factor_evidence_level if gate else None,
            "allocation_multiplier": gate.allocation_multiplier if gate else None,
            "review_after_days": gate.review_after_days if gate else None,
            "factor_turnover": result.factor_turnover,
            "oos_trade_count": result.oos_trade_count,
            "break_even_cost_bps": result.break_even_cost_bps,
            "actual_cost_bps": result.actual_cost_bps,
            "cost_margin_bps": cost_margin_bps,
            "failures": failures,
            "warnings": warnings,
        })

    return {
        "total": len(gatechecks),
        "passed": sum(1 for gate in gatechecks if gate.passed),
        "failure_counts": [
            {"rule_id": rule_id, "count": count}
            for rule_id, count in failure_counts.most_common()
        ],
        "warning_counts": [
            {"rule_id": rule_id, "count": count}
            for rule_id, count in warning_counts.most_common()
        ],
        "risk_tier_counts": dict(Counter(row["risk_tier"] or "unclassified" for row in rows)),
        "metric_summary": {
            "net_sharpe": _numeric_summary(row["net_sharpe"] for row in rows),
            "gross_sharpe": _numeric_summary(row["gross_sharpe"] for row in rows),
            "cost_drag_sharpe": _numeric_summary(row["cost_drag_sharpe"] for row in rows),
            "factor_turnover": _numeric_summary(row["factor_turnover"] for row in rows),
            "break_even_cost_bps": _numeric_summary(row["break_even_cost_bps"] for row in rows),
            "actual_cost_bps": _numeric_summary(row["actual_cost_bps"] for row in rows),
            "cost_margin_bps": _numeric_summary(row["cost_margin_bps"] for row in rows),
            "oos_trade_count": _numeric_summary(row["oos_trade_count"] for row in rows),
        },
        "top_by_net_sharpe": sorted(rows, key=lambda row: row["net_sharpe"], reverse=True)[:10],
        "top_by_cost_margin": sorted(rows, key=lambda row: row["cost_margin_bps"], reverse=True)[:10],
        "rows": rows,
    }


def _numeric_summary(values) -> dict[str, float | int | None]:
    clean = sorted(
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    )
    if not clean:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "count": len(clean),
        "min": clean[0],
        "p25": clean[int((len(clean) - 1) * 0.25)],
        "median": clean[int((len(clean) - 1) * 0.50)],
        "p75": clean[int((len(clean) - 1) * 0.75)],
        "max": clean[-1],
    }


def _format_optional_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:+.2f}" if np.isfinite(number) else "n/a"


def _build_tasks(
    candidates: list[CandidateStrategySpec],
    frame: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_meta: dict,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None = None,
    *,
    trial_counts_by_candidate: dict[str, dict[str, int]] | None = None,
    data_quality_notes: list[DataQualityNote] | None = None,
) -> list[tuple[np.ndarray, dict, int, dict[str, int], list[dict]]]:
    """Construct regime-conditional trading signals for each candidate.

    Returns list of (signal_array, candidate_dict, index, trial_counts, data_quality_notes).
    """
    tasks = []
    note_dicts = [note.model_dump(mode="json") for note in (data_quality_notes or [])]
    for i, c in enumerate(candidates):
        signal_arr = _build_signal_for(c, frame, features_df, feature_meta, i, forward_regimes, funding_rate)
        trial_counts = (trial_counts_by_candidate or {}).get(
            c.candidate_id,
            {
                "effective_trials_count": 1,
                "global_cumulative_trials_count": 1,
            },
        )
        tasks.append((signal_arr, c.model_dump(mode="json"), i, trial_counts, note_dicts))
    return tasks


# ── signal construction ─────────────────────────────────────────────


# Family aliases and normalize_family() are now in mining.py.
# Imported at module level as ``normalize_family``.
_normalize_family = normalize_family

# Lookback variations per method type (legacy path only)
_METHOD_LOOKBACKS: dict[str, list[int]] = {
    "factor_scoring": [12],
    "parameter_sweep": [6, 12, 24, 48, 96],
    "ic_analysis": [12, 48],
    "rank_ic_analysis": [12, 48],
}


def _apply_transform(
    raw: pd.Series,
    direction: int,
    transform: str,
    params: dict,
) -> pd.Series:
    """Normalize a raw indicator value into a trading signal in (-1, 1)."""
    if transform == "tanh_zscore":
        window = params.get("zscore_window", 288)
        scale = params.get("tanh_scale", 2.0)
        mu = raw.rolling(window, min_periods=20).mean()
        sigma = raw.rolling(window, min_periods=20).std().replace(0, np.nan)
        z = ((raw - mu) / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Apply smoothing (e.g. 1 hour = 12 bars) to prevent rapid flipping
        smooth_z = z.ewm(span=12, min_periods=1).mean()
        sig = direction * np.tanh(smooth_z / scale)
        return _apply_signal_controls(sig.fillna(0.0), params)
    elif transform == "rank":
        window = params.get("zscore_window", 288)
        signal = raw.rolling(window, min_periods=20).rank(pct=True) * 2 - 1
    elif transform == "raw_clip":
        signal = raw.fillna(0.0).clip(-1, 1)
    else:
        signal = raw.fillna(0.0).clip(-1, 1)
    return _apply_signal_controls((direction * signal).fillna(0.0), params)


def _apply_signal_controls(signal: pd.Series, params: dict) -> pd.Series:
    try:
        smooth_span = int(params.get("smooth_span", 1))
    except (TypeError, ValueError):
        smooth_span = 1
    if smooth_span > 1:
        signal = signal.ewm(span=smooth_span, min_periods=1).mean()

    try:
        threshold = float(params.get("signal_threshold", 0.0))
    except (TypeError, ValueError):
        threshold = 0.0
    if threshold > 0.0:
        signal = signal.where(signal.abs() >= threshold, 0.0)

    return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)


def _apply_candidate_filters(
    signal: pd.Series,
    params: dict,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None,
) -> pd.Series:
    filtered = signal.copy()
    regime_filter = _filter_values(params.get("regime_filter"))
    if regime_filter:
        regimes = pd.Series(forward_regimes.to_numpy(), index=filtered.index).astype(str)
        filtered = filtered.where(regimes.isin(regime_filter), 0.0)

    funding_state_filter = _filter_values(params.get("funding_state_filter"))
    if funding_state_filter:
        states = funding_state_labels(funding_rate, filtered.index).astype(str)
        filtered = filtered.where(states.isin(funding_state_filter), 0.0)

    funding_trend_filter = _filter_values(params.get("funding_trend_filter"))
    if funding_trend_filter:
        trends = funding_trend_labels(funding_rate, filtered.index).astype(str)
        filtered = filtered.where(trends.isin(funding_trend_filter), 0.0)

    return filtered.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)


def _filter_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, list | tuple | set):
        return {str(item) for item in value}
    return {str(value)}


def _build_signal_for(
    candidate: CandidateStrategySpec,
    frame: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_meta: dict,
    index: int,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None = None,
) -> np.ndarray:
    """Construct a trading signal for a single candidate.

    Dispatch order:
      1. ``signal_source="feature"`` — explicit indicator from features_df
      2. ``signal_source="factor_signal"`` — factor_signal() with explicit params
      3. composite — multi-factor blend
      4. legacy — implicit family/index routing (backward compat)
    """
    signal_source = candidate.params.get("signal_source")

    # ── Priority 1: explicit feature indicator ──────────────────────
    if signal_source == "feature":
        indicator_name = candidate.params.get("indicator_name")
        if indicator_name is None or indicator_name not in features_df.columns:
            available = len(features_df.columns)
            raise ValueError(
                f"Candidate {candidate.candidate_id} requests indicator "
                f"'{indicator_name}' but it is not in features_df ({available} cols). "
                f"Fail loud: do not silently fall back."
            )
        raw = features_df[indicator_name]
        direction = candidate.params.get("direction", 1)
        transform = candidate.params.get("transform", "tanh_zscore")
        signal = _apply_transform(raw, direction, transform, candidate.params)
        canonical = _normalize_family(candidate.hypothesis_family)
        if canonical:
            signal = _apply_regime_modulation(signal, forward_regimes, canonical)
        signal = _apply_candidate_filters(signal, candidate.params, forward_regimes, funding_rate)
        return signal.to_numpy(dtype=float)

    # ── Priority 2: explicit factor_signal ──────────────────────────
    if signal_source == "factor_signal":
        family = candidate.params.get("factor_family")
        lookback = candidate.params.get("factor_lookback", 12)
        if family is None:
            raise ValueError(
                f"Candidate {candidate.candidate_id} has signal_source='factor_signal' "
                f"but missing 'factor_family' in params."
            )
        signal = factor_signal(frame, family=family, lookback=lookback, funding_rate=funding_rate)
        direction = int(candidate.params.get("direction", 1))
        signal = _apply_signal_controls(direction * signal.fillna(0), candidate.params).clip(-3, 3)
        canonical = _normalize_family(candidate.hypothesis_family)
        if canonical:
            signal = _apply_regime_modulation(signal, forward_regimes, canonical)
        signal = _apply_candidate_filters(signal, candidate.params, forward_regimes, funding_rate)
        return signal.to_numpy(dtype=float)

    # ── Composite ───────────────────────────────────────────────────
    if candidate.hypothesis_family == "composite" or candidate.params.get("components"):
        return _build_composite_signal(candidate, frame, features_df, feature_meta, index, forward_regimes, funding_rate)

    # ── Legacy fallback (no signal_source in params) ────────────────
    canonical = _normalize_family(candidate.hypothesis_family)

    if canonical is not None:
        lookbacks = _METHOD_LOOKBACKS.get(candidate.method_id, [12])
        lookback = lookbacks[index % len(lookbacks)]
        signal = factor_signal(frame, family=canonical, lookback=lookback, funding_rate=funding_rate)
        signal = signal.fillna(0).clip(-3, 3)
        signal = _apply_regime_modulation(signal, forward_regimes, canonical)
        signal = _apply_candidate_filters(signal, candidate.params, forward_regimes, funding_rate)
        return signal.to_numpy(dtype=float)

    # Fallback: pick an engineered feature by family match
    family_features = [
        col for col, m in feature_meta.items()
        if m.get("family", "") == candidate.hypothesis_family
    ]
    if not family_features:
        family_features = [
            col for col, m in feature_meta.items()
            if m.get("family", "") in ("trend_following", "mean_reversion", "volatility_regime", "volume_confirmation")
        ]
    feature_col = family_features[index % len(family_features)]
    signal = features_df[feature_col].fillna(0).clip(-3, 3)
    signal = _apply_candidate_filters(signal, candidate.params, forward_regimes, funding_rate)
    return signal.to_numpy(dtype=float)


def _build_composite_signal(
    candidate: CandidateStrategySpec,
    frame: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_meta: dict,
    index: int,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None,
) -> np.ndarray:
    components = candidate.params.get("components") or []
    weights = candidate.params.get("weights") or []
    if not isinstance(components, list) or not isinstance(weights, list) or len(components) < 2:
        return np.zeros(len(frame), dtype=float)

    component_signals: list[np.ndarray] = []
    component_weights: list[float] = []
    for offset, (component_payload, raw_weight) in enumerate(zip(components, weights, strict=False)):
        try:
            component = CandidateStrategySpec.model_validate(component_payload)
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if component.candidate_id == candidate.candidate_id or component.hypothesis_family == "composite":
            continue
        signal = _build_signal_for(
            component,
            frame,
            features_df,
            feature_meta,
            index + offset,
            forward_regimes,
            funding_rate,
        )
        component_signals.append(signal)
        component_weights.append(weight)

    if len(component_signals) < 2:
        return np.zeros(len(frame), dtype=float)
    weights_arr = np.asarray(component_weights, dtype=float)
    gross = float(np.sum(np.abs(weights_arr)))
    if gross <= 1e-12:
        return np.zeros(len(frame), dtype=float)
    weights_arr = weights_arr / gross
    combined = np.zeros(len(frame), dtype=float)
    for weight, signal in zip(weights_arr, component_signals, strict=True):
        combined += weight * signal
    controlled = _apply_signal_controls(pd.Series(combined, index=frame.index), candidate.params)
    filtered = _apply_candidate_filters(controlled, candidate.params, forward_regimes, funding_rate)
    return np.clip(filtered.to_numpy(dtype=float), -3.0, 3.0)


# ── regime-conditional signal modulation ────────────────────────────

# Per-family regime multipliers: (bull_mult, bear_mult, sideways_mult, high_vol_mult)
_REGIME_SIGNAL_MODULATION: dict[str, tuple[float, float, float, float]] = {
    "momentum":        (1.5, 0.3, 1.0, 0.5),   # strong in trends, weak in bear
    "mean_reversion":  (0.5, 1.5, 1.0, 0.7),   # strong in bear/reversals
    "volatility":      (0.7, 0.7, 0.5, 1.5),   # strong in high vol
    "funding_basis":   (1.0, 1.2, 1.0, 0.5),   # stronger in bear (funding mean-reverts)
    "volume_confirmation": (1.3, 0.5, 0.8, 0.6),
}


def _apply_regime_modulation(
    signal: pd.Series,
    forward_regimes: pd.Series,
    canonical_family: str,
) -> pd.Series:
    """Scale signal based on forward-predicted regime."""
    multipliers = _REGIME_SIGNAL_MODULATION.get(canonical_family, (1.0, 1.0, 1.0, 1.0))
    regime_map = {"bull": 0, "bear": 1, "sideways": 2, "high_vol": 3}

    # Align indices
    common = signal.index.intersection(forward_regimes.index)
    modulated = signal.copy()
    for regime, idx in regime_map.items():
        mask = forward_regimes.loc[common] == regime
        modulated.loc[common[mask.values]] *= multipliers[idx]

    return modulated.clip(-3, 3)


# ── regime model fitting ────────────────────────────────────────────

def _fit_regime_model(
    frame: pd.DataFrame,
    tail: int | None,
    log_fn,
) -> pd.Series:
    """Fit HMM on an initial prefix and expose only lagged live regimes."""
    from factor_mining.regime.hmm import MarkovRegimeDetector

    detector = MarkovRegimeDetector(n_states=3, random_state=42)
    n_rows = len(frame)
    if n_rows < 100:
        return pd.Series("unknown", index=frame.index)

    max_fit_rows = min(tail or 50_000, 50_000, n_rows)
    if max_fit_rows < 100:
        return pd.Series("unknown", index=frame.index)
    fit_rows = min(max(100, n_rows // 3), max_fit_rows)
    fit_frame = frame.iloc[:fit_rows]
    log_fn(f"  HMM fitting on {len(fit_frame):,} bars...")
    detector.fit(fit_frame)
    fit_states = detector.predict(fit_frame)
    labels = detector.label_states(fit_states, fit_frame)
    log_fn(f"  HMM states: {labels}")

    # Forward regime is computed with a detector trained only on the prefix.
    # The prefix itself is unavailable for OOS decisions, and the final signal is
    # lagged one bar before downstream signal modulation.
    regimes = detector.rolling_forward_regime(frame, horizon=12)
    regimes.iloc[:fit_rows] = "unknown"
    return regimes.shift(1).fillna("unknown")


def _run_backtests_parallel(
    tasks: list[tuple],
    frame: pd.DataFrame,
    settings: Settings,
    max_workers: int | None,
    funding_df: pd.DataFrame | None = None,
) -> list[BacktestResult]:
    """Execute backtests in parallel, collecting successful results."""
    n_workers = min(max_workers or os.cpu_count() or 4, len(tasks))
    result_by_idx: dict[int, BacktestResult] = {}

    slim_tasks = [(arr, cdict, trial_counts, notes) for arr, cdict, _, trial_counts, notes in tasks]

    if n_workers <= 1:
        _init_worker(frame, settings, funding_df)
        for idx, task in enumerate(slim_tasks):
            c = CandidateStrategySpec.model_validate(tasks[idx][1])
            out = _run_one_backtest(task)
            if isinstance(out, Exception):
                _log(f"  [{idx + 1}/{len(tasks)}] {c.candidate_id[:16]}... SKIP: {out}")
            else:
                result_by_idx[idx] = out
        return [result_by_idx[i] for i in sorted(result_by_idx)]

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(frame, settings, funding_df)) as executor:
        future_to_idx = {executor.submit(_run_one_backtest, t): tasks[i][2] for i, t in enumerate(slim_tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            c = CandidateStrategySpec.model_validate(tasks[idx][1])
            out = future.result()
            if isinstance(out, Exception):
                _log(f"  [{idx + 1}/{len(tasks)}] {c.candidate_id[:16]}... SKIP: {out}")
            else:
                result_by_idx[idx] = out

    return [result_by_idx[i] for i in sorted(result_by_idx)]


def _check_mining_boundaries(
    new_candidates: list[CandidateStrategySpec],
    trial_counts: dict[str, int],
    round_backtests: list[BacktestResult],
    hypotheses: list[HypothesisSpec],
) -> bool:
    """Check whether mining should stop for any hypothesis family."""
    return not _filter_candidates_by_mining_boundaries(
        new_candidates,
        trial_counts,
        round_backtests,
        hypotheses,
        log_blocks=False,
    )


def _filter_candidates_by_mining_boundaries(
    new_candidates: list[CandidateStrategySpec],
    trial_counts: dict[str, int],
    round_backtests: list[BacktestResult],
    hypotheses: list[HypothesisSpec],
    *,
    log_blocks: bool,
) -> list[CandidateStrategySpec]:
    """Drop boundary-breaching fresh hypotheses without stopping repair/survivor rounds."""
    allowed: list[CandidateStrategySpec] = []
    blocked: Counter[tuple[str, str]] = Counter()
    for c in new_candidates:
        if _is_repair_candidate(c):
            allowed.append(c)
            continue
        fam = c.hypothesis_family
        trials = trial_counts.get(fam, 0)
        # Find matching backtest for OOS/IS ratio
        matching = [r for r in round_backtests if r.candidate_id == c.candidate_id]
        oos_ratio = 1.0
        be_multiple = 2.0
        if matching:
            r = matching[0]
            oos_ratio = getattr(r, "prior_posterior_ic_ratio", 1.0)
            be_multiple = r.break_even_cost_bps / max(r.actual_cost_bps, 1e-12)

        cont, reason = should_continue_mining(
            fam, cumulative_trials=trials, oos_ic_ratio=oos_ratio,
            break_even_multiple=be_multiple,
        )
        if cont:
            allowed.append(c)
        else:
            blocked[(fam, reason)] += 1
    if log_blocks:
        for (fam, reason), count in blocked.most_common():
            _log(f"  Block {count} new [{fam}] candidates: {reason}")
    return allowed


def _is_repair_candidate(candidate: CandidateStrategySpec) -> bool:
    generated_by = str(candidate.params.get("generated_by", ""))
    variant = str(candidate.params.get("search_variant", ""))
    return generated_by in {
        "pre_gate_repair",
        "near_miss_repair",
        "optimizer_repair",
        "minimax_survivor_adjustment",
        "minimax_survivor_composite",
    } or variant.startswith("repair_")


def _archive_top(result: PipelineResult, settings: Settings, archive_top: int) -> int:
    """Archive the top-scoring experiments across all rounds."""
    from factor_mining.archive import archive_experiment

    archived = 0
    for c, r, s in result.top_candidates[:archive_top]:
        gc = next((g for g in result.gatechecks if g.experiment_id == r.experiment_id), None)
        if gc is None:
            continue
        try:
            archive_experiment(result=r, gatecheck=gc, hardscore=s, settings=settings)
            archived += 1
        except Exception:
            pass
    return archived


def _step_header(n: int, description: str) -> None:
    _separator()
    print(f"\n[{n}] {description}...")
    _emit_event("step", "info", description, {"step": n})


def _separator() -> None:
    print("-" * 66)


def _log(msg: str) -> None:
    print(f"  {msg}")
    _emit_event("log", "info", msg, None)


def _emit_event(phase: str, level: str, message: str, payload: dict[str, Any] | None = None) -> None:
    if _EVENT_SINK is not None:
        try:
            _EVENT_SINK(phase, level, message, payload)
        except Exception:
            pass
