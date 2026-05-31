"""Full pipeline orchestrator: DeepSeek/default hypotheses → backtest → gatecheck → hardscore → optimize → archive.

Supports separate discovery and optimization round budgets. Discovery rounds can
generate broad candidates and repairs; optimization rounds tune survivor outputs
and stop early on convergence.
"""

from __future__ import annotations

import os
import hashlib
import json
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations, product
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
from factor_mining.mining import (
    build_indicator_candidates,
    build_v1_candidates,
    default_hypotheses,
    factor_signal,
    filter_candidates_for_lab_factors,
    generate_hypotheses_with_deepseek,
    normalize_family,
)
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


@dataclass
class SignalBuildContext:
    frame: pd.DataFrame
    features_df: pd.DataFrame
    feature_meta: dict
    forward_regimes: pd.Series
    funding_rate: pd.Series | None
    factor_signal_cache: dict[tuple[str, int], pd.Series] = field(default_factory=dict)
    feature_transform_cache: dict[str, pd.Series] = field(default_factory=dict)
    filter_mask_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = field(default_factory=dict)
    candidate_signal_cache: dict[str, np.ndarray] = field(default_factory=dict)
    cache_hits: Counter[str] = field(default_factory=Counter)
    cache_misses: Counter[str] = field(default_factory=Counter)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _in_flight: dict[tuple[str, Any], threading.Event] = field(default_factory=dict)
    _cache_errors: dict[tuple[str, Any], Exception] = field(default_factory=dict)

    def cached(self, cache_name: str, cache: dict, key: Any, compute: Callable[[], Any]) -> Any:
        token = (cache_name, key)
        while True:
            with self._lock:
                if key in cache:
                    self.cache_hits[cache_name] += 1
                    return cache[key]
                if token in self._cache_errors:
                    raise self._cache_errors[token]
                event = self._in_flight.get(token)
                if event is None:
                    event = threading.Event()
                    self._in_flight[token] = event
                    self.cache_misses[cache_name] += 1
                    break
            event.wait()

        try:
            value = compute()
        except Exception as exc:
            with self._lock:
                self._cache_errors[token] = exc
                self._in_flight.pop(token, None)
                event.set()
            raise

        with self._lock:
            cache[key] = value
            self._in_flight.pop(token, None)
            event.set()
        return value

    def cached_filter_mask(self, kind: str, values: set[str]) -> np.ndarray:
        normalized = tuple(sorted(str(item) for item in values))
        key = (kind, normalized)

        def compute() -> np.ndarray:
            if kind == "regime":
                labels = pd.Series(self.forward_regimes.to_numpy(), index=self.frame.index).astype(str)
            elif kind == "funding_state":
                labels = funding_state_labels(self.funding_rate, self.frame.index).astype(str)
            elif kind == "funding_trend":
                labels = funding_trend_labels(self.funding_rate, self.frame.index).astype(str)
            else:
                raise ValueError(f"Unknown filter mask kind: {kind}")
            return labels.isin(set(normalized)).to_numpy(dtype=bool)

        return self.cached("filter_mask", self.filter_mask_cache, key, compute)


@dataclass(frozen=True)
class SignalBuildSkip:
    index: int
    candidate_id: str
    error: ValueError


# ── multiprocessing worker state ────────────────────────────────────

_worker_frame: pd.DataFrame | None = None
_worker_settings: Settings | None = None
_worker_funding: pd.DataFrame | None = None
_EVENT_SINK: Callable[[str, str, str, dict[str, Any] | None], None] | None = None
_RUN_ID: str | None = None

_CHECKPOINT_SCHEMA_VERSION = 1
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
_LOCAL_TUNING_PARENT_LIMIT = 8
_LOCAL_TUNING_MAX_PER_PARENT = 64
_LOCAL_TUNING_TOTAL_LIMIT = 256
_LOCAL_TUNING_TOP_K_PER_PARENT = 3
_LOCAL_TUNING_LOOKBACKS = (6, 12, 24, 48, 96)
_LOCAL_TUNING_SMOOTH_SPANS = (1, 12, 24, 48, 96)
_LOCAL_TUNING_SIGNAL_THRESHOLDS = (0.0, 0.10, 0.20, 0.30)
_LOCAL_TUNING_POSITION_BUFFERS = (0.05, 0.10, 0.20, 0.30)
_LOCAL_TUNING_ZSCORE_WINDOWS = (96, 288, 576)
_LOCAL_TUNING_TANH_SCALES = (1.0, 2.0, 3.0)
_LOCAL_TUNING_MIN_PRIORITY = 1.75
_LOCAL_TUNING_MIN_WEAK_PRIORITY = 1.25
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


def _filter_candidates_for_direction_scope(
    candidates: list[CandidateStrategySpec],
    direction_scope: dict[str, Any] | None,
) -> list[CandidateStrategySpec]:
    if not direction_scope:
        return candidates
    factor_ids = list(direction_scope.get("factor_ids") or [])
    if not factor_ids:
        return candidates
    return filter_candidates_for_lab_factors(candidates, factor_ids)


def _run_checkpoint_args(**kwargs: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None
    }


def _settings_hash(settings: Settings) -> str:
    payload = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _args_hash(run_args: dict[str, Any]) -> str:
    payload = json.dumps(run_args, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_checkpoint_id(run_id: str, name: str) -> str:
    return f"checkpoint:{run_id}:{name}"


def _stage_checkpoint_id(run_id: str, *, round_num: int, symbol: str, market: str, stage: str) -> str:
    return f"checkpoint:{run_id}:round{round_num}:{symbol}:{market}:{stage}"


def _base_checkpoint_meta(settings: Settings, run_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "settings_hash": _settings_hash(settings),
        "args_hash": _args_hash(run_args),
    }


def _checkpoint_fingerprint(
    settings: Settings,
    *,
    run_args: dict[str, Any],
    symbol: str,
    market: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        **_base_checkpoint_meta(settings, run_args),
        "symbol": symbol,
        "market": market,
        "row_count": int(len(frame)),
        "open_time_min": None if frame.empty else int(frame["open_time"].min()),
        "open_time_max": None if frame.empty else int(frame["open_time"].max()),
    }


def _save_run_checkpoint_payload(
    store: MetadataStore | None,
    run_id: str | None,
    name: str,
    *,
    settings: Settings,
    run_args: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    if store is None or not run_id:
        return
    store.save_artifact(
        _run_checkpoint_id(run_id, name),
        "pipeline_checkpoint",
        {
            "fingerprint": _base_checkpoint_meta(settings, run_args),
            "payload": payload,
        },
    )


def _load_run_checkpoint_payload(
    store: MetadataStore | None,
    run_id: str | None,
    name: str,
    *,
    settings: Settings,
    run_args: dict[str, Any],
) -> dict[str, Any] | None:
    if store is None or not run_id:
        return None
    artifact = store.load_artifact(_run_checkpoint_id(run_id, name))
    if artifact is None:
        return None
    expected = _base_checkpoint_meta(settings, run_args)
    if artifact.get("fingerprint") != expected:
        raise ValueError(f"Checkpoint {name} fingerprint mismatch for run {run_id}")
    payload = artifact.get("payload")
    return payload if isinstance(payload, dict) else None


def _save_stage_checkpoint(
    store: MetadataStore | None,
    run_id: str | None,
    *,
    round_num: int,
    symbol: str,
    market: str,
    stage: str,
    fingerprint: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    if store is None or not run_id:
        return
    store.save_artifact(
        _stage_checkpoint_id(run_id, round_num=round_num, symbol=symbol, market=market, stage=stage),
        "pipeline_checkpoint",
        {
            "fingerprint": fingerprint,
            "payload": payload,
        },
    )


def _load_stage_checkpoint(
    store: MetadataStore | None,
    run_id: str | None,
    *,
    round_num: int,
    symbol: str,
    market: str,
    stage: str,
    fingerprint: dict[str, Any],
) -> dict[str, Any] | None:
    if store is None or not run_id:
        return None
    artifact_id = _stage_checkpoint_id(run_id, round_num=round_num, symbol=symbol, market=market, stage=stage)
    artifact = store.load_artifact(artifact_id)
    if artifact is None:
        return None
    if artifact.get("fingerprint") != fingerprint:
        raise ValueError(f"Checkpoint {artifact_id} fingerprint mismatch")
    payload = artifact.get("payload")
    return payload if isinstance(payload, dict) else None


def run_pipeline(
    settings: Settings,
    *,
    use_llm: bool = True,
    max_workers: int | None = None,
    tail: int | None = None,
    sample_bars: int | None = None,
    sample_mode: str = "block",
    seed: int = 42,
    archive_top: int = 3,
    research_brief: str | None = None,
    hypothesis_count: int = 5,
    iterations: int = 1,
    discovery_rounds: int | None = None,
    optimization_rounds: int | None = None,
    store: MetadataStore | None = None,
    event_sink: Callable[[str, str, str, dict[str, Any] | None], None] | None = None,
    run_id: str | None = None,
    resume_run_id: str | None = None,
    seed_hypotheses: list[HypothesisSpec] | None = None,
    direction_scope: dict[str, Any] | None = None,
    stop_event: threading.Event | None = None,
) -> PipelineResult:
    """Execute the full factor mining workflow with optional iterative optimization.

    Steps:
      1. Hypothesis generation (DeepSeek or defaults)
      2. Build initial candidates + load data + generate features
      3-6. Mining round(s): backtest → gatecheck → hardscore → optimize

    Args:
        iterations: Legacy total round budget. When split controls are omitted,
            this runs one discovery round plus iterations - 1 optimization rounds.
        discovery_rounds: Rounds that can generate broad candidates and repairs.
        optimization_rounds: Rounds that tune survivor candidates and stop on convergence.
    """
    if tail is not None and sample_bars is not None:
        raise ValueError("--tail and --sample-bars are mutually exclusive")
    if sample_bars is not None and sample_mode != "block":
        raise ValueError("sample_mode must be 'block'")

    global _EVENT_SINK, _RUN_ID
    previous_sink = _EVENT_SINK
    previous_run_id = _RUN_ID
    _EVENT_SINK = event_sink
    _RUN_ID = run_id
    t_start = time.perf_counter()
    result = PipelineResult()
    normalized_scope = _normalize_direction_scope(direction_scope)
    effective_settings = _settings_for_direction_scope(settings, normalized_scope)
    effective_research_brief = _research_brief_for_direction_scope(research_brief, normalized_scope)
    resolved_discovery_rounds, resolved_optimization_rounds, resolved_total_rounds = _resolve_round_controls(
        iterations=iterations,
        discovery_rounds=discovery_rounds,
        optimization_rounds=optimization_rounds,
    )
    try:
        return _run_pipeline_impl(
            effective_settings,
            use_llm=use_llm,
            max_workers=max_workers,
            tail=tail,
            sample_bars=sample_bars,
            sample_mode=sample_mode,
            seed=seed,
            archive_top=archive_top,
            research_brief=effective_research_brief,
            hypothesis_count=hypothesis_count,
            iterations=resolved_total_rounds,
            discovery_rounds=resolved_discovery_rounds,
            optimization_rounds=resolved_optimization_rounds,
            store=store,
            result=result,
            t_start=t_start,
            run_id=run_id,
            resume_run_id=resume_run_id,
            seed_hypotheses=seed_hypotheses,
            direction_scope=normalized_scope,
            stop_event=stop_event,
        )
    finally:
        _EVENT_SINK = previous_sink
        _RUN_ID = previous_run_id


def _resolve_round_controls(
    *,
    iterations: int,
    discovery_rounds: int | None,
    optimization_rounds: int | None,
) -> tuple[int, int, int]:
    legacy_total = max(1, int(iterations or 1))
    if discovery_rounds is None and optimization_rounds is None:
        discovery = 1
        optimization = max(0, legacy_total - 1)
    else:
        discovery = max(1, int(discovery_rounds if discovery_rounds is not None else 1))
        optimization = max(0, int(optimization_rounds if optimization_rounds is not None else 0))
    return discovery, optimization, discovery + optimization


def _run_pipeline_impl(
    settings: Settings,
    *,
    use_llm: bool,
    max_workers: int | None,
    tail: int | None,
    sample_bars: int | None,
    sample_mode: str,
    seed: int,
    archive_top: int,
    research_brief: str | None,
    hypothesis_count: int,
    iterations: int,
    discovery_rounds: int,
    optimization_rounds: int,
    store: MetadataStore | None,
    result: PipelineResult,
    t_start: float,
    run_id: str | None,
    resume_run_id: str | None,
    seed_hypotheses: list[HypothesisSpec] | None,
    direction_scope: dict[str, Any] | None,
    stop_event: threading.Event | None = None,
) -> PipelineResult:
    run_args = _run_checkpoint_args(
        use_llm=use_llm,
        max_workers=max_workers,
        tail=tail,
        sample_bars=sample_bars,
        sample_mode=sample_mode,
        seed=seed,
        archive_top=archive_top,
        research_brief=research_brief,
        hypothesis_count=hypothesis_count,
        iterations=iterations,
        discovery_rounds=discovery_rounds,
        optimization_rounds=optimization_rounds,
        direction_scope=direction_scope,
    )
    checkpoint_source_run_id = resume_run_id or run_id
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

    hypotheses_checkpoint = _load_run_checkpoint_payload(
        store,
        checkpoint_source_run_id if resume_run_id else None,
        "hypotheses",
        settings=settings,
        run_args=run_args,
    )
    if hypotheses_checkpoint is not None:
        result.hypotheses = [
            HypothesisSpec.model_validate(item)
            for item in hypotheses_checkpoint.get("items", [])
        ]
        _log(f"Resumed {len(result.hypotheses)} hypotheses from {resume_run_id}")
    elif seed_hypotheses is not None:
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
        if run_id:
            _save_run_checkpoint_payload(
                store,
                run_id,
                "hypotheses",
                settings=settings,
                run_args=run_args,
                payload={"items": [h.model_dump(mode="json") for h in result.hypotheses]},
            )

    # ── Step 2: Candidates + Data (once) ────────────────────────────
    _step_header(2, "Building candidates and loading data")
    t0 = time.perf_counter()

    initial_checkpoint = _load_run_checkpoint_payload(
        store,
        checkpoint_source_run_id if resume_run_id else None,
        "initial_candidates",
        settings=settings,
        run_args=run_args,
    )
    if initial_checkpoint is not None:
        initial_candidates = [
            CandidateStrategySpec.model_validate(item)
            for item in initial_checkpoint.get("items", [])
        ]
        _log(f"Resumed {len(initial_candidates)} initial candidates from {resume_run_id}")
        data_contexts = _load_data_contexts(
            initial_candidates,
            settings,
            tail=tail,
            sample_bars=sample_bars,
            sample_mode=sample_mode,
            seed=seed,
        )
        survivor_seed_candidates = [
            c for c in survivor_seed_candidates
            if _data_key(c) in data_contexts
        ]
    else:
        initial_candidates = build_v1_candidates(
            result.hypotheses, symbols=settings.data.symbols, interval=settings.data.default_interval,
        )
        initial_candidates = _annotate_candidates_for_direction_scope(initial_candidates, direction_scope)
        if survivor_seed_candidates:
            initial_candidates = _dedupe_candidates(survivor_seed_candidates + initial_candidates)
        _log(f"{len(initial_candidates)} initial candidates ({len(result.hypotheses)} hypotheses × {len(settings.data.symbols)} symbols × methods)")

        data_contexts = _load_data_contexts(
            initial_candidates,
            settings,
            tail=tail,
            sample_bars=sample_bars,
            sample_mode=sample_mode,
            seed=seed,
        )
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
        indicator_candidates = _filter_candidates_for_direction_scope(indicator_candidates, direction_scope)
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
        if run_id:
            _save_run_checkpoint_payload(
                store,
                run_id,
                "initial_candidates",
                settings=settings,
                run_args=run_args,
                payload={"items": [c.model_dump(mode="json") for c in initial_candidates]},
            )

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
    previous_optimization_signature: str | None = None

    for round_num in range(1, iterations + 1):
        phase = "discovery" if round_num <= discovery_rounds else "optimization"
        phase_round = round_num if phase == "discovery" else round_num - discovery_rounds
        phase_total = discovery_rounds if phase == "discovery" else optimization_rounds
        _step_header(
            2 + round_num,
            f"{phase.title()} round {phase_round}/{phase_total} — {len(current_candidates)} candidates",
        )
        _check_stop(stop_event)

        round_backtests: list[BacktestResult] = []
        round_gatechecks: list[GateCheckResult] = []
        round_hardscores: list[HardScoreReport] = []
        round_candidates: list[CandidateStrategySpec] = []
        new_candidates: list[CandidateStrategySpec] = []
        child_history: list[dict[str, Any]] = []

        grouped_candidates = _group_candidates_by_data(current_candidates)
        runnable_groups: list[tuple[tuple[str, str], MarketDataContext, list[CandidateStrategySpec]]] = []
        for key, symbol_candidates in grouped_candidates.items():
            _check_stop(stop_event)
            context = data_contexts.get(key)
            if context is None:
                _log(f"  Skip {key[0]}/{key[1]}: no local parquet data")
                continue
            _log(f"  {context.symbol}/{context.market}: {len(symbol_candidates)} candidates")
            runnable_groups.append((key, context, symbol_candidates))

        symbol_workers, symbol_max_workers = _symbol_round_parallelism(len(runnable_groups), max_workers)
        trial_counts_lock = threading.Lock() if symbol_workers > 1 else None

        def run_symbol_group(
            context: MarketDataContext,
            symbol_candidates: list[CandidateStrategySpec],
        ) -> dict[str, Any]:
            return _run_mining_round(
                current_candidates=symbol_candidates,
                frame=context.frame,
                features_df=context.features_df,
                feature_meta=context.feature_meta,
                forward_regimes=context.forward_regimes,
                funding_df=context.funding_df,
                funding_rate=context.funding_rate,
                data_quality_notes=context.data_quality_notes,
                settings=settings,
                max_workers=symbol_max_workers,
                store=store,
                iteration=round_num - 1,
                round_num=round_num,
                phase=phase,
                allow_pre_gate_repair=phase == "discovery",
                allow_optimizer_repairs=phase == "discovery",
                allow_next_hypotheses=phase == "discovery",
                artifact_scope=f"round{round_num}_{context.symbol}_{context.market}",
                cumulative_trial_counts=cumulative_trial_counts,
                survivor_candidate_ids=survivor_candidate_ids,
                previous_actions=list(result.optimization_history),
                run_id=run_id,
                resume_run_id=resume_run_id,
                run_args=run_args,
                stop_event=stop_event,
                trial_counts_lock=trial_counts_lock,
            )

        round_data_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        if symbol_workers > 1:
            _log(
                f"  Running {len(runnable_groups)} data groups in parallel "
                f"({symbol_workers} symbol workers, {symbol_max_workers} backtest workers/group)"
            )
            with ThreadPoolExecutor(max_workers=symbol_workers) as executor:
                future_to_key = {
                    executor.submit(run_symbol_group, context, symbol_candidates): key
                    for key, context, symbol_candidates in runnable_groups
                }
                for future in as_completed(future_to_key):
                    round_data_by_key[future_to_key[future]] = future.result()
        else:
            for key, context, symbol_candidates in runnable_groups:
                round_data_by_key[key] = run_symbol_group(context, symbol_candidates)

        for key, _context, _symbol_candidates in runnable_groups:
            round_data = round_data_by_key[key]

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
        generated_candidates = list(new_candidates)

        # Check boundary conditions for continued mining
        if iterations > 1 and new_candidates:
            new_candidates = _filter_candidates_by_mining_boundaries(
                new_candidates, cumulative_trial_counts, round_backtests, result.hypotheses, log_blocks=True,
            )

        history_entry = _combine_round_history(
            round_num,
            round_num - 1,
            child_history,
            generated_candidates,
            phase=phase,
            next_candidates_count=len(new_candidates),
        )
        converged = False
        if phase == "optimization":
            signature = _candidate_output_signature(new_candidates)
            history_entry["output_signature"] = signature
            if previous_optimization_signature == signature and round_num < iterations:
                history_entry["converged"] = True
                converged = True
            previous_optimization_signature = signature
        result.optimization_history.append(history_entry)

        if not generated_candidates and round_num < iterations:
            _log("No new candidates from optimization — stopping early.")
            break
        if generated_candidates and not new_candidates:
            _log("Boundary conditions triggered — stopping early.")
            break
        if converged:
            _log("Optimization converged — stopping early.")
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
        _log(f"  Optimizer combos: {len(combos)}")
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


def verify_research_survivors(
    settings: Settings,
    *,
    store: MetadataStore,
    max_workers: int | None = None,
    tail: int | None = None,
    sample_bars: int | None = None,
    sample_mode: str = "block",
    seed: int = 42,
    event_sink: Callable[[str, str, str, dict[str, Any] | None], None] | None = None,
    run_id: str | None = None,
) -> PipelineResult:
    """Re-evaluate active research survivors without running discovery or optimization."""
    if tail is not None and sample_bars is not None:
        raise ValueError("--tail and --sample-bars are mutually exclusive")
    if sample_bars is not None and sample_mode != "block":
        raise ValueError("sample_mode must be 'block'")

    global _EVENT_SINK, _RUN_ID
    previous_sink = _EVENT_SINK
    previous_run_id = _RUN_ID
    _EVENT_SINK = event_sink
    _RUN_ID = run_id
    t_start = time.perf_counter()
    result = PipelineResult()
    try:
        _step_header(1, "Verifying active Research Survivors")
        records = store.list_research_survivors(status="active")
        candidates = _survivor_seed_candidates(records, result.errors)
        if not candidates:
            _log("No active research survivors with valid candidate payloads.")
            result.elapsed_s = time.perf_counter() - t_start
            return result

        _log(f"Loaded {len(candidates)} survivor candidates")
        data_contexts = _load_data_contexts(
            candidates,
            settings,
            tail=tail,
            sample_bars=sample_bars,
            sample_mode=sample_mode,
            seed=seed,
        )

        all_candidates: list[CandidateStrategySpec] = []
        all_backtests: list[BacktestResult] = []
        all_gatechecks: list[GateCheckResult] = []
        all_factor_evidence: list[FactorEvidenceReport] = []
        all_research_gates: list[ResearchGateResult] = []
        all_research_survivors: list[dict[str, Any]] = []

        for key, symbol_candidates in _group_candidates_by_data(candidates).items():
            context = data_contexts.get(key)
            if context is None:
                _log(f"  Skip {key[0]}/{key[1]}: no local parquet data")
                continue
            _log(f"  {context.symbol}/{context.market}: verifying {len(symbol_candidates)} survivors")
            split_plan = _build_data_split_plan(context.frame, regimes=context.forward_regimes)
            final_frame = _masked_frame(context.frame, split_plan.final_oos_mask)
            final_regimes = _masked_series(context.forward_regimes, split_plan.final_oos_mask)
            final_funding_rate = _masked_series(context.funding_rate, split_plan.final_oos_mask)
            symbol_candidates, skipped_funding = _filter_unfunded_factor_signal_candidates(
                symbol_candidates,
                context.funding_rate,
            )
            if skipped_funding:
                _log(f"  Funding survivor candidates skipped: {skipped_funding}")
            trial_counts = _candidate_trial_count_snapshots(symbol_candidates, store, settings)
            full_tasks = _build_tasks(
                symbol_candidates,
                context.frame,
                context.features_df,
                context.feature_meta,
                context.forward_regimes,
                context.funding_rate,
                trial_counts_by_candidate=trial_counts,
                data_quality_notes=context.data_quality_notes,
                max_workers=max_workers,
            )
            final_tasks = _slice_tasks(full_tasks, split_plan.final_oos_mask)
            round_backtests = _run_backtests_parallel(
                final_tasks,
                final_frame,
                settings,
                max_workers,
                context.funding_df,
            )
            if round_backtests:
                _apply_batch_pbo(final_frame, final_tasks, round_backtests, settings, context.funding_df)

            result_ids = {item.candidate_id for item in round_backtests}
            round_candidates = [
                candidate for candidate in symbol_candidates
                if candidate.candidate_id in result_ids
            ]
            round_evidence = build_factor_evidence_reports(
                frame=final_frame,
                tasks=final_tasks,
                candidates=round_candidates,
                results=round_backtests,
                settings=settings,
                forward_regimes=final_regimes,
                funding_rate=final_funding_rate,
                funding_df=context.funding_df,
            )

            from factor_mining.registry import get_method
            from factor_mining.validation.gatecheck import apply_fdr, apply_risk_stratified_gatechecks, run_gatecheck

            fdr_map = apply_fdr(round_backtests, settings)
            methods_map = {method.method_id: method for method in METHOD_REGISTRY}
            round_gatechecks: list[GateCheckResult] = []
            for backtest in round_backtests:
                method = methods_map.get(backtest.method_id) or get_method(backtest.method_id)
                fdr_p = fdr_map.get(
                    backtest.experiment_id,
                    combined_ic_tstat_pvalue(backtest.ic_tstat_nw, backtest.rankic_tstat_nw),
                )
                round_gatechecks.append(run_gatecheck(backtest, settings, method=method, fdr_adjusted_pvalue=fdr_p))
            apply_risk_stratified_gatechecks(round_backtests, round_gatechecks, round_evidence, settings)
            round_research_gates = apply_research_gate(round_backtests, round_gatechecks, round_evidence)
            persistent_records = build_research_survivor_records(
                candidates_by_id={candidate.candidate_id: candidate for candidate in round_candidates},
                results=round_backtests,
                research_gates=round_research_gates,
                fdr_map=fdr_map,
                settings=settings,
            )
            survivor_payloads = research_survivor_payloads(
                {candidate.candidate_id: candidate for candidate in round_candidates},
                round_backtests,
                round_research_gates,
            )
            _augment_research_survivor_payloads(survivor_payloads, persistent_records)
            _update_research_survivor_store(
                store=store,
                records=persistent_records,
                rechecked_candidate_ids={candidate.candidate_id for candidate in round_candidates},
                research_gates=round_research_gates,
                results=round_backtests,
                fdr_map=fdr_map,
                settings=settings,
            )

            all_candidates.extend(round_candidates)
            all_backtests.extend(round_backtests)
            all_gatechecks.extend(round_gatechecks)
            all_factor_evidence.extend(round_evidence)
            all_research_gates.extend(round_research_gates)
            all_research_survivors.extend(survivor_payloads)
            _log(f"  Verified {len(round_backtests)} survivors for {context.symbol}/{context.market}")

        result.candidates = all_candidates
        result.backtests = all_backtests
        result.gatechecks = all_gatechecks
        result.factor_evidence = all_factor_evidence
        result.research_gates = all_research_gates
        result.n_gatecheck_passed = sum(1 for gate in all_gatechecks if gate.passed)
        result.total_rounds = 1 if all_backtests else 0
        result.elapsed_s = time.perf_counter() - t_start
        artifact_id = f"survivor_verify_{run_id or uuid.uuid4().hex[:12]}"
        store.save_artifact(artifact_id, "survivor_verify", {
            "candidates": [candidate.model_dump(mode="json") for candidate in all_candidates],
            "backtests": [backtest.model_dump(mode="json") for backtest in all_backtests],
            "gatechecks": [gate.model_dump(mode="json") for gate in all_gatechecks],
            "research_gate": [gate.model_dump(mode="json") for gate in all_research_gates],
            "research_survivors": all_research_survivors,
        })
        _log(f"Survivor verification complete: {len(all_backtests)} evaluated")
        return result
    finally:
        _EVENT_SINK = previous_sink
        _RUN_ID = previous_run_id


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
    phase: str = "discovery",
    allow_pre_gate_repair: bool = True,
    allow_optimizer_repairs: bool = True,
    allow_next_hypotheses: bool = True,
    artifact_scope: str | None = None,
    survivor_candidate_ids: set[str] | None = None,
    previous_actions: list[dict] | None = None,
    run_id: str | None = None,
    resume_run_id: str | None = None,
    run_args: dict[str, Any] | None = None,
    stop_event: threading.Event | None = None,
    trial_counts_lock: Any | None = None,
) -> dict[str, Any]:
    """Execute one complete mining round: backtest → gatecheck → hardscore → optimize."""
    artifact_scope = artifact_scope or f"round{round_num}"
    _check_stop(stop_event)
    survivor_candidate_ids = survivor_candidate_ids or set()
    run_args = run_args or {}

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
    checkpoint_symbol = current_candidates[0].symbol if current_candidates else "unknown"
    checkpoint_market = current_candidates[0].market if current_candidates else "unknown"
    checkpoint_source = resume_run_id or run_id
    checkpoint_fingerprint = _checkpoint_fingerprint(
        settings,
        run_args=run_args,
        symbol=checkpoint_symbol,
        market=checkpoint_market,
        frame=frame,
    )

    current_candidates, skipped_funding = _filter_unfunded_factor_signal_candidates(current_candidates, funding_rate)
    if skipped_funding:
        _log(
            f"  Funding factor_signal candidates skipped: {skipped_funding}; "
            "supplemental funding features still allowed"
        )

    trial_counts_by_candidate = _record_candidate_trials(
        current_candidates,
        store,
        settings,
        cumulative_trial_counts,
        trial_counts_lock=trial_counts_lock,
    )
    full_tasks = _build_tasks(
        current_candidates,
        frame,
        features_df,
        feature_meta,
        forward_regimes,
        funding_rate,
        trial_counts_by_candidate=trial_counts_by_candidate,
        data_quality_notes=data_quality_notes,
        max_workers=max_workers,
    )
    discovery_tasks = _slice_tasks(full_tasks, split_plan.discovery_mask)
    discovery_checkpoint = _load_stage_checkpoint(
        store,
        checkpoint_source if resume_run_id else None,
        round_num=round_num,
        symbol=checkpoint_symbol,
        market=checkpoint_market,
        stage="discovery_backtests",
        fingerprint=checkpoint_fingerprint,
    )
    if discovery_checkpoint is not None:
        discovery_backtests = [
            BacktestResult.model_validate(item)
            for item in discovery_checkpoint.get("items", [])
        ]
        _log(f"  Discovery backtests: resumed {len(discovery_backtests)} from checkpoint")
    else:
        discovery_backtests = _run_backtests_parallel(discovery_tasks, discovery_frame, settings, max_workers, funding_df)
        _save_stage_checkpoint(
            store,
            run_id,
            round_num=round_num,
            symbol=checkpoint_symbol,
            market=checkpoint_market,
            stage="discovery_backtests",
            fingerprint=checkpoint_fingerprint,
            payload={"items": [result.model_dump(mode="json") for result in discovery_backtests]},
        )

    # Align candidates with successful backtests
    discovery_backtest_ids = {r.candidate_id for r in discovery_backtests}
    discovery_candidates = [c for c in current_candidates if c.candidate_id in discovery_backtest_ids]

    _log(f"  Discovery backtests: {len(discovery_backtests)}/{len(current_candidates)} completed "
         f"({time.perf_counter() - t0:.0f}s)")
    _check_stop(stop_event)

    if discovery_backtests:
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

    pre_gate_candidates: list[CandidateStrategySpec] = []
    if allow_pre_gate_repair:
        pre_gate_checkpoint = _load_stage_checkpoint(
            store,
            checkpoint_source if resume_run_id else None,
            round_num=round_num,
            symbol=checkpoint_symbol,
            market=checkpoint_market,
            stage="pre_gate_candidates",
            fingerprint=checkpoint_fingerprint,
        )
        if pre_gate_checkpoint is not None:
            pre_gate_candidates = [
                CandidateStrategySpec.model_validate(item)
                for item in pre_gate_checkpoint.get("items", [])
            ]
            _log(f"  Pre-Gate repair candidates: resumed {len(pre_gate_candidates)} from checkpoint")
        else:
            pre_gate_repairs = _build_pre_gate_repair_candidates(
                discovery_candidates,
                discovery_backtests,
                initial_factor_evidence,
            )
            local_tuning_candidates = _build_local_grid_tuning_candidates(
                discovery_candidates,
                discovery_backtests,
                initial_factor_evidence,
            )
            pre_gate_candidates = pre_gate_repairs + local_tuning_candidates
            _save_stage_checkpoint(
                store,
                run_id,
                round_num=round_num,
                symbol=checkpoint_symbol,
                market=checkpoint_market,
                stage="pre_gate_candidates",
                fingerprint=checkpoint_fingerprint,
                payload={"items": [candidate.model_dump(mode="json") for candidate in pre_gate_candidates]},
            )
    else:
        _log("  Pre-Gate repair skipped for optimization round")
    pre_gate_generated = len(pre_gate_candidates)
    pre_gate_generated_by_kind = Counter(
        str(candidate.params.get("generated_by", "unknown"))
        for candidate in pre_gate_candidates
    )
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
        repair_counts = _record_candidate_trials(
            pre_gate_candidates,
            store,
            settings,
            cumulative_trial_counts,
            trial_counts_lock=trial_counts_lock,
        )
        repair_full_tasks = _build_tasks(
            pre_gate_candidates,
            frame,
            features_df,
            feature_meta,
            forward_regimes,
            funding_rate,
            trial_counts_by_candidate=repair_counts,
            data_quality_notes=data_quality_notes,
            max_workers=max_workers,
        )
        validation_full_tasks.extend(repair_full_tasks)
        validation_candidates.extend(pre_gate_candidates)
        _log(
            f"  Pre-Gate repair generated: {len(pre_gate_candidates)} candidates "
            f"{dict(pre_gate_generated_by_kind)} "
            f"({time.perf_counter() - t_repair:.0f}s)"
        )

    validation_tasks = _slice_tasks(validation_full_tasks, split_plan.repair_validation_mask)
    validation_checkpoint = _load_stage_checkpoint(
        store,
        checkpoint_source if resume_run_id else None,
        round_num=round_num,
        symbol=checkpoint_symbol,
        market=checkpoint_market,
        stage="validation_backtests",
        fingerprint=checkpoint_fingerprint,
    )
    if validation_checkpoint is not None:
        validation_backtests = [
            BacktestResult.model_validate(item)
            for item in validation_checkpoint.get("items", [])
        ]
        _log(f"  Repair validation backtests: resumed {len(validation_backtests)} from checkpoint")
    else:
        validation_backtests = _run_backtests_parallel(
            validation_tasks,
            repair_validation_frame,
            settings,
            max_workers,
            funding_df,
        )
        if validation_backtests:
            _apply_batch_pbo(repair_validation_frame, validation_tasks, validation_backtests, settings, funding_df)
        _save_stage_checkpoint(
            store,
            run_id,
            round_num=round_num,
            symbol=checkpoint_symbol,
            market=checkpoint_market,
            stage="validation_backtests",
            fingerprint=checkpoint_fingerprint,
            payload={"items": [result.model_dump(mode="json") for result in validation_backtests]},
        )
    pre_gate_ids = {candidate.candidate_id for candidate in pre_gate_candidates}
    pre_gate_completed = sum(1 for result in validation_backtests if result.candidate_id in pre_gate_ids)
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
                if candidate.params.get("generated_by") in {"pre_gate_repair", "local_grid_tuning"}
            ],
            "generated_by_kind": dict(pre_gate_generated_by_kind),
            "diagnostics": repair_merge_diagnostics,
        })

    validation_result_by_candidate = {
        result.candidate_id: result
        for result in merge_plan.validation_results
    }

    final_tasks = _slice_tasks(merge_plan.full_tasks, split_plan.final_oos_mask)
    final_checkpoint = _load_stage_checkpoint(
        store,
        checkpoint_source if resume_run_id else None,
        round_num=round_num,
        symbol=checkpoint_symbol,
        market=checkpoint_market,
        stage="final_backtests",
        fingerprint=checkpoint_fingerprint,
    )
    if final_checkpoint is not None:
        final_backtests = [
            BacktestResult.model_validate(item)
            for item in final_checkpoint.get("items", [])
        ]
        _log(f"  Final OOS backtests: resumed {len(final_backtests)} from checkpoint")
    else:
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
        _save_stage_checkpoint(
            store,
            run_id,
            round_num=round_num,
            symbol=checkpoint_symbol,
            market=checkpoint_market,
            stage="final_backtests",
            fingerprint=checkpoint_fingerprint,
            payload={"items": [result.model_dump(mode="json") for result in final_backtests]},
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

    # ── Trajectory Records (Evolutionary Alpha lineage) ──────────
    round_trajectories: list[dict[str, Any]] = []
    if store:
        from factor_mining.trajectory_ledger import TrajectoryLedger  # noqa: F811

        trajectory_ledger = TrajectoryLedger(store, settings)
        parent_candidates_by_id = {c.candidate_id: c for c in current_candidates}
        records, skipped_trajectory_candidates = trajectory_ledger.create_records_for_candidates(
            round_candidates,
            round_backtests,
            round_factor_evidence,
            round_research_gates,
            round_near_misses,
            parent_candidates_by_id,
            artifact_scope=artifact_scope,
        )
        for record in records:
            trajectory_ledger.save(record)
            round_trajectories.append(record.model_dump(mode="json"))
        if skipped_trajectory_candidates:
            _log(
                "  Trajectory records: skipped missing backtests for "
                + ", ".join(item[:16] for item in skipped_trajectory_candidates[:5])
            )
        _log(f"  Trajectory records: {len(round_trajectories)} saved")

    # ── Optimize: signal-side ───────────────────────────────────────
    _check_stop(stop_event)
    t0 = time.perf_counter()
    from factor_mining.optimizers.traditional_optimizer import (
        apply_exit_adjustments,
        apply_optimization_result,
        build_optimization_context,
        optimize_exits_traditionally,
        optimize_traditionally,
    )

    ctx = build_optimization_context(
        round_candidates, round_backtests, round_gatechecks, iteration,
        previous_actions=previous_actions,
        research_gates=round_research_gates, near_misses=round_near_misses,
    )
    if not allow_optimizer_repairs:
        ctx["repair_adjustments"] = []
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
    optimization = optimize_traditionally(ctx, mode="full")
    if not allow_next_hypotheses:
        optimization = dict(optimization)
        optimization["next_hypotheses"] = []
    _log(f"  Traditional optimizer signal: {optimization.get('action', 'unknown')}")

    signal_candidates, opt_summary = apply_optimization_result(
        optimization,
        round_candidates,
        round_backtests,
        allow_repairs=allow_optimizer_repairs,
        allow_next_hypotheses=allow_next_hypotheses,
    )
    new_candidates = list(signal_candidates)
    _log(f"  Signal optimization: {opt_summary['combinations_created']} combos, "
         f"{opt_summary['adjustments_applied']} adjustments, "
         f"{opt_summary.get('repairs_created', 0)} repairs, "
         f"{opt_summary.get('evolutions_created', 0)} evolutions, "
         f"{opt_summary['hypotheses_suggested']} new hypotheses")

    # ── Optimize: exit-side ──────────────────────────────────────────
    exit_opt = optimize_exits_traditionally(ctx)
    _log(f"  Traditional optimizer exit: {len(exit_opt.get('exit_adjustments', []))} adjustments")
    new_candidates = apply_exit_adjustments(exit_opt, new_candidates, settings)

    _log(f"  Total optimization: {len(new_candidates) - len(round_candidates)} new candidates "
         f"({time.perf_counter() - t0:.0f}s)")

    history_entry = {
        "round": round_num,
        "iteration": iteration,
        "phase": phase,
        "symbol": round_candidates[0].symbol if round_candidates else None,
        "market": round_candidates[0].market if round_candidates else None,
        "num_candidates": len(round_candidates),
        "num_backtests": len(round_backtests),
        "num_pre_gate_repairs": pre_gate_completed,
        "num_pre_gate_repairs_generated": pre_gate_generated,
        "num_pre_gate_repairs_generated_by_kind": dict(pre_gate_generated_by_kind),
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
        "optimizer_outcomes": ctx.get("optimizer_outcomes", []),
        "optimizer_outcome_counts": ctx.get("optimizer_outcome_counts", {}),
        "optimizer_proposal_counts": optimization.get("proposal_counts", {}),
        "optimization": optimization,
        "summary": opt_summary,
        "new_candidates_count": len(new_candidates),
        "trajectory_ids": [t.get("trajectory_id") for t in round_trajectories if isinstance(t, dict)],
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


def _symbol_round_parallelism(group_count: int, max_workers: int | None) -> tuple[int, int | None]:
    if group_count <= 1:
        return 1, max_workers

    total_workers = max(1, int(max_workers or os.cpu_count() or 4))
    symbol_workers = min(group_count, total_workers)
    backtest_workers_per_group = max(1, total_workers // symbol_workers)
    return symbol_workers, backtest_workers_per_group


def _record_candidate_trials(
    candidates: list[CandidateStrategySpec],
    store: MetadataStore | None,
    settings: Settings,
    cumulative_trial_counts: dict[str, int],
    *,
    trial_counts_lock: Any | None = None,
) -> dict[str, dict[str, int]]:
    if trial_counts_lock is not None:
        with trial_counts_lock:
            return _record_candidate_trials(
                candidates,
                store,
                settings,
                cumulative_trial_counts,
                trial_counts_lock=None,
            )

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


def _candidate_trial_count_snapshots(
    candidates: list[CandidateStrategySpec],
    store: MetadataStore | None,
    settings: Settings,
) -> dict[str, dict[str, int]]:
    ledger = TrialLedger(store, settings) if store is not None else None
    counts_by_candidate: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        complexity_score = _candidate_complexity_score(candidate)
        candidate.params["complexity_score"] = complexity_score
        if ledger is not None:
            counts = ledger.counts_for(candidate.hypothesis_family)
        else:
            counts = {
                "family_trials_count": 1,
                "rolling_90d_trials_count": 1,
                "effective_trials_count": 1,
                "global_cumulative_trials_count": 1,
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


def _build_local_grid_tuning_candidates(
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
    evidence_reports: list[FactorEvidenceReport],
    *,
    parent_limit: int = _LOCAL_TUNING_PARENT_LIMIT,
    max_per_parent: int = _LOCAL_TUNING_MAX_PER_PARENT,
    total_limit: int = _LOCAL_TUNING_TOTAL_LIMIT,
) -> list[CandidateStrategySpec]:
    """Build bounded per-parent parameter grids for validation-split tuning."""
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    evidence_by_exp = {report.experiment_id: report for report in evidence_reports}
    ordered = sorted(
        results,
        key=lambda result: _pre_gate_priority(result, evidence_by_exp.get(result.experiment_id)),
        reverse=True,
    )

    tuning_candidates: list[CandidateStrategySpec] = []
    signatures: set[tuple[str, str]] = set()
    parents_seen = 0

    for result in ordered:
        parent = candidate_by_id.get(result.candidate_id)
        if parent is None:
            continue
        evidence = evidence_by_exp.get(result.experiment_id)
        if not _passes_local_tuning_evidence(parent, result, evidence):
            continue

        raw_grid = _local_tuning_param_grid(parent, result, evidence)
        ranked_grid = sorted(
            raw_grid,
            key=lambda params: _local_grid_acquisition_score(parent, result, evidence, params),
            reverse=True,
        )
        kept_for_parent = 0
        emitted_for_parent = False

        for params in ranked_grid:
            candidate = _spawn_local_grid_tuning_candidate(parent, params, result, evidence)
            if not _repair_respects_family(parent, candidate):
                continue
            complexity = _candidate_complexity_score(candidate)
            candidate.params["complexity_score"] = complexity
            if complexity > _MAX_FINAL_COMPLEXITY:
                continue

            signature = (
                parent.candidate_id,
                _json_dumps_sorted(_repair_signature_params(candidate.params)),
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            tuning_candidates.append(candidate)
            kept_for_parent += 1
            emitted_for_parent = True

            if kept_for_parent >= max_per_parent or len(tuning_candidates) >= total_limit:
                break

        if emitted_for_parent:
            parents_seen += 1
        if parents_seen >= parent_limit or len(tuning_candidates) >= total_limit:
            break

    return tuning_candidates


def _passes_local_tuning_evidence(
    parent: CandidateStrategySpec,
    result: BacktestResult,
    evidence: FactorEvidenceReport | None,
) -> bool:
    if evidence is None:
        return False
    if _candidate_complexity_score(parent) > _MAX_FINAL_COMPLEXITY:
        return False

    gross_sharpe = result.metrics_gross.sharpe if result.metrics_gross is not None else result.metrics_primary.sharpe
    max_abs_ic = _evidence_max_abs_ic(evidence)
    max_abs_rankic = _max_abs_mapping(evidence.rankic_by_horizon)
    has_stat_signal = (
        max_abs_ic >= _PRE_GATE_MIN_ABS_IC
        or max_abs_rankic >= _PRE_GATE_MIN_ABS_IC
        or bool(evidence.evidence_flags.get("ic_ci_excludes_zero"))
        or bool(evidence.evidence_flags.get("decay_curve_supported"))
    )
    has_economic_signal = (
        gross_sharpe >= 0.60
        or (
            evidence.turnover_adjusted_return > 0.0
            and (gross_sharpe >= 0.40 or max(max_abs_ic, max_abs_rankic) >= 0.006)
        )
    )
    if not (has_stat_signal or has_economic_signal):
        return False

    priority = _pre_gate_priority(result, evidence)
    if priority >= _LOCAL_TUNING_MIN_PRIORITY:
        return True
    return _local_tuning_problem_severity(result) > 0.0 and priority >= _LOCAL_TUNING_MIN_WEAK_PRIORITY


def _local_tuning_param_grid(
    parent: CandidateStrategySpec,
    result: BacktestResult,
    evidence: FactorEvidenceReport | None,
) -> list[dict[str, Any]]:
    lookback_values = _local_tuning_lookback_values(parent, evidence)
    smooth_values = _ordered_numeric_values(parent.params.get("smooth_span", 1), _LOCAL_TUNING_SMOOTH_SPANS, int)
    threshold_values = _ordered_numeric_values(
        parent.params.get("signal_threshold", 0.0),
        _LOCAL_TUNING_SIGNAL_THRESHOLDS,
        float,
    )
    buffer_values = _ordered_numeric_values(
        parent.params.get("position_buffer", 0.05),
        _LOCAL_TUNING_POSITION_BUFFERS,
        float,
    )
    zscore_values = _local_tuning_zscore_values(parent)
    tanh_values = _local_tuning_tanh_values(parent)
    base_signature = _json_dumps_sorted(_repair_signature_params(parent.params))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for lookback, smooth_span, signal_threshold, position_buffer, zscore_window, tanh_scale in product(
        lookback_values,
        smooth_values,
        threshold_values,
        buffer_values,
        zscore_values,
        tanh_values,
    ):
        params: dict[str, Any] = {
            "smooth_span": int(smooth_span),
            "signal_threshold": round(float(signal_threshold), 6),
            "position_buffer": round(float(position_buffer), 6),
        }
        if lookback is not None:
            params["factor_lookback"] = int(lookback)
        if zscore_window is not None:
            params["zscore_window"] = int(zscore_window)
        if tanh_scale is not None:
            params["tanh_scale"] = round(float(tanh_scale), 6)
        signature = _json_dumps_sorted(_repair_signature_params({**parent.params, **params}))
        if signature == base_signature or signature in seen:
            continue
        seen.add(signature)
        rows.append(params)

    return rows


def _local_tuning_lookback_values(
    parent: CandidateStrategySpec,
    evidence: FactorEvidenceReport | None,
) -> list[int | None]:
    has_lookback = (
        parent.params.get("signal_source") == "factor_signal"
        or parent.params.get("factor_lookback") is not None
    )
    if not has_lookback:
        return [None]

    values: list[int] = []
    current = parent.params.get("factor_lookback", 12)
    if _is_int_like(current):
        values.append(int(current))
    if evidence is not None and evidence.best_horizon_bars is not None:
        values.append(int(evidence.best_horizon_bars))
    values.extend(_LOCAL_TUNING_LOOKBACKS)
    return _ordered_unique_ints(values)


def _local_tuning_zscore_values(parent: CandidateStrategySpec) -> list[int | None]:
    if parent.params.get("signal_source") != "feature":
        return [None]
    transform = str(parent.params.get("transform", "tanh_zscore"))
    if transform not in {"tanh_zscore", "rank"}:
        return [None]
    values: list[int] = []
    current = parent.params.get("zscore_window", 288)
    if _is_int_like(current):
        values.append(int(current))
    values.extend(_LOCAL_TUNING_ZSCORE_WINDOWS)
    return _ordered_unique_ints(values)


def _local_tuning_tanh_values(parent: CandidateStrategySpec) -> list[float | None]:
    if parent.params.get("signal_source") != "feature":
        return [None]
    if str(parent.params.get("transform", "tanh_zscore")) != "tanh_zscore":
        return [None]
    return _ordered_numeric_values(parent.params.get("tanh_scale", 2.0), _LOCAL_TUNING_TANH_SCALES, float)


def _ordered_numeric_values(current: Any, defaults: tuple[Any, ...], caster: Callable[[Any], Any]) -> list[Any]:
    values: list[Any] = []
    try:
        values.append(caster(current))
    except (TypeError, ValueError):
        pass
    values.extend(caster(value) for value in defaults)

    ordered: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _ordered_unique_ints(values: list[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _spawn_local_grid_tuning_candidate(
    parent: CandidateStrategySpec,
    params: dict[str, Any],
    result: BacktestResult,
    evidence: FactorEvidenceReport | None,
) -> CandidateStrategySpec:
    candidate = parent.model_copy(deep=True)
    candidate.candidate_id = f"c_grid_{uuid.uuid4().hex[:12]}"
    candidate.candidate_type = "grid_tuning"
    candidate.parent_candidate_id = parent.candidate_id
    candidate.params.update(params)
    candidate.params["parent_id"] = parent.candidate_id
    candidate.params["generated_by"] = "local_grid_tuning"
    candidate.params["search_variant"] = "local_grid"
    candidate.params["tuning_objective"] = "validation_dsr_cost_turnover_score"
    candidate.params["optimizer_reason"] = "per_parent_local_grid_validation_tuning"
    param_diff = _optimizer_style_param_diff(parent.params, params)
    proposal_signature = _local_grid_proposal_signature(parent.candidate_id, param_diff)
    candidate.params["optimizer_proposal_id"] = f"opt_{proposal_signature[:12]}"
    candidate.params["optimizer_proposal_signature"] = proposal_signature
    candidate.params["optimizer_proposal_kind"] = "local_grid_tuning"
    candidate.params["optimizer_root_parent_id"] = parent.candidate_id
    candidate.params["optimizer_param_diff"] = param_diff
    candidate.params["optimizer_param_change"] = _param_change(parent.params, params)
    candidate.params["optimizer_variant_key"] = _local_grid_variant_key(param_diff)
    candidate.params["parent_validation_baseline"] = {
        "discovery_net_sharpe": result.metrics_primary.sharpe,
        "discovery_gross_sharpe": result.metrics_gross.sharpe if result.metrics_gross is not None else None,
        "discovery_deflated_sharpe": result.deflated_sharpe,
        "discovery_turnover": result.factor_turnover,
        "discovery_break_even_cost_bps": result.break_even_cost_bps,
        "discovery_actual_cost_bps": result.actual_cost_bps,
        "evidence_max_abs_ic": _evidence_max_abs_ic(evidence),
        "evidence_best_horizon_bars": evidence.best_horizon_bars if evidence is not None else None,
    }
    candidate.params["optimizer_parent_metrics"] = _metrics_payload(result)
    candidate.params["local_grid_acquisition_score"] = _local_grid_acquisition_score(
        parent,
        result,
        evidence,
        params,
    )
    return candidate


def _optimizer_style_param_diff(parent_params: dict[str, Any], child_params: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "smooth_span",
        "signal_threshold",
        "position_buffer",
        "factor_lookback",
        "zscore_window",
        "tanh_scale",
    }
    return {
        key: child_params[key]
        for key in keys
        if key in child_params and parent_params.get(key) != child_params[key]
    }


def _param_change(parent_params: dict[str, Any], child_params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for key, child_value in child_params.items():
        parent_value = parent_params.get(key)
        if parent_value != child_value:
            diff[key] = {"from": parent_value, "to": child_value}
    return diff


def _local_grid_proposal_signature(root_parent_id: str, param_diff: dict[str, Any]) -> str:
    payload = {
        "kind": "local_grid_tuning",
        "root_parent_id": root_parent_id,
        "param_diff": param_diff,
    }
    return hashlib.sha256(_json_dumps_sorted(payload).encode("utf-8")).hexdigest()


def _local_grid_variant_key(param_diff: dict[str, Any]) -> str:
    if not param_diff:
        return "local_grid"
    parts = [f"{key}={param_diff[key]}" for key in sorted(param_diff)]
    return f"local_grid_{'_'.join(parts)}"


def _metrics_payload(result: BacktestResult | None) -> dict[str, float | None]:
    if result is None:
        return {
            "sharpe": None,
            "gross_sharpe": None,
            "max_dd": None,
            "factor_turnover": None,
            "cost_margin_bps": None,
        }
    return {
        "sharpe": result.metrics_primary.sharpe,
        "gross_sharpe": result.metrics_gross.sharpe if result.metrics_gross is not None else None,
        "max_dd": result.metrics_primary.max_drawdown,
        "factor_turnover": result.factor_turnover,
        "cost_margin_bps": result.break_even_cost_bps - 2.0 * result.actual_cost_bps,
    }


def _local_grid_acquisition_score(
    parent: CandidateStrategySpec,
    result: BacktestResult,
    evidence: FactorEvidenceReport | None,
    params: dict[str, Any],
) -> float:
    score = _pre_gate_priority(result, evidence)
    smooth_span = max(1, int(params.get("smooth_span", parent.params.get("smooth_span", 1)) or 1))
    signal_threshold = float(params.get("signal_threshold", parent.params.get("signal_threshold", 0.0)) or 0.0)
    position_buffer = float(params.get("position_buffer", parent.params.get("position_buffer", 0.05)) or 0.05)
    conservatism = np.log2(float(smooth_span)) + 4.0 * signal_threshold + 2.0 * position_buffer
    problem_severity = _local_tuning_problem_severity(result)
    score += (0.45 * problem_severity - 0.10) * conservatism

    if evidence is not None and evidence.best_horizon_bars is not None and params.get("factor_lookback") is not None:
        try:
            lookback = max(1, int(params["factor_lookback"]))
            best = max(1, int(evidence.best_horizon_bars))
            score += 1.0 / (1.0 + abs(np.log2(lookback / best)))
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    diff_size = len(_optimizer_style_param_diff(parent.params, params))
    score += 0.05 * diff_size
    score -= 0.25 * max(0, _candidate_complexity_score(_candidate_with_params(parent, params)) - 2)
    return float(score)


def _local_tuning_problem_severity(result: BacktestResult) -> float:
    actual_cost = float(result.actual_cost_bps)
    denominator = 2.0 * actual_cost + 1e-12
    cost_ratio = max(0.0, 1.0 - float(result.break_even_cost_bps) / denominator) if actual_cost > 0.0 else 0.0
    turnover_severity = max(0.0, (float(result.factor_turnover) - 0.10) / 0.10)
    return min(1.0, max(cost_ratio, turnover_severity))


def _candidate_with_params(parent: CandidateStrategySpec, params: dict[str, Any]) -> CandidateStrategySpec:
    candidate = parent.model_copy(deep=True)
    candidate.params.update(params)
    return candidate


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
    repair.candidate_type = "repair"
    repair.parent_candidate_id = parent.candidate_id
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
        "zscore_window",
        "tanh_scale",
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
        score = _repair_selection_score(repair, result_by_candidate[repair.candidate_id], corr)
        repair.params["validation_selection_score"] = score
        repair_rows.append((
            score,
            repair,
            corr,
        ))

    per_parent_repair_count: Counter[str] = Counter()
    per_parent_tuning_count: Counter[str] = Counter()
    merged = 0
    rejected = 0
    for _, repair, parent_corr in sorted(repair_rows, key=lambda item: item[0], reverse=True):
        parent_id = str(repair.params.get("parent_id") or "")
        result = result_by_candidate[repair.candidate_id]
        parent = original_by_id.get(parent_id) or candidate_by_id.get(parent_id)
        pbo = float(result.pbo if result.pbo is not None else 1.0)
        is_local_tuning = _is_local_grid_tuning(repair)
        reasons: list[str] = []
        if parent is None or parent_id not in result_by_candidate:
            reasons.append("parent_missing_validation")
        elif not _repair_respects_family(parent, repair):
            reasons.append("cross_family_mutation")
        if _candidate_complexity_score(repair) > _MAX_FINAL_COMPLEXITY:
            reasons.append("complexity_cap")
        if pbo > _REPAIR_MAX_PBO:
            reasons.append("high_validation_pbo")
        if not is_local_tuning:
            if parent_corr is None:
                reasons.append("missing_parent_correlation")
            elif abs(parent_corr) >= _REPAIR_MAX_PARENT_CORR:
                reasons.append("low_incremental_orthogonality")
            if per_parent_repair_count[parent_id] >= _PRE_GATE_REPAIR_MAX_PER_PARENT:
                reasons.append("repair_parent_ratio_cap")
        elif per_parent_tuning_count[parent_id] >= _LOCAL_TUNING_TOP_K_PER_PARENT:
            reasons.append("local_tuning_parent_ratio_cap")

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

        if is_local_tuning:
            per_parent_tuning_count[parent_id] += 1
            parent_result = result_by_candidate.get(parent_id)
            repair.params["optimizer_parent_metrics"] = _metrics_payload(parent_result)
            repair.params["local_grid_validation_delta"] = _local_grid_validation_delta(result, parent_result)
        else:
            per_parent_repair_count[parent_id] += 1
        merged += 1
        repair.params["merge_pool_status"] = "merged"
        repair.params["repair_validation_pbo"] = pbo
        repair.params["parent_signal_correlation"] = parent_corr
        repair.params["merge_pool_score"] = _repair_selection_score(repair, result, parent_corr)
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


def _is_local_grid_tuning(candidate: CandidateStrategySpec) -> bool:
    return candidate.params.get("generated_by") == "local_grid_tuning"


def _repair_selection_score(
    repair: CandidateStrategySpec,
    result: BacktestResult,
    parent_corr: float | None,
) -> float:
    if _is_local_grid_tuning(repair):
        return _local_tuning_validation_score(result, parent_corr)
    return _repair_merge_score(result, parent_corr)


def _local_tuning_validation_score(result: BacktestResult, parent_corr: float | None) -> float:
    gross_sharpe = result.metrics_gross.sharpe if result.metrics_gross is not None else result.metrics_primary.sharpe
    pbo = float(result.pbo if result.pbo is not None else 1.0)
    cost_margin = result.break_even_cost_bps - 2.0 * result.actual_cost_bps
    turnover_penalty = max(0.0, float(result.factor_turnover) - 0.10)
    drawdown_penalty = max(0.0, abs(float(result.metrics_primary.max_drawdown)) - 0.12)
    corr_penalty = 0.0 if parent_corr is None else 0.10 * abs(parent_corr)
    # Deflated Sharpe is the primary objective; gross Sharpe is only a small
    # tie-breaker so observed Sharpe is not counted twice.
    return float(
        result.deflated_sharpe
        + 0.15 * max(0.0, gross_sharpe)
        + 0.02 * max(-10.0, min(25.0, cost_margin))
        - 1.50 * turnover_penalty
        - 2.00 * drawdown_penalty
        - 0.50 * pbo
        - corr_penalty
    )


def _local_grid_validation_delta(
    result: BacktestResult,
    parent_result: BacktestResult | None,
) -> dict[str, float | None]:
    parent_metrics = _metrics_payload(parent_result)
    current = _metrics_payload(result)
    return {
        "delta_sharpe": _delta_metric(current.get("sharpe"), parent_metrics.get("sharpe")),
        "delta_turnover": _delta_metric(current.get("factor_turnover"), parent_metrics.get("factor_turnover")),
        "delta_max_dd": _delta_metric(current.get("max_dd"), parent_metrics.get("max_dd")),
        "delta_cost_margin_bps": _delta_metric(current.get("cost_margin_bps"), parent_metrics.get("cost_margin_bps")),
    }


def _delta_metric(current: Any, previous: Any) -> float | None:
    if current is None or previous is None:
        return None
    return float(current) - float(previous)


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
        "generated_by": repair.params.get("generated_by"),
        "search_variant": repair.params.get("search_variant"),
        "validation_selection_score": repair.params.get("validation_selection_score"),
        "merge_pool_score": repair.params.get("merge_pool_score"),
        "local_grid_validation_delta": repair.params.get("local_grid_validation_delta"),
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
    """Compute the effective trial count for the final merge-pool DSR penalty.

    Policy (explicit):
    - `tested_candidates`  = actual distinct candidates that competed in the merge pool;
      this is the primary measure of independent search paths.
    - `sum(cumulative_trial_counts)` = conservative family-based floor; only used if it
      exceeds `tested_candidates` (rare, happens when many families compete).
    - We deliberately do NOT take max(observed_effective_trials) from validation results
      because those already incorporate the same cumulative counts — taking that max would
      double-apply the family-level penalty.
    """
    family_floor = sum(int(v) for v in cumulative_trial_counts.values())
    return max(1, int(tested_candidates), family_floor)


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
        # Write merge-pool trial count back to trial_diagnostics for artifact transparency.
        result.trial_diagnostics["merge_pool_effective_trials"] = effective_trials_count
        result.trial_diagnostics["effective_trials_at_eval"] = result.effective_trials_at_eval
        result.trial_diagnostics["global_trials_at_eval"] = result.global_trials_at_eval
        result.trial_diagnostics["dsr"] = float(result.deflated_sharpe)


def _json_dumps_sorted(payload: Any) -> str:
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


def _sample_frame_blocks(
    frame: pd.DataFrame,
    *,
    sample_bars: int,
    interval_ms: int,
    seed: int,
) -> pd.DataFrame:
    if sample_bars <= 0:
        raise ValueError("sample_bars must be positive")
    if sample_bars >= len(frame):
        return frame.reset_index(drop=True)

    block_len = max(1, int(round(7 * 86_400_000 / max(interval_ms, 1))))
    block_len = min(block_len, len(frame))
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    max_start = max(0, len(frame) - block_len)
    # Oversample blocks, then de-duplicate and trim after chronological sort.
    sampled = pd.DataFrame()
    attempts = 0
    while len(sampled) < sample_bars and attempts < max(16, len(frame) // max(block_len, 1) * 8):
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end = min(len(frame), start + block_len)
        piece = frame.iloc[start:end]
        pieces.append(piece)
        sampled = pd.concat(pieces, ignore_index=False).sort_values("open_time").drop_duplicates("open_time")
        attempts += 1
    if len(sampled) < sample_bars:
        sampled = (
            pd.concat([sampled, frame], ignore_index=False)
            .sort_values("open_time")
            .drop_duplicates("open_time")
        )
    sampled = sampled.head(sample_bars).reset_index(drop=True)
    return sampled


def _load_data_contexts(
    candidates: list[CandidateStrategySpec],
    settings: Settings,
    *,
    tail: int | None,
    sample_bars: int | None = None,
    sample_mode: str = "block",
    seed: int = 42,
) -> dict[tuple[str, str], MarketDataContext]:
    if tail is not None and sample_bars is not None:
        raise ValueError("--tail and --sample-bars are mutually exclusive")
    if sample_bars is not None and sample_mode != "block":
        raise ValueError("sample_mode must be 'block'")

    contexts: dict[tuple[str, str], MarketDataContext] = {}
    for symbol, market in sorted({(c.symbol, c.market) for c in candidates}):
        try:
            full_frame = load_frame(
                settings,
                symbol=symbol,
                market=market,
                tail=None if sample_bars is not None else tail,
            )
        except FileNotFoundError as exc:
            _log(f"Skip {symbol}/{market}: {exc}")
            continue

        quality_frame = full_frame
        if sample_bars is not None:
            frame = _sample_frame_blocks(
                full_frame,
                sample_bars=sample_bars,
                interval_ms=interval_to_ms(settings.data.default_interval),
                seed=seed,
            )
            _log(
                f"Data sample {symbol}/{market}: {len(frame):,}/{len(full_frame):,} rows "
                f"from full coverage {full_frame['open_time'].min()} → {full_frame['open_time'].max()}"
            )
        else:
            frame = full_frame

        n_rows = len(frame)
        _log(f"Data {symbol}/{market}: {n_rows:,} rows, {frame['open_time'].min()} → {frame['open_time'].max()}")
        quality_notes = kline_quality_notes(
            quality_frame,
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
            _log(f"Funding {symbol}: not available; funding factor_signal candidates skipped, supplemental funding features still allowed")

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
    *,
    phase: str = "discovery",
    next_candidates_count: int | None = None,
) -> dict[str, Any]:
    optimization = child_history[-1].get("optimization", {}) if child_history else {}
    optimizer_outcomes = _combined_optimizer_outcomes(child_history)
    return {
        "round": round_num,
        "iteration": iteration,
        "phase": phase,
        "children": child_history,
        "optimization": optimization,
        "new_candidates_count": len(new_candidates),
        "next_candidates_count": len(new_candidates) if next_candidates_count is None else next_candidates_count,
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
        "optimizer_outcomes": optimizer_outcomes,
        "optimizer_outcome_counts": _sum_optimizer_outcome_counts(child_history),
        "optimizer_proposal_counts": _sum_optimizer_proposal_counts(child_history),
    }


def _candidate_output_signature(candidates: list[CandidateStrategySpec]) -> str:
    payloads = [
        _strip_candidate_output_ids(candidate.model_dump(mode="json"))
        for candidate in candidates
    ]
    payloads.sort(key=_json_dumps_sorted)
    raw = _json_dumps_sorted({"candidates": payloads})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strip_candidate_output_ids(value: Any) -> Any:
    if isinstance(value, dict):
        stripped: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"candidate_id", "parent_candidate_id", "parent_id", "experiment_id"}:
                continue
            if str(key).startswith("optimizer_"):
                continue
            if key == "factor_ids":
                continue
            stripped[str(key)] = _strip_candidate_output_ids(item)
        return stripped
    if isinstance(value, list):
        return [_strip_candidate_output_ids(item) for item in value]
    return value


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


def _combined_optimizer_outcomes(child_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for child in child_history:
        for outcome in child.get("optimizer_outcomes", []) or []:
            if isinstance(outcome, dict):
                outcomes.append(outcome)
    return outcomes


def _sum_optimizer_outcome_counts(child_history: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for child in child_history:
        counts.update(child.get("optimizer_outcome_counts", {}))
    return dict(counts)


def _sum_optimizer_proposal_counts(child_history: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for child in child_history:
        counts.update(child.get("optimizer_proposal_counts", {}))
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
    max_workers: int | None = None,
) -> list[tuple[np.ndarray, dict, int, dict[str, int], list[dict]]]:
    """Construct regime-conditional trading signals for each candidate.

    Returns list of (signal_array, candidate_dict, index, trial_counts, data_quality_notes).
    """
    if not candidates:
        return []

    t0 = time.perf_counter()
    context = SignalBuildContext(
        frame=frame,
        features_df=features_df,
        feature_meta=feature_meta,
        forward_regimes=forward_regimes,
        funding_rate=funding_rate,
    )
    note_dicts = [note.model_dump(mode="json") for note in (data_quality_notes or [])]
    worker_count = min(max(1, int(max_workers or os.cpu_count() or 4)), len(candidates))
    trial_counts_by_candidate = trial_counts_by_candidate or {}

    def build_one(i: int, c: CandidateStrategySpec) -> tuple[np.ndarray, dict, int, dict[str, int], list[dict]] | SignalBuildSkip:
        try:
            signal_arr = _build_signal_for(
                c,
                frame,
                features_df,
                feature_meta,
                i,
                forward_regimes,
                funding_rate,
                build_context=context,
            )
        except ValueError as exc:
            return SignalBuildSkip(index=i, candidate_id=c.candidate_id, error=exc)
        trial_counts = trial_counts_by_candidate.get(
            c.candidate_id,
            {
                "effective_trials_count": 1,
                "global_cumulative_trials_count": 1,
            },
        )
        return (signal_arr, c.model_dump(mode="json"), i, trial_counts, note_dicts)

    results: list[tuple[np.ndarray, dict, int, dict[str, int], list[dict]] | SignalBuildSkip] = []
    if worker_count <= 1:
        results = [build_one(i, c) for i, c in enumerate(candidates)]
    else:
        by_idx: dict[int, tuple[np.ndarray, dict, int, dict[str, int], list[dict]] | SignalBuildSkip] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(build_one, i, c): i
                for i, c in enumerate(candidates)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                by_idx[idx] = future.result()
        results = [by_idx[i] for i in sorted(by_idx)]

    tasks = []
    skipped = 0
    for item in results:
        if isinstance(item, SignalBuildSkip):
            skipped += 1
            _log(f"[{item.index + 1}/{len(candidates)}] {item.candidate_id[:16]}... SKIP signal: {item.error}")
            continue
        tasks.append(item)

    elapsed = time.perf_counter() - t0
    _log(
        "  Built signal tasks: "
        f"{len(tasks)}/{len(candidates)} in {elapsed:.2f}s "
        f"(workers={worker_count}, skipped={skipped}, "
        f"cache_hits={dict(context.cache_hits)}, cache_misses={dict(context.cache_misses)})"
    )
    return tasks


def _filter_unfunded_factor_signal_candidates(
    candidates: list[CandidateStrategySpec],
    funding_rate: pd.Series | None,
) -> tuple[list[CandidateStrategySpec], int]:
    if _has_usable_funding_rate(funding_rate):
        return candidates, 0
    filtered = [candidate for candidate in candidates if not _candidate_requires_funding_rate(candidate)]
    return filtered, len(candidates) - len(filtered)


def _has_usable_funding_rate(funding_rate: pd.Series | None) -> bool:
    if funding_rate is None:
        return False
    values = pd.Series(funding_rate).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(values.abs().sum()) > 1e-12


def _candidate_requires_funding_rate(candidate: CandidateStrategySpec, *, depth: int = 0) -> bool:
    return _params_require_funding_rate(candidate.hypothesis_family, candidate.params, depth=depth)


def _params_require_funding_rate(hypothesis_family: str, params: dict, *, depth: int = 0) -> bool:
    if depth > 4:
        return False
    signal_source = params.get("signal_source")
    if signal_source == "feature":
        return False
    if signal_source == "factor_signal":
        return params.get("factor_family") == "funding_basis"
    components = params.get("components")
    if isinstance(components, list):
        for payload in components:
            try:
                component = CandidateStrategySpec.model_validate(payload)
            except (TypeError, ValueError):
                continue
            if _candidate_requires_funding_rate(component, depth=depth + 1):
                return True
        return False
    return signal_source is None and _normalize_family(hypothesis_family) == "funding_basis"


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
    build_context: SignalBuildContext | None = None,
) -> pd.Series:
    filtered = signal.copy()
    regime_filter = _filter_values(params.get("regime_filter"))
    if regime_filter:
        if build_context is not None and len(filtered) == len(build_context.frame):
            filtered = filtered.where(build_context.cached_filter_mask("regime", regime_filter), 0.0)
        else:
            regimes = pd.Series(forward_regimes.to_numpy(), index=filtered.index).astype(str)
            filtered = filtered.where(regimes.isin(regime_filter), 0.0)

    funding_state_filter = _filter_values(params.get("funding_state_filter"))
    if funding_state_filter:
        if build_context is not None and len(filtered) == len(build_context.frame):
            filtered = filtered.where(build_context.cached_filter_mask("funding_state", funding_state_filter), 0.0)
        else:
            states = funding_state_labels(funding_rate, filtered.index).astype(str)
            filtered = filtered.where(states.isin(funding_state_filter), 0.0)

    funding_trend_filter = _filter_values(params.get("funding_trend_filter"))
    if funding_trend_filter:
        if build_context is not None and len(filtered) == len(build_context.frame):
            filtered = filtered.where(build_context.cached_filter_mask("funding_trend", funding_trend_filter), 0.0)
        else:
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


def _stable_signal_key(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _candidate_signal_cache_key(candidate: CandidateStrategySpec, index: int) -> str:
    return _stable_signal_key({
        "candidate": candidate.model_dump(mode="json"),
        "index": index,
    })


def _feature_transform_cache_key(candidate: CandidateStrategySpec, indicator_name: str, direction: Any, transform: Any) -> str:
    params = candidate.params
    return _stable_signal_key({
        "indicator_name": indicator_name,
        "direction": direction,
        "transform": transform,
        "zscore_window": params.get("zscore_window", 288),
        "tanh_scale": params.get("tanh_scale", 2.0),
        "smooth_span": params.get("smooth_span", 1),
        "signal_threshold": params.get("signal_threshold", 0.0),
    })


def _cached_factor_signal(
    build_context: SignalBuildContext,
    *,
    family: str,
    lookback: int,
) -> pd.Series:
    key = (str(family), int(lookback))
    return build_context.cached(
        "factor_signal",
        build_context.factor_signal_cache,
        key,
        lambda: factor_signal(
            build_context.frame,
            family=family,
            lookback=lookback,
            funding_rate=build_context.funding_rate,
        ),
    )


def _cached_feature_transform(
    build_context: SignalBuildContext,
    *,
    candidate: CandidateStrategySpec,
    indicator_name: str,
    raw: pd.Series,
    direction: Any,
    transform: Any,
) -> pd.Series:
    key = _feature_transform_cache_key(candidate, indicator_name, direction, transform)
    return build_context.cached(
        "feature_transform",
        build_context.feature_transform_cache,
        key,
        lambda: _apply_transform(raw, direction, transform, candidate.params),
    )


def _build_signal_for(
    candidate: CandidateStrategySpec,
    frame: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_meta: dict,
    index: int,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None = None,
    *,
    build_context: SignalBuildContext | None = None,
) -> np.ndarray:
    context = build_context or SignalBuildContext(
        frame=frame,
        features_df=features_df,
        feature_meta=feature_meta,
        forward_regimes=forward_regimes,
        funding_rate=funding_rate,
    )
    key = _candidate_signal_cache_key(candidate, index)
    return context.cached(
        "candidate_signal",
        context.candidate_signal_cache,
        key,
        lambda: _build_signal_for_uncached(
            candidate,
            frame,
            features_df,
            feature_meta,
            index,
            forward_regimes,
            funding_rate,
            build_context=context,
        ),
    )


def _build_signal_for_uncached(
    candidate: CandidateStrategySpec,
    frame: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_meta: dict,
    index: int,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None = None,
    *,
    build_context: SignalBuildContext,
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
        signal = _cached_feature_transform(
            build_context,
            candidate=candidate,
            indicator_name=indicator_name,
            raw=raw,
            direction=direction,
            transform=transform,
        )
        canonical = _normalize_family(candidate.hypothesis_family)
        if canonical:
            signal = _apply_regime_modulation(signal, forward_regimes, canonical)
        signal = _apply_candidate_filters(signal, candidate.params, forward_regimes, funding_rate, build_context)
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
        signal = _cached_factor_signal(build_context, family=family, lookback=int(lookback))
        direction = int(candidate.params.get("direction", 1))
        signal = _apply_signal_controls(direction * signal.fillna(0), candidate.params).clip(-3, 3)
        canonical = _normalize_family(candidate.hypothesis_family)
        if canonical:
            signal = _apply_regime_modulation(signal, forward_regimes, canonical)
        signal = _apply_candidate_filters(signal, candidate.params, forward_regimes, funding_rate, build_context)
        return signal.to_numpy(dtype=float)

    # ── Composite ───────────────────────────────────────────────────
    if candidate.hypothesis_family == "composite" or candidate.params.get("components"):
        return _build_composite_signal(
            candidate,
            frame,
            features_df,
            feature_meta,
            index,
            forward_regimes,
            funding_rate,
            build_context=build_context,
        )

    # ── Legacy fallback (no signal_source in params) ────────────────
    canonical = _normalize_family(candidate.hypothesis_family)

    if canonical is not None:
        lookbacks = _METHOD_LOOKBACKS.get(candidate.method_id, [12])
        lookback = lookbacks[index % len(lookbacks)]
        signal = _cached_factor_signal(build_context, family=canonical, lookback=int(lookback))
        signal = signal.fillna(0).clip(-3, 3)
        signal = _apply_regime_modulation(signal, forward_regimes, canonical)
        signal = _apply_candidate_filters(signal, candidate.params, forward_regimes, funding_rate, build_context)
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
    if not family_features:
        raise ValueError(
            f"Candidate {candidate.candidate_id} has unknown hypothesis_family "
            f"'{candidate.hypothesis_family}' and no engineered fallback features."
        )
    feature_col = family_features[index % len(family_features)]
    signal = features_df[feature_col].fillna(0).clip(-3, 3)
    signal = _apply_candidate_filters(signal, candidate.params, forward_regimes, funding_rate, build_context)
    return signal.to_numpy(dtype=float)


def _build_composite_signal(
    candidate: CandidateStrategySpec,
    frame: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_meta: dict,
    index: int,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None,
    *,
    build_context: SignalBuildContext,
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
            build_context=build_context,
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
    filtered = _apply_candidate_filters(controlled, candidate.params, forward_regimes, funding_rate, build_context)
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

    detector = MarkovRegimeDetector(n_states=5, random_state=42)
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
        "traditional_survivor_adjustment",
        "traditional_survivor_composite",
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
            event_payload = dict(payload or {})
            if _RUN_ID is not None:
                event_payload.setdefault("run_id", _RUN_ID)
            _EVENT_SINK(phase, level, message, event_payload)
        except Exception:
            pass
