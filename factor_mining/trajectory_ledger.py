"""Trajectory curator that retroactively classifies pipeline mutations into a
structured genealogy without changing any production behaviour.

Every candidate that reaches the final OOS window receives a
``TrajectoryRecord`` regardless of GateCheck outcome.  Failed pre-gate repair
attempts are NOT promoted to trajectories — they remain inline logs.
"""

from __future__ import annotations

import hashlib
from typing import Any

from factor_mining.config import Settings
from factor_mining.models import (
    BacktestResult,
    CandidateStrategySpec,
    FactorEvidenceReport,
    GateCheckResult,
    NearMissAnalysis,
    ResearchGateResult,
    TrajectoryRecord,
)
from factor_mining.storage import MetadataStore


class TrajectoryLedger:
    """Creates TrajectoryRecords from existing pipeline outputs and stores them."""

    def __init__(self, store: MetadataStore | None, settings: Settings) -> None:
        self._store = store
        self._settings = settings

    @property
    def store(self) -> MetadataStore | None:
        return self._store

    # ── operator classification ─────────────────────────────────────

    def classify_operator(
        self,
        candidate: CandidateStrategySpec,
        parent_candidates: dict[str, CandidateStrategySpec],
    ) -> str:
        """Infer the six-way semantic evolution operator."""
        operator, _detail = self.classify_operator_with_detail(candidate, parent_candidates)
        return operator

    def classify_operator_with_detail(
        self,
        candidate: CandidateStrategySpec,
        parent_candidates: dict[str, CandidateStrategySpec],
    ) -> tuple[str, str | None]:
        """Infer semantic operator plus source/detail metadata.

        The top-level operator stays in the six freeze-point categories.  Existing
        local-grid, repair, optimizer, and composite labels live in
        ``operator_detail`` so selector budgets can reason over one stable enum.
        """
        ctype = candidate.candidate_type
        kind = str(candidate.params.get("optimizer_proposal_kind", ""))
        generated_by = str(candidate.params.get("generated_by", ""))
        parent_id = candidate.parent_candidate_id
        has_lab_direction = bool(candidate.params.get("lab_direction_factor_ids"))

        if ctype == "original":
            if parent_id is not None:
                return "MUTATION_AT_DSL", "OPTIMIZER_EVOLUTION"
            if has_lab_direction:
                return "SEED_INFORMED", "LAB_DIRECTION"
            return "SEED", None

        if ctype == "grid_tuning":
            return "MUTATION_AT_DSL", "GRID_TUNING"
        if ctype == "repair":
            return "MUTATION_AT_DSL", "PRE_GATE_REPAIR"
        if ctype == "composite":
            return "CROSSOVER", "COMPOSITE_EQUI_WEIGHT"

        if ctype == "optimizer":
            if kind == "crossover" or generated_by == "crossover_dsl_composite":
                return "CROSSOVER", "DSL_COMPOSITE"
            if kind == "hill_climb":
                return "MUTATION_AT_DSL", "OPTIMIZER_HILL_CLIMB"
            if kind in ("evolution", "turnover_control"):
                return "MUTATION_AT_DSL", "OPTIMIZER_EVOLUTION"
            if kind == "repair":
                return "MUTATION_AT_DSL", "PRE_GATE_REPAIR"
            if kind == "local_grid_tuning":
                return "MUTATION_AT_DSL", "GRID_TUNING"
            if generated_by in ("near_miss_repair", "optimizer_repair"):
                return "MUTATION_AT_DSL", "PRE_GATE_REPAIR"
            return "MUTATION_AT_DSL", "OPTIMIZER_EVOLUTION"

        if parent_id is not None:
            return "MUTATION_AT_DSL", "OPTIMIZER_EVOLUTION"

        return "SEED", None

    # ── record creation ─────────────────────────────────────────────

    def create_record(
        self,
        candidate: CandidateStrategySpec,
        backtest: BacktestResult | None,
        evidence: FactorEvidenceReport | None,
        research_gate: ResearchGateResult | None,
        near_miss: NearMissAnalysis | None,
        parent_candidates: dict[str, CandidateStrategySpec],
        *,
        artifact_scope: str = "",
    ) -> TrajectoryRecord:
        """Build a single lineage record for an evaluated candidate."""

        operator, operator_detail = self.classify_operator_with_detail(candidate, parent_candidates)
        parent_ids = _resolve_parent_ids(candidate)
        parent_trajectory_ids = self._resolve_parent_trajectory_ids(parent_ids)
        classification = _classify_outcome(candidate, backtest, research_gate)
        diagnosis = _build_diagnosis(candidate, backtest, near_miss)

        scope = artifact_scope or "unknown"
        experiment_id = backtest.experiment_id if backtest is not None else None
        artifact_references = [
            f"backtests_{scope}",
            f"factor_evidence_{scope}",
            f"gatechecks_{scope}",
            f"research_gate_{scope}",
            f"near_misses_{scope}",
            f"hardscores_{scope}",
        ]

        return TrajectoryRecord(
            trajectory_id=_trajectory_id(scope, candidate.candidate_id, experiment_id),
            candidate_id=candidate.candidate_id,
            experiment_id=experiment_id,
            artifact_scope=scope,
            parent_ids=parent_ids,
            parent_trajectory_ids=parent_trajectory_ids,
            operator=operator,
            operator_detail=operator_detail,
            source_candidate_type=candidate.candidate_type,
            freeze_point=_extract_freeze_point(candidate),
            hypothesis_family=candidate.hypothesis_family,
            method_id=candidate.method_id,
            symbol=candidate.symbol,
            classification=classification,
            promotion_reason=_promotion_reason(research_gate),
            diagnosis_text=diagnosis,
            candidate_snapshot=candidate.model_dump(mode="json"),
            backtest_result=backtest.model_dump(mode="json") if backtest is not None else None,
            evidence_snapshot=evidence.model_dump(mode="json") if evidence is not None else None,
            research_gate_snapshot=research_gate.model_dump(mode="json") if research_gate is not None else None,
            near_miss_snapshot=near_miss.model_dump(mode="json") if near_miss is not None else None,
            trial_ids=[],  # reserved: backtest engine records trials but does not return trial ids yet
            trial_refs=_trial_refs(backtest),
            artifact_references=artifact_references,
        )

    def create_records_for_candidates(
        self,
        candidates: list[CandidateStrategySpec],
        backtests: list[BacktestResult],
        evidence_reports: list[FactorEvidenceReport],
        research_gates: list[ResearchGateResult],
        near_misses: list[NearMissAnalysis],
        parent_candidates: dict[str, CandidateStrategySpec],
        *,
        artifact_scope: str = "",
    ) -> tuple[list[TrajectoryRecord], list[str]]:
        """Create records by candidate id, never by positional list alignment."""

        backtest_by_candidate_id = {item.candidate_id: item for item in backtests}
        evidence_by_candidate_id = {item.candidate_id: item for item in evidence_reports}
        research_gate_by_candidate_id = {item.candidate_id: item for item in research_gates}
        near_miss_by_candidate_id = {item.candidate_id: item for item in near_misses}

        records: list[TrajectoryRecord] = []
        skipped_candidate_ids: list[str] = []
        for candidate in candidates:
            backtest = backtest_by_candidate_id.get(candidate.candidate_id)
            if backtest is None:
                skipped_candidate_ids.append(candidate.candidate_id)
                continue
            records.append(
                self.create_record(
                    candidate,
                    backtest,
                    evidence_by_candidate_id.get(candidate.candidate_id),
                    research_gate_by_candidate_id.get(candidate.candidate_id),
                    near_miss_by_candidate_id.get(candidate.candidate_id),
                    parent_candidates,
                    artifact_scope=artifact_scope,
                )
            )
        return records, skipped_candidate_ids

    # ── persistence ─────────────────────────────────────────────────

    def save(self, record: TrajectoryRecord) -> None:
        """Persist a trajectory record to the metadata store (no-op if no store)."""
        if self._store is not None:
            self._store.save_trajectory(record)

    def _resolve_parent_trajectory_ids(self, parent_ids: list[str]) -> list[str]:
        if self._store is None:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for parent_id in parent_ids:
            for record in self._store.list_trajectories(candidate_id=parent_id, limit=1):
                if record.trajectory_id not in seen:
                    out.append(record.trajectory_id)
                    seen.add(record.trajectory_id)
                break
        return out


# ── private helpers ─────────────────────────────────────────────────


def _resolve_parent_ids(candidate: CandidateStrategySpec) -> list[str]:
    """Walk the optimizer lineage metadata to produce an ordered parent id list."""
    seen: set[str] = set()
    ids: list[str] = []

    explicit_parent_ids = candidate.params.get("parent_ids")
    if isinstance(explicit_parent_ids, list):
        for item in explicit_parent_ids:
            parent_id = str(item or "").strip()
            if parent_id and parent_id not in seen and parent_id != candidate.candidate_id:
                ids.append(parent_id)
                seen.add(parent_id)

    # Explicit parent from params lineage
    root = str(
        candidate.params.get("optimizer_root_parent_id")
        or candidate.params.get("parent_id")
        or candidate.parent_candidate_id
        or ""
    ).strip()
    if root and root not in seen and root != candidate.candidate_id:
        ids.append(root)
        seen.add(root)

    # Direct parent
    parent = (candidate.parent_candidate_id or "").strip()
    if parent and parent not in seen and parent != candidate.candidate_id:
        ids.append(parent)
        seen.add(parent)

    return ids


def _trajectory_id(artifact_scope: str, candidate_id: str, experiment_id: str | None) -> str:
    key = "|".join([artifact_scope or "unknown", candidate_id, experiment_id or "no_experiment"])
    return f"traj_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


def _trial_refs(backtest: BacktestResult | None) -> list[dict[str, Any]]:
    if backtest is None:
        return []
    return [{
        "experiment_id": backtest.experiment_id,
        "effective_trials_at_eval": backtest.effective_trials_at_eval,
        "global_trials_at_eval": backtest.global_trials_at_eval,
    }]


def _classify_outcome(
    candidate: CandidateStrategySpec,
    backtest: BacktestResult | None,
    research_gate: ResearchGateResult | None,
) -> str:
    """Map the candidate's evaluation outcome to a trajectory classification."""
    if research_gate is not None:
        status = research_gate.status
        if status in ("production_passed", "research_survivor", "rejected"):
            return status
    # Fallback: use candidate_type hints
    if candidate.candidate_type in ("repair", "grid_tuning"):
        merged = candidate.params.get("merge_pool_status")
        if merged == "merged":
            return "research_survivor"
        return "pre_gate_skipped"
    if backtest is not None and backtest.deflated_sharpe <= 0.0:
        return "gatecheck_failed"
    return "rejected"


def _promotion_reason(research_gate: ResearchGateResult | None) -> str | None:
    if research_gate is None:
        return None
    if research_gate.status == "production_passed":
        return "production_gate_passed"
    if research_gate.status == "research_survivor":
        return ", ".join(research_gate.reasons[:3]) if research_gate.reasons else "research_survivor"
    return None


def _build_diagnosis(
    candidate: CandidateStrategySpec,
    backtest: BacktestResult | None,
    near_miss: NearMissAnalysis | None,
) -> str | None:
    parts: list[str] = []
    if backtest is not None:
        parts.append(f"sr={backtest.metrics_primary.sharpe:+.2f}")
        parts.append(f"dsr={backtest.deflated_sharpe:+.3f}")
        parts.append(f"turnover={backtest.factor_turnover:.3f}")
        parts.append(f"cost_bps={backtest.actual_cost_bps:.1f}")
    if near_miss is not None and near_miss.primary_reason != "production_passed":
        parts.append(f"near_miss={near_miss.primary_reason}")
        if near_miss.repair_actions:
            parts.append(f"repair_actions={','.join(near_miss.repair_actions[:2])}")
    if candidate.params.get("merge_pool_reasons"):
        parts.append(f"merge_reasons={','.join(candidate.params['merge_pool_reasons'])}")
    return "; ".join(parts) if parts else None


def _extract_freeze_point(candidate: CandidateStrategySpec) -> dict[str, Any]:
    """Extract freeze-point metadata from a candidate's DSL and lineage fields.

    For non-DSL candidates the freeze point captures the hypothesis and
    structural params that a future mutation operator should preserve.
    """
    fp: dict[str, Any] = {
        "hypothesis_id": candidate.hypothesis_id,
        "hypothesis_family": candidate.hypothesis_family,
        "method_id": candidate.method_id,
        "symbol": candidate.symbol,
        "dsl_fingerprint": candidate.dsl_fingerprint,
        "dsl_version": candidate.dsl_version,
    }
    # Carry forward the parent's freeze depth if this is a child mutation
    parent_depth = candidate.params.get("freeze_depth")
    if isinstance(parent_depth, int) and parent_depth > 0:
        fp["freeze_depth"] = parent_depth
    return fp
