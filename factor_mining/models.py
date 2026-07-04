from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator


UTC = timezone.utc


class HypothesisSpec(BaseModel):
    hypothesis_id: str
    hypothesis_family: str
    economic_mechanism: str
    testable_prediction: str
    null_hypothesis: str
    expected_ic_range: tuple[float, float]
    expected_decay_halflife_bars: int
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    generated_by: str = "manual"
    mechanism_taxonomy: str | None = None
    required_data_families: list[str] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CandidateStrategySpec(BaseModel):
    candidate_id: str
    hypothesis_id: str
    method_id: str
    hypothesis_family: str
    symbol: str
    market: Literal["spot", "um_futures"] = "um_futures"
    interval: str = "5m"
    params: dict[str, Any] = Field(default_factory=dict)
    max_feature_lookback_bars: int = 288
    is_ml: bool = False
    candidate_type: Literal["original", "repair", "grid_tuning", "composite", "optimizer"] = "original"
    parent_candidate_id: str | None = None
    # Root candidate id of the search lineage. Derived candidates (repairs,
    # grid tuning) inherit their root's id so the trial ledger can count
    # independent search paths instead of raw evaluations.
    lineage_id: str | None = None
    # ── DSL fields (optional, None for non-DSL candidates) ──────────
    dsl_expression: str | None = None
    dsl_canonical_expression: str | None = None
    dsl_ast: dict[str, Any] | None = None
    dsl_fingerprint: str | None = None
    dsl_version: str | None = None


class MetricsBlock(BaseModel):
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_vol: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    trade_count: int = 0
    pnl: float = 0.0


class OOSWindowMetrics(BaseModel):
    """Metrics for a single chronological sub-window of the final OOS period."""
    window_index: int
    start_bar: int
    end_bar: int
    sharpe: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    ic_tstat: float | None = None


class WindowStabilityDiagnostics(BaseModel):
    """Aggregate stability metrics across OOS sub-windows (diagnostic-only, not a GateCheck rule)."""
    n_windows: int = 0
    window_sharpe_mean: float = 0.0
    window_sharpe_std: float = 0.0
    window_positive_rate: float = 0.0  # fraction of windows with Sharpe > 0
    window_trade_coverage: float = 0.0  # fraction of windows with at least 1 trade
    stability_score: float = 0.0  # composite 0-1 score; higher = more consistent
    per_window: list[OOSWindowMetrics] = Field(default_factory=list)


class DataQualityNote(BaseModel):
    scope: str
    severity: Literal["info", "warn", "fail"] = "warn"
    message: str
    degraded_ratio: float = 0.0


class BacktestResult(BaseModel):
    experiment_id: str
    candidate_id: str
    hypothesis_family: str
    method_id: str
    symbol: str
    market: str
    interval: str
    primary_position_mode: str = "vol_target"
    secondary_position_mode: str = "fixed_notional"
    metrics_primary: MetricsBlock
    metrics_secondary: MetricsBlock | None = None
    metrics_gross: MetricsBlock | None = None
    ic_tstat_nw: float = 0.0
    rankic_tstat_nw: float = 0.0
    sharpe_ci_5_95: tuple[float, float] = (0.0, 0.0)
    probabilistic_sharpe: float = 0.0
    # Expected-max haircut Sharpe (sqrt(2 ln N / n) penalty), not Bailey's
    # PSR-based DSR — see stats.metrics.deflated_sharpe_ratio. Diagnostic:
    # annualized on intraday bars it dwarfs any honest SR, so it must not gate.
    deflated_sharpe: float = 0.0
    # True Bailey & López de Prado DSR: P(observed SR beats the expected max
    # of effective_trials_at_eval null trials), skew/kurtosis-adjusted.
    # G1 gates on this clearing settings.gatecheck.dsr_prob_min.
    deflated_sharpe_prob: float | None = None
    return_skew: float = 0.0
    return_kurtosis: float = 3.0
    effective_trials_at_eval: int = 0
    global_trials_at_eval: int = 0
    pbo: float | None = None
    permutation_test_pvalue: float = 1.0
    regime_conditional_metrics: dict[str, MetricsBlock] = Field(default_factory=dict)
    avg_participation_rate: float = 0.0
    estimated_capacity_usd: float = 0.0
    factor_turnover: float = 0.0
    break_even_cost_bps: float = 0.0
    avg_holding_period_bars: float = 0.0
    return_autocorr_lag1: float = 0.0
    data_quality_notes: list[DataQualityNote] = Field(default_factory=list)
    leakage_checks_passed: bool = True
    split_overlap_detected: bool = False
    oos_trade_count: int = 0
    actual_cost_bps: float = 0.0
    prior_posterior_ic_ratio: float = 1.0
    window_stability: WindowStabilityDiagnostics | None = None
    # Bounded (downsampled) final-OOS compound-equity path so the product can render
    # a real chart for any result — including a reproduced run, which returns only
    # this BacktestResult and no heavyweight detail payload. Empty by default for
    # backward compatibility with archived results and the cross-repo consumer.
    equity_curve: list[float] = Field(default_factory=list)
    trial_diagnostics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def max_data_quality_degraded_ratio(self) -> float:
        if not self.data_quality_notes:
            return 0.0
        return max(note.degraded_ratio for note in self.data_quality_notes)


class FactorEvidenceReport(BaseModel):
    experiment_id: str
    candidate_id: str
    hypothesis_family: str
    method_id: str
    symbol: str
    market: str
    interval: str
    horizons_bars: list[int] = Field(default_factory=list)
    ic_by_horizon: dict[str, float] = Field(default_factory=dict)
    ic_ci_by_horizon: dict[str, tuple[float, float]] = Field(default_factory=dict)
    rankic_by_horizon: dict[str, float] = Field(default_factory=dict)
    quantile_spread_by_horizon: dict[str, float] = Field(default_factory=dict)
    signal_decay_profile: dict[str, float] = Field(default_factory=dict)
    best_horizon_bars: int | None = None
    regime_conditional_ic: dict[str, dict[str, float]] = Field(default_factory=dict)
    funding_conditional_ic: dict[str, dict[str, float]] = Field(default_factory=dict)
    long_only_metrics: MetricsBlock = Field(default_factory=MetricsBlock)
    short_only_metrics: MetricsBlock = Field(default_factory=MetricsBlock)
    turnover_adjusted_return: float = 0.0
    decay_quality: float = 0.0
    long_short_spread_sharpe: float = 0.0
    regime_conflict: bool = False
    evidence_score: float = 0.0
    evidence_flags: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    conflict_reasons: list[str] = Field(default_factory=list)
    gross_net_decomposition: dict[str, float | int | None] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchGateResult(BaseModel):
    experiment_id: str
    candidate_id: str
    status: Literal["production_passed", "research_survivor", "rejected"]
    production_gate_passed: bool = False
    production_gate_failures: list[str] = Field(default_factory=list)
    research_score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    evidence_flags: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchSurvivorRecord(BaseModel):
    candidate_id: str
    experiment_id: str
    status: Literal["active", "promoted", "retired"] = "active"
    candidate_payload: dict[str, Any] = Field(default_factory=dict)
    paper_trade_start_date: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    current_trades: int = 0
    required_additional_trades: int = 0
    required_oos_days: int = 90
    recheck_trigger: str = "on_next_pipeline_run"
    promotion_criteria: str = "NW FDR P < 0.10 AND trades >= threshold"
    promotion_ready: bool = False
    survivor_reason: str = ""
    research_score: float = 0.0
    fdr_pvalue: float | None = None
    sharpe: float = 0.0
    dsr: float = 0.0
    cost_margin_bps: float | None = None
    production_gate_failures: list[str] = Field(default_factory=list)
    evidence_flags: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    status_reason: str | None = None
    # Recheck bookkeeping (defaults so pre-factory payload_json rows parse):
    # holdout-grade rejections in a row; K of them demotes to retired.
    consecutive_recheck_failures: int = 0
    last_recheck_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NearMissAnalysis(BaseModel):
    experiment_id: str
    candidate_id: str
    primary_reason: Literal[
        "production_passed",
        "cost_destroyed_edge",
        "excess_turnover",
        "horizon_mismatch",
        "regime_mixing",
        "funding_state_mixing",
        "long_short_asymmetry",
        "statistically_underpowered_survivor",
        "insufficient_trades",
        "overfit_or_unstable",
        "weak_but_stable_ic",
        "no_evidence",
    ]
    reasons: list[str] = Field(default_factory=list)
    actionable: bool = False
    suggested_params: dict[str, Any] = Field(default_factory=dict)
    suggested_param_variants: list[dict[str, Any]] = Field(default_factory=list)
    repair_actions: list[str] = Field(default_factory=list)
    diagnostics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GateCheckItem(BaseModel):
    rule_id: str
    status: Literal["pass", "fail", "warn"]
    message: str
    value: float | str | None = None
    threshold: float | str | None = None


class GateCheckResult(BaseModel):
    experiment_id: str
    candidate_id: str = ""
    passed: bool
    items: list[GateCheckItem]
    raw_passed: bool | None = None
    risk_tier: Literal["unclassified", "full_pass", "conditional_pass", "fail"] = "unclassified"
    factor_evidence_level: Literal["unknown", "weak", "moderate", "strong"] = "unknown"
    allocation_multiplier: float | None = None
    review_after_days: int | None = None
    tier_reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field
    @property
    def failures(self) -> list[GateCheckItem]:
        return [item for item in self.items if item.status == "fail"]

    @computed_field
    @property
    def warnings(self) -> list[GateCheckItem]:
        return [item for item in self.items if item.status == "warn"]


class HardScoreReport(BaseModel):
    experiment_id: str
    score: float
    haircut_sharpe: float
    fdr_adjusted_pvalue: float
    prior_posterior_ic_ratio: float
    effective_trials_count: int
    global_cumulative_trials_count: int
    allocation_multiplier: float | None = None
    blocked_method_reason: str | None = None
    archive_reproducibility_status: Literal["not_checked", "valid", "invalid"] = "not_checked"
    gatecheck_failures: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ArchiveManifest(BaseModel):
    experiment_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_sha: str | None = None
    data_manifest: dict[str, Any] = Field(default_factory=dict)
    config_hash: str
    result_hash: str
    metric_tolerance: float = 1e-6


class TrialRecord(BaseModel):
    trial_id: str
    candidate_id: str
    experiment_id: str | None = None
    hypothesis_family: str
    method_id: str
    # Root of the search lineage this evaluation belongs to; defaults to the
    # candidate's own id at storage time when unset.
    lineage_id: str | None = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


_TRAJECTORY_OPERATOR_LITERAL = Literal[
    "SEED", "SEED_INFORMED",
    "MUTATION_AT_HYPOTHESIS", "MUTATION_AT_MECHANISM", "MUTATION_AT_DSL",
    "CROSSOVER",
]

_TRAJECTORY_CLASSIFICATION_LITERAL = Literal[
    "production_passed", "research_survivor", "rejected",
    "gatecheck_failed", "pre_gate_skipped",
]


class TrajectoryRecord(BaseModel):
    """Lineage record for a single evaluated candidate after final OOS classification.

    Every candidate that reaches the final OOS window receives a trajectory
    record, regardless of whether it passes GateCheck.  Failed pre-gate repair
    attempts are NOT promoted to trajectories — they remain inline logs.
    """

    trajectory_id: str
    candidate_id: str
    experiment_id: str | None = None
    artifact_scope: str = ""
    parent_ids: list[str] = Field(default_factory=list)
    parent_trajectory_ids: list[str] = Field(default_factory=list)
    operator: _TRAJECTORY_OPERATOR_LITERAL = "SEED"
    operator_detail: str | None = None
    source_candidate_type: str | None = None
    freeze_point: dict[str, Any] = Field(default_factory=dict)
    hypothesis_family: str = ""
    method_id: str = ""
    symbol: str = ""
    classification: _TRAJECTORY_CLASSIFICATION_LITERAL = "rejected"
    promotion_reason: str | None = None
    diagnosis_text: str | None = None
    candidate_snapshot: dict[str, Any] = Field(default_factory=dict)
    backtest_result: dict[str, Any] | None = None
    evidence_snapshot: dict[str, Any] | None = None
    research_gate_snapshot: dict[str, Any] | None = None
    near_miss_snapshot: dict[str, Any] | None = None
    trial_ids: list[str] = Field(default_factory=list)
    trial_refs: list[dict[str, Any]] = Field(default_factory=list)
    artifact_references: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_operator(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy_map = {
            "GRID_TUNING": "MUTATION_AT_DSL",
            "PRE_GATE_REPAIR": "MUTATION_AT_DSL",
            "OPTIMIZER_HILL_CLIMB": "MUTATION_AT_DSL",
            "OPTIMIZER_EVOLUTION": "MUTATION_AT_DSL",
            "COMPOSITE_EQUI_WEIGHT": "CROSSOVER",
        }
        operator = data.get("operator")
        if operator in legacy_map:
            data = dict(data)
            data.setdefault("operator_detail", operator)
            data["operator"] = legacy_map[operator]
        return data


class DataCoverageRecord(BaseModel):
    market: str
    dataset: str
    symbol: str
    interval: str | None = None
    year: int
    month: int
    source_url: str
    checksum_verified: bool
    parquet_path: str | None = None
    row_count: int = 0
    status: Literal["downloaded", "normalized", "missing", "failed"] = "downloaded"
    message: str | None = None
