from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


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
    candidate_type: Literal["original", "repair", "composite", "optimizer"] = "original"
    parent_candidate_id: str | None = None


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
    deflated_sharpe: float = 0.0
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
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
