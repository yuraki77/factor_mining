from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    markets: list[Literal["spot", "um_futures"]] = Field(default_factory=lambda: ["spot", "um_futures"])
    default_interval: str = "5m"
    on_demand_intervals: list[str] = Field(default_factory=lambda: ["1m"])
    start_date: str = "2020-01-01"
    raw_dir: Path = Path("data/raw")
    parquet_dir: Path = Path("data/parquet")
    sqlite_path: Path = Path("data/factor_mining.sqlite3")


class TrialLedgerConfig(BaseModel):
    partition: Literal["family_and_rolling_window"] = "family_and_rolling_window"
    window_days: int = 90


class WalkForwardConfig(BaseModel):
    mode: Literal["rolling"] = "rolling"
    train_months: int = 12
    validation_months: int = 3
    test_months: int = 3
    purge_bars_floor: int = 288
    embargo_bars: int = 288
    min_folds: int = 4

    def purge_bars(self, max_feature_lookback_bars: int) -> int:
        return max(self.purge_bars_floor, 2 * max_feature_lookback_bars)


class CPCVConfig(BaseModel):
    n_groups: int = 8
    test_groups: int = 2
    pbo_threshold: float = 0.40
    ml_pbo_threshold: float = 0.30


class RegimeConfig(BaseModel):
    method: Literal["btc_rolling_60d"] = "btc_rolling_60d"
    bull_threshold: float = 0.20
    bear_threshold: float = -0.20
    high_vol_rank: float = 0.85


class BootstrapConfig(BaseModel):
    method: Literal["stationary_block"] = "stationary_block"
    n_resamples: int = 1000
    min_block_length_bars: int = 288
    block_length_multiplier: float = 2.0
    ci_levels: tuple[float, float] = (0.05, 0.95)

    def block_length_bars(self, avg_holding_period_bars: float) -> int:
        return max(self.min_block_length_bars, int(self.block_length_multiplier * avg_holding_period_bars))


class NeweyWestConfig(BaseModel):
    kernel: Literal["bartlett"] = "bartlett"
    bandwidth: Literal["fixed_cube_root"] = "fixed_cube_root"


class PermutationTestConfig(BaseModel):
    n_permutations: int = 100
    permute_target: Literal["factor_values"] = "factor_values"
    test_statistic: Literal["mean_ic"] = "mean_ic"


class PositionSizingConfig(BaseModel):
    primary: Literal["vol_target"] = "vol_target"
    secondary: Literal["fixed_notional"] = "fixed_notional"
    target_annual_vol: float = 0.15
    vol_window_days: int = 30
    max_leverage: float = 3.0
    symbol_max_leverage: dict[str, float] = Field(default_factory=dict)
    fixed_notional_usd: float = 10_000.0

    def max_leverage_for(self, symbol: str) -> float:
        overrides = {str(key).upper(): float(value) for key, value in self.symbol_max_leverage.items()}
        for key in _symbol_lookup_keys(symbol):
            if key in overrides:
                value = overrides[key]
                return max(0.0, value) if math.isfinite(value) else self.max_leverage
        return self.max_leverage


class ExitConfig(BaseModel):
    stop_loss_pct: float = -0.05
    max_hold_bars: int = 0
    tp_tiers: list[list[float]] = Field(default_factory=list)
    trailing_stop_pct: float = 0.0
    trailing_after_first_tp: bool = True


class CostConfig(BaseModel):
    maker_bps: float = 2.0
    taker_bps: float = 5.0
    slippage_base_bps: float = 1.0
    slippage_k: float = 25.0
    slippage_gamma: float = 0.5


class CapacityConfig(BaseModel):
    adv_participation: float = 0.05
    min_capacity_usd: float = 10_000.0


class MLComplexityConfig(BaseModel):
    max_depth: int = 3
    max_features: int = 32
    min_train_samples: int = 5000


class DeepSeekConfig(BaseModel):
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    hypothesis_model: str = "deepseek-chat"
    hardscore_model: str = "deepseek-chat"


class LLMConfig(BaseModel):
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)


class GateCheckConfig(BaseModel):
    fdr_q: float = 0.05
    # G1: minimum Bailey-LdP deflated-Sharpe probability (PSR at the
    # expected-max-SR threshold). 0.95 is the B&LdP confidence convention.
    dsr_prob_min: float = 0.95
    ic_tstat_nw_min: float = 2.0
    rankic_tstat_nw_min: float = 2.0
    sharpe_ci_5_min: float = 0.0
    min_oos_trades: int = 100
    max_regime_pnl_concentration: float = 0.80
    break_even_cost_multiple: float = 2.0
    prior_posterior_ic_max_ratio: float = 5.0
    return_autocorr_warn_abs: float = 0.10
    data_quality_degraded_warn_ratio: float = 0.10
    data_quality_degraded_block_ratio: float = 0.20
    research_survivor_promotion_fdr_p: float = 0.10
    research_survivor_min_oos_days: int = 90


class ExitBoundsConfig(BaseModel):
    stop_loss_pct_min: float = -0.15
    stop_loss_pct_max: float = -0.01
    max_hold_bars_min: int = 10
    max_hold_bars_max: int = 2000
    tp_tier_pct_min: float = 0.005
    tp_tier_pct_max: float = 0.20
    tp_tier_fraction_min: float = 0.10
    tp_tier_fraction_max: float = 0.90
    max_tp_tiers: int = 4
    trailing_stop_pct_min: float = 0.005
    trailing_stop_pct_max: float = 0.10


class EvolutionaryConfig(BaseModel):
    """Controls for the evolutionary alpha workflow (Phase 3).

    All evolutionary behaviour is gated behind ``enabled`` — when False
    (default) the pipeline is unchanged.
    """

    enabled: bool = False
    budget_per_round: int = 20
    crossover_parent_count: int = 4
    freeze_depth: int = 3
    llm_mutations: bool = False
    max_constants_per_expression: int = 3
    max_complexity: int = 30
    max_categorical_comparisons: int = 4
    output_correlation_reject_threshold: float = 0.8
    output_correlation_warn_threshold: float = 0.6


class Settings(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    trial_ledger: TrialLedgerConfig = Field(default_factory=TrialLedgerConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    cpcv: CPCVConfig = Field(default_factory=CPCVConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    newey_west: NeweyWestConfig = Field(default_factory=NeweyWestConfig)
    permutation_test: PermutationTestConfig = Field(default_factory=PermutationTestConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    ml_complexity: MLComplexityConfig = Field(default_factory=MLComplexityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    gatecheck: GateCheckConfig = Field(default_factory=GateCheckConfig)
    exit_bounds: ExitBoundsConfig = Field(default_factory=ExitBoundsConfig)
    evolutionary: EvolutionaryConfig = Field(default_factory=EvolutionaryConfig)


def apply_trade_overrides(
    settings: Settings,
    *,
    btc_leverage: float | None = None,
    eth_leverage: float | None = None,
    taker_bps: float | None = None,
    slippage_base_bps: float | None = None,
    slippage_k: float | None = None,
    slippage_gamma: float | None = None,
) -> Settings:
    """Return a settings copy with run-scoped leverage and execution-cost overrides."""
    leverage = dict(settings.position_sizing.symbol_max_leverage)
    if btc_leverage is not None:
        leverage["BTCUSDT"] = _nonnegative_float(btc_leverage, "btc_leverage")
    if eth_leverage is not None:
        leverage["ETHUSDT"] = _nonnegative_float(eth_leverage, "eth_leverage")

    position_updates = {}
    if leverage != settings.position_sizing.symbol_max_leverage:
        position_updates["symbol_max_leverage"] = leverage

    cost_updates = {}
    for key, value in {
        "taker_bps": taker_bps,
        "slippage_base_bps": slippage_base_bps,
        "slippage_k": slippage_k,
        "slippage_gamma": slippage_gamma,
    }.items():
        if value is not None:
            cost_updates[key] = _nonnegative_float(value, key)

    updates = {}
    if position_updates:
        updates["position_sizing"] = settings.position_sizing.model_copy(update=position_updates)
    if cost_updates:
        updates["costs"] = settings.costs.model_copy(update=cost_updates)
    return settings.model_copy(update=updates) if updates else settings


def load_settings(path: Path | None = None) -> Settings:
    load_dotenv(Path(".env"))
    if path is None:
        path = Path("configs/default.yaml")
    if not path.exists():
        return Settings()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return Settings.model_validate(payload)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _symbol_lookup_keys(symbol: str) -> list[str]:
    normalized = str(symbol or "").upper().replace("/", "").replace("-", "")
    keys = [normalized]
    matched_suffix = False
    for suffix in ("USDT", "USD", "BUSD"):
        if normalized.endswith(suffix) and normalized != suffix:
            base = normalized[: -len(suffix)]
            if base:
                keys.append(base)
            matched_suffix = True
            break
    if normalized and not matched_suffix:
        keys.extend([f"{normalized}USDT", f"{normalized}USD"])
    return keys


def _nonnegative_float(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return number
