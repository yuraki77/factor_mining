from __future__ import annotations

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
    n_permutations: int = 5000
    permute_target: Literal["factor_values"] = "factor_values"
    test_statistic: Literal["mean_ic"] = "mean_ic"
    rejection_threshold: float = 0.05


class PositionSizingConfig(BaseModel):
    primary: Literal["vol_target"] = "vol_target"
    secondary: Literal["fixed_notional"] = "fixed_notional"
    target_annual_vol: float = 0.15
    vol_window_days: int = 30
    max_leverage: float = 3.0
    fixed_notional_usd: float = 10_000.0


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


class MiniMaxConfig(BaseModel):
    base_url: str = "https://api.minimaxi.com/v1"
    api_key_env: str = "MINIMAX_API_KEY"
    optimizer_model: str = "MiniMax-M2.7"


class LLMConfig(BaseModel):
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    minimax: MiniMaxConfig = Field(default_factory=MiniMaxConfig)


class GateCheckConfig(BaseModel):
    fdr_q: float = 0.05
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
    costs: CostConfig = Field(default_factory=CostConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    ml_complexity: MLComplexityConfig = Field(default_factory=MLComplexityConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    gatecheck: GateCheckConfig = Field(default_factory=GateCheckConfig)


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
