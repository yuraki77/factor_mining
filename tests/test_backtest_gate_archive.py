import pandas as pd
import pytest

from factor_mining.archive import archive_experiment, reproduce_archive
from factor_mining.backtest.engine import evaluate_strategy_path, run_backtest
from factor_mining.config import BootstrapConfig, DataConfig, GateCheckConfig, PermutationTestConfig, PositionSizingConfig, Settings
from factor_mining.models import BacktestResult, CandidateStrategySpec, DataQualityNote, FactorEvidenceReport, GateCheckResult, MetricsBlock
from factor_mining.registry import get_method
from factor_mining.storage import MetadataStore
from factor_mining.trial_ledger import TrialLedger
from factor_mining.validation.gatecheck import apply_fdr, apply_risk_stratified_gatechecks, run_gatecheck
from factor_mining.hardscore import hardscore


def small_settings(tmp_path) -> Settings:
    return Settings(
        data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"),
        bootstrap=BootstrapConfig(n_resamples=20),
        permutation_test=PermutationTestConfig(n_permutations=20),
        gatecheck=GateCheckConfig(ic_tstat_nw_min=0.0, rankic_tstat_nw_min=0.0, min_oos_trades=1),
    )


def make_frame(n: int = 400) -> pd.DataFrame:
    open_prices = [100 + idx * 0.1 for idx in range(n)]
    return pd.DataFrame(
        {
            "open_time": [1_700_000_000_000 + idx * 300_000 for idx in range(n)],
            "open": open_prices,
            "high": [price * 1.001 for price in open_prices],
            "low": [price * 0.999 for price in open_prices],
            "close": [price * 1.0005 for price in open_prices],
            "volume": [10.0] * n,
            "quote_volume": [1_000_000.0] * n,
        }
    )


def test_backtest_uses_next_bar_execution_and_records_trial(tmp_path) -> None:
    settings = small_settings(tmp_path)
    store = MetadataStore(settings.data.sqlite_path)
    ledger = TrialLedger(store, settings)
    frame = make_frame()
    frame.loc[1, "open"] = frame.loc[0, "open"] * 2
    frame.loc[2:, "open"] = frame.loc[1, "open"]
    signals = pd.Series([1.0] + [0.0] * (len(frame) - 1))
    candidate = CandidateStrategySpec(
        candidate_id="c1",
        hypothesis_id="h1",
        method_id="rule_mining",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"expected_ic_mid": 0.02, "oos_trade_count": 3, "pbo": 0.2},
    )
    result = run_backtest(frame, signals, candidate, settings, trial_ledger=ledger)
    assert result.metrics_secondary.total_return < 0.10
    assert result.global_trials_at_eval == 1
    assert result.pbo is None
    assert result.oos_trade_count != 3


def test_backtest_records_real_regime_pnl_and_trade_counts(tmp_path) -> None:
    settings = small_settings(tmp_path)
    frame = make_frame(120)
    signals = pd.Series(([1.0, -1.0] * 60)[: len(frame)])
    candidate = CandidateStrategySpec(
        candidate_id="c_regime",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
    )

    result = run_backtest(frame, signals, candidate, settings)

    total_regime_pnl = sum(block.pnl for block in result.regime_conditional_metrics.values())
    total_regime_trades = sum(block.trade_count for block in result.regime_conditional_metrics.values())
    assert total_regime_pnl == pytest.approx(result.metrics_primary.pnl)
    assert total_regime_trades > 0


def test_permutation_test_uses_executable_lagged_signal(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_permutation_test(factor_values, forward_returns, *, n_permutations, seed=42):
        captured["factor_values"] = list(pd.Series(factor_values))
        return 0.5

    monkeypatch.setattr(
        "factor_mining.backtest.engine.permutation_test_mean_ic",
        fake_permutation_test,
    )
    settings = small_settings(tmp_path)
    frame = make_frame(80)
    signals = pd.Series([idx / 10.0 for idx in range(len(frame))])
    candidate = CandidateStrategySpec(
        candidate_id="c_perm",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
    )

    run_backtest(frame, signals, candidate, settings)

    assert captured["factor_values"][0] == 0.0
    assert captured["factor_values"][1] == signals.iloc[0]


def test_vol_target_does_not_use_next_open_return(tmp_path) -> None:
    settings = Settings(
        data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"),
        position_sizing=PositionSizingConfig(target_annual_vol=0.02, vol_window_days=1, max_leverage=100.0),
    )
    frame = make_frame(80)
    for idx in range(1, len(frame)):
        frame.loc[idx, "open"] = frame.loc[idx - 1, "open"] * (1.0 + (0.002 if idx % 2 else -0.0015))
    shocked = frame.copy()
    shocked.loc[20:, "open"] = shocked.loc[20:, "open"] * 1.8
    signals = pd.Series([1.0] * len(frame))
    candidate = CandidateStrategySpec(
        candidate_id="c_vol",
        hypothesis_id="h1",
        method_id="rule_mining",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
    )

    base_path = evaluate_strategy_path(frame, signals, candidate, settings)
    shocked_path = evaluate_strategy_path(shocked, signals, candidate, settings)

    assert shocked_path.position.iloc[19] == pytest.approx(base_path.position.iloc[19])


def test_gatecheck_warns_autocorr_and_data_quality_without_hard_fail() -> None:
    result = BacktestResult(
        experiment_id="exp1",
        candidate_id="c1",
        hypothesis_family="momentum",
        method_id="rule_mining",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=2.0, trade_count=200, pnl=100.0),
        ic_tstat_nw=3.0,
        rankic_tstat_nw=3.0,
        sharpe_ci_5_95=(0.2, 2.5),
        probabilistic_sharpe=0.99,
        deflated_sharpe=0.95,
        pbo=0.2,
        permutation_test_pvalue=0.01,
        regime_conditional_metrics={"bull": MetricsBlock(pnl=50), "bear": MetricsBlock(pnl=50)},
        estimated_capacity_usd=50_000,
        break_even_cost_bps=50,
        actual_cost_bps=5,
        return_autocorr_lag1=0.2,
        data_quality_notes=[DataQualityNote(scope="x", message="degraded", degraded_ratio=0.2)],
        oos_trade_count=120,
        prior_posterior_ic_ratio=2.0,
    )
    gate = run_gatecheck(result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)
    assert gate.passed
    warning_ids = {item.rule_id for item in gate.warnings}
    assert {"G13", "G14"} <= warning_ids


def _gate_ready_result(*, experiment_id: str = "exp-tier", pbo: float = 0.5) -> BacktestResult:
    return BacktestResult(
        experiment_id=experiment_id,
        candidate_id="c-tier",
        hypothesis_family="momentum",
        method_id="rule_mining",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=1.5, trade_count=200, pnl=100.0),
        metrics_gross=MetricsBlock(sharpe=1.8, trade_count=200, pnl=120.0),
        ic_tstat_nw=3.0,
        rankic_tstat_nw=3.0,
        sharpe_ci_5_95=(0.2, 2.5),
        probabilistic_sharpe=0.95,
        deflated_sharpe=0.7,
        pbo=pbo,
        permutation_test_pvalue=0.01,
        regime_conditional_metrics={"bull": MetricsBlock(pnl=50), "bear": MetricsBlock(pnl=50)},
        estimated_capacity_usd=50_000,
        break_even_cost_bps=50,
        actual_cost_bps=5,
        oos_trade_count=120,
        prior_posterior_ic_ratio=2.0,
    )


def _evidence(experiment_id: str, *, strong: bool = False) -> FactorEvidenceReport:
    return FactorEvidenceReport(
        experiment_id=experiment_id,
        candidate_id="c-tier",
        hypothesis_family="momentum",
        method_id="rule_mining",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        ic_by_horizon={"12": 0.02 if strong else 0.012},
        rankic_by_horizon={"12": 0.018 if strong else 0.011},
        quantile_spread_by_horizon={"12": 2.5},
        turnover_adjusted_return=1.0 if strong else 0.0,
        decay_quality=0.5 if strong else 0.0,
        long_short_spread_sharpe=0.5 if strong else 0.0,
        evidence_flags={
            "positive_turnover_adjusted_return": strong,
            "decay_curve_supported": strong,
            "long_short_spread": strong,
            "no_conflict": True,
        },
    )


def test_apply_fdr_uses_family_groups_with_effective_floor() -> None:
    target = _gate_ready_result(experiment_id="target").model_copy(update={
        "hypothesis_family": "momentum",
        "ic_tstat_nw": 2.9,
        "rankic_tstat_nw": 0.0,
    })
    distractors = [
        _gate_ready_result(experiment_id=f"other-{idx}").model_copy(update={
            "hypothesis_family": "mean_reversion",
            "ic_tstat_nw": -1.0,
            "rankic_tstat_nw": -1.0,
        })
        for idx in range(20)
    ]

    fdr_map = apply_fdr([target, *distractors], Settings())

    assert 0.03 < fdr_map[target.experiment_id] < 0.05


def test_nonblocking_gate_items_warn_without_raw_failure() -> None:
    result = _gate_ready_result(pbo=0.20).model_copy(update={
        "ic_tstat_nw": 0.0,
        "rankic_tstat_nw": 0.0,
        "oos_trade_count": 10,
        "prior_posterior_ic_ratio": 10.0,
        "return_autocorr_lag1": 0.2,
        "regime_conditional_metrics": {"bull": MetricsBlock(pnl=100)},
    })

    gate = run_gatecheck(result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.50)

    ids = {item.rule_id for item in gate.items}
    warning_ids = {item.rule_id for item in gate.warnings}
    assert "G12" not in ids
    assert {"G3", "G4", "G4R", "G6", "G7", "G9", "G13"} <= warning_ids
    assert gate.raw_passed is True
    assert gate.failures == []


def test_gatecheck_blocks_pbo_and_data_quality_block_threshold() -> None:
    pbo_result = _gate_ready_result(pbo=0.50)
    pbo_gate = run_gatecheck(pbo_result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)
    assert pbo_gate.raw_passed is False
    assert [item.rule_id for item in pbo_gate.failures] == ["G2"]

    warn_result = _gate_ready_result(pbo=0.20).model_copy(update={
        "data_quality_notes": [DataQualityNote(scope="x", message="degraded", degraded_ratio=0.15)]
    })
    warn_gate = run_gatecheck(warn_result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)
    assert warn_gate.raw_passed is True
    assert next(item for item in warn_gate.items if item.rule_id == "G14").status == "warn"

    block_result = warn_result.model_copy(update={
        "data_quality_notes": [DataQualityNote(scope="x", message="degraded", degraded_ratio=0.25)]
    })
    block_gate = run_gatecheck(block_result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)
    assert block_gate.raw_passed is False
    assert [item.rule_id for item in block_gate.failures] == ["G14"]


def test_risk_stratified_gate_allows_conditional_pass_for_moderate_evidence() -> None:
    result = _gate_ready_result(pbo=0.35)
    gate = run_gatecheck(result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)

    apply_risk_stratified_gatechecks([result], [gate], [_evidence(result.experiment_id)], Settings())

    assert gate.raw_passed is True
    assert gate.passed is True
    assert gate.risk_tier == "conditional_pass"
    assert gate.factor_evidence_level == "moderate"
    assert gate.allocation_multiplier == 0.25
    assert gate.review_after_days == 60
    g16 = next(item for item in gate.items if item.rule_id == "G16")
    assert g16.status == "warn"
    assert "evidence_moderate" in g16.message
    assert "pbo=0.350" in g16.message


def test_risk_stratified_gate_full_pass_requires_strong_evidence_and_low_pbo() -> None:
    result = _gate_ready_result(pbo=0.20)
    gate = run_gatecheck(result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)

    apply_risk_stratified_gatechecks([result], [gate], [_evidence(result.experiment_id, strong=True)], Settings())
    score = hardscore(result, gate, fdr_adjusted_pvalue=0.01, settings=Settings())

    assert gate.passed is True
    assert gate.risk_tier == "full_pass"
    assert gate.allocation_multiplier == 1.0
    assert score.allocation_multiplier == 1.0


def test_risk_stratified_gate_blocks_weak_evidence_even_with_low_pbo() -> None:
    result = _gate_ready_result(pbo=0.20)
    gate = run_gatecheck(result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)
    weak = FactorEvidenceReport(
        experiment_id=result.experiment_id,
        candidate_id=result.candidate_id,
        hypothesis_family=result.hypothesis_family,
        method_id=result.method_id,
        symbol=result.symbol,
        market=result.market,
        interval=result.interval,
        ic_by_horizon={"12": 0.0},
    )

    apply_risk_stratified_gatechecks([result], [gate], [weak], Settings())

    assert gate.passed is False
    assert gate.risk_tier == "fail"
    assert gate.allocation_multiplier == 0.0


def test_conditional_pass_hardscore_is_scaled_by_allocation() -> None:
    result = _gate_ready_result(pbo=0.35)
    gate = run_gatecheck(result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)
    apply_risk_stratified_gatechecks([result], [gate], [_evidence(result.experiment_id)], Settings())

    score = hardscore(result, gate, fdr_adjusted_pvalue=0.01, settings=Settings())

    assert score.score > 0.0
    assert score.allocation_multiplier == 0.25


def test_hardscore_rewards_continuous_evidence_components() -> None:
    settings = Settings()
    gate = GateCheckResult(experiment_id="exp-score", passed=True, allocation_multiplier=1.0, items=[])
    base = _gate_ready_result(experiment_id="exp-score", pbo=0.20).model_copy(update={
        "ic_tstat_nw": 0.0,
        "rankic_tstat_nw": 0.0,
        "oos_trade_count": 25,
        "regime_conditional_metrics": {"bull": MetricsBlock(pnl=100)},
        "prior_posterior_ic_ratio": 5.0,
        "return_autocorr_lag1": 0.20,
    })
    stronger = base.model_copy(update={
        "ic_tstat_nw": 5.0,
        "oos_trade_count": 100,
        "regime_conditional_metrics": {"bull": MetricsBlock(pnl=50), "bear": MetricsBlock(pnl=50)},
        "prior_posterior_ic_ratio": 1.0,
        "return_autocorr_lag1": 0.0,
    })

    low = hardscore(base, gate, fdr_adjusted_pvalue=0.05, settings=settings)
    high = hardscore(stronger, gate, fdr_adjusted_pvalue=0.05, settings=settings)
    blocked = hardscore(base, gate.model_copy(update={"passed": False, "allocation_multiplier": 0.0}), fdr_adjusted_pvalue=0.05, settings=settings)

    assert high.score > low.score
    assert blocked.score == 0.0


def test_archive_reproduce_validates_hash(tmp_path) -> None:
    result = BacktestResult(
        experiment_id="exp-archive",
        candidate_id="c1",
        hypothesis_family="momentum",
        method_id="rule_mining",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=2.0, trade_count=200, pnl=100.0),
    )
    gate = run_gatecheck(result.model_copy(update={
        "ic_tstat_nw": 3.0,
        "rankic_tstat_nw": 3.0,
        "sharpe_ci_5_95": (0.2, 2.5),
        "pbo": 0.2,
        "permutation_test_pvalue": 0.01,
        "regime_conditional_metrics": {"bull": MetricsBlock(pnl=50), "bear": MetricsBlock(pnl=50)},
        "estimated_capacity_usd": 50_000,
        "break_even_cost_bps": 50,
        "actual_cost_bps": 5,
        "oos_trade_count": 120,
    }), Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)
    score = hardscore(result, gate, fdr_adjusted_pvalue=0.01, settings=Settings())
    archive_experiment(result=result, gatecheck=gate, hardscore=score, settings=Settings(), root=tmp_path / "archives")
    assert reproduce_archive("exp-archive", root=tmp_path / "archives")["status"] == "valid"
