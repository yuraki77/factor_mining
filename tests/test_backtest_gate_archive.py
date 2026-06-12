import pandas as pd
import pytest

from factor_mining.archive import archive_experiment, verify_archive
from factor_mining.backtest.engine import _EQUITY_CURVE_MAX_POINTS, _apply_exit_rules, _bounded_equity_curve, evaluate_strategy_path, run_backtest, walk_forward_oos_mask
from factor_mining.config import BootstrapConfig, DataConfig, GateCheckConfig, PermutationTestConfig, PositionSizingConfig, Settings, WalkForwardConfig
from factor_mining.models import BacktestResult, CandidateStrategySpec, DataQualityNote, FactorEvidenceReport, GateCheckResult, MetricsBlock
from factor_mining.registry import get_method
from factor_mining.storage import MetadataStore
from factor_mining.trial_ledger import TrialLedger
from factor_mining.validation.gatecheck import apply_risk_stratified_gatechecks, run_gatecheck
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


def test_walk_forward_oos_mask_purges_between_test_windows() -> None:
    month_ms = 30 * 86_400_000
    frame = pd.DataFrame({"open_time": [1_700_000_000_000 + idx * month_ms for idx in range(15)]})
    settings = Settings(
        walk_forward=WalkForwardConfig(
            train_months=0,
            validation_months=0,
            test_months=2,
            purge_bars_floor=2,
            embargo_bars=3,
        )
    )
    candidate = CandidateStrategySpec(
        candidate_id="c-wf",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        max_feature_lookback_bars=1,
    )

    mask = walk_forward_oos_mask(frame, settings, candidate)
    oos_indices = list(mask[mask].index)

    assert oos_indices[:4] == [2, 3, 9, 10]
    assert oos_indices[2] - oos_indices[1] - 1 == 5


def test_gatecheck_result_carries_candidate_id() -> None:
    result = BacktestResult(
        experiment_id="exp-gate",
        candidate_id="cand-gate",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=1.0, trade_count=100),
        metrics_gross=MetricsBlock(sharpe=1.2),
        deflated_sharpe=1.0,
        pbo=0.1,
        oos_trade_count=100,
        estimated_capacity_usd=1_000_000.0,
        break_even_cost_bps=10.0,
        actual_cost_bps=1.0,
    )

    gate = run_gatecheck(result, Settings(), method=get_method("factor_scoring"), fdr_adjusted_pvalue=0.01)

    assert gate.candidate_id == "cand-gate"


def test_exit_rules_reset_take_profit_after_natural_flat() -> None:
    position = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    open_returns = pd.Series([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])

    adjusted = _apply_exit_rules(position, open_returns, tp_tiers=[(0.02, 0.50)])

    assert adjusted.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0, 1.0, 1.0])


def test_exit_rules_stop_blocks_reentry_until_signal_resets() -> None:
    position = pd.Series([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    open_returns = pd.Series([-0.06, 0.0, 0.0, 0.0, 0.0, 0.0])

    adjusted = _apply_exit_rules(position, open_returns, stop_loss_pct=-0.05)

    assert adjusted.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])


def test_exit_rules_max_hold_allows_reentry_on_continuous_signal() -> None:
    position = pd.Series([1.0, 1.0, 1.0, 1.0])
    open_returns = pd.Series([0.0, 0.0, 0.0, 0.0])

    adjusted = _apply_exit_rules(position, open_returns, max_hold_bars=2)

    assert adjusted.tolist() == pytest.approx([1.0, 1.0, 0.0, 1.0])


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
    # I5: every result carries a bounded equity curve so the product (and a
    # reproduced run, which returns only this BacktestResult) can chart it.
    assert 0 < len(result.equity_curve) <= _EQUITY_CURVE_MAX_POINTS
    assert all(value > 0.0 for value in result.equity_curve)


def test_bounded_equity_curve_caps_points_and_keeps_endpoints() -> None:
    """I5: the equity curve is downsampled to a fixed cap so an archived/serialized
    BacktestResult stays small regardless of OOS length, while the first and last
    points — and therefore the exact final compounded value — are preserved. WHY it
    matters: the product charts this curve directly for reproduced runs, so it must
    be both bounded (no unbounded payload growth) and faithful at the endpoints."""
    returns = pd.Series([0.001] * 5000)
    curve = _bounded_equity_curve(returns)
    assert len(curve) <= _EQUITY_CURVE_MAX_POINTS
    assert curve[0] == pytest.approx(1.001)
    assert curve[-1] == pytest.approx(1.001**5000)
    short = _bounded_equity_curve(pd.Series([0.01, -0.02, 0.03]))
    assert short == pytest.approx([1.01, 1.01 * 0.98, 1.01 * 0.98 * 1.03])


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


def test_symbol_specific_leverage_caps_vol_target_position(tmp_path) -> None:
    settings = Settings(
        data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"),
        position_sizing=PositionSizingConfig(
            target_annual_vol=10.0,
            vol_window_days=1,
            max_leverage=1.0,
            symbol_max_leverage={"ETHUSDT": 4.0},
        ),
    )
    frame = make_frame(80)
    signals = pd.Series([1.0] * len(frame))
    btc_candidate = CandidateStrategySpec(
        candidate_id="c_btc_lev",
        hypothesis_id="h1",
        method_id="rule_mining",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
    )
    eth_candidate = btc_candidate.model_copy(update={"candidate_id": "c_eth_lev", "symbol": "ETHUSDT"})

    btc_path = evaluate_strategy_path(frame, signals, btc_candidate, settings)
    eth_path = evaluate_strategy_path(frame, signals, eth_candidate, settings)

    assert btc_path.position.max() == pytest.approx(1.0)
    assert eth_path.position.max() == pytest.approx(4.0)


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


def test_risk_stratified_gate_allows_conditional_pass_for_moderate_evidence() -> None:
    result = _gate_ready_result(pbo=0.30)
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
    assert "pbo=0.300" in g16.message


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
    result = _gate_ready_result(pbo=0.30)
    gate = run_gatecheck(result, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)
    apply_risk_stratified_gatechecks([result], [gate], [_evidence(result.experiment_id)], Settings())

    score = hardscore(result, gate, fdr_adjusted_pvalue=0.01, settings=Settings())

    assert score.score > 0.0
    assert score.allocation_multiplier == 0.25


def test_hardscore_distinguishes_strong_dsr_without_hard_saturation() -> None:
    low = _gate_ready_result(experiment_id="exp-low", pbo=0.20).model_copy(update={"deflated_sharpe": 1.0})
    high = _gate_ready_result(experiment_id="exp-high", pbo=0.20).model_copy(update={"deflated_sharpe": 3.0})
    low_gate = GateCheckResult(experiment_id=low.experiment_id, passed=True, items=[], allocation_multiplier=1.0)
    high_gate = GateCheckResult(experiment_id=high.experiment_id, passed=True, items=[], allocation_multiplier=1.0)

    low_score = hardscore(low, low_gate, fdr_adjusted_pvalue=0.01, settings=Settings())
    high_score = hardscore(high, high_gate, fdr_adjusted_pvalue=0.01, settings=Settings())

    assert high_score.score > low_score.score


def test_hardscore_distinguishes_strong_ic_without_hard_saturation() -> None:
    low = _gate_ready_result(experiment_id="exp-low", pbo=0.20).model_copy(update={"ic_tstat_nw": 4.0, "rankic_tstat_nw": 4.0})
    high = _gate_ready_result(experiment_id="exp-high", pbo=0.20).model_copy(update={"ic_tstat_nw": 8.0, "rankic_tstat_nw": 8.0})
    low_gate = GateCheckResult(experiment_id=low.experiment_id, passed=True, items=[], allocation_multiplier=1.0)
    high_gate = GateCheckResult(experiment_id=high.experiment_id, passed=True, items=[], allocation_multiplier=1.0)

    low_score = hardscore(low, low_gate, fdr_adjusted_pvalue=0.01, settings=Settings())
    high_score = hardscore(high, high_gate, fdr_adjusted_pvalue=0.01, settings=Settings())

    assert high_score.score > low_score.score


def test_hardscore_score_does_not_depend_on_haircut_or_psr_components() -> None:
    low = _gate_ready_result(experiment_id="exp-low", pbo=0.20).model_copy(
        update={
            "metrics_primary": MetricsBlock(sharpe=0.5, trade_count=200, pnl=100.0),
            "probabilistic_sharpe": 0.10,
        }
    )
    high = _gate_ready_result(experiment_id="exp-high", pbo=0.20).model_copy(
        update={
            "metrics_primary": MetricsBlock(sharpe=5.0, trade_count=200, pnl=100.0),
            "probabilistic_sharpe": 0.99,
        }
    )
    low_gate = GateCheckResult(experiment_id=low.experiment_id, passed=True, items=[], allocation_multiplier=1.0)
    high_gate = GateCheckResult(experiment_id=high.experiment_id, passed=True, items=[], allocation_multiplier=1.0)

    low_score = hardscore(low, low_gate, fdr_adjusted_pvalue=0.01, settings=Settings())
    high_score = hardscore(high, high_gate, fdr_adjusted_pvalue=0.01, settings=Settings())

    assert high_score.haircut_sharpe > low_score.haircut_sharpe
    assert high_score.score == low_score.score


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
    filled = result.model_copy(update={
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
    })
    gate = run_gatecheck(filled, Settings(), method=get_method("rule_mining"), fdr_adjusted_pvalue=0.01)
    score = hardscore(filled, gate, fdr_adjusted_pvalue=0.01, settings=Settings())
    archive_experiment(result=filled, gatecheck=gate, hardscore=score, settings=Settings(), root=tmp_path / "archives")
    assert verify_archive("exp-archive", root=tmp_path / "archives")["status"] == "valid"


def test_archive_bundle_writes_candidate_and_data_manifest(tmp_path) -> None:
    """I2: the archive bundle must carry the full candidate spec (candidate.json)
    and a real data manifest, so a later reproduce can reconstruct the exact
    CandidateStrategySpec and pin the data extent — BacktestResult omits params."""
    import json

    from factor_mining.models import CandidateStrategySpec

    result = BacktestResult(
        experiment_id="exp-bundle",
        candidate_id="c1",
        hypothesis_family="momentum",
        method_id="rule_mining",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=2.0, trade_count=200, pnl=100.0),
    )
    gate = GateCheckResult(experiment_id=result.experiment_id, passed=True, items=[], allocation_multiplier=1.0)
    score = hardscore(result, gate, fdr_adjusted_pvalue=0.01, settings=Settings())
    candidate = CandidateStrategySpec(
        candidate_id="c1",
        hypothesis_id="h1",
        method_id="rule_mining",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        params={"signal_source": "factor_signal", "factor_family": "momentum", "factor_lookback": 24},
    )
    data_manifest = {"symbol": "BTCUSDT", "market": "um_futures", "data_end_ms": 1_777_593_300_000, "rows": 1000}
    root = tmp_path / "archives"
    archive_experiment(
        result=result,
        gatecheck=gate,
        hardscore=score,
        settings=Settings(),
        candidate=candidate,
        data_manifest=data_manifest,
        root=root,
    )
    bundle = root / "exp-bundle"
    saved_candidate = json.loads((bundle / "candidate.json").read_text())
    assert saved_candidate["candidate_id"] == "c1"
    assert saved_candidate["params"]["factor_family"] == "momentum"  # params survive the round-trip
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["data_manifest"]["data_end_ms"] == 1_777_593_300_000  # real extent, not {}
