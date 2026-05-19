import numpy as np
import pandas as pd
import pytest

from factor_mining.config import BootstrapConfig, PermutationTestConfig, Settings
from factor_mining.evidence import build_factor_evidence_report
from factor_mining.models import BacktestResult, CandidateStrategySpec, MetricsBlock


def _frame(n: int = 240) -> pd.DataFrame:
    signal = np.sin(np.arange(n) / 8.0)
    forward_like = np.roll(signal, 1) * 0.001
    prices = [100.0]
    for ret in forward_like[1:]:
        prices.append(prices[-1] * (1.0 + float(ret)))
    return pd.DataFrame({
        "open_time": [1_700_000_000_000 + idx * 300_000 for idx in range(n)],
        "open": prices,
        "high": [price * 1.01 for price in prices],
        "low": [price * 0.99 for price in prices],
        "close": [price * 1.001 for price in prices],
        "volume": [100.0] * n,
        "quote_volume": [1_000_000.0] * n,
    })


def test_factor_evidence_report_contains_required_diagnostics() -> None:
    frame = _frame()
    signal = pd.Series(np.sin(np.arange(len(frame)) / 8.0), index=frame.index)
    regimes = pd.Series(["low_vol"] * 80 + ["mid_vol"] * 80 + ["high_vol"] * 80, index=frame.index)
    funding_rate = pd.Series(np.linspace(-2.0, 2.0, len(frame)), index=frame.index)
    candidate = CandidateStrategySpec(
        candidate_id="c_evidence",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    result = BacktestResult(
        experiment_id="exp-evidence",
        candidate_id="c_evidence",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.2, total_return=0.01),
        metrics_gross=MetricsBlock(sharpe=0.7, total_return=0.03),
        factor_turnover=0.05,
        avg_holding_period_bars=12.0,
        break_even_cost_bps=8.0,
        actual_cost_bps=2.0,
        oos_trade_count=120,
    )

    report = build_factor_evidence_report(
        frame=frame,
        signal=signal,
        candidate=candidate,
        result=result,
        settings=Settings(
            bootstrap=BootstrapConfig(n_resamples=20),
            permutation_test=PermutationTestConfig(n_permutations=20),
        ),
        forward_regimes=regimes,
        funding_rate=funding_rate,
        funding_df=None,
        horizons=(1, 3, 6, 12),
    )

    assert report.candidate_id == "c_evidence"
    assert report.horizons_bars == [1, 3, 6, 12]
    assert set(report.ic_by_horizon) == {"1", "3", "6", "12"}
    assert set(report.ic_ci_by_horizon) == {"1", "3", "6", "12"}
    assert set(report.rankic_by_horizon) == {"1", "3", "6", "12"}
    assert set(report.quantile_spread_by_horizon) == {"1", "3", "6", "12"}
    assert report.best_horizon_bars in {1, 3, 6, 12}
    assert "low_vol" in report.regime_conditional_ic
    assert any(key.startswith("state:") for key in report.funding_conditional_ic)
    assert any(key.startswith("trend:") for key in report.funding_conditional_ic)
    assert report.long_only_metrics.trade_count >= 0
    assert report.short_only_metrics.trade_count >= 0
    assert report.decay_quality >= 0.0
    assert "ic_ci_excludes_zero" in report.evidence_flags
    assert isinstance(report.regime_conflict, bool)
    assert report.gross_net_decomposition["cost_drag_sharpe"] == pytest.approx(0.5)
    assert report.gross_net_decomposition["cost_margin_bps"] == 4.0
