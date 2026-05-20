import pandas as pd

from factor_mining.backtest.engine import build_backtest_detail
from factor_mining.config import DataConfig, Settings
from factor_mining.models import BacktestResult, CandidateStrategySpec, GateCheckItem, GateCheckResult, MetricsBlock
from factor_mining.storage import MetadataStore
from factor_mining.ui import _json_safe, _recover_interrupted_stop_requests, build_dashboard_state


def _frame(n: int = 120) -> pd.DataFrame:
    prices = [100.0 + idx * 0.2 for idx in range(n)]
    return pd.DataFrame(
        {
            "open_time": [1_700_000_000_000 + idx * 300_000 for idx in range(n)],
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": [price * 1.002 for price in prices],
            "volume": [100.0] * n,
            "quote_volume": [1_000_000.0] * n,
        }
    )


def test_pipeline_run_events_are_ordered_and_active_run_clears(tmp_path) -> None:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.create_pipeline_run("run-1", {"tail": 100})

    assert store.active_pipeline_run()["run_id"] == "run-1"

    store.append_pipeline_event("run-1", phase="step", message="start", payload={"step": 1})
    store.append_pipeline_event("run-1", phase="log", message="finish")

    events = store.load_pipeline_events("run-1")
    assert [event["seq"] for event in events] == [1, 2]
    assert events[0]["payload"] == {"step": 1}

    store.update_pipeline_run("run-1", "completed")
    assert store.active_pipeline_run() is None
    assert store.list_pipeline_runs()[0]["status"] == "completed"


def test_metadata_store_prunes_unprotected_artifacts(tmp_path) -> None:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.save_artifact("experiment_detail_keep", "experiment_detail", {"value": 1})
    store.save_artifact("experiment_detail_drop", "experiment_detail", {"value": 2})
    store.save_artifact("latest_backtests", "backtests", {"items": []})

    deleted = store.prune_artifacts(
        kind="experiment_detail",
        keep_artifact_ids={"experiment_detail_keep"},
        max_unprotected_rows=0,
    )

    assert deleted == 1
    assert store.load_artifact("experiment_detail_keep") == {"value": 1}
    assert store.load_artifact("experiment_detail_drop") is None
    assert store.load_artifact("latest_backtests") == {"items": []}


def test_pipeline_stop_request_keeps_run_active_until_worker_finishes(tmp_path) -> None:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.create_pipeline_run("hosted-1", {"mode": "hosted"})

    store.request_pipeline_stop("hosted-1")

    assert store.pipeline_run_status("hosted-1") == "stopping"
    assert store.active_pipeline_run()["run_id"] == "hosted-1"

    store.update_pipeline_run("hosted-1", "stopped")
    assert store.active_pipeline_run() is None


def test_dashboard_recovers_interrupted_stop_request(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    store.create_pipeline_run("hosted-1", {"mode": "hosted"})
    store.request_pipeline_stop("hosted-1")

    assert store.active_pipeline_run()["run_id"] == "hosted-1"

    assert _recover_interrupted_stop_requests(settings) == 1

    assert store.active_pipeline_run() is None
    assert store.pipeline_run_status("hosted-1") == "stopped"
    assert store.load_pipeline_events("hosted-1")[-1]["message"] == "Recovered interrupted stop request on dashboard startup."


def test_dashboard_state_includes_gatecheck_diagnostics(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    store.save_artifact("latest_gatecheck_diagnostics", "gatecheck_diagnostics", {
        "total": 1,
        "passed": 0,
        "rows": [{
            "experiment_id": "exp-1",
            "candidate_id": "cand-1",
            "gross_sharpe": 0.3,
            "cost_drag_sharpe": 1.1,
            "cost_margin_bps": -2.0,
            "factor_turnover": 0.2,
            "search_variant": "low_turnover",
            "signal_source": "factor_signal",
        }],
        "failure_counts": [{"rule_id": "G8", "count": 1}],
        "metric_summary": {},
    })

    state = build_dashboard_state(settings, store)

    assert state["bundle"]["gatecheck_diagnostics"]["total"] == 1


def test_dashboard_state_includes_research_survivors(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    store.save_artifact("latest_research_survivors", "research_survivors", {
        "items": [{
            "candidate_id": "cand-1",
            "hypothesis_family": "momentum",
            "research_score": 2.4,
        }],
    })

    state = build_dashboard_state(settings, store)

    assert state["bundle"]["research_survivors"][0]["candidate_id"] == "cand-1"


def test_dashboard_state_includes_factor_evidence(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    store.save_artifact("latest_factor_evidence", "factor_evidence", {
        "items": [{
            "candidate_id": "cand-1",
            "experiment_id": "exp-1",
            "ic_by_horizon": {"1": 0.02},
            "funding_conditional_ic": {"state:positive": {"1": 0.03}},
        }],
    })

    state = build_dashboard_state(settings, store)

    assert state["bundle"]["factor_evidence"][0]["candidate_id"] == "cand-1"


def test_dashboard_state_includes_research_gate(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    store.save_artifact("latest_research_gate", "research_gate", {
        "items": [{
            "candidate_id": "cand-1",
            "experiment_id": "exp-1",
            "status": "research_survivor",
            "research_score": 2.0,
            "reasons": ["ic_signal"],
        }],
    })

    state = build_dashboard_state(settings, store)

    assert state["bundle"]["research_gate"][0]["status"] == "research_survivor"


def test_dashboard_state_includes_near_misses(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    store.save_artifact("latest_near_misses", "near_misses", {
        "items": [{
            "candidate_id": "cand-1",
            "experiment_id": "exp-1",
            "primary_reason": "cost_destroyed_edge",
            "actionable": True,
            "repair_actions": ["reduce_turnover"],
        }],
    })

    state = build_dashboard_state(settings, store)

    assert state["bundle"]["near_misses"][0]["primary_reason"] == "cost_destroyed_edge"


def test_dashboard_state_falls_back_to_computed_diagnostics(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    candidate = CandidateStrategySpec(
        candidate_id="cand-1",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        params={"search_variant": "low_turnover", "signal_source": "factor_signal"},
    )
    result = BacktestResult(
        experiment_id="exp-1",
        candidate_id="cand-1",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=-0.4),
        break_even_cost_bps=1.0,
        actual_cost_bps=2.0,
    )
    gate = GateCheckResult(
        experiment_id="exp-1",
        passed=False,
        items=[GateCheckItem(rule_id="G8", status="fail", message="cost")],
    )
    store.save_artifact("latest_candidates", "candidates", {"items": [candidate.model_dump(mode="json")]})
    store.save_artifact("latest_backtests", "backtests", {"items": [result.model_dump(mode="json")]})
    store.save_artifact("latest_gatechecks", "gatechecks", {"items": [gate.model_dump(mode="json")]})

    state = build_dashboard_state(settings, store)
    diagnostics = state["bundle"]["gatecheck_diagnostics"]

    assert diagnostics["generated_by"] == "ui_fallback"
    assert diagnostics["failure_counts"] == [{"rule_id": "G8", "count": 1}]
    assert diagnostics["rows"][0]["search_variant"] == "low_turnover"
    assert diagnostics["rows"][0]["cost_margin_bps"] == -3.0


def test_dashboard_json_payload_replaces_nan_with_null(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    result = BacktestResult(
        experiment_id="exp-nan",
        candidate_id="cand-nan",
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=float("nan")),
        return_autocorr_lag1=float("nan"),
    )
    store.save_artifact("latest_backtests", "backtests", {"items": [result.model_dump(mode="json")]})

    payload = _json_safe(build_dashboard_state(settings, store))

    import json
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded


def test_backtest_detail_artifact_has_chart_series_and_trade_list(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    frame = _frame()
    signals = pd.Series([0.0] * 12 + [1.0] * 36 + [-1.0] * 36 + [0.0] * 36)
    candidate = CandidateStrategySpec(
        candidate_id="candidate-1",
        hypothesis_id="hyp-1",
        method_id="parameter_sweep",
        hypothesis_family="mean_reversion",
        symbol="BTCUSDT",
        params={"lookback": 12},
    )
    result = BacktestResult(
        experiment_id="experiment-1",
        candidate_id="candidate-1",
        hypothesis_family="mean_reversion",
        method_id="parameter_sweep",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=1.2, trade_count=3),
    )

    detail = build_backtest_detail(frame, signals, candidate, settings, result, max_chart_rows=50, max_trades=20)

    assert detail["experiment_id"] == "experiment-1"
    assert detail["total_rows"] == 120
    assert len(detail["ohlcv"]) == 50
    assert len(detail["series"]) == 50
    assert {"equity", "drawdown", "position", "signal"} <= set(detail["series"][0])
    assert {trade["side"] for trade in detail["trades"]} == {"buy", "sell"}
