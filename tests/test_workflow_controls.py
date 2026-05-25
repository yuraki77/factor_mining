import numpy as np
import pandas as pd
from typer.testing import CliRunner

from factor_mining.cli import app
from factor_mining.config import DataConfig, Settings
from factor_mining.models import BacktestResult, CandidateStrategySpec, MetricsBlock, ResearchSurvivorRecord
from factor_mining.pipeline import MarketDataContext, PipelineResult, verify_research_survivors
from factor_mining.storage import MetadataStore
from factor_mining.ui import _run_args


def test_cli_run_creates_pipeline_run_and_events(tmp_path, monkeypatch) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))

    def fake_run_pipeline(settings_arg, **kwargs):
        assert settings_arg.data.sqlite_path == settings.data.sqlite_path
        assert kwargs["run_id"]
        assert kwargs["discovery_rounds"] == 2
        assert kwargs["optimization_rounds"] == 4
        kwargs["event_sink"]("step", "info", "fake pipeline step", {"ok": True})
        return PipelineResult(elapsed_s=0.1)

    monkeypatch.setattr("factor_mining.cli.load_settings", lambda: settings)
    monkeypatch.setattr("factor_mining.pipeline.run_pipeline", fake_run_pipeline)

    result = CliRunner().invoke(app, ["mine", "run", "--tail", "10", "--discovery-rounds", "2", "--optimization-rounds", "4"])

    assert result.exit_code == 0
    store = MetadataStore(settings.data.sqlite_path)
    runs = store.list_pipeline_runs()
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    events = store.load_pipeline_events(runs[0]["run_id"])
    assert any(event["message"] == "fake pipeline step" for event in events)


def test_dashboard_run_args_parse_split_round_controls() -> None:
    args = _run_args({
        "discovery_rounds": 2,
        "optimization_rounds": 4,
        "tail": 0,
    })

    assert args["discovery_rounds"] == 2
    assert args["optimization_rounds"] == 4
    assert args["tail"] is None


def test_cli_run_rejects_tail_with_sample_bars(tmp_path, monkeypatch) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    monkeypatch.setattr("factor_mining.cli.load_settings", lambda: settings)

    result = CliRunner().invoke(app, ["mine", "run", "--tail", "10", "--sample-bars", "10"])

    assert result.exit_code != 0
    assert "--tail and --sample-bars are mutually exclusive" in result.output


def test_verify_research_survivors_promotes_without_recording_new_trials(tmp_path, monkeypatch) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    candidate = CandidateStrategySpec(
        candidate_id="cand-survivor",
        hypothesis_id="hyp-1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
        market="um_futures",
    )
    store.upsert_research_survivors([
        ResearchSurvivorRecord(
            candidate_id=candidate.candidate_id,
            experiment_id="exp-old",
            status="active",
            candidate_payload=candidate.model_dump(mode="json"),
        )
    ])

    frame = pd.DataFrame({
        "open_time": [1_700_000_000_000 + idx * 300_000 for idx in range(60)],
        "open": np.linspace(100.0, 110.0, 60),
        "high": np.linspace(101.0, 111.0, 60),
        "low": np.linspace(99.0, 109.0, 60),
        "close": np.linspace(100.5, 110.5, 60),
        "volume": [100.0] * 60,
        "quote_volume": [1_000_000.0] * 60,
    })
    context = MarketDataContext(
        symbol="BTCUSDT",
        market="um_futures",
        frame=frame,
        features_df=pd.DataFrame(index=frame.index),
        feature_meta={},
        forward_regimes=pd.Series(["unknown"] * len(frame), index=frame.index),
        funding_df=None,
        funding_rate=pd.Series([0.0] * len(frame), index=frame.index),
        data_quality_notes=[],
    )

    def fake_backtests(tasks, final_frame, settings_arg, max_workers, funding_df=None):
        return [BacktestResult(
            experiment_id="exp-new",
            candidate_id=candidate.candidate_id,
            hypothesis_family=candidate.hypothesis_family,
            method_id=candidate.method_id,
            symbol=candidate.symbol,
            market=candidate.market,
            interval=candidate.interval,
            metrics_primary=MetricsBlock(sharpe=2.0, trade_count=120),
            metrics_gross=MetricsBlock(sharpe=2.1),
            ic_tstat_nw=4.0,
            rankic_tstat_nw=4.0,
            sharpe_ci_5_95=(0.5, 2.5),
            deflated_sharpe=1.5,
            pbo=0.1,
            oos_trade_count=120,
            estimated_capacity_usd=1_000_000.0,
            break_even_cost_bps=20.0,
            actual_cost_bps=1.0,
        )]

    monkeypatch.setattr("factor_mining.pipeline._load_data_contexts", lambda *args, **kwargs: {("BTCUSDT", "um_futures"): context})
    monkeypatch.setattr(
        "factor_mining.pipeline._build_tasks",
        lambda candidates, *args, **kwargs: [(np.ones(len(frame)), candidates[0].model_dump(mode="json"), 0, {}, [])],
    )
    monkeypatch.setattr("factor_mining.pipeline._run_backtests_parallel", fake_backtests)
    monkeypatch.setattr("factor_mining.pipeline._apply_batch_pbo", lambda *args, **kwargs: None)
    monkeypatch.setattr("factor_mining.pipeline.build_factor_evidence_reports", lambda *args, **kwargs: [])

    result = verify_research_survivors(settings, store=store, max_workers=1)

    records = store.list_research_survivors(status=None)
    assert result.backtests[0].candidate_id == candidate.candidate_id
    assert records[0].status == "promoted"
    assert records[0].status_reason in {"production_gate_passed", "promotion_criteria_met"}
    assert store.trial_counts(candidate.hypothesis_family) == (0, 0, 0)
