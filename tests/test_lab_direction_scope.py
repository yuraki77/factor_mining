from __future__ import annotations

from factor_mining.config import Settings
from factor_mining.models import HypothesisSpec
import factor_mining.pipeline as pipeline


def _hypothesis() -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="h_lab_seed",
        hypothesis_family="momentum",
        economic_mechanism="Seeded Lab hypothesis",
        testable_prediction="Positive drift after trend confirmation",
        null_hypothesis="No predictive drift",
        expected_ic_range=(0.005, 0.02),
        expected_decay_halflife_bars=24,
    )


def test_run_pipeline_defaults_remain_unscoped(monkeypatch):
    captured = {}

    def fake_impl(settings, **kwargs):
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return kwargs["result"]

    monkeypatch.setattr(pipeline, "_run_pipeline_impl", fake_impl)

    pipeline.run_pipeline(Settings(), use_llm=False)

    assert captured["settings"].data.symbols == ["BTCUSDT", "ETHUSDT"]
    assert captured["kwargs"]["seed_hypotheses"] is None
    assert captured["kwargs"]["direction_scope"] is None
    assert captured["kwargs"]["research_brief"] is None


def test_run_pipeline_applies_lab_direction_scope(monkeypatch):
    captured = {}

    def fake_impl(settings, **kwargs):
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return kwargs["result"]

    monkeypatch.setattr(pipeline, "_run_pipeline_impl", fake_impl)
    seed = [_hypothesis()]

    pipeline.run_pipeline(
        Settings(),
        use_llm=True,
        seed_hypotheses=seed,
        direction_scope={
            "factor_ids": ["rsi14", "ma_above_200"],
            "symbols": ["ethusdt"],
            "objective": "MAX_SHARPE",
            "brief": "Only mine the winning Lab ballot direction.",
        },
    )

    assert captured["settings"].data.symbols == ["ETHUSDT"]
    assert captured["kwargs"]["seed_hypotheses"] == seed
    assert captured["kwargs"]["direction_scope"]["factor_ids"] == ["rsi14", "ma_above_200"]
    assert "Only mine the winning Lab ballot direction." in captured["kwargs"]["research_brief"]
    assert "ETHUSDT" in captured["kwargs"]["research_brief"]


def test_direction_scope_annotation_adds_lab_payload_fields():
    candidate = pipeline.CandidateStrategySpec(
        candidate_id="c1",
        hypothesis_id="h1",
        method_id="factor_scoring",
        hypothesis_family="momentum",
        symbol="ETHUSDT",
        market="spot",
        interval="5m",
        params={"signal_source": "factor_signal"},
    )

    annotated = pipeline._annotate_candidates_for_direction_scope(
        [candidate],
        {"factor_ids": ["rsi14"], "symbols": ["ETHUSDT"], "objective": "MAX_SHARPE"},
    )

    assert annotated[0].params["lab_direction_factor_ids"] == ["rsi14"]
    assert annotated[0].params["lab_direction_symbols"] == ["ETHUSDT"]
    assert annotated[0].params["lab_direction_objective"] == "MAX_SHARPE"
    assert candidate.params == {"signal_source": "factor_signal"}
