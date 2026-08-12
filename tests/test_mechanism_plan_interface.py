"""The public plan seam must detect tampering and fail closed on execution."""

from __future__ import annotations

import json

import pytest

from factor_mining.models import CandidateStrategySpec, HypothesisSpec
from factor_mining.pipeline import (
    MECHANISM_PLAN_EXECUTION_SUPPORTED,
    run_pipeline,
)
from factor_mining.research.mechanism_compiler import (
    FrozenProxyTemplate,
    MechanismContract,
    MechanismExecutionPermit,
    MechanismExperimentPlan,
    MechanismNovelty,
    SemanticProxyDescriptor,
)


def _plan() -> MechanismExperimentPlan:
    mechanism_id = "interface.test"
    revision_id = "interface.test:v1"
    proxy_id = "interface.proxy"
    descriptor = SemanticProxyDescriptor(
        proxy_id=proxy_id,
        mechanism_id=mechanism_id,
        causal_driver_id="interface.driver",
        proxy_family_id="interface.family",
        mechanism_novelty=MechanismNovelty.EXISTING,
    )
    permit = MechanismExecutionPermit.freeze(
        manifest_sha256="a" * 64,
        registry_sha256="b" * 64,
        exposure_index_sha256="c" * 64,
        mechanism_id=mechanism_id,
        revision_id=revision_id,
        proxy_ids=(proxy_id,),
        inherited_exposure_ids=(),
        inherited_lineage_count=0,
        inherited_window_count=0,
    )
    hypothesis = HypothesisSpec(
        hypothesis_id="h_interface",
        hypothesis_family="interface",
        economic_mechanism="constraint to flow to price",
        testable_prediction="signed next-open response",
        null_hypothesis="no response",
        expected_ic_range=(0.0, 0.1),
        expected_decay_halflife_bars=1,
    )
    candidate = CandidateStrategySpec(
        candidate_id="c_interface",
        hypothesis_id=hypothesis.hypothesis_id,
        method_id="factor_scoring",
        hypothesis_family=hypothesis.hypothesis_family,
        symbol="BTCUSDT",
        params={
            "mechanism_id": mechanism_id,
            "mechanism_revision_id": revision_id,
            "proxy_id": proxy_id,
            "mechanism_execution_permit_sha256": permit.permit_sha256,
            "mechanism_lineage_floor": 0,
        },
    )
    return MechanismExperimentPlan.freeze(
        revision_id=revision_id,
        mechanism_id=mechanism_id,
        mechanism_contract=MechanismContract(
            mechanism_id=mechanism_id,
            name="interface",
            market_participant="participant",
            binding_constraint="constraint",
            causal_chain="constraint to flow to price",
            predicted_direction="continuation",
            predicted_horizon="one bar",
            stronger_where="condition",
            weaker_or_absent_where="control",
            required_proxies=(proxy_id,),
            falsification_conditions=("no response",),
        ),
        execution_permit=permit,
        signature_event_id="signature-event",
        proxy_templates=(
            FrozenProxyTemplate(
                descriptor=descriptor,
                hypothesis_family=hypothesis.hypothesis_family,
                method_id=candidate.method_id,
                symbols=(candidate.symbol,),
                interval=candidate.interval,
                params={},
                max_feature_lookback_bars=candidate.max_feature_lookback_bars,
                expected_ic_range=(0.0, 0.1),
                expected_decay_halflife_bars=1,
            ),
        ),
        hypotheses=(hypothesis,),
        candidates=(candidate,),
    )


def test_round_trip_preserves_the_frozen_digest(tmp_path) -> None:
    path = tmp_path / "plan.json"
    expected = _plan()
    expected.to_json(path)

    loaded = MechanismExperimentPlan.from_json(path)

    assert loaded.plan_sha256 == expected.plan_sha256
    assert loaded.candidates[0].candidate_id == "c_interface"


def test_tampering_fails_before_a_plan_can_be_consumed(tmp_path) -> None:
    path = tmp_path / "plan.json"
    plan = _plan()
    plan.to_json(path)
    payload = json.loads(path.read_text())
    payload["candidates"][0]["symbol"] = "ETHUSDT"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="digest mismatch"):
        MechanismExperimentPlan.from_json(path)


def test_interface_release_rejects_outcome_execution() -> None:
    assert MECHANISM_PLAN_EXECUTION_SUPPORTED is False
    with pytest.raises(RuntimeError, match="interface-only"):
        run_pipeline(object(), mechanism_plan=_plan())
