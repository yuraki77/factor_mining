"""Consumer contract for a separately authorized mechanism experiment plan.

This public module validates the frozen plan envelope that downstream systems
exchange with Factor Mining.  It deliberately does not compile or execute a
plan: those controls remain in the protected research lane.  Keeping the wire
contract here lets integrations reject tampering and API drift without
publishing research outcomes or silently treating an interface as permission
to run research.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from factor_mining.models import CandidateStrategySpec, HypothesisSpec


SCHEMA_VERSION = "fm.mechanism-experiment-plan/1"
PERMIT_SCHEMA_VERSION = "fm.mechanism-execution-permit/1"
_VARIANT_KINDS = frozenset(
    {
        "cadence",
        "composite",
        "grid_tuning",
        "monotonic_transform",
        "optimizer",
        "parameter_change",
        "regime_filter",
        "repair",
        "smoothing",
    }
)
_RESERVED_PARAMS = frozenset(
    {
        "causal_driver_id",
        "mechanism_id",
        "mechanism_revision_id",
        "proxy_id",
        "semantic_proxy_family_id",
        "mechanism_execution_permit_sha256",
        "mechanism_lineage_floor",
    }
)


class MechanismNovelty(str, Enum):
    EXISTING = "existing"
    REVISION = "revision"
    NEW = "new"


class MechanismContract(BaseModel):
    mechanism_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    market_participant: str = Field(min_length=1)
    binding_constraint: str = Field(min_length=1)
    causal_chain: str = Field(min_length=1)
    predicted_direction: str = Field(min_length=1)
    predicted_horizon: str = Field(min_length=1)
    stronger_where: str = Field(min_length=1)
    weaker_or_absent_where: str = Field(min_length=1)
    required_proxies: tuple[str, ...] = Field(min_length=1)
    falsification_conditions: tuple[str, ...] = Field(min_length=1)
    information_visible_at_generation: str = ""
    data_version: str = ""
    parent_mechanism_id: str | None = None


class MechanismExecutionPermit(BaseModel):
    schema_version: str = PERMIT_SCHEMA_VERSION
    permit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mechanism_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    proxy_ids: tuple[str, ...] = Field(min_length=1)
    inherited_exposure_ids: tuple[str, ...]
    inherited_lineage_count: int = Field(ge=0)
    inherited_window_count: int = Field(ge=0)

    @classmethod
    def freeze(cls, **payload: Any) -> "MechanismExecutionPermit":
        unsigned = {"schema_version": PERMIT_SCHEMA_VERSION, **payload}
        draft = cls.model_construct(permit_sha256="0" * 64, **unsigned)
        digest = _sha256_json(
            draft.model_dump(mode="json", exclude={"permit_sha256"})
        )
        return cls(permit_sha256=digest, **unsigned)

    @model_validator(mode="after")
    def validate_permit(self) -> "MechanismExecutionPermit":
        if self.schema_version != PERMIT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported mechanism execution permit: {self.schema_version}"
            )
        expected = _sha256_json(
            self.model_dump(mode="json", exclude={"permit_sha256"})
        )
        if self.permit_sha256 != expected:
            raise ValueError("mechanism execution permit digest mismatch")
        if len(set(self.proxy_ids)) != len(self.proxy_ids):
            raise ValueError("mechanism execution permit proxy IDs must be unique")
        return self


class SemanticProxyDescriptor(BaseModel):
    proxy_id: str = Field(min_length=1)
    mechanism_id: str = Field(min_length=1)
    causal_driver_id: str = Field(min_length=1)
    proxy_family_id: str = Field(min_length=1)
    mechanism_novelty: MechanismNovelty
    variant_kinds: tuple[str, ...] = ()
    parent_proxy_ids: tuple[str, ...] = ()
    component_proxy_ids: tuple[str, ...] = ()
    independence_basis: str | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> "SemanticProxyDescriptor":
        unknown = sorted(set(self.variant_kinds).difference(_VARIANT_KINDS))
        if unknown:
            raise ValueError("unknown semantic variant kinds: " + ", ".join(unknown))
        if self.mechanism_novelty == MechanismNovelty.NEW:
            if not self.independence_basis:
                raise ValueError("a new mechanism requires an independence basis")
            if self.variant_kinds or self.parent_proxy_ids:
                raise ValueError(
                    "an inherited variant cannot claim a new mechanism identity"
                )
        return self


class FrozenProxyTemplate(BaseModel):
    descriptor: SemanticProxyDescriptor
    hypothesis_family: str = Field(min_length=1)
    method_id: str = Field(min_length=1)
    symbols: tuple[str, ...] = Field(min_length=1)
    market: Literal["spot", "um_futures"] = "um_futures"
    interval: str = Field(min_length=1)
    params: dict[str, Any]
    max_feature_lookback_bars: int = Field(gt=0)
    expected_ic_range: tuple[float, float]
    expected_decay_halflife_bars: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_template(self) -> "FrozenProxyTemplate":
        if self.expected_ic_range[0] > self.expected_ic_range[1]:
            raise ValueError("expected IC range lower bound exceeds upper bound")
        collisions = sorted(_RESERVED_PARAMS.intersection(self.params))
        if collisions:
            raise ValueError(
                "proxy template cannot override compiler-owned params: "
                + ", ".join(collisions)
            )
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("proxy template symbols must be unique")
        return self


class MechanismExperimentPlan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision_id: str = Field(min_length=1)
    mechanism_id: str = Field(min_length=1)
    mechanism_contract: MechanismContract
    execution_permit: MechanismExecutionPermit
    signature_event_id: str = Field(min_length=1)
    proxy_templates: tuple[FrozenProxyTemplate, ...] = Field(min_length=1)
    hypotheses: tuple[HypothesisSpec, ...] = Field(min_length=1)
    candidates: tuple[CandidateStrategySpec, ...] = Field(min_length=1)

    @classmethod
    def freeze(cls, **payload: Any) -> "MechanismExperimentPlan":
        unsigned = {"schema_version": SCHEMA_VERSION, **payload}
        draft = cls.model_construct(plan_sha256="0" * 64, **unsigned)
        digest = _sha256_json(
            draft.model_dump(mode="json", exclude={"plan_sha256"})
        )
        return cls(plan_sha256=digest, **unsigned)

    @model_validator(mode="after")
    def validate_plan(self) -> "MechanismExperimentPlan":
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported mechanism plan: {self.schema_version}")
        expected = _sha256_json(
            self.model_dump(mode="json", exclude={"plan_sha256"})
        )
        if self.plan_sha256 != expected:
            raise ValueError("mechanism experiment plan digest mismatch")
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("compiled candidate IDs must be unique")
        if self.execution_permit.revision_id != self.revision_id:
            raise ValueError("plan revision differs from execution permit")
        if self.execution_permit.mechanism_id != self.mechanism_id:
            raise ValueError("plan mechanism differs from execution permit")
        if self.mechanism_contract.mechanism_id != self.mechanism_id:
            raise ValueError("plan mechanism differs from mechanism contract")
        template_proxy_ids = {
            item.descriptor.proxy_id for item in self.proxy_templates
        }
        if template_proxy_ids != set(self.execution_permit.proxy_ids):
            raise ValueError("plan proxy cohort differs from execution permit")
        hypothesis_ids = {item.hypothesis_id for item in self.hypotheses}
        for candidate in self.candidates:
            if candidate.hypothesis_id not in hypothesis_ids:
                raise ValueError("compiled candidate refers to an absent hypothesis")
            if (
                candidate.params.get("mechanism_id") != self.mechanism_id
                or candidate.params.get("mechanism_revision_id") != self.revision_id
            ):
                raise ValueError("compiled candidate changed mechanism identity")
            if candidate.params.get("proxy_id") not in template_proxy_ids:
                raise ValueError("compiled candidate uses an unregistered proxy")
            if (
                candidate.params.get("mechanism_execution_permit_sha256")
                != self.execution_permit.permit_sha256
            ):
                raise ValueError("compiled candidate changed execution permit")
            if (
                candidate.params.get("mechanism_lineage_floor")
                != self.execution_permit.inherited_lineage_count
            ):
                raise ValueError("compiled candidate changed mechanism lineage floor")
        return self

    @classmethod
    def from_json(cls, path: Path) -> "MechanismExperimentPlan":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
