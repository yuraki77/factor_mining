"""Evolutionary operations — CROSSOVER_DSL_COMPOSITE and freeze-point extraction.

Phase 3: deterministic crossover of DSL-bearing candidates and freeze-point
contracts for LLM mutation operators.
"""

from __future__ import annotations

from typing import Any

import uuid

from factor_mining.config import Settings
from factor_mining.dsl import canonicalize, structural_fingerprint, parse
from factor_mining.dsl.catalog import OPERATORS
from factor_mining.dsl.complexity import categorical_comparison_count, free_constant_count
from factor_mining.dsl.fingerprint_store import FingerprintStore
from factor_mining.dsl.renderer import render
from factor_mining.models import CandidateStrategySpec

# ── freeze-point contract ──────────────────────────────────────────


def extract_freeze_point(
    parent: CandidateStrategySpec,
    *,
    freeze_depth: int = 3,
) -> dict[str, Any]:
    """Extract freeze-point metadata from a parent candidate.

    The freeze point is the immutable prefix that future LLM mutations
    must preserve.  For non-DSL candidates this captures the hypothesis
    and structural params; for DSL candidates it captures the first
    *freeze_depth* AST nodes.
    """
    fp: dict[str, Any] = {
        "parent_candidate_id": parent.candidate_id,
        "hypothesis_id": parent.hypothesis_id,
        "hypothesis_family": parent.hypothesis_family,
        "method_id": parent.method_id,
        "symbol": parent.symbol,
        "freeze_depth": freeze_depth,
        "params_prefix": _fixed_params(parent.params),
    }

    if parent.dsl_expression and parent.dsl_ast:
        parent_ast = canonicalize(parent.dsl_ast)
        fp["dsl_expression"] = parent.dsl_expression
        fp["dsl_ast_prefix"] = _slice_ast_prefix(parent_ast, freeze_depth)
        fp["dsl_fingerprint"] = parent.dsl_fingerprint
        fp["dsl_version"] = parent.dsl_version
    for key in ("economic_mechanism", "testable_prediction", "optimizer_reason"):
        value = parent.params.get(key)
        if value:
            fp[key] = value

    # Carry forward the parent's freeze depth lineage
    parent_depth = parent.params.get("freeze_depth")
    if isinstance(parent_depth, int) and parent_depth > 0:
        fp["lineage_freeze_depth"] = parent_depth

    return fp


def _fixed_params(params: dict[str, Any]) -> dict[str, Any]:
    """Extract the subset of params that mutations should preserve."""
    fixed_keys = {
        "signal_source", "indicator_name", "indicator_family",
        "factor_family", "direction", "transform", "side_mode",
        "signal_role", "hypothesis_family",
    }
    return {k: v for k, v in params.items() if k in fixed_keys}


def _slice_ast_prefix(ast: dict[str, Any], depth: int) -> dict[str, Any] | None:
    """Return the first *depth* levels of the AST as a prefix tree.

    If *depth* is 0, returns None (no freeze).  If the AST is shallower
    than *depth*, returns the full AST.
    """
    if depth <= 0 or ast is None:
        return None
    if depth == 1:
        # Only freeze the root node type/op
        t = ast.get("type", "")
        if t == "factor":
            return {"type": "factor", "value": ast.get("value")}
        if t == "constant":
            return {"type": "constant", "value": ast.get("value")}
        if t == "func_call":
            return {"type": "func_call", "name": ast.get("name"), "args": []}
        if t == "binary_op":
            return {"type": "binary_op", "op": ast.get("op"), "left": None, "right": None}
        return {"type": t}

    # Recursively freeze children at depth-1
    result = dict(ast)
    if "operand" in result:
        result["operand"] = _slice_ast_prefix(result["operand"], depth - 1)
    if "left" in result:
        result["left"] = _slice_ast_prefix(result["left"], depth - 1)
    if "right" in result:
        result["right"] = _slice_ast_prefix(result["right"], depth - 1)
    if "args" in result:
        result["args"] = [
            _slice_ast_prefix(a, depth - 1) for a in result["args"]
        ]
    return result


# ── CROSSOVER_DSL_COMPOSITE ────────────────────────────────────────


def crossover_dsl_composite(
    parent_a: CandidateStrategySpec,
    parent_b: CandidateStrategySpec,
    settings: Settings,
    *,
    fingerprint_store: FingerprintStore | None = None,
    max_children: int | None = None,
) -> list[CandidateStrategySpec]:
    """Create composite candidates via deterministic DSL crossover.

    When both parents have DSL expressions:
      1. Parse both DSL -> AST
      2. Create children: A + B (weighted blend) and B + A
      3. Canonicalize, fingerprint, assign CROSSOVER operator

    When only one parent has DSL, creates a composite from that parent
    with the traditional composite path.

    Always returns at most 2 children.
    """
    # Check DSL compatibility
    a_has_dsl = parent_a.dsl_expression is not None and parent_a.dsl_fingerprint is not None
    b_has_dsl = parent_b.dsl_expression is not None and parent_b.dsl_fingerprint is not None

    if not a_has_dsl and not b_has_dsl:
        return []

    children: list[CandidateStrategySpec] = []
    fstore = fingerprint_store or FingerprintStore()
    parent_fingerprints = {
        fp for fp in (parent_a.dsl_fingerprint, parent_b.dsl_fingerprint)
        if isinstance(fp, str) and fp
    }
    for fp in parent_fingerprints:
        fstore.register_parent(fp)

    def add_child(ast: dict[str, Any], label: str) -> None:
        if max_children is not None and len(children) >= max_children:
            return
        canon = canonicalize(ast)
        fp = structural_fingerprint(canon)
        if not _passes_evolutionary_gates(canon, fp, settings, parent_fingerprints, fstore):
            return
        child = _spawn_crossover_child(parent_a, parent_b, render(canon), canon, fp, label, settings)
        children.append(child)
        fstore.register(fp, child.candidate_id)

    if a_has_dsl and b_has_dsl:
        # DSL-DSL crossover
        try:
            ast_a = canonicalize(parse(parent_a.dsl_expression))
            ast_b = canonicalize(parse(parent_b.dsl_expression))
        except Exception:
            return []

        # Create addition crossover: A + B
        add_ast: dict[str, Any] = {
            "type": "binary_op", "op": "+",
            "left": ast_a, "right": ast_b,
        }
        add_child(
            add_ast,
            f"crossover_add_{parent_a.candidate_id[:8]}_{parent_b.candidate_id[:8]}",
        )

        if max_children is None or len(children) < max_children:
            # Subtraction crossover: A - B
            sub_ast: dict[str, Any] = {
                "type": "binary_op", "op": "-",
                "left": ast_a, "right": ast_b,
            }
            add_child(
                sub_ast,
                f"crossover_sub_{parent_a.candidate_id[:8]}_{parent_b.candidate_id[:8]}",
            )

    elif a_has_dsl:
        # Only parent A has DSL — create traditional-style composite
        add_child(
            parent_a.dsl_ast or parse(parent_a.dsl_expression or ""),
            f"crossover_single_{parent_a.candidate_id[:8]}",
        )

    return children


def _passes_evolutionary_gates(
    ast: dict[str, Any],
    fingerprint: str,
    settings: Settings,
    parent_fingerprints: set[str],
    fingerprint_store: FingerprintStore,
) -> bool:
    evo = settings.evolutionary
    if fingerprint in parent_fingerprints:
        return False
    if _contains_cross_sectional_operator(ast):
        return False
    if free_constant_count(ast) > evo.max_constants_per_expression:
        return False
    if categorical_comparison_count(ast) > evo.max_categorical_comparisons:
        return False
    from factor_mining.dsl import complexity_score

    if complexity_score(ast) > evo.max_complexity:
        return False
    return fingerprint_store.is_novel(fingerprint)


def _contains_cross_sectional_operator(ast: dict[str, Any]) -> bool:
    if ast["type"] == "func_call":
        if OPERATORS[ast["name"]].cross_sectional:
            return True
        return any(_contains_cross_sectional_operator(arg) for arg in ast["args"])
    if ast["type"] == "binary_op":
        return _contains_cross_sectional_operator(ast["left"]) or _contains_cross_sectional_operator(ast["right"])
    return False


def _spawn_crossover_child(
    parent_a: CandidateStrategySpec,
    parent_b: CandidateStrategySpec,
    dsl_expr: str,
    dsl_ast: dict[str, Any] | None,
    dsl_fp: str,
    label: str,
    settings: Settings,
) -> CandidateStrategySpec:
    """Create a child candidate from crossover parents."""
    child = CandidateStrategySpec(
        candidate_id=f"c_xo_{uuid.uuid4().hex[:12]}",
        hypothesis_id=parent_a.hypothesis_id,
        method_id="factor_scoring",
        hypothesis_family=parent_a.hypothesis_family,
        symbol=parent_a.symbol,
        market=parent_a.market,
        interval=parent_a.interval,
        candidate_type="optimizer",
        parent_candidate_id=parent_a.candidate_id,
        dsl_expression=dsl_expr,
        dsl_ast=dsl_ast,
        dsl_fingerprint=dsl_fp,
        dsl_version="0.1.0",
        params={
            "parent_ids": [parent_a.candidate_id, parent_b.candidate_id],
            "crossover_label": label,
            "generated_by": "crossover_dsl_composite",
            "search_variant": "crossover_dsl",
            "optimizer_proposal_kind": "crossover",
            "optimizer_root_parent_id": parent_a.candidate_id,
            "expected_ic_mid": 0.02,
            "smooth_span": 48,
            "signal_threshold": 0.25,
            "position_buffer": 0.20,
        },
    )
    # Canonical expression via render
    child.dsl_canonical_expression = dsl_expr
    return child
