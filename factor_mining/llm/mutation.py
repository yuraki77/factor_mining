"""LLM-driven factor mutation — MUTATION_AT_MECHANISM and MUTATION_AT_DSL.

Phase 3: the LLM receives a freeze-point (hypothesis + mechanism + fixed params)
and regenerates only the unfrozen suffix — typically a DSL expression that
operationalises the mechanism.

Deterministic pipeline integration ensures:
- Parent and freeze point are determined before the LLM call
- LLM output is parsed, canonicalised, and fingerprinted
- Novelty is checked against archives, survivors, batch, and parent bank
- The child candidate receives full trajectory lineage
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from factor_mining.config import Settings
from factor_mining.dsl import (
    canonicalize,
    complexity_score,
    parse,
    structural_fingerprint,
    LEAVES,
    OPERATORS,
    WINDOWS,
)
from factor_mining.dsl.complexity import categorical_comparison_count, free_constant_count
from factor_mining.dsl.consistency import check_consistent
from factor_mining.dsl.fingerprint_store import FingerprintStore
from factor_mining.dsl.renderer import render
from factor_mining.models import CandidateStrategySpec
from factor_mining.optimizers.evolutionary_operations import _contains_cross_sectional_operator
from factor_mining.storage import MetadataStore


def mutate_with_mechanism(
    parent: CandidateStrategySpec,
    freeze_point: dict[str, Any],
    settings: Settings,
    *,
    store: MetadataStore | None = None,
    fingerprint_store: FingerprintStore | None = None,
) -> CandidateStrategySpec | None:
    """Generate one child via MUTATION_AT_MECHANISM.

    The freeze point captures the parent's hypothesis + mechanism.
    The LLM generates a DSL expression that operationalises the idea,
    then the system parses/canonicalises/fingerprints it.

    Returns None if the LLM output is invalid or not novel.
    """
    evo = settings.evolutionary
    prompt = _build_mechanism_mutation_prompt(freeze_point, settings)
    dsl_expr = _call_llm_for_dsl(prompt, settings)

    if dsl_expr is None:
        return None

    try:
        ast = parse(dsl_expr)
    except Exception:
        return None

    ok, _reason = check_consistent(
        ast,
        _first_str(freeze_point.get("mechanism_taxonomy"), parent.params.get("mechanism_taxonomy")),
        _first_list(freeze_point.get("required_data_families"), parent.params.get("required_data_families")),
    )
    if not ok:
        return None

    canon = canonicalize(ast)
    fp = structural_fingerprint(canon)
    parent_fp = parent.dsl_fingerprint

    if isinstance(parent_fp, str) and fp == parent_fp:
        return None
    if not _ast_prefix_matches(canon, freeze_point.get("dsl_ast_prefix")):
        return None

    # Complexity gate
    if complexity_score(canon) > evo.max_complexity:
        return None

    # Free constant cap
    if free_constant_count(canon) > evo.max_constants_per_expression:
        return None
    if categorical_comparison_count(canon) > evo.max_categorical_comparisons:
        return None
    if _contains_cross_sectional_operator(canon):
        return None

    # Novelty check
    fstore = fingerprint_store or FingerprintStore(store)
    if isinstance(parent_fp, str) and parent_fp:
        fstore.register_parent(parent_fp)
    if not fstore.is_novel(fp):
        return None

    child = CandidateStrategySpec(
        candidate_id=f"c_mut_{uuid.uuid4().hex[:12]}",
        hypothesis_id=parent.hypothesis_id,
        method_id="factor_scoring",
        hypothesis_family=parent.hypothesis_family,
        symbol=parent.symbol,
        market=parent.market,
        interval=parent.interval,
        candidate_type="optimizer",
        parent_candidate_id=parent.candidate_id,
        dsl_expression=dsl_expr.strip(),
        dsl_canonical_expression=None,  # set below
        dsl_ast=canon,
        dsl_fingerprint=fp,
        dsl_version="0.1.0",
        params={
            "parent_id": parent.candidate_id,
            "generated_by": "llm_mutation_at_mechanism",
            "search_variant": "llm_mechanism_mutation",
            "freeze_depth": evo.freeze_depth,
            "optimizer_proposal_kind": "mutation_at_mechanism",
            "optimizer_root_parent_id": parent.candidate_id,
            "optimizer_reason": "LLM-generated DSL from hypothesis mechanism",
            "expected_ic_mid": 0.02,
        },
    )
    child.dsl_canonical_expression = render(canon)
    fstore.register(fp, child.candidate_id)
    return child


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _first_list(*values: Any) -> list[str] | None:
    for value in values:
        if isinstance(value, list):
            return [str(item) for item in value]
    return None


# ── prompt building ────────────────────────────────────────────────


def _build_mechanism_mutation_prompt(
    freeze_point: dict[str, Any],
    settings: Settings,
) -> str:
    """Build a structured prompt for MUTATION_AT_MECHANISM."""
    hypothesis = freeze_point.get("hypothesis_family", "unknown")
    mechanism = freeze_point.get("economic_mechanism") or freeze_point.get("testable_prediction") or ""
    symbol = freeze_point.get("symbol", "BTCUSDT")
    fixed_params = freeze_point.get("params_prefix", {})
    freeze_depth = freeze_point.get("freeze_depth", 0)

    leaves = ", ".join(sorted(LEAVES))
    windows = ", ".join(str(w) for w in sorted(WINDOWS))
    ops = ", ".join(
        f"{name}({spec.arity})" for name, spec in sorted(OPERATORS.items())
        if not spec.cross_sectional
    )

    return (
        f"Generate a factor expression in a time-series DSL for crypto futures.\n\n"
        f"## Hypothesis\n{hypothesis}\n\n"
        f"## Mechanism\n{mechanism}\n\n"
        f"## Symbol\n{symbol} on Binance perpetual futures, 5m bars\n\n"
        f"## Frozen Parameters\n{json.dumps(fixed_params, default=str)}\n"
        f"Freeze depth: {freeze_depth}\n\n"
        f"## Available Leaves\n{leaves}\n\n"
        f"## Available Windows\n{windows}\n\n"
        f"## Available Operators (time-series only)\n{ops}\n\n"
        f"## Constraints\n"
        f"- Use ONLY the listed operator names with EXACT arities shown\n"
        f"- Window arguments MUST be from the allowed grid: {windows}\n"
        f"- Free constants (non-trivial numeric literals not in window position) max 3\n"
        f"- Factor leaves ALWAYS start with $\n"
        f"- No bare identifiers — every leaf is $something\n"
        f"- Comparison operators: GT (greater-than), EQ (equal), AND, OR, NOT\n"
        f"- Cross-sectional operators (RANK, ZSCORE, SCALE) are NOT available; "
        f"use time-series operators only\n\n"
        f"Return ONLY valid JSON with one key: "
        f'{{"dsl_expression": "<DSL expression string>"}}'
    )


def _call_llm_for_dsl(prompt: str, settings: Settings) -> str | None:
    """Call the LLM to generate a DSL expression string.

    Uses the configured DeepSeek provider.  Returns None on failure.
    """
    try:
        from factor_mining.llm.providers import provider_from_settings

        provider = provider_from_settings("deepseek", settings)
        if not provider.is_configured:
            return None

        response = provider.chat_json(
            model=settings.llm.deepseek.hypothesis_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative factor researcher. "
                        "Return ONLY valid DSL expressions. "
                        "Do NOT explain your reasoning. "
                        "Do NOT use markdown formatting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content")
        return _extract_dsl_expression(content)
    except Exception:
        return None


def _extract_dsl_expression(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, dict):
        value = content.get("dsl_expression") or content.get("expression") or content.get("dsl")
        return str(value).strip() if value else None
    text = str(content).strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict):
        value = payload.get("dsl_expression") or payload.get("expression") or payload.get("dsl")
        return str(value).strip() if value else None
    if isinstance(payload, str):
        return payload.strip()
    return None


def _ast_prefix_matches(ast: dict[str, Any], prefix: Any) -> bool:
    if prefix is None:
        return True
    if not isinstance(prefix, dict) or not isinstance(ast, dict):
        return ast == prefix
    for key, expected in prefix.items():
        if expected is None:
            continue
        if key == "args" and expected == []:
            continue
        actual = ast.get(key)
        if isinstance(expected, dict):
            if not _ast_prefix_matches(actual, expected):
                return False
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) < len(expected):
                return False
            for actual_child, expected_child in zip(actual, expected, strict=False):
                if not _ast_prefix_matches(actual_child, expected_child):
                    return False
        elif actual != expected:
            return False
    return True
