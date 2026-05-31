"""Deterministic evolutionary budget allocator for Phase 3.

Allocates candidate slots across four operator categories (exploit, variation,
targeted_repair, seed) and selects parent candidates for each.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_BUDGET = {
    "exploit": 0.40,
    "variation": 0.30,
    "targeted_repair": 0.20,
    "seed": 0.10,
}


def allocate_evolutionary_budget(
    context: dict[str, Any],
    *,
    total_budget: int = 20,
    budget_weights: dict[str, float] | None = None,
) -> dict[str, int]:
    """Allocate candidate slots across evolutionary operator categories.

    Returns ``{category: int}`` summing to *total_budget*.

    If any category has insufficient available candidates the surplus
    redistributes to ``"exploit"``.
    """
    weights = dict(budget_weights or _DEFAULT_BUDGET)
    research_survivors = context.get("research_survivors") or []
    near_misses = context.get("near_misses") or []

    available: dict[str, int] = {
        "exploit": len(research_survivors),
        "variation": len(research_survivors),
        "targeted_repair": sum(1 for nm in near_misses if isinstance(nm, dict) and nm.get("actionable")),
        "seed": 999,  # always available (hypotheses are unbounded)
    }

    raw: dict[str, int] = {}
    surplus = 0
    for category, weight in sorted(weights.items()):
        target = max(0, int(round(total_budget * weight)))
        allocated = min(target, available.get(category, 0))
        raw[category] = allocated
        surplus += target - allocated

    # Redistribute surplus to exploit
    if surplus > 0 and available.get("exploit", 0) > raw.get("exploit", 0):
        raw["exploit"] = min(
            raw["exploit"] + surplus,
            available["exploit"],
            total_budget - sum(v for k, v in raw.items() if k != "exploit"),
        )

    # Ensure total sums to total_budget (or less if all categories exhausted)
    allocated = sum(raw.values())
    if allocated < total_budget and available.get("exploit", 0) > raw.get("exploit", 0):
        raw["exploit"] += min(total_budget - allocated, available["exploit"] - raw["exploit"])

    return raw


def select_evolutionary_parents(
    context: dict[str, Any],
    *,
    operator: str,
    count: int,
) -> list[dict[str, Any]]:
    """Select parent candidates for a given evolutionary operator.

    Selection rules (deterministic):
    - ``exploit``: top-N by research_score from research_survivors
    - ``variation``: survivors with least-used parameter neighbourhoods
    - ``targeted_repair``: actionable near-misses sorted by
      expected improvement
    - ``seed``: hypotheses with no existing candidate recorded in
      trajectory history
    """
    if operator == "exploit":
        survivors = list(context.get("research_survivors") or [])
        survivors.sort(key=_survivor_sort_key, reverse=True)
        return survivors[:count]

    if operator == "variation":
        survivors = list(context.get("research_survivors") or [])
        # Prefer survivors with lower complexity + higher score
        survivors.sort(key=lambda s: (
            _param_variation_score(s),
            -_survivor_sort_key(s),
        ))
        return survivors[:count]

    if operator == "targeted_repair":
        near_misses = list(context.get("near_misses") or [])
        actionable = [nm for nm in near_misses if isinstance(nm, dict) and nm.get("actionable")]
        return actionable[:count]

    if operator == "seed":
        # Return minimal hypothesis dicts for seed generation
        factors = context.get("factors") or []
        seen_families: set[str] = set()
        seeds: list[dict[str, Any]] = []
        for f in factors:
            family = str(f.get("hypothesis_family") or "")
            if family and family not in seen_families:
                seen_families.add(family)
                seeds.append({"hypothesis_family": family, "hypothesis_id": f.get("hypothesis_id", "")})
        return seeds[:count]

    return []


def _survivor_sort_key(survivor: dict[str, Any]) -> float:
    score = float(survivor.get("research_score") or 0.0)
    passed = 1.0 if survivor.get("gatecheck_passed") else 0.0
    return passed * 100.0 + score


def _param_variation_score(survivor: dict[str, Any]) -> float:
    """Lower score = less explored = preferred for variation."""
    params = survivor.get("params") or {}
    if not isinstance(params, dict):
        return 0.0
    # Count how many optimizer variant keys have been tried
    variant_key = str(params.get("optimizer_variant_key") or "")
    complexity = int(params.get("complexity_score") or 1)
    return float(len(variant_key) + complexity)
