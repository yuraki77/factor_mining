"""Lightweight hypothesis-to-DSL consistency checks."""

from __future__ import annotations

from typing import Any


LEAF_TO_FAMILY = {
    "$open": "price",
    "$high": "price",
    "$low": "price",
    "$close": "price",
    "$returns": "price",
    "$vwap": "price",
    "$volume": "volume",
    "$dollar_volume": "volume",
    "$turnover": "volume",
    "$funding_rate": "funding",
    "$open_interest": "positioning",
    "$regime_state": "regime",
}


def check_consistent(
    ast: dict[str, Any],
    taxonomy: str | None,
    required_families: list[str] | None,
) -> tuple[bool, str]:
    """Return whether a DSL AST uses the data families required by a hypothesis."""
    del taxonomy
    if not required_families:
        return True, ""
    used_families = {
        LEAF_TO_FAMILY.get(leaf)
        for leaf in _extract_leaves(ast)
    }
    used_families.discard(None)
    missing = sorted({str(family) for family in required_families} - used_families)
    if missing:
        return False, f"DSL expression is missing required data family/families: {', '.join(missing)}"
    return True, ""


def _extract_leaves(ast: dict[str, Any]) -> set[str]:
    node_type = ast["type"]
    if node_type == "factor":
        return {str(ast["value"])}
    if node_type == "binary_op":
        return _extract_leaves(ast["left"]) | _extract_leaves(ast["right"])
    if node_type == "func_call":
        leaves: set[str] = set()
        for arg in ast["args"]:
            leaves |= _extract_leaves(arg)
        return leaves
    return set()
