"""Complexity and inventory helpers for DSL ASTs."""

from __future__ import annotations

from typing import Any

from factor_mining.dsl.catalog import OPERATORS, TRIVIAL_CONSTANTS, WINDOWS


def complexity_score(ast: dict[str, Any]) -> int:
    """Simple weighted complexity score."""
    return (
        2 * sum(operator_count(ast).values())
        + max_depth(ast)
        + len(extract_features(ast))
        + len(extract_lookbacks(ast))
        + free_constant_count(ast)
    )


def max_depth(ast: dict[str, Any]) -> int:
    node_type = ast["type"]
    if node_type in ("factor", "constant"):
        return 0
    if node_type == "binary_op":
        return 1 + max(max_depth(ast["left"]), max_depth(ast["right"]))
    if node_type == "func_call":
        return 1 + max((max_depth(arg) for arg in ast["args"]), default=0)
    return 0


def operator_count(ast: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}

    def visit(node: dict[str, Any]) -> None:
        if node["type"] == "binary_op":
            counts[node["op"]] = counts.get(node["op"], 0) + 1
            visit(node["left"])
            visit(node["right"])
        elif node["type"] == "func_call":
            counts[node["name"]] = counts.get(node["name"], 0) + 1
            for arg in node["args"]:
                visit(arg)

    visit(ast)
    return counts


def extract_features(ast: dict[str, Any]) -> set[str]:
    features: set[str] = set()

    def visit(node: dict[str, Any]) -> None:
        if node["type"] == "factor":
            features.add(node["value"])
        elif node["type"] == "binary_op":
            visit(node["left"])
            visit(node["right"])
        elif node["type"] == "func_call":
            for arg in node["args"]:
                visit(arg)

    visit(ast)
    return features


def extract_lookbacks(ast: dict[str, Any]) -> set[int]:
    lookbacks: set[int] = set()

    def visit(node: dict[str, Any]) -> None:
        if node["type"] == "binary_op":
            visit(node["left"])
            visit(node["right"])
        elif node["type"] == "func_call":
            spec = OPERATORS[node["name"]]
            for i, arg in enumerate(node["args"]):
                if i in spec.window_args and arg["type"] == "constant":
                    value = int(arg["value"])
                    if value in WINDOWS:
                        lookbacks.add(value)
                visit(arg)

    visit(ast)
    return lookbacks


def free_constant_count(ast: dict[str, Any], *, known_factors: set[str] | None = None) -> int:
    """Count numeric literals that are neither windows nor trivial constants."""
    del known_factors
    return _free_constant_count(ast)


def _free_constant_count(ast: dict[str, Any], *, parent: dict[str, Any] | None = None, arg_index: int | None = None) -> int:
    if ast["type"] == "constant":
        value = ast["value"]
        if value in TRIVIAL_CONSTANTS:
            return 0
        if _is_window(value, parent, arg_index):
            return 0
        if _is_regime_category(value, parent):
            return 0
        return 1
    if ast["type"] == "binary_op":
        return _free_constant_count(ast["left"], parent=ast) + _free_constant_count(ast["right"], parent=ast)
    if ast["type"] == "func_call":
        return sum(_free_constant_count(arg, parent=ast, arg_index=i) for i, arg in enumerate(ast["args"]))
    return 0


def categorical_comparison_count(ast: dict[str, Any]) -> int:
    count = 0

    def visit(node: dict[str, Any]) -> None:
        nonlocal count
        if node["type"] == "func_call":
            if node["name"] == "EQ":
                count += 1
            for arg in node["args"]:
                visit(arg)
        elif node["type"] == "binary_op":
            visit(node["left"])
            visit(node["right"])

    visit(ast)
    return count


def _is_window(value: float, parent: dict[str, Any] | None, arg_index: int | None) -> bool:
    if parent is None or parent.get("type") != "func_call" or arg_index is None:
        return False
    return arg_index in OPERATORS[parent["name"]].window_args and value.is_integer() and int(value) in WINDOWS


def _is_regime_category(value: float, parent: dict[str, Any] | None) -> bool:
    if parent is None or parent.get("type") != "func_call" or parent["name"] != "EQ":
        return False
    return value.is_integer() and any(
        arg.get("type") == "factor" and arg.get("value") == "$regime_state"
        for arg in parent["args"]
    )
