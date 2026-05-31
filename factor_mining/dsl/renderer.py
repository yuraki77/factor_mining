"""Render a DSL AST back to canonical surface syntax."""

from __future__ import annotations

from typing import Any

_PREC = {"+": 1, "-": 1, "*": 2, "/": 2}


def render(ast: dict[str, Any], *, compact: bool = True) -> str:
    """Render AST to a parseable DSL expression."""
    del compact
    return _render(ast)


def _render(ast: dict[str, Any], *, parent_op: str | None = None, side: str = "") -> str:
    node_type = ast["type"]
    if node_type == "factor":
        return ast["value"]
    if node_type == "constant":
        value = ast["value"]
        if value == int(value):
            return str(int(value))
        return f"{value:.10g}"
    if node_type == "binary_op":
        op = ast["op"]
        left = _render(ast["left"], parent_op=op, side="left")
        right = _render(ast["right"], parent_op=op, side="right")
        text = f"{left} {op} {right}"
        if parent_op and _needs_parens(op, parent_op, side):
            return f"({text})"
        return text
    if node_type == "func_call":
        args = ", ".join(_render(arg) for arg in ast["args"])
        return f"{ast['name']}({args})"
    raise ValueError(f"unknown AST node type {node_type!r}")


def _needs_parens(child_op: str, parent_op: str, side: str) -> bool:
    child_prec = _PREC.get(child_op, 99)
    parent_prec = _PREC.get(parent_op, 99)
    if child_prec < parent_prec:
        return True
    return side == "right" and child_prec == parent_prec and parent_op in {"-", "/"}
