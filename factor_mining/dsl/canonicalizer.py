"""Conservative canonicalization for DSL ASTs."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any

_COMMUTATIVE_BINARY = frozenset({"+", "*"})
_COMMUTATIVE_FUNCS = frozenset({"AND", "OR", "EQ"})
_IDEMPOTENT_FUNCS = frozenset({"RANK", "ZSCORE"})


def canonicalize(ast: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical copy of *ast*.

    Rewrites are intentionally conservative and avoid rules that change NaN
    behavior, such as ``x * 0 -> 0`` or ``x / x -> 1``.
    """
    current = deepcopy(ast)
    for _ in range(10):
        next_ast = _rewrite(current)
        if _stable_json(next_ast) == _stable_json(current):
            return next_ast
        current = next_ast
    raise RuntimeError("canonicalization did not converge within 10 iterations")


def structural_fingerprint(ast: dict[str, Any]) -> str:
    """Deterministic fingerprint for the canonical AST."""
    payload = _stable_json(canonicalize(ast))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _rewrite(ast: dict[str, Any]) -> dict[str, Any]:
    node_type = ast["type"]
    if node_type in ("factor", "constant"):
        return deepcopy(ast)
    if node_type == "binary_op":
        return _rewrite_binary(ast)
    if node_type == "func_call":
        return _rewrite_func(ast)
    raise ValueError(f"unknown AST node type {node_type!r}")


def _rewrite_binary(ast: dict[str, Any]) -> dict[str, Any]:
    op = ast["op"]
    left = _rewrite(ast["left"])
    right = _rewrite(ast["right"])

    if op == "-":
        return _rewrite({"type": "binary_op", "op": "+", "left": left, "right": _neg(right)})

    if _is_const(left) and _is_const(right):
        folded = _fold_binary_const(op, left["value"], right["value"])
        if folded is not None:
            return folded

    if op == "+":
        if _is_const_value(left, 0.0):
            return right
        if _is_const_value(right, 0.0):
            return left
    if op == "*":
        if _is_const_value(left, 1.0):
            return right
        if _is_const_value(right, 1.0):
            return left
    if op == "/" and _is_const_value(right, 1.0):
        return left

    if op in _COMMUTATIVE_BINARY and _sort_key(right) < _sort_key(left):
        left, right = right, left
    return {"type": "binary_op", "op": op, "left": left, "right": right}


def _rewrite_func(ast: dict[str, Any]) -> dict[str, Any]:
    name = ast["name"]
    args = [_rewrite(arg) for arg in ast["args"]]

    if name == "LT":
        return _rewrite({"type": "func_call", "name": "GT", "args": [args[1], args[0]]})

    if len(args) == 1:
        arg = args[0]
        if name == "NEG" and _is_func(arg, "NEG"):
            return arg["args"][0]
        if name == "NOT" and _is_func(arg, "NOT"):
            return arg["args"][0]
        if name == "ABS" and _is_func(arg, "ABS"):
            return arg
        if name == "ABS" and _is_func(arg, "NEG"):
            return {"type": "func_call", "name": "ABS", "args": [arg["args"][0]]}
        if name == "SIGN" and _is_func(arg, "SIGN"):
            return arg
        if name in _IDEMPOTENT_FUNCS and _is_func(arg, name):
            return arg
        if _is_const(arg):
            folded = _fold_unary_const(name, arg["value"])
            if folded is not None:
                return folded

    if name == "NOT" and len(args) == 1:
        arg = args[0]
        if _is_func(arg, "AND"):
            return {"type": "func_call", "name": "OR", "args": [_not(arg["args"][0]), _not(arg["args"][1])]}
        if _is_func(arg, "OR"):
            return {"type": "func_call", "name": "AND", "args": [_not(arg["args"][0]), _not(arg["args"][1])]}

    if name == "WHERE" and len(args) == 3:
        cond, true_branch, false_branch = args
        if _same_tree(true_branch, false_branch):
            return true_branch
        if _is_func(cond, "NOT"):
            return {"type": "func_call", "name": "WHERE", "args": [cond["args"][0], false_branch, true_branch]}
        if _is_const_value(cond, 1.0):
            return true_branch
        if _is_const_value(cond, 0.0):
            return false_branch

    if name == "TS_CORR" and len(args) == 3 and _sort_key(args[1]) < _sort_key(args[0]):
        args = [args[1], args[0], args[2]]
    elif name in _COMMUTATIVE_FUNCS and len(args) == 2 and _sort_key(args[1]) < _sort_key(args[0]):
        args = [args[1], args[0]]

    return {"type": "func_call", "name": name, "args": args}


def _fold_binary_const(op: str, left: float, right: float) -> dict[str, Any] | None:
    if op == "+":
        return {"type": "constant", "value": left + right}
    if op == "-":
        return {"type": "constant", "value": left - right}
    if op == "*":
        return {"type": "constant", "value": left * right}
    if op == "/" and right != 0.0:
        return {"type": "constant", "value": left / right}
    return None


def _fold_unary_const(name: str, value: float) -> dict[str, Any] | None:
    if name == "NEG":
        return {"type": "constant", "value": -value}
    if name == "ABS":
        return {"type": "constant", "value": abs(value)}
    if name == "SIGN":
        return {"type": "constant", "value": 1.0 if value > 0 else -1.0 if value < 0 else 0.0}
    if name == "NOT":
        return {"type": "constant", "value": 0.0 if value != 0.0 and math.isfinite(value) else 1.0}
    if name == "LOG" and value > 0.0:
        return {"type": "constant", "value": math.log(value)}
    if name == "SQRT" and value >= 0.0:
        return {"type": "constant", "value": math.sqrt(value)}
    return None


def _neg(ast: dict[str, Any]) -> dict[str, Any]:
    return {"type": "func_call", "name": "NEG", "args": [ast]}


def _not(ast: dict[str, Any]) -> dict[str, Any]:
    return {"type": "func_call", "name": "NOT", "args": [ast]}


def _is_const(ast: dict[str, Any]) -> bool:
    return ast["type"] == "constant"


def _is_const_value(ast: dict[str, Any], value: float) -> bool:
    return _is_const(ast) and ast["value"] == value


def _is_func(ast: dict[str, Any], name: str) -> bool:
    return ast["type"] == "func_call" and ast["name"] == name


def _same_tree(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _stable_json(left) == _stable_json(right)


def _sort_key(ast: dict[str, Any]) -> str:
    return hashlib.blake2b(_stable_json(ast).encode("utf-8"), digest_size=16).hexdigest()


def _stable_json(ast: dict[str, Any]) -> str:
    return json.dumps(ast, sort_keys=True, separators=(",", ":"), allow_nan=False)
