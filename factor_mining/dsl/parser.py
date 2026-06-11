"""Recursive-descent parser for the phase-2 factor DSL."""

from __future__ import annotations

import math
import re
from typing import Any

from factor_mining.dsl.catalog import (
    ALIASES,
    BINARY_OPERATOR_ALIASES,
    FREE_CONST_GRID,
    LEAVES,
    MAX_FREE_CONSTANTS,
    OPERATORS,
    TRIVIAL_CONSTANTS,
    WINDOWS,
)

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<number> \d+ (?:\.\d+)? (?:[eE][+-]?\d+)? )
      | (?P<leaf>   \$[a-zA-Z_]\w* )
      | (?P<ident>  [a-zA-Z_]\w* )
      | (?P<op>     ==|[+\-*/()<>,] )
      | (?P<error>  \S+ )
    )
    """,
    re.VERBOSE,
)


_ALLOW_CROSS_SECTIONAL = False


def parse(source: str, *, allow_cross_sectional: bool = False) -> dict[str, Any]:
    """Parse a DSL expression into a validated AST.

    The phase-2 parser intentionally accepts only whitelisted ``$`` leaves,
    catalog operators, discrete window constants, and snapped free constants.
    """
    global _ALLOW_CROSS_SECTIONAL
    previous_allow_cross_sectional = _ALLOW_CROSS_SECTIONAL
    _ALLOW_CROSS_SECTIONAL = allow_cross_sectional
    try:
        tokens = _tokenize(source)
        if not tokens:
            raise ValueError("empty expression")
        ast, pos = _parse_expression(tokens, 0)
        if pos < len(tokens):
            _fail(tokens, pos, "unexpected trailing tokens")
        ast = _normalize_comparison_funcs(ast)
        _validate_tree(ast)
        ast = _snap_constants(ast)
        free_constants = _count_free_constants(ast)
        if free_constants > MAX_FREE_CONSTANTS:
            raise ValueError(f"too many free constants: {free_constants} > {MAX_FREE_CONSTANTS}")
        return ast
    finally:
        _ALLOW_CROSS_SECTIONAL = previous_allow_cross_sectional


def _tokenize(source: str) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for match in _TOKEN_RE.finditer(source):
        if match.lastgroup == "error":
            raise ValueError(f"unexpected character at position {match.start()}: {match.group('error')!r}")
        if match.lastgroup is None:
            continue
        tokens.append({
            "kind": match.lastgroup,
            "value": match.group(match.lastgroup),
            "pos": match.start(),
        })
    return tokens


def _parse_expression(tokens: list[dict[str, Any]], pos: int) -> tuple[dict[str, Any], int]:
    return _parse_logical_or(tokens, pos)


def _parse_logical_or(tokens: list[dict[str, Any]], pos: int) -> tuple[dict[str, Any], int]:
    left, pos = _parse_logical_and(tokens, pos)
    while _match_ident(tokens, pos, "or"):
        pos += 1
        right, pos = _parse_logical_and(tokens, pos)
        left = {"type": "func_call", "name": "OR", "args": [left, right]}
    return left, pos


def _parse_logical_and(tokens: list[dict[str, Any]], pos: int) -> tuple[dict[str, Any], int]:
    left, pos = _parse_comparison(tokens, pos)
    while _match_ident(tokens, pos, "and"):
        pos += 1
        right, pos = _parse_comparison(tokens, pos)
        left = {"type": "func_call", "name": "AND", "args": [left, right]}
    return left, pos


def _parse_comparison(tokens: list[dict[str, Any]], pos: int) -> tuple[dict[str, Any], int]:
    left, pos = _parse_additive(tokens, pos)
    if pos < len(tokens) and tokens[pos]["kind"] == "op" and tokens[pos]["value"] in (">", "<", "=="):
        op = BINARY_OPERATOR_ALIASES[tokens[pos]["value"]]
        pos += 1
        right, pos = _parse_additive(tokens, pos)
        left = {"type": "func_call", "name": op, "args": [left, right]}
    return left, pos


def _parse_additive(tokens: list[dict[str, Any]], pos: int) -> tuple[dict[str, Any], int]:
    left, pos = _parse_multiplicative(tokens, pos)
    while pos < len(tokens) and tokens[pos]["kind"] == "op" and tokens[pos]["value"] in ("+", "-"):
        op = tokens[pos]["value"]
        pos += 1
        right, pos = _parse_multiplicative(tokens, pos)
        left = {"type": "binary_op", "op": op, "left": left, "right": right}
    return left, pos


def _parse_multiplicative(tokens: list[dict[str, Any]], pos: int) -> tuple[dict[str, Any], int]:
    left, pos = _parse_unary(tokens, pos)
    while pos < len(tokens) and tokens[pos]["kind"] == "op" and tokens[pos]["value"] in ("*", "/"):
        op = tokens[pos]["value"]
        pos += 1
        right, pos = _parse_unary(tokens, pos)
        left = {"type": "binary_op", "op": op, "left": left, "right": right}
    return left, pos


def _parse_unary(tokens: list[dict[str, Any]], pos: int) -> tuple[dict[str, Any], int]:
    if pos < len(tokens) and tokens[pos]["kind"] == "op" and tokens[pos]["value"] in ("+", "-"):
        op = tokens[pos]["value"]
        pos += 1
        operand, pos = _parse_unary(tokens, pos)
        if op == "+":
            return operand, pos
        if operand["type"] == "constant":
            return {"type": "constant", "value": -operand["value"]}, pos
        return {"type": "func_call", "name": "NEG", "args": [operand]}, pos
    if _match_ident(tokens, pos, "not"):
        pos += 1
        operand, pos = _parse_unary(tokens, pos)
        return {"type": "func_call", "name": "NOT", "args": [operand]}, pos
    return _parse_atom(tokens, pos)


def _parse_atom(tokens: list[dict[str, Any]], pos: int) -> tuple[dict[str, Any], int]:
    if pos >= len(tokens):
        _fail(tokens, pos, "unexpected end of expression")
    token = tokens[pos]
    if token["kind"] == "number":
        value = float(token["value"])
        if not math.isfinite(value):
            _fail(tokens, pos, f"constant must be finite: {token['value']}")
        return {"type": "constant", "value": value}, pos + 1
    if token["kind"] == "leaf":
        leaf = token["value"].lower()
        if leaf not in LEAVES:
            _fail(tokens, pos, f"unknown leaf {token['value']!r}")
        return {"type": "factor", "value": leaf}, pos + 1
    if token["kind"] == "ident":
        name = token["value"]
        if pos + 1 < len(tokens) and tokens[pos + 1]["kind"] == "op" and tokens[pos + 1]["value"] == "(":
            return _parse_func_call(tokens, pos, name)
        if name.lower() in ("and", "or", "not"):
            _fail(tokens, pos, f"unexpected keyword {name!r}")
        _fail(tokens, pos, f"bare identifiers are not valid leaves: {name!r}; use a whitelisted $leaf")
    if token["kind"] == "op" and token["value"] == "(":
        ast, pos = _parse_expression(tokens, pos + 1)
        pos = _expect(tokens, pos, "op", ")")
        return ast, pos
    _fail(tokens, pos, f"unexpected token: {token['value']!r}")


def _parse_func_call(tokens: list[dict[str, Any]], pos: int, raw_name: str) -> tuple[dict[str, Any], int]:
    name = ALIASES.get(raw_name.upper(), raw_name.upper())
    if name not in OPERATORS:
        _fail(tokens, pos, f"unknown operator {raw_name!r}")
    if OPERATORS[name].cross_sectional and not _ALLOW_CROSS_SECTIONAL:
        _fail(tokens, pos, f"cross-sectional operator {name!r} is not available in single-symbol mode")
    pos += 1
    pos = _expect(tokens, pos, "op", "(")
    args: list[dict[str, Any]] = []
    if not (pos < len(tokens) and tokens[pos]["kind"] == "op" and tokens[pos]["value"] == ")"):
        arg, pos = _parse_expression(tokens, pos)
        args.append(arg)
        while pos < len(tokens) and tokens[pos]["kind"] == "op" and tokens[pos]["value"] == ",":
            arg, pos = _parse_expression(tokens, pos + 1)
            args.append(arg)
    pos = _expect(tokens, pos, "op", ")")
    spec = OPERATORS[name]
    if len(args) != spec.arity:
        _fail(tokens, pos - 1, f"{name}() takes exactly {spec.arity} argument(s), got {len(args)}")
    return {"type": "func_call", "name": name, "args": args}, pos


def _normalize_comparison_funcs(ast: dict[str, Any]) -> dict[str, Any]:
    if ast["type"] == "func_call":
        name = ast["name"]
        args = [_normalize_comparison_funcs(a) for a in ast["args"]]
        if name == "LT":
            return {"type": "func_call", "name": "GT", "args": [args[1], args[0]]}
        return {"type": "func_call", "name": name, "args": args}
    if ast["type"] == "binary_op":
        return {
            "type": "binary_op",
            "op": ast["op"],
            "left": _normalize_comparison_funcs(ast["left"]),
            "right": _normalize_comparison_funcs(ast["right"]),
        }
    return ast


def _validate_tree(ast: dict[str, Any], *, parent: dict[str, Any] | None = None, arg_index: int | None = None) -> None:
    node_type = ast["type"]
    if node_type == "factor":
        if ast["value"] not in LEAVES:
            raise ValueError(f"unknown leaf {ast['value']!r}")
        return
    if node_type == "constant":
        _validate_constant(ast["value"], parent=parent, arg_index=arg_index)
        return
    if node_type == "binary_op":
        if ast["op"] not in {"+", "-", "*", "/"}:
            raise ValueError(f"unsupported binary operator {ast['op']!r}")
        _validate_tree(ast["left"], parent=ast, arg_index=None)
        _validate_tree(ast["right"], parent=ast, arg_index=None)
        return
    if node_type == "func_call":
        name = ast["name"]
        spec = OPERATORS.get(name)
        if spec is None:
            raise ValueError(f"unknown operator {name!r}")
        if len(ast["args"]) != spec.arity:
            raise ValueError(f"{name}() takes exactly {spec.arity} argument(s)")
        for i, arg in enumerate(ast["args"]):
            _validate_tree(arg, parent=ast, arg_index=i)
        return
    raise ValueError(f"unknown AST node type {node_type!r}")


def _validate_constant(value: float, *, parent: dict[str, Any] | None, arg_index: int | None) -> None:
    if not math.isfinite(value):
        raise ValueError(f"constant must be finite: {value!r}")
    if _is_window_argument(parent=parent, arg_index=arg_index) and not _is_window_constant(value, parent=parent, arg_index=arg_index):
        raise ValueError(f"window constant {value:g} is not in the allowed window set")
    if _is_window_constant(value, parent=parent, arg_index=arg_index):
        return
    if _is_categorical_regime_constant(value, parent=parent):
        return
    if value in TRIVIAL_CONSTANTS:
        return
    nearest = min(FREE_CONST_GRID, key=lambda grid_value: abs(grid_value - abs(value)))
    tolerance = max(abs(nearest) * 0.05, 1e-12)
    if abs(abs(value) - nearest) > tolerance:
        raise ValueError(f"free constant {value:g} is not on the allowed grid")


def _is_window_constant(value: float, *, parent: dict[str, Any] | None, arg_index: int | None) -> bool:
    if not _is_window_argument(parent=parent, arg_index=arg_index):
        return False
    spec = OPERATORS[parent["name"]]
    return arg_index in spec.window_args and value.is_integer() and int(value) in WINDOWS


def _is_window_argument(*, parent: dict[str, Any] | None, arg_index: int | None) -> bool:
    if parent is None or parent.get("type") != "func_call" or arg_index is None:
        return False
    return arg_index in OPERATORS[parent["name"]].window_args


def _is_categorical_regime_constant(value: float, *, parent: dict[str, Any] | None) -> bool:
    if parent is None or parent.get("type") != "func_call" or parent["name"] != "EQ":
        return False
    if not value.is_integer():
        return False
    return any(arg.get("type") == "factor" and arg.get("value") == "$regime_state" for arg in parent["args"])


def _count_free_constants(ast: dict[str, Any], *, parent: dict[str, Any] | None = None, arg_index: int | None = None) -> int:
    if ast["type"] == "constant":
        value = ast["value"]
        if value in TRIVIAL_CONSTANTS:
            return 0
        if _is_window_constant(value, parent=parent, arg_index=arg_index):
            return 0
        if _is_categorical_regime_constant(value, parent=parent):
            return 0
        return 1
    if ast["type"] == "binary_op":
        return _count_free_constants(ast["left"], parent=ast) + _count_free_constants(ast["right"], parent=ast)
    if ast["type"] == "func_call":
        return sum(_count_free_constants(arg, parent=ast, arg_index=i) for i, arg in enumerate(ast["args"]))
    return 0


def _snap_constants(ast: dict[str, Any], *, parent: dict[str, Any] | None = None, arg_index: int | None = None) -> dict[str, Any]:
    if ast["type"] == "constant":
        value = ast["value"]
        if value in TRIVIAL_CONSTANTS or _is_window_constant(value, parent=parent, arg_index=arg_index) or _is_categorical_regime_constant(value, parent=parent):
            return dict(ast)
        nearest = min(FREE_CONST_GRID, key=lambda grid_value: abs(grid_value - abs(value)))
        snapped = math.copysign(nearest, value)
        return {"type": "constant", "value": snapped}
    if ast["type"] == "binary_op":
        return {
            "type": "binary_op",
            "op": ast["op"],
            "left": _snap_constants(ast["left"], parent=ast),
            "right": _snap_constants(ast["right"], parent=ast),
        }
    if ast["type"] == "func_call":
        return {
            "type": "func_call",
            "name": ast["name"],
            "args": [_snap_constants(arg, parent=ast, arg_index=i) for i, arg in enumerate(ast["args"])],
        }
    return dict(ast)


def _match_ident(tokens: list[dict[str, Any]], pos: int, value: str) -> bool:
    return pos < len(tokens) and tokens[pos]["kind"] == "ident" and tokens[pos]["value"].lower() == value


def _expect(tokens: list[dict[str, Any]], pos: int, kind: str, value: str) -> int:
    if not (pos < len(tokens) and tokens[pos]["kind"] == kind and tokens[pos]["value"] == value):
        got = tokens[pos]["value"] if pos < len(tokens) else "end of input"
        _fail(tokens, pos, f"expected {value!r}, got {got!r}")
    return pos + 1


def _fail(tokens: list[dict[str, Any]], pos: int, message: str) -> None:
    if pos < len(tokens):
        raise ValueError(f"parse error at position {tokens[pos]['pos']}: {message}")
    raise ValueError(f"parse error at end of input: {message}")
