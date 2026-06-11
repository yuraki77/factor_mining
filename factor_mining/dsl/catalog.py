"""Catalog constants for the phase-2 factor DSL."""

from __future__ import annotations

from dataclasses import dataclass

DSL_VERSION = "0.1.0"

WINDOWS = frozenset({1, 3, 5, 10, 20, 60, 120, 250})
FREE_CONST_GRID = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
TRIVIAL_CONSTANTS = frozenset({-1.0, 0.0, 1.0})
MAX_FREE_CONSTANTS = 3

# Optional leaves stay in the syntax catalog for portability, but runtime data
# availability is still gated by the evaluator's market-data/features lookup.
OPTIONAL_LEAVES = frozenset({
    "$vwap",
    "$funding_rate",
    "$open_interest",
})

LEAVES = frozenset({
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$vwap",
    "$returns",
    "$funding_rate",
    "$open_interest",
    "$regime_state",
})


@dataclass(frozen=True)
class OperatorSpec:
    arity: int
    window_args: frozenset[int] = frozenset()
    cross_sectional: bool = False
    commutative_args: tuple[int, ...] = ()


OPERATORS: dict[str, OperatorSpec] = {
    "DELTA": OperatorSpec(2, frozenset({1})),
    "DELAY": OperatorSpec(2, frozenset({1})),
    "TS_MEAN": OperatorSpec(2, frozenset({1})),
    "TS_STD": OperatorSpec(2, frozenset({1})),
    "TS_MAX": OperatorSpec(2, frozenset({1})),
    "TS_MIN": OperatorSpec(2, frozenset({1})),
    "TS_RANK": OperatorSpec(2, frozenset({1})),
    "TS_CORR": OperatorSpec(3, frozenset({2}), commutative_args=(0, 1)),
    "TS_ZSCORE": OperatorSpec(2, frozenset({1})),
    "TS_PCTCHANGE": OperatorSpec(2, frozenset({1})),
    "RANK": OperatorSpec(1, cross_sectional=True),
    "ZSCORE": OperatorSpec(1, cross_sectional=True),
    "SCALE": OperatorSpec(1, cross_sectional=True),
    "ABS": OperatorSpec(1),
    "SIGN": OperatorSpec(1),
    "LOG": OperatorSpec(1),
    "SQRT": OperatorSpec(1),
    "NEG": OperatorSpec(1),
    "EQ": OperatorSpec(2, commutative_args=(0, 1)),
    "GT": OperatorSpec(2),
    "LT": OperatorSpec(2),
    "AND": OperatorSpec(2, commutative_args=(0, 1)),
    "OR": OperatorSpec(2, commutative_args=(0, 1)),
    "NOT": OperatorSpec(1),
    "WHERE": OperatorSpec(3),
}

ALIASES = {
    "SMA": "TS_MEAN",
    "CORRELATION": "TS_CORR",
}

# TODO: Add macro expansion for EMA, RSI, ATR, BBAND, and VWAP_DEV once the DSL
# supports multi-node aliases rather than simple operator renames.

BINARY_OPERATOR_ALIASES = {
    ">": "GT",
    "<": "LT",
    "==": "EQ",
    "and": "AND",
    "or": "OR",
}
