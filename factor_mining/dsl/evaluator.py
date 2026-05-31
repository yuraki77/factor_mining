"""Per-symbol evaluator for the phase-2 factor DSL.

Cross-sectional operators are part of the DSL catalog but require a panel
adapter. This module evaluates single-symbol series expressions and fails loud
when a cross-sectional operator is used.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_mining.dsl.catalog import LEAVES, OPERATORS


def evaluate(
    ast: dict[str, Any],
    frame: pd.DataFrame,
    features_df: pd.DataFrame | None = None,
    feature_meta: dict[str, Any] | None = None,
) -> pd.Series:
    """Evaluate *ast* against one symbol's market-data frame."""
    del feature_meta
    index = frame.index
    features = features_df if features_df is not None else pd.DataFrame(index=index)
    result = _eval_node(ast, frame, features, index)
    return pd.Series(result, index=index, dtype=float)


def supported_operators() -> dict[str, dict[str, Any]]:
    """Return the phase-2 operator catalog with evaluator support flags."""
    return {
        name: {
            "arity": spec.arity,
            "cross_sectional": spec.cross_sectional,
            "supported_by_single_symbol_evaluator": not spec.cross_sectional,
        }
        for name, spec in OPERATORS.items()
    }


def _eval_node(ast: dict[str, Any], frame: pd.DataFrame, features: pd.DataFrame, index: pd.Index) -> np.ndarray:
    node_type = ast["type"]
    if node_type == "factor":
        return _lookup_leaf(ast["value"], frame, features, index)
    if node_type == "constant":
        return np.full(len(index), float(ast["value"]), dtype=float)
    if node_type == "binary_op":
        return _eval_binary(
            ast["op"],
            _eval_node(ast["left"], frame, features, index),
            _eval_node(ast["right"], frame, features, index),
        )
    if node_type == "func_call":
        name = ast["name"]
        args = [_eval_node(arg, frame, features, index) for arg in ast["args"]]
        return _eval_func(name, args)
    raise ValueError(f"unknown AST node type: {node_type!r}")


def _lookup_leaf(name: str, frame: pd.DataFrame, features: pd.DataFrame, index: pd.Index) -> np.ndarray:
    if name not in LEAVES:
        raise ValueError(f"unknown DSL leaf {name!r}")
    column_name = name[1:]
    if name == "$returns" and column_name not in frame.columns and column_name not in features.columns:
        close = _lookup_leaf("$close", frame, features, index)
        shifted = _delay(close, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.log(close / shifted)
        result[~np.isfinite(result)] = np.nan
        return result
    if column_name in frame.columns:
        return frame[column_name].reindex(index).to_numpy(dtype=float)
    if column_name in features.columns:
        return features[column_name].reindex(index).to_numpy(dtype=float)
    raise ValueError(f"required DSL leaf {name!r} is unavailable in market data")


def _eval_binary(op: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        with np.errstate(divide="ignore", invalid="ignore"):
            result = left / right
        result[~np.isfinite(result)] = np.nan
        return result
    raise ValueError(f"unknown binary operator: {op!r}")


def _eval_func(name: str, args: list[np.ndarray]) -> np.ndarray:
    if OPERATORS[name].cross_sectional:
        raise ValueError(f"cross-sectional operator {name}() requires a panel adapter")
    fn = _FUNC_DISPATCH.get(name)
    if fn is None:
        raise ValueError(f"unknown function: {name!r}")
    return fn(*args)


def _scalar(arg: np.ndarray, name: str) -> float:
    if arg.size == 0 or not np.all(arg == arg.flat[0]):
        raise ValueError(f"{name} argument must be a scalar constant")
    return float(arg.flat[0])


def _window(arg: np.ndarray, name: str = "window") -> int:
    value = _scalar(arg, name)
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _delta(x: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    n = _window(n_arg)
    result = np.full(len(x), np.nan, dtype=float)
    result[n:] = x[n:] - x[:-n]
    return result


def _delay(x: np.ndarray, n: int) -> np.ndarray:
    result = np.full(len(x), np.nan, dtype=float)
    result[n:] = x[:-n]
    return result


def _delay_func(x: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    return _delay(x, _window(n_arg))


def _rolling_unary(x: np.ndarray, n_arg: np.ndarray, fn: Callable[[pd.core.window.rolling.Rolling], pd.Series]) -> np.ndarray:
    n = _window(n_arg)
    return fn(pd.Series(x).rolling(n, min_periods=n)).to_numpy(dtype=float)


def _ts_mean(x: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    return _rolling_unary(x, n_arg, lambda rolling: rolling.mean())


def _ts_std(x: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    return _rolling_unary(x, n_arg, lambda rolling: rolling.std(ddof=1))


def _ts_max(x: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    return _rolling_unary(x, n_arg, lambda rolling: rolling.max())


def _ts_min(x: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    return _rolling_unary(x, n_arg, lambda rolling: rolling.min())


def _ts_rank(x: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    n = _window(n_arg)
    result = np.full(len(x), np.nan, dtype=float)
    for idx in range(n - 1, len(x)):
        window = x[idx - n + 1 : idx + 1]
        if np.isfinite(window).all():
            ordered = np.sort(window)
            result[idx] = np.searchsorted(ordered, window[-1], side="right") / n
    return result


def _ts_corr(x: np.ndarray, y: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    n = _window(n_arg)
    sx = pd.Series(x)
    sy = pd.Series(y)
    return sx.rolling(n, min_periods=n).corr(sy).to_numpy(dtype=float)


def _ts_zscore(x: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    mean = _ts_mean(x, n_arg)
    std = _ts_std(x, n_arg)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (x - mean) / std
    result[~np.isfinite(result)] = np.nan
    return result


def _ts_pctchange(x: np.ndarray, n_arg: np.ndarray) -> np.ndarray:
    shifted = _delay(x, _window(n_arg))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = x / shifted - 1.0
    result[~np.isfinite(result)] = np.nan
    return result


def _log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0.0, np.log(x), np.nan)


def _sqrt(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(x >= 0.0, np.sqrt(x), np.nan)


def _bool(x: np.ndarray) -> np.ndarray:
    return (x != 0.0) & np.isfinite(x)


def _gt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(x > y, 1.0, 0.0)


def _eq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(x == y, 1.0, 0.0)


def _and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(_bool(x) & _bool(y), 1.0, 0.0)


def _or(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(_bool(x) | _bool(y), 1.0, 0.0)


def _not(x: np.ndarray) -> np.ndarray:
    return np.where(_bool(x), 0.0, 1.0)


def _where(cond: np.ndarray, true_branch: np.ndarray, false_branch: np.ndarray) -> np.ndarray:
    return np.where(_bool(cond), true_branch, false_branch)


_FUNC_DISPATCH: dict[str, Callable[..., np.ndarray]] = {
    "DELTA": _delta,
    "DELAY": _delay_func,
    "TS_MEAN": _ts_mean,
    "TS_STD": _ts_std,
    "TS_MAX": _ts_max,
    "TS_MIN": _ts_min,
    "TS_RANK": _ts_rank,
    "TS_CORR": _ts_corr,
    "TS_ZSCORE": _ts_zscore,
    "TS_PCTCHANGE": _ts_pctchange,
    "ABS": np.abs,
    "SIGN": np.sign,
    "LOG": _log,
    "SQRT": _sqrt,
    "NEG": np.negative,
    "EQ": _eq,
    "GT": _gt,
    "LT": lambda x, y: _gt(y, x),
    "AND": _and,
    "OR": _or,
    "NOT": _not,
    "WHERE": _where,
}
