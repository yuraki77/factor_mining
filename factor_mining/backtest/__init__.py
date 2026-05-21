"""Backtest engine."""

from factor_mining.backtest.cross_sectional import (
    CrossSectionalBacktestResult,
    run_cross_sectional_backtest,
    to_legacy_backtest_result,
)

__all__ = [
    "CrossSectionalBacktestResult",
    "run_cross_sectional_backtest",
    "to_legacy_backtest_result",
]
