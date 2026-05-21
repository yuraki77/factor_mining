"""Wide (time × symbol) panel construction for cross-sectional factor tests.

The single-asset engine consumes per-symbol DataFrames keyed by ``open_time``.
For cross-sectional tests we need an aligned (time × symbol) panel of factor
scores and forward returns. This module is the *only* place where per-symbol
frames get stacked — everything downstream consumes panels.

The point-in-time universe is encoded by NaN: a symbol that has not listed at
time t (or has been delisted) is NaN in both the factor and the returns panel.
Downstream code must therefore handle NaN as "not in universe" rather than
forward-filling or zero-filling.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorPanel:
    """Aligned cross-sectional factor + forward-return panel.

    Both ``factor`` and ``forward_returns`` are (time × symbol) DataFrames
    sharing the same row index (sorted ``open_time``) and column order.
    ``universe_mask`` is True where the symbol is tradeable at time t — the
    intersection of "factor is finite", "forward return is finite", and any
    external listing-date filter.
    """

    factor: pd.DataFrame
    forward_returns: pd.DataFrame
    universe_mask: pd.DataFrame

    @property
    def index(self) -> pd.Index:
        return self.factor.index

    @property
    def symbols(self) -> list[str]:
        return list(self.factor.columns)

    def universe_size(self) -> pd.Series:
        return self.universe_mask.sum(axis=1).astype(int)


def build_panel(
    frames_by_symbol: Mapping[str, pd.DataFrame],
    signals_by_symbol: Mapping[str, pd.Series],
    *,
    forward_horizon_bars: int = 1,
    listing_dates: Mapping[str, int] | None = None,
) -> FactorPanel:
    """Stack per-symbol frames into a cross-sectional panel.

    Args:
        frames_by_symbol: symbol → OHLCV frame (must contain ``open_time`` and
            ``open`` columns).  The single-asset engine's frames are the right
            shape.
        signals_by_symbol: symbol → factor score Series aligned to that
            symbol's frame (same length, same row order).  The signals are
            assumed to be the *factor score* at decision time t, before any
            ``shift(1)``; this function takes care of aligning the forward
            return to the factor.
        forward_horizon_bars: how many bars ahead to measure the return.  1 =
            next-bar open-to-open return (matches the single-asset engine's
            ``frame["open"].shift(-1) / frame["open"] - 1``).
        listing_dates: optional symbol → ``open_time`` ms threshold; bars with
            ``open_time < listing_dates[symbol]`` are excluded from the
            universe even if data exists.  Use this to avoid survivorship bias
            when historical data starts before the asset was actually listed
            on the venue.

    Returns:
        A ``FactorPanel`` whose ``factor[t, i]`` is observable at t and whose
        ``forward_returns[t, i]`` is the return realized between t and
        t+horizon (open-to-open).  ``universe_mask[t, i]`` is True iff both
        are finite and the listing date is satisfied.
    """
    if frames_by_symbol.keys() != signals_by_symbol.keys():
        missing = set(frames_by_symbol) ^ set(signals_by_symbol)
        raise ValueError(f"Frame and signal dicts must share keys; differ on: {missing}")
    if not frames_by_symbol:
        raise ValueError("At least one symbol required")
    horizon = int(forward_horizon_bars)
    if horizon < 1:
        raise ValueError("forward_horizon_bars must be >= 1")

    factor_cols: dict[str, pd.Series] = {}
    return_cols: dict[str, pd.Series] = {}
    for symbol, frame in frames_by_symbol.items():
        if "open_time" not in frame.columns or "open" not in frame.columns:
            raise ValueError(f"Frame for {symbol} missing open_time/open columns")
        ordered = frame.sort_values("open_time").reset_index(drop=True)
        signal = pd.Series(
            np.asarray(signals_by_symbol[symbol], dtype=float), index=ordered.index
        )
        if len(signal) != len(ordered):
            raise ValueError(f"Signal length mismatch for {symbol}")
        opens = ordered["open"].astype(float)
        fwd = opens.shift(-horizon) / opens - 1.0
        keyed_factor = pd.Series(signal.to_numpy(), index=ordered["open_time"].astype("int64"))
        keyed_fwd = pd.Series(fwd.to_numpy(), index=ordered["open_time"].astype("int64"))
        if listing_dates and symbol in listing_dates:
            cutoff = int(listing_dates[symbol])
            mask = keyed_factor.index >= cutoff
            keyed_factor = keyed_factor.where(mask)
            keyed_fwd = keyed_fwd.where(mask)
        keyed_factor = keyed_factor[~keyed_factor.index.duplicated(keep="last")]
        keyed_fwd = keyed_fwd[~keyed_fwd.index.duplicated(keep="last")]
        factor_cols[symbol] = keyed_factor
        return_cols[symbol] = keyed_fwd

    factor_df = pd.concat(factor_cols, axis=1).sort_index()
    returns_df = pd.concat(return_cols, axis=1).sort_index()
    factor_df, returns_df = factor_df.align(returns_df, join="outer")
    factor_df = factor_df[sorted(factor_df.columns)]
    returns_df = returns_df[factor_df.columns]
    universe = factor_df.notna() & returns_df.notna()
    return FactorPanel(factor=factor_df, forward_returns=returns_df, universe_mask=universe)
