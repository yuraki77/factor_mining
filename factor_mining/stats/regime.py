from __future__ import annotations

import pandas as pd

from factor_mining.config import RegimeConfig


def label_btc_regime(frame: pd.DataFrame, config: RegimeConfig) -> pd.Series:
    close = pd.Series(frame["close"].to_numpy(dtype=float), index=frame.index)
    bars_60d = 60 * 24 * 12
    min_periods = max(10, bars_60d // 10)
    returns_60d = close.pct_change(periods=bars_60d)
    vol_60d = close.pct_change().rolling(bars_60d, min_periods=min_periods).std()
    # Trailing (point-in-time) percentile rank: each bar's 60-day vol is ranked
    # only against vol observed up to that bar. The previous full-sample
    # ``.rank(pct=True)`` ranked against the entire series — including future bars —
    # so the "high_vol" label at bar t leaked information from bars after t (Q6).
    vol_rank = vol_60d.expanding(min_periods=min_periods).rank(pct=True)
    regime = pd.Series("sideways", index=frame.index)
    regime.loc[returns_60d > config.bull_threshold] = "bull"
    regime.loc[returns_60d < config.bear_threshold] = "bear"
    high_vol_mask = (vol_rank > config.high_vol_rank) & (regime == "sideways")
    regime.loc[high_vol_mask] = "high_vol"
    return regime.fillna("sideways")

