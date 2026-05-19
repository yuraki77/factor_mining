from __future__ import annotations

import pandas as pd

from factor_mining.config import RegimeConfig


def label_btc_regime(frame: pd.DataFrame, config: RegimeConfig) -> pd.Series:
    close = pd.Series(frame["close"].to_numpy(dtype=float), index=frame.index)
    bars_60d = 60 * 24 * 12
    returns_60d = close.pct_change(periods=bars_60d)
    vol_60d = close.pct_change().rolling(bars_60d, min_periods=max(10, bars_60d // 10)).std()
    vol_rank = vol_60d.rank(pct=True)
    regime = pd.Series("sideways", index=frame.index)
    regime.loc[returns_60d > config.bull_threshold] = "bull"
    regime.loc[returns_60d < config.bear_threshold] = "bear"
    high_vol_mask = (vol_rank > config.high_vol_rank) & (regime == "sideways")
    regime.loc[high_vol_mask] = "high_vol"
    return regime.fillna("sideways")

