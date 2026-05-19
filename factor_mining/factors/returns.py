"""Forward returns computation for factor evaluation."""
import numpy as np
import pandas as pd


def forward_returns(close: pd.Series, horizons: list[int], log: bool = True) -> pd.DataFrame:
    """Compute forward returns at multiple horizons.

    Args:
        close: Price series.
        horizons: List of bar offsets (1 = next bar).
        log: If True, log returns; else simple returns.

    Returns:
        DataFrame with columns fwd_{h} for each horizon.
    """
    prices = close.values
    result = {}
    for h in horizons:
        fwd = np.full(len(prices), np.nan)
        if log:
            fwd[:-h] = np.log(prices[h:] / prices[:-h])
        else:
            fwd[:-h] = (prices[h:] / prices[:-h]) - 1.0
        result[f"fwd_{h}"] = fwd
    return pd.DataFrame(result, index=close.index)
