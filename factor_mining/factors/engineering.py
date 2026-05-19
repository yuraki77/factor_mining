"""
Feature engineering — 18 indicators + 9 derived features from OHLCV.

Adapted from back_lab/factors/engineering.py for factor_mining data format.
Input: DataFrame with [open, high, low, close, volume] columns.
Output: DataFrame of factor values, same index.
"""
import numpy as np
import pandas as pd


# ── Momentum ────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    result.name = f"rsi_{period}"
    return result


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    e_fast = close.ewm(span=fast, adjust=False).mean()
    e_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = e_fast - e_slow
    result = macd_line - macd_line.ewm(span=signal, adjust=False).mean()
    result.name = f"macd_hist_{fast}_{slow}_{signal}"
    return result


def _macd_line(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    result = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    result.name = f"macd_line_{fast}_{slow}"
    return result


def _stoch_k(df: pd.DataFrame, k: int = 14, smooth: int = 3) -> pd.Series:
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    fast_k = 100.0 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    result = fast_k.rolling(smooth).mean()
    result.name = f"stoch_k_{k}"
    return result


def _stoch_d(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.Series:
    result = _stoch_k(df, k).rolling(d).mean()
    result.name = f"stoch_d_{k}_{d}"
    return result


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    result = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    result.name = f"cci_{period}"
    return result


def _roc(close: pd.Series, period: int = 10) -> pd.Series:
    result = 100.0 * (close / close.shift(period) - 1.0)
    result.name = f"roc_{period}"
    return result


def _willr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    result = -100.0 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)
    result.name = f"willr_{period}"
    return result


# ── Trend ────────────────────────────────────────────────────────────────

def _sma(close: pd.Series, period: int) -> pd.Series:
    result = close.rolling(period).mean()
    result.name = f"sma_{period}"
    return result


def _ema(close: pd.Series, period: int) -> pd.Series:
    result = close.ewm(span=period, adjust=False).mean()
    result.name = f"ema_{period}"
    return result


def _price_sma_ratio(close: pd.Series, period: int) -> pd.Series:
    ma = _sma(close, period)
    result = (close / ma.replace(0, np.nan)) - 1.0
    result.name = f"price_sma_ratio_{period}"
    return result


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = _true_range(df)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan)
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    result = dx.ewm(alpha=1 / period, adjust=False).mean()
    result.name = f"adx_{period}"
    return result


def _aroon_up(df: pd.DataFrame, period: int = 25) -> pd.Series:
    high_roll = df["high"].rolling(period)
    bars_since_high = period - high_roll.apply(lambda x: x.argmax(), raw=True)
    result = 100.0 * bars_since_high / period
    result.name = f"aroon_up_{period}"
    return result


def _aroon_down(df: pd.DataFrame, period: int = 25) -> pd.Series:
    low_roll = df["low"].rolling(period)
    bars_since_low = period - low_roll.apply(lambda x: x.argmin(), raw=True)
    result = 100.0 * bars_since_low / period
    result.name = f"aroon_down_{period}"
    return result


# ── Volatility ───────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    result = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    result.name = "true_range"
    return result


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    result = _true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    result.name = f"atr_{period}"
    return result


def _bb_width(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    middle = _sma(close, period)
    rstd = close.rolling(period).std()
    result = (2.0 * std * rstd) / middle.replace(0, np.nan)
    result.name = f"bb_width_{period}"
    return result


def _bb_pct(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    middle = _sma(close, period)
    rstd = close.rolling(period).std()
    upper = middle + std * rstd
    lower = middle - std * rstd
    result = (close - lower) / (upper - lower).replace(0, np.nan)
    result.name = f"bb_pct_{period}"
    return result


def _hist_vol(close: pd.Series, period: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    result = log_ret.rolling(period).std() * np.sqrt(105120)
    result.name = f"hist_vol_{period}"
    return result


# ── Volume ───────────────────────────────────────────────────────────────

def _vol_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    result = volume / volume.rolling(period).mean().replace(0, np.nan)
    result.name = f"vol_ratio_{period}"
    return result


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    result = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    result.name = "obv"
    return result


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_mf = tp * df["volume"]
    tp_diff = tp.diff()
    pos_mf = raw_mf.where(tp_diff > 0, 0.0).rolling(period).sum()
    neg_mf = raw_mf.where(tp_diff < 0, 0.0).rolling(period).sum()
    mf_ratio = pos_mf / neg_mf.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + mf_ratio))
    result.name = f"mfi_{period}"
    return result


def _cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    result = (clv * df["volume"]).rolling(period).sum() / df["volume"].rolling(period).sum()
    result.name = f"cmf_{period}"
    return result


# ── Kline microstructure ────────────────────────────────────────────────

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _rolling_zscore(series: pd.Series, period: int) -> pd.Series:
    mean = series.rolling(period, min_periods=max(12, period // 6)).mean()
    std = series.rolling(period, min_periods=max(12, period // 6)).std().replace(0, np.nan)
    return (series - mean) / std


def _taker_buy_pressure(df: pd.DataFrame) -> pd.Series:
    result = _safe_divide(df["taker_buy_volume"], df["volume"]).clip(0.0, 1.0)
    result.name = "taker_buy_pressure"
    return result


def _taker_quote_pressure(df: pd.DataFrame) -> pd.Series:
    result = _safe_divide(df["taker_buy_quote_volume"], df["quote_volume"]).clip(0.0, 1.0)
    result.name = "taker_quote_pressure"
    return result


def _order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    result = (2.0 * _taker_buy_pressure(df) - 1.0).clip(-1.0, 1.0)
    result.name = "order_flow_imbalance"
    return result


def _order_flow_imbalance_z(df: pd.DataFrame, period: int) -> pd.Series:
    result = _rolling_zscore(_order_flow_imbalance(df), period)
    result.name = f"order_flow_imbalance_z_{period}"
    return result


def _aggressive_buy_volume_z(df: pd.DataFrame, period: int) -> pd.Series:
    result = _rolling_zscore(df["taker_buy_quote_volume"], period)
    result.name = f"aggressive_buy_volume_z_{period}"
    return result


def _quote_volume_z(df: pd.DataFrame, period: int) -> pd.Series:
    result = _rolling_zscore(df["quote_volume"], period)
    result.name = f"quote_volume_z_{period}"
    return result


def _quote_volume_chg(df: pd.DataFrame, period: int) -> pd.Series:
    result = df["quote_volume"].pct_change(period)
    result.name = f"quote_volume_chg_{period}"
    return result


def _trade_density_z(df: pd.DataFrame, period: int) -> pd.Series:
    density = _safe_divide(df["trade_count"], df["quote_volume"]).replace([np.inf, -np.inf], np.nan)
    result = _rolling_zscore(density, period)
    result.name = f"trade_density_z_{period}"
    return result


def _has_kline_microstructure(df: pd.DataFrame) -> bool:
    return {
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    }.issubset(df.columns)


# ── Derived ──────────────────────────────────────────────────────────────

def _log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    result = np.log(close / close.shift(period))
    result.name = f"log_ret_{period}"
    return result


def _close_position(df: pd.DataFrame) -> pd.Series:
    result = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    result.name = "close_position"
    return result


def _hl_ratio(df: pd.DataFrame) -> pd.Series:
    result = df["high"] / df["low"].replace(0, np.nan)
    result.name = "hl_ratio"
    return result


def _gap(df: pd.DataFrame) -> pd.Series:
    result = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan)
    result.name = "gap"
    return result


def _intraday_range(df: pd.DataFrame) -> pd.Series:
    result = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    result.name = "intraday_range"
    return result


def _vol_trend_corr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    result = df["close"].rolling(period).corr(df["volume"])
    result.name = f"vol_trend_corr_{period}"
    return result


def _eff_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    result = direction / volatility.replace(0, np.nan)
    result.name = f"eff_ratio_{period}"
    return result


# ── Metadata ─────────────────────────────────────────────────────────────

INDICATOR_META: dict[str, dict] = {
    "rsi":           {"family": "mean_reversion",    "direction": "negative_when_high", "regime": "range_bound"},
    "stoch_k":       {"family": "mean_reversion",    "direction": "negative_when_high", "regime": "range_bound"},
    "stoch_d":       {"family": "mean_reversion",    "direction": "negative_when_high", "regime": "range_bound"},
    "cci":           {"family": "mean_reversion",    "direction": "negative_when_high", "regime": "range_bound"},
    "willr":         {"family": "mean_reversion",    "direction": "positive",           "regime": "range_bound"},
    "roc":           {"family": "trend_following",   "direction": "positive",           "regime": "trending"},
    "sma":           {"family": "trend_following",   "direction": "positive",           "regime": "trending"},
    "ema":           {"family": "trend_following",   "direction": "positive",           "regime": "trending"},
    "price_sma_ratio":{"family":"trend_following",   "direction": "positive",           "regime": "trending"},
    "macd_hist":     {"family": "trend_following",   "direction": "positive",           "regime": "trending"},
    "macd_line":     {"family": "trend_following",   "direction": "positive",           "regime": "trending"},
    "adx":           {"family": "trend_following",   "direction": "positive",           "regime": "trending"},
    "aroon_up":      {"family": "trend_following",   "direction": "positive",           "regime": "trending"},
    "aroon_down":    {"family": "trend_following",   "direction": "negative",           "regime": "trending"},
    "atr":           {"family": "volatility_regime", "direction": "neutral",            "regime": "any"},
    "true_range":    {"family": "volatility_regime", "direction": "neutral",            "regime": "any"},
    "bb_width":      {"family": "volatility_regime", "direction": "neutral",            "regime": "any"},
    "bb_pct":        {"family": "mean_reversion",    "direction": "negative_when_high", "regime": "range_bound"},
    "hist_vol":      {"family": "volatility_regime", "direction": "neutral",            "regime": "any"},
    "obv":           {"family": "volume_confirmation","direction":"positive",           "regime": "any"},
    "mfi":           {"family": "volume_confirmation","direction":"negative_when_high", "regime": "range_bound"},
    "cmf":           {"family": "volume_confirmation","direction":"positive",           "regime": "any"},
    "vol_ratio":     {"family": "volume_confirmation","direction":"positive",           "regime": "any"},
    "vol_trend_corr":{"family": "volume_confirmation","direction":"positive",           "regime": "trending"},
    "taker_buy_pressure":{"family":"volume_confirmation","direction":"positive",         "regime": "any"},
    "taker_quote_pressure":{"family":"volume_confirmation","direction":"positive",       "regime": "any"},
    "order_flow_imbalance":{"family":"volume_confirmation","direction":"positive",       "regime": "any"},
    "aggressive_buy_volume_z":{"family":"volume_confirmation","direction":"positive",    "regime": "any"},
    "quote_volume_z":{"family": "volume_confirmation","direction":"positive",           "regime": "any"},
    "quote_volume_chg":{"family": "volume_confirmation","direction":"positive",         "regime": "any"},
    "trade_density_z":{"family": "volume_confirmation","direction":"positive",          "regime": "any"},
    "log_ret":       {"family": "trend_following",   "direction": "positive",           "regime": "trending"},
    "hl_ratio":      {"family": "volatility_regime", "direction": "neutral",            "regime": "any"},
    "close_position":{"family": "mean_reversion",    "direction": "positive",           "regime": "range_bound"},
    "gap":           {"family": "mean_reversion",    "direction": "negative",           "regime": "range_bound"},
    "intraday_range":{"family": "volatility_regime", "direction": "neutral",            "regime": "any"},
    "eff_ratio":     {"family": "trend_following",   "direction": "positive",           "regime": "trending"},
}

DEFAULT_PARAMS = {
    "rsi_periods": [7, 14, 21],
    "stoch_k_periods": [14, 21], "stoch_d_period": 3,
    "cci_periods": [14, 20],
    "roc_periods": [5, 10, 20],
    "willr_periods": [14, 28],
    "macd_configs": [(12, 26, 9), (5, 35, 5)],
    "sma_periods": [10, 20, 50, 100, 200],
    "ema_periods": [12, 26, 50, 100],
    "price_sma_periods": [20, 50, 100],
    "adx_period": 14, "aroon_period": 25,
    "atr_periods": [14, 20],
    "bb_periods": [20, 50], "bb_std": 2.0,
    "hist_vol_periods": [10, 20, 50],
    "vol_ratio_periods": [10, 20],
    "mfi_periods": [14, 21], "cmf_periods": [20, 50],
    "microstructure_z_periods": [48, 288],
    "quote_volume_chg_periods": [12, 48],
    "eff_ratio_periods": [10, 20],
}


def _base_name(col: str) -> str:
    for prefix in ["bb_width", "bb_pct", "price_sma", "macd_hist", "macd_line",
                    "stoch_k", "stoch_d", "aroon_up", "aroon_down",
                    "true_range", "hist_vol", "vol_ratio", "vol_trend_corr",
                    "taker_buy_pressure", "taker_quote_pressure",
                    "order_flow_imbalance", "aggressive_buy_volume_z",
                    "quote_volume_z", "quote_volume_chg", "trade_density_z",
                    "log_ret", "hl_ratio", "close_position", "intraday_range", "eff_ratio"]:
        if col.startswith(prefix):
            return prefix
    import re
    parts = re.split(r"[_\d]", col)
    return parts[0] if parts else col


# ── Orchestrator ──────────────────────────────────────────────────────────

def generate_features(df: pd.DataFrame, include: list[str] | None = None,
                      params: dict | None = None) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Generate all indicators from OHLCV DataFrame.

    Returns (features_df, meta) where meta maps column name → metadata dict.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    close, high, low, open_, volume = df["close"], df["high"], df["low"], df["open"], df["volume"]
    features, meta = {}, {}

    def _add(series: pd.Series):
        if series.name in features:
            return
        base = _base_name(series.name)
        features[series.name] = series.values
        meta[series.name] = dict(INDICATOR_META.get(base, {}))
        meta[series.name].setdefault("family", "hybrid")
        meta[series.name].setdefault("direction", "neutral")
        meta[series.name].setdefault("regime", "any")

    def _ok(name: str) -> bool:
        return include is None or name in include

    if _ok("rsi"):
        for period in p["rsi_periods"]:
            _add(_rsi(close, period))
    if _ok("stoch"):
        for kp in p["stoch_k_periods"]:
            _add(_stoch_k(df, kp))
            _add(_stoch_d(df, kp, p["stoch_d_period"]))
    if _ok("cci"):
        for period in p["cci_periods"]:
            _add(_cci(df, period))
    if _ok("roc"):
        for period in p["roc_periods"]:
            _add(_roc(close, period))
    if _ok("willr"):
        for period in p["willr_periods"]:
            _add(_willr(df, period))
    if _ok("macd"):
        for fast, slow, sig in p["macd_configs"]:
            _add(_macd_hist(close, fast, slow, sig))
            _add(_macd_line(close, fast, slow))
    if _ok("sma"):
        for period in p["sma_periods"]:
            _add(_sma(close, period))
    if _ok("ema"):
        for period in p["ema_periods"]:
            _add(_ema(close, period))
    if _ok("price_sma_ratio"):
        for period in p["price_sma_periods"]:
            _add(_price_sma_ratio(close, period))
    if _ok("adx"):
        _add(_adx(df, p["adx_period"]))
    if _ok("aroon"):
        _add(_aroon_up(df, p["aroon_period"]))
        _add(_aroon_down(df, p["aroon_period"]))
    if _ok("atr"):
        for period in p["atr_periods"]:
            _add(_atr(df, period))
    if _ok("bbands"):
        for period in p["bb_periods"]:
            _add(_bb_width(close, period, p["bb_std"]))
            _add(_bb_pct(close, period, p["bb_std"]))
    if _ok("hist_vol"):
        for period in p["hist_vol_periods"]:
            _add(_hist_vol(close, period))
    if _ok("obv"):
        _add(_obv(close, volume))
    if _ok("mfi"):
        for period in p["mfi_periods"]:
            _add(_mfi(df, period))
    if _ok("cmf"):
        for period in p["cmf_periods"]:
            _add(_cmf(df, period))
    if _ok("vol_ratio"):
        for period in p["vol_ratio_periods"]:
            _add(_vol_ratio(volume, period))
    if _ok("vol_trend_corr"):
        _add(_vol_trend_corr(df, 20))
    if _ok("microstructure") and _has_kline_microstructure(df):
        _add(_taker_buy_pressure(df))
        _add(_taker_quote_pressure(df))
        _add(_order_flow_imbalance(df))
        for period in p["microstructure_z_periods"]:
            _add(_order_flow_imbalance_z(df, period))
            _add(_aggressive_buy_volume_z(df, period))
            _add(_quote_volume_z(df, period))
            _add(_trade_density_z(df, period))
        for period in p["quote_volume_chg_periods"]:
            _add(_quote_volume_chg(df, period))
    if _ok("log_ret"):
        for period in [1, 3, 12, 48, 288]:
            _add(_log_returns(close, period))
    if _ok("close_position"):
        _add(_close_position(df))
    if _ok("hl_ratio"):
        _add(_hl_ratio(df))
    if _ok("gap"):
        _add(_gap(df))
    if _ok("intraday_range"):
        _add(_intraday_range(df))
    if _ok("eff_ratio"):
        for period in p["eff_ratio_periods"]:
            _add(_eff_ratio(close, period))

    result = pd.DataFrame(features, index=df.index)
    return result, meta
