from __future__ import annotations

import numpy as np
import pandas as pd

from factor_mining.backtest.engine import evaluate_strategy_path
from factor_mining.config import Settings
from factor_mining.models import BacktestResult, CandidateStrategySpec, FactorEvidenceReport, MetricsBlock
from factor_mining.stats.metrics import annualization_factor, max_drawdown, sharpe_ratio


DEFAULT_EVIDENCE_HORIZONS = (1, 3, 6, 12, 24, 48, 96)
_MAX_IC_BOOTSTRAP_RESAMPLES = 64
_CONFLICT_IC_THRESHOLD = 0.015


def build_factor_evidence_reports(
    *,
    frame: pd.DataFrame,
    tasks: list[tuple],
    candidates: list[CandidateStrategySpec],
    results: list[BacktestResult],
    settings: Settings,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None,
    funding_df: pd.DataFrame | None,
    horizons: tuple[int, ...] = DEFAULT_EVIDENCE_HORIZONS,
) -> list[FactorEvidenceReport]:
    """Build factor research evidence without changing production GateCheck behavior."""
    signal_by_candidate = {
        cdict["candidate_id"]: signal_arr
        for signal_arr, cdict, *_ in tasks
    }
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    reports: list[FactorEvidenceReport] = []
    for result in results:
        candidate = candidate_by_id.get(result.candidate_id)
        signal_arr = signal_by_candidate.get(result.candidate_id)
        if candidate is None or signal_arr is None:
            continue
        reports.append(
            build_factor_evidence_report(
                frame=frame,
                signal=pd.Series(signal_arr, index=frame.index),
                candidate=candidate,
                result=result,
                settings=settings,
                forward_regimes=forward_regimes,
                funding_rate=funding_rate,
                funding_df=funding_df,
                horizons=horizons,
            )
        )
    return reports


def build_factor_evidence_report(
    *,
    frame: pd.DataFrame,
    signal: pd.Series,
    candidate: CandidateStrategySpec,
    result: BacktestResult,
    settings: Settings,
    forward_regimes: pd.Series,
    funding_rate: pd.Series | None,
    funding_df: pd.DataFrame | None,
    horizons: tuple[int, ...] = DEFAULT_EVIDENCE_HORIZONS,
) -> FactorEvidenceReport:
    path = evaluate_strategy_path(frame, signal, candidate, settings, funding=funding_df)
    aligned_frame = path.frame
    executable_signal = path.signals.shift(1).fillna(0.0)

    ic_by_horizon: dict[str, float] = {}
    ic_ci_by_horizon: dict[str, tuple[float, float]] = {}
    rankic_by_horizon: dict[str, float] = {}
    quantile_spread_by_horizon: dict[str, float] = {}
    n_resamples = min(max(20, settings.bootstrap.n_resamples), _MAX_IC_BOOTSTRAP_RESAMPLES)
    for horizon in horizons:
        forward = _forward_open_return(aligned_frame, horizon)
        key = str(horizon)
        ic_by_horizon[key] = _pearson_ic(executable_signal, forward)
        ic_ci_by_horizon[key] = _bootstrap_ic_ci(
            executable_signal,
            forward,
            n_resamples=n_resamples,
            ci=settings.bootstrap.ci_levels,
            seed=42 + int(horizon),
        )
        rankic_by_horizon[key] = _rank_ic(executable_signal, forward)
        quantile_spread_by_horizon[key] = _quantile_spread_bps(executable_signal, forward)

    abs_ics = {key: abs(value) for key, value in ic_by_horizon.items()}
    max_abs_ic = max(abs_ics.values(), default=0.0)
    signal_decay_profile = {
        key: (value / max_abs_ic if max_abs_ic > 0 else 0.0)
        for key, value in abs_ics.items()
    }
    best_horizon = max(abs_ics, key=abs_ics.get) if abs_ics else None

    regimes = _aligned_labels(forward_regimes, aligned_frame.index, default="unknown")
    funding_states = _funding_state_labels(funding_rate, aligned_frame.index)
    funding_trends = _funding_trend_labels(funding_rate, aligned_frame.index)

    long_position = path.position.clip(lower=0.0)
    short_position = path.position.clip(upper=0.0)
    long_returns = _strategy_returns_for_position(aligned_frame, path.open_returns, long_position, settings, funding_df)
    short_returns = _strategy_returns_for_position(aligned_frame, path.open_returns, short_position, settings, funding_df)
    long_metrics = _metrics_from_returns(
        long_returns,
        interval=candidate.interval,
        trade_count=_trade_count(long_position),
    )
    short_metrics = _metrics_from_returns(
        short_returns,
        interval=candidate.interval,
        trade_count=_trade_count(short_position),
    )
    turnover_adjusted_return = result.metrics_primary.total_return / max(result.factor_turnover, 1e-6)
    decay_quality = _decay_quality(abs_ics)
    long_short_spread_sharpe = long_metrics.sharpe - short_metrics.sharpe
    conflict_reasons = _evidence_conflict_reasons(ic_by_horizon, {
        **_conditional_ic(executable_signal, aligned_frame, regimes, horizons),
        **_conditional_ic(executable_signal, aligned_frame, funding_states, horizons, prefix="state"),
        **_conditional_ic(executable_signal, aligned_frame, funding_trends, horizons, prefix="trend"),
    })
    evidence_flags = _evidence_flags(
        ic_ci_by_horizon=ic_ci_by_horizon,
        turnover_adjusted_return=turnover_adjusted_return,
        decay_quality=decay_quality,
        long_short_spread_sharpe=long_short_spread_sharpe,
        conflict_reasons=conflict_reasons,
    )

    return FactorEvidenceReport(
        experiment_id=result.experiment_id,
        candidate_id=result.candidate_id,
        hypothesis_family=result.hypothesis_family,
        method_id=result.method_id,
        symbol=result.symbol,
        market=result.market,
        interval=result.interval,
        horizons_bars=list(horizons),
        ic_by_horizon=ic_by_horizon,
        ic_ci_by_horizon=ic_ci_by_horizon,
        rankic_by_horizon=rankic_by_horizon,
        quantile_spread_by_horizon=quantile_spread_by_horizon,
        signal_decay_profile=signal_decay_profile,
        best_horizon_bars=int(best_horizon) if best_horizon is not None else None,
        regime_conditional_ic=_conditional_ic(executable_signal, aligned_frame, regimes, horizons),
        funding_conditional_ic={
            **_conditional_ic(executable_signal, aligned_frame, funding_states, horizons, prefix="state"),
            **_conditional_ic(executable_signal, aligned_frame, funding_trends, horizons, prefix="trend"),
        },
        long_only_metrics=long_metrics,
        short_only_metrics=short_metrics,
        turnover_adjusted_return=turnover_adjusted_return,
        decay_quality=decay_quality,
        long_short_spread_sharpe=long_short_spread_sharpe,
        regime_conflict=bool(conflict_reasons),
        evidence_score=sum(1.0 for value in evidence_flags.values() if value is True),
        evidence_flags=evidence_flags,
        conflict_reasons=conflict_reasons,
        gross_net_decomposition=_gross_net_decomposition(result),
    )


def _forward_open_return(frame: pd.DataFrame, horizon: int) -> pd.Series:
    open_price = pd.Series(frame["open"].to_numpy(dtype=float), index=frame.index)
    return (open_price.shift(-horizon) / open_price - 1.0).replace([np.inf, -np.inf], np.nan)


def _pearson_ic(signal: pd.Series, returns: pd.Series) -> float:
    sig, ret = _valid_pair(signal, returns)
    if len(sig) < 3 or sig.std(ddof=0) == 0 or ret.std(ddof=0) == 0:
        return 0.0
    return float(np.corrcoef(sig, ret)[0, 1])


def _rank_ic(signal: pd.Series, returns: pd.Series) -> float:
    sig, ret = _valid_pair(signal, returns)
    if len(sig) < 3 or sig.std(ddof=0) == 0 or ret.std(ddof=0) == 0:
        return 0.0
    corr = pd.Series(sig).rank().corr(pd.Series(ret).rank())
    return 0.0 if pd.isna(corr) else float(corr)


def _valid_pair(signal: pd.Series, returns: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"signal": signal, "returns": returns}).replace([np.inf, -np.inf], np.nan).dropna()
    return frame["signal"].to_numpy(dtype=float), frame["returns"].to_numpy(dtype=float)


def _bootstrap_ic_ci(
    signal: pd.Series,
    returns: pd.Series,
    *,
    n_resamples: int,
    ci: tuple[float, float],
    seed: int,
) -> tuple[float, float]:
    sig, ret = _valid_pair(signal, returns)
    n = len(sig)
    if n < 20 or sig.std(ddof=0) == 0 or ret.std(ddof=0) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    block_len = max(5, int(np.sqrt(n)))
    values: list[float] = []
    for _ in range(n_resamples):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            indices.extend((start + offset) % n for offset in range(block_len))
        idx = np.asarray(indices[:n], dtype=int)
        if sig[idx].std(ddof=0) == 0 or ret[idx].std(ddof=0) == 0:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(sig[idx], ret[idx])[0, 1]))
    return (float(np.quantile(values, ci[0])), float(np.quantile(values, ci[1])))


def _quantile_spread_bps(signal: pd.Series, returns: pd.Series, *, quantiles: int = 5) -> float:
    frame = pd.DataFrame({"signal": signal, "returns": returns}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < quantiles * 3 or frame["signal"].nunique() < 2:
        return 0.0
    try:
        buckets = pd.qcut(frame["signal"].rank(method="first"), quantiles, labels=False)
    except ValueError:
        return 0.0
    low = frame.loc[buckets == int(buckets.min()), "returns"]
    high = frame.loc[buckets == int(buckets.max()), "returns"]
    if low.empty or high.empty:
        return 0.0
    return float((high.mean() - low.mean()) * 10_000.0)


def _conditional_ic(
    signal: pd.Series,
    frame: pd.DataFrame,
    labels: pd.Series,
    horizons: tuple[int, ...],
    *,
    prefix: str | None = None,
    min_obs: int = 50,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    labels = pd.Series(labels.to_numpy(), index=frame.index).fillna("unknown")
    for label in sorted(str(item) for item in labels.dropna().unique()):
        mask = labels.astype(str) == label
        if int(mask.sum()) < min_obs:
            continue
        key = f"{prefix}:{label}" if prefix else label
        values: dict[str, float] = {}
        for horizon in horizons:
            forward = _forward_open_return(frame, horizon)
            values[str(horizon)] = _pearson_ic(signal.loc[mask], forward.loc[mask])
        out[key] = values
    return out


def _decay_quality(abs_ics: dict[str, float]) -> float:
    clean = [float(value) for value in abs_ics.values() if np.isfinite(float(value))]
    total = sum(clean)
    if total <= 0.0:
        return 0.0
    concentration = max(clean) / total
    return float(max(0.0, min(1.0, 1.0 - concentration)))


def _evidence_conflict_reasons(
    ic_by_horizon: dict[str, float],
    conditional_ic: dict[str, dict[str, float]],
) -> list[str]:
    reasons: list[str] = []
    horizon_values = [float(value) for value in ic_by_horizon.values() if abs(float(value)) >= _CONFLICT_IC_THRESHOLD]
    if horizon_values and min(horizon_values) < 0.0 < max(horizon_values):
        reasons.append("horizon_ic_sign_conflict")

    conditional_values = [
        float(value)
        for nested in conditional_ic.values()
        for value in nested.values()
        if abs(float(value)) >= _CONFLICT_IC_THRESHOLD
    ]
    if conditional_values and min(conditional_values) < 0.0 < max(conditional_values):
        reasons.append("conditional_ic_sign_conflict")
    return reasons


def _evidence_flags(
    *,
    ic_ci_by_horizon: dict[str, tuple[float, float]],
    turnover_adjusted_return: float,
    decay_quality: float,
    long_short_spread_sharpe: float,
    conflict_reasons: list[str],
) -> dict[str, float | int | str | bool | None]:
    ci_excludes_zero = any(
        (low > 0.0 or high < 0.0)
        for low, high in ic_ci_by_horizon.values()
    )
    return {
        "ic_ci_excludes_zero": ci_excludes_zero,
        "positive_turnover_adjusted_return": turnover_adjusted_return > 0.0,
        "decay_curve_supported": decay_quality >= 0.25,
        "long_short_spread": abs(long_short_spread_sharpe) >= 0.4,
        "no_conflict": not conflict_reasons,
        "conflict_count": len(conflict_reasons),
    }


def _aligned_labels(labels: pd.Series, index: pd.Index, *, default: str) -> pd.Series:
    if labels is None or len(labels) == 0:
        return pd.Series(default, index=index)
    return pd.Series(labels.to_numpy(), index=index).fillna(default)


def _funding_state_labels(funding_rate: pd.Series | None, index: pd.Index) -> pd.Series:
    if funding_rate is None or len(funding_rate) == 0 or float(pd.Series(funding_rate).abs().sum()) <= 1e-12:
        return pd.Series("unavailable", index=index)
    z = pd.Series(funding_rate.to_numpy(dtype=float), index=index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    labels = pd.Series("neutral", index=index)
    labels.loc[z <= -1.5] = "extreme_negative"
    labels.loc[(z > -1.5) & (z <= -0.5)] = "negative"
    labels.loc[(z >= 0.5) & (z < 1.5)] = "positive"
    labels.loc[z >= 1.5] = "extreme_positive"
    return labels


def funding_state_labels(funding_rate: pd.Series | None, index: pd.Index) -> pd.Series:
    return _funding_state_labels(funding_rate, index)


def _funding_trend_labels(funding_rate: pd.Series | None, index: pd.Index) -> pd.Series:
    if funding_rate is None or len(funding_rate) == 0 or float(pd.Series(funding_rate).abs().sum()) <= 1e-12:
        return pd.Series("unavailable", index=index)
    z = pd.Series(funding_rate.to_numpy(dtype=float), index=index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    diff = z.diff().fillna(0.0)
    labels = pd.Series("stable", index=index)
    labels.loc[diff > 0.05] = "rising"
    labels.loc[diff < -0.05] = "falling"
    return labels


def funding_trend_labels(funding_rate: pd.Series | None, index: pd.Index) -> pd.Series:
    return _funding_trend_labels(funding_rate, index)


def _strategy_returns_for_position(
    frame: pd.DataFrame,
    open_returns: pd.Series,
    position: pd.Series,
    settings: Settings,
    funding: pd.DataFrame | None,
) -> pd.Series:
    turnover = position.diff().abs().fillna(position.abs())
    order_notional = settings.position_sizing.fixed_notional_usd * turnover
    participation = (order_notional / frame["quote_volume"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    slippage_bps = settings.costs.slippage_base_bps + settings.costs.slippage_k * (
        participation.clip(lower=0.0) ** settings.costs.slippage_gamma
    )
    cost_returns = turnover * (settings.costs.taker_bps + slippage_bps) / 10_000.0
    returns = position * open_returns.fillna(0.0) + _funding_returns(position, frame, funding) - cost_returns
    return returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _funding_returns(position: pd.Series, frame: pd.DataFrame, funding: pd.DataFrame | None) -> pd.Series:
    impact = pd.Series(0.0, index=frame.index)
    if funding is None or funding.empty:
        return impact
    rate_by_time = dict(zip(funding["calc_time"], funding["last_funding_rate"], strict=False))
    for idx, open_time in frame["open_time"].items():
        rate = rate_by_time.get(int(open_time))
        if rate is not None:
            impact.loc[idx] = -float(position.loc[idx]) * float(rate)
    return impact


def _metrics_from_returns(returns: pd.Series, *, interval: str, trade_count: int) -> MetricsBlock:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    periods = annualization_factor(interval)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
    ann_return = float((1.0 + returns.mean()) ** periods - 1.0) if not returns.empty else 0.0
    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods)) if len(returns) > 1 else 0.0
    mdd = max_drawdown(equity)
    return MetricsBlock(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe=sharpe_ratio(returns, periods_per_year=periods),
        max_drawdown=mdd,
        calmar=ann_return / abs(mdd) if mdd < 0 else 0.0,
        trade_count=trade_count,
        pnl=0.0,
    )


def _trade_count(position: pd.Series) -> int:
    return int((position.diff().abs() > 1e-12).sum())


def _gross_net_decomposition(result: BacktestResult) -> dict[str, float | int | None]:
    gross = result.metrics_gross
    net = result.metrics_primary
    gross_sharpe = gross.sharpe if gross is not None else None
    gross_return = gross.total_return if gross is not None else None
    return {
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net.sharpe,
        "cost_drag_sharpe": None if gross_sharpe is None else gross_sharpe - net.sharpe,
        "gross_total_return": gross_return,
        "net_total_return": net.total_return,
        "gross_minus_net_total_return": None if gross_return is None else gross_return - net.total_return,
        "factor_turnover": result.factor_turnover,
        "avg_holding_period_bars": result.avg_holding_period_bars,
        "break_even_cost_bps": result.break_even_cost_bps,
        "actual_cost_bps": result.actual_cost_bps,
        "cost_margin_bps": result.break_even_cost_bps - 2.0 * result.actual_cost_bps,
        "oos_trade_count": result.oos_trade_count,
    }
