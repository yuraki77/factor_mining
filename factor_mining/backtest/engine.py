from __future__ import annotations

from dataclasses import dataclass
import uuid

import numpy as np
import pandas as pd

from factor_mining.config import Settings
from factor_mining.hypotheses.discovered import boundary_conditions
from factor_mining.models import (
    BacktestResult,
    CandidateStrategySpec,
    DataQualityNote,
    MetricsBlock,
    OOSWindowMetrics,
    TrialRecord,
    WindowStabilityDiagnostics,
)
from factor_mining.stats.metrics import (
    _block_bootstrap_sharpes,
    annualization_factor,
    deflated_sharpe_probability,
    deflated_sharpe_ratio,
    haircut_sharpe,
    max_drawdown,
    newey_west_tstat,
    permutation_test_mean_ic,
    probabilistic_sharpe_ratio,
    return_autocorrelation_lag1,
    return_moments,
    rolling_pearson_ic,
    rolling_rank_ic,
    sharpe_ratio,
)
from factor_mining.stats.regime import label_btc_regime
from factor_mining.trial_ledger import TrialLedger


# Rolling IC lookback. Adjacent rolling-IC observations share all but one bar of
# this window, so the IC series is overlapping (~MA(window-1)) with non-zero
# autocovariance out to ~window lags. The Newey-West t-stat on that series must
# use a bandwidth of this horizon — not the generic n**(1/3) rule — or it
# under-counts the overlap variance and over-states IC significance (Q4). Kept as
# one constant so the window and the bandwidth can never drift apart.
_IC_WINDOW_BARS = 288

# Cap on the number of points stored in BacktestResult.equity_curve (I5). Keeps a
# serialized/archived result small regardless of OOS length while preserving the
# curve's shape and its exact final value.
_EQUITY_CURVE_MAX_POINTS = 512


@dataclass(frozen=True)
class StrategyPath:
    frame: pd.DataFrame
    signals: pd.Series
    open_returns: pd.Series
    position: pd.Series
    fixed_position: pd.Series
    strategy_returns: pd.Series
    fixed_returns: pd.Series
    avg_cost_bps: float
    avg_participation: float


_CALMAR_CAP = 1000.0


def _metrics_from_returns(returns: pd.Series, *, interval: str, trade_count: int, pnl: float) -> MetricsBlock:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    periods = annualization_factor(interval)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
    # Geometric annualized return: annualize the realized compound growth, not
    # the arithmetic per-period mean. Compounding the mean — (1 + mean) ** periods
    # — overstates the annualized figure (arithmetic mean ≥ geometric mean) and
    # diverges further the more volatile the series.
    n_obs = len(returns)
    if n_obs > 0 and equity.iloc[-1] > 0.0:
        ann_return = float(equity.iloc[-1] ** (periods / n_obs) - 1.0)
    elif n_obs > 0:
        ann_return = -1.0  # equity wiped out → annualized total loss
    else:
        ann_return = 0.0
    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods)) if len(returns) > 1 else 0.0
    sharpe = sharpe_ratio(returns, periods_per_year=periods)
    mdd = max_drawdown(equity)
    # I5 (P2-6): zero drawdown with positive returns used to report calmar=0,
    # indistinguishable from "no edge". Cap instead of infinity so the field
    # stays JSON-serializable and sortable.
    if mdd < 0:
        calmar = min(ann_return / abs(mdd), _CALMAR_CAP)
    else:
        calmar = _CALMAR_CAP if ann_return > 0 else 0.0
    return MetricsBlock(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
        calmar=calmar,
        trade_count=trade_count,
        pnl=pnl,
    )


def _avg_holding_period(position: pd.Series) -> float:
    active = position.abs() > 1e-12
    lengths: list[int] = []
    current = 0
    for value in active:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return float(np.mean(lengths)) if lengths else 0.0


def compute_oos_window_diagnostics(
    returns: pd.Series,
    position: pd.Series,
    interval: str,
    *,
    n_windows: int = 4,
    ic_series: pd.Series | None = None,
) -> WindowStabilityDiagnostics:
    """Split the OOS period into chronological windows and compute stability metrics.

    Args:
        returns: Strategy net returns for the final OOS period.
        position: Position series (same length/index as returns).
        interval: Bar interval string used to determine annualization.
        n_windows: How many chronological windows to split into (minimum 2).
        ic_series: Optional per-bar rolling IC; used to compute per-window IC t-stat.

    Returns:
        WindowStabilityDiagnostics with per-window details and aggregate scores.
    """
    n_windows = max(2, n_windows)
    periods = annualization_factor(interval)
    n = len(returns)
    if n < n_windows:
        # Not enough bars — return a degenerate but valid diagnostics object.
        return WindowStabilityDiagnostics(n_windows=0)

    per_window: list[OOSWindowMetrics] = []
    window_size = n // n_windows
    for i in range(n_windows):
        start = i * window_size
        end = n if i == n_windows - 1 else (i + 1) * window_size
        w_ret = returns.iloc[start:end].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        w_pos = position.iloc[start:end]
        trade_mask = w_pos.diff().abs().fillna(w_pos.abs()) > 1e-12
        w_trade_count = int(trade_mask.sum())
        if len(w_ret) < 2 or w_ret.std() < 1e-12:
            w_sharpe = 0.0
        else:
            w_sharpe = sharpe_ratio(w_ret, periods_per_year=periods)
        equity = (1.0 + w_ret).cumprod()
        w_total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
        w_mdd = max_drawdown(equity)
        w_ic_tstat: float | None = None
        if ic_series is not None:
            w_ic = ic_series.iloc[start:end].dropna()
            if len(w_ic) >= 4:
                w_ic_tstat = float(newey_west_tstat(w_ic, lag=_IC_WINDOW_BARS))
        per_window.append(OOSWindowMetrics(
            window_index=i,
            start_bar=start,
            end_bar=end,
            sharpe=w_sharpe,
            total_return=w_total_return,
            max_drawdown=w_mdd,
            trade_count=w_trade_count,
            ic_tstat=w_ic_tstat,
        ))

    sharpes = np.array([w.sharpe for w in per_window], dtype=float)
    positive_rate = float(np.mean(sharpes > 0))
    trade_coverage = float(np.mean([w.trade_count > 0 for w in per_window]))
    sharpe_mean = float(np.mean(sharpes))
    sharpe_std = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    # Stability score: [0,1]. Penalises high std and low positive rate.
    # score = positive_rate * clip(sharpe_mean / (1 + sharpe_std), 0, 1)
    if sharpe_mean <= 0:
        stability_score = 0.0
    else:
        norm = sharpe_mean / (1.0 + sharpe_std)
        stability_score = float(np.clip(positive_rate * norm, 0.0, 1.0))

    return WindowStabilityDiagnostics(
        n_windows=len(per_window),
        window_sharpe_mean=sharpe_mean,
        window_sharpe_std=sharpe_std,
        window_positive_rate=positive_rate,
        window_trade_coverage=trade_coverage,
        stability_score=stability_score,
        per_window=per_window,
    )


def _apply_funding(position: pd.Series, frame: pd.DataFrame, funding: pd.DataFrame | None) -> pd.Series:
    """Funding cash flows mapped to the first bar at-or-after each settlement.

    Vectorized via searchsorted on ``open_time`` (A6/A8). The previous
    exact-match dict lookup silently dropped any funding event whose
    settlement time fell inside a data gap or off the bar grid — the cash
    flow simply vanished from the backtest. Events now settle on the next
    available bar (several events inside one gap accumulate there); events
    before the window's first bar or after its last bar are outside the
    evaluation and are dropped.
    """
    impact = pd.Series(0.0, index=frame.index)
    if funding is None or funding.empty:
        return impact
    open_times = frame["open_time"].to_numpy(dtype=np.int64)
    calc_times = funding["calc_time"].to_numpy(dtype=np.int64)
    rates = funding["last_funding_rate"].to_numpy(dtype=float)
    bar_idx = np.searchsorted(open_times, calc_times, side="left")
    valid = (calc_times >= open_times[0]) & (bar_idx < len(open_times))
    if not valid.any():
        return impact
    flows = np.zeros(len(open_times), dtype=float)
    np.add.at(flows, bar_idx[valid], rates[valid])
    impact.iloc[:] = -position.to_numpy(dtype=float) * flows
    return impact


def _apply_position_buffer(target_position: pd.Series, threshold: float = 0.10) -> pd.Series:
    """Lazy execution: only change position if target deviates by more than threshold."""
    target = target_position.to_numpy()
    actual = np.zeros_like(target)
    current = 0.0
    for i in range(len(target)):
        if abs(target[i] - current) > threshold:
            current = target[i]
        actual[i] = current
    return pd.Series(actual, index=target_position.index)


def _apply_exit_rules(
    position: pd.Series,
    open_returns: pd.Series,
    *,
    stop_loss_pct: float = 0.0,
    max_hold_bars: int = 0,
    tp_tiers: list[tuple[float, float]] | None = None,
    trailing_stop_pct: float = 0.0,
    trailing_after_first_tp: bool = True,
) -> pd.Series:
    """Apply exit overrides to *position*: stop-loss, max-hold, batch TP, trailing stop.

    Called after position_buffer, before strategy_returns.

    Layers (checked in order each bar):

    1. **stop_loss** — cumulative PnL < *stop_loss_pct* → force flat.
    2. **max_hold_bars** — bars held >= *max_hold_bars* → force flat.
    3. **tp_tiers** — cumulative PnL crosses a tier threshold → reduce
       position by the tier's close-fraction.  Tiers fire at most once per
       trade cluster.  e.g. ``[(0.02, 0.50), (0.05, 0.30)]`` means close
       50% at +2% PnL, then 30% of remaining at +5%.
    4. **trailing_stop** — after first TP hit (or from entry if
       *trailing_after_first_tp* is False), peak PnL drops by
       *trailing_stop_pct* → force flat.

    Sign-flips (long↔short) reset all state as a new trade cluster.
    """
    tiers = list(tp_tiers or [])
    tiers.sort(key=lambda t: t[0])
    any_active = stop_loss_pct < 0.0 or max_hold_bars > 0 or tiers or trailing_stop_pct > 0.0
    if not any_active:
        return position
    pos = position.to_numpy(dtype=float)
    ret = open_returns.to_numpy(dtype=float)
    result = pos.copy()
    n = len(pos)
    in_position = False
    cum_pnl_pct = 0.0       # actual PnL on scaled position (for stop/trailing)
    cum_pnl_full = 0.0      # PnL on original position (for TP thresholds)
    bars_held = 0
    peak_pnl_pct = 0.0      # peak actual PnL (for trailing stop)
    trailing_active = False
    tier_hit_mask = 0
    position_scale = 1.0
    active_sign = 0.0
    blocked_sign = 0.0

    def reset_trade() -> None:
        nonlocal in_position, cum_pnl_pct, cum_pnl_full, bars_held
        nonlocal peak_pnl_pct, trailing_active, tier_hit_mask, position_scale, active_sign
        in_position = False
        cum_pnl_pct = 0.0
        cum_pnl_full = 0.0
        bars_held = 0
        peak_pnl_pct = 0.0
        trailing_active = False
        tier_hit_mask = 0
        position_scale = 1.0
        active_sign = 0.0

    for i in range(n):
        desired_sign = 1.0 if pos[i] > 0 else -1.0 if pos[i] < 0 else 0.0
        if blocked_sign != 0.0:
            if desired_sign == 0.0:
                blocked_sign = 0.0
            elif desired_sign == blocked_sign:
                result[i] = 0.0
                continue
            else:
                blocked_sign = 0.0
        if in_position and i > 0:
            prev_ret = ret[i - 1]
            if np.isfinite(prev_ret):
                pnl_bar = float(result[i - 1] * prev_ret)
                full_bar = float(pos[i - 1] * prev_ret)
                cum_pnl_pct += pnl_bar
                cum_pnl_full += full_bar
                bars_held += 1
                if cum_pnl_pct > peak_pnl_pct:
                    peak_pnl_pct = cum_pnl_pct
            stopped = False
            block_reentry = False
            if 0 < max_hold_bars <= bars_held:
                stopped = True
            if stop_loss_pct < 0.0 and cum_pnl_pct < stop_loss_pct:
                stopped = True
                block_reentry = True
            if trailing_active and trailing_stop_pct > 0.0:
                if cum_pnl_pct < peak_pnl_pct - trailing_stop_pct:
                    stopped = True
                    block_reentry = True
            if stopped:
                result[i] = 0.0
                stopped_sign = active_sign
                reset_trade()
                if block_reentry and desired_sign == stopped_sign:
                    blocked_sign = stopped_sign
                continue
            if desired_sign == 0.0:
                result[i] = 0.0
                reset_trade()
                continue
            if active_sign != 0.0 and desired_sign != active_sign:
                reset_trade()
            for tier_idx, (threshold, fraction) in enumerate(tiers):
                if (tier_hit_mask >> tier_idx) & 1:
                    continue
                if cum_pnl_full >= threshold:
                    position_scale *= 1.0 - float(fraction)
                    tier_hit_mask |= 1 << tier_idx
                    if trailing_after_first_tp and not trailing_active:
                        trailing_active = True
                        peak_pnl_pct = cum_pnl_pct
            if in_position:
                result[i] = pos[i] * position_scale
        if not in_position and abs(pos[i]) > 1e-8:
            in_position = True
            cum_pnl_pct = 0.0
            cum_pnl_full = 0.0
            bars_held = 0
            peak_pnl_pct = 0.0
            trailing_active = not trailing_after_first_tp
            tier_hit_mask = 0
            position_scale = 1.0
            active_sign = desired_sign
            result[i] = pos[i]
    return pd.Series(result, index=position.index)


def _resolve_exit_params(
    candidate: CandidateStrategySpec, settings: Settings,
) -> tuple[float, int, list[tuple[float, float]], float, bool]:
    """Return (stop_loss_pct, max_hold_bars, tp_tiers, trailing_stop_pct, trailing_after_first_tp)."""
    ex = settings.exit
    sl = float(candidate.params.get("stop_loss_pct", ex.stop_loss_pct))
    mh = int(candidate.params.get("max_hold_bars", ex.max_hold_bars))
    raw_tiers = candidate.params.get("tp_tiers", ex.tp_tiers)
    if raw_tiers is None:
        raw_tiers = []
    tiers = [tuple(float(v) for v in t[:2]) for t in raw_tiers if len(t) >= 2]
    tr = float(candidate.params.get("trailing_stop_pct", ex.trailing_stop_pct))
    ta = bool(candidate.params.get("trailing_after_first_tp", ex.trailing_after_first_tp))
    return sl, mh, tiers, tr, ta


def _pre_gap_flags(frame: pd.DataFrame, interval: str) -> np.ndarray:
    """True at bars whose NEXT bar is non-contiguous (G1/P2-2): any 1-bar
    forward quantity computed at such a bar spans a data outage and is not a
    per-interval return."""
    open_times = frame["open_time"].to_numpy(dtype=np.int64)
    flags = np.zeros(open_times.size, dtype=bool)
    if open_times.size < 2:
        return flags
    interval_ms = round(365 * 86_400_000 / annualization_factor(interval))
    flags[:-1] = np.diff(open_times) > 1.5 * interval_ms
    return flags


def _freeze_reopen_entries(position: pd.Series, reopen_indices: np.ndarray) -> pd.Series:
    """Block NEW risk on the first bar after a data gap (G1/P2-2): the signal
    there was built on pre-gap information and a fill at the reopen print is
    fantasy. Exits and reductions stay allowed — the position is clamped to
    the interval between zero and the previous bar's position."""
    if reopen_indices.size == 0:
        return position
    pos = position.copy()
    for idx in reopen_indices:
        if idx <= 0 or idx >= len(pos):
            continue
        prev = float(pos.iloc[idx - 1])
        lo, hi = min(0.0, prev), max(0.0, prev)
        pos.iloc[idx] = float(np.clip(pos.iloc[idx], lo, hi))
    return pos


def realized_vol_series(frame: pd.DataFrame, settings: Settings, interval: str) -> pd.Series:
    """Trailing annualized realized vol used for vol targeting.

    Exposed so the pipeline can compute it once on the full frame and slice
    the *state* along with the signal (F3/P1-5): re-warming inside every
    evaluation slice zeroed leverage over each slice's warm-up bars even
    though the trailing history existed on the full frame."""
    periods = annualization_factor(interval)
    bars_per_day = max(1.0, periods / 365.0)
    vol_window = max(2, int(settings.position_sizing.vol_window_days * bars_per_day))
    known_open_returns = frame["open"].pct_change().shift(1)
    return known_open_returns.rolling(vol_window, min_periods=10).std() * np.sqrt(periods)


def evaluate_strategy_path(
    frame: pd.DataFrame,
    signals: pd.Series,
    candidate: CandidateStrategySpec,
    settings: Settings,
    *,
    funding: pd.DataFrame | None = None,
    realized_vol: pd.Series | None = None,
) -> StrategyPath:
    frame = frame.sort_values("open_time").reset_index(drop=True).copy()
    signals = pd.Series(signals.to_numpy(dtype=float), index=frame.index).clip(-1, 1).fillna(0.0)
    signals = _apply_side_mode(signals, candidate)
    if not _short_allowed(candidate.hypothesis_family):
        signals = signals.clip(lower=0)

    open_returns = frame["open"].shift(-1) / frame["open"] - 1.0
    executable_signal = signals.shift(1).fillna(0.0)
    if realized_vol is None:
        realized_vol = realized_vol_series(frame, settings, candidate.interval)
    else:
        # Precomputed full-frame state sliced by the caller; align positionally.
        realized_vol = pd.Series(np.asarray(realized_vol, dtype=float), index=frame.index)
    leverage = (settings.position_sizing.target_annual_vol / realized_vol.replace(0, np.nan)).clip(
        upper=settings.position_sizing.max_leverage_for(candidate.symbol)
    ).fillna(0.0)
    raw_vol_target = (executable_signal * leverage).fillna(0.0)
    position_buffer = _position_buffer_threshold(candidate)
    vol_target_position = _apply_position_buffer(raw_vol_target, threshold=position_buffer)

    raw_fixed = executable_signal.fillna(0.0)
    fixed_position = _apply_position_buffer(raw_fixed, threshold=position_buffer)

    sl_pct, max_hold, tiers, tr_pct, tr_after_tp = _resolve_exit_params(candidate, settings)
    if sl_pct < 0.0 or max_hold > 0 or tiers or tr_pct > 0.0:
        vol_target_position = _apply_exit_rules(
            vol_target_position, open_returns,
            stop_loss_pct=sl_pct, max_hold_bars=max_hold,
            tp_tiers=tiers, trailing_stop_pct=tr_pct, trailing_after_first_tp=tr_after_tp,
        )
        fixed_position = _apply_exit_rules(
            fixed_position, open_returns,
            stop_loss_pct=sl_pct, max_hold_bars=max_hold,
            tp_tiers=tiers, trailing_stop_pct=tr_pct, trailing_after_first_tp=tr_after_tp,
        )

    reopen_indices = np.where(_pre_gap_flags(frame, candidate.interval))[0] + 1
    if reopen_indices.size:
        vol_target_position = _freeze_reopen_entries(vol_target_position, reopen_indices)
        fixed_position = _freeze_reopen_entries(fixed_position, reopen_indices)

    primary_returns, primary_cost_bps, avg_participation = _strategy_returns(
        frame,
        open_returns,
        vol_target_position,
        settings=settings,
        funding=funding,
    )
    secondary_returns, _, _ = _strategy_returns(
        frame,
        open_returns,
        fixed_position,
        settings=settings,
        funding=funding,
    )
    return StrategyPath(
        frame=frame,
        signals=signals,
        open_returns=open_returns,
        position=vol_target_position,
        fixed_position=fixed_position,
        strategy_returns=primary_returns,
        fixed_returns=secondary_returns,
        avg_cost_bps=primary_cost_bps,
        avg_participation=avg_participation,
    )


def _bounded_equity_curve(returns: pd.Series, *, max_points: int = _EQUITY_CURVE_MAX_POINTS) -> list[float]:
    """Downsampled compound-equity path (cumprod of 1+r), matching the series
    ``build_backtest_detail`` charts. Bounded to ``max_points`` so a serialized or
    archived BacktestResult stays small regardless of OOS length; the first and
    last points are always kept (stride-sampled interior), so the final compounded
    value is exact. NaN/inf returns are treated as a flat (0%) bar to preserve the
    timeline."""
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    if clean.size == 0:
        return []
    equity = np.cumprod(1.0 + clean)
    if equity.size > max_points:
        idx = np.unique(np.linspace(0, equity.size - 1, max_points).round().astype(int))
        equity = equity[idx]
    return [float(value) for value in equity]


def run_backtest(
    frame: pd.DataFrame,
    signals: pd.Series,
    candidate: CandidateStrategySpec,
    settings: Settings,
    *,
    trial_ledger: TrialLedger | None = None,
    trial_counts: dict[str, int] | None = None,
    funding: pd.DataFrame | None = None,
    data_quality_notes: list[DataQualityNote] | None = None,
    btc_regime_frame: pd.DataFrame | None = None,
    realized_vol: pd.Series | None = None,
    regime_labels: pd.Series | None = None,
) -> BacktestResult:
    path = evaluate_strategy_path(frame, signals, candidate, settings, funding=funding, realized_vol=realized_vol)
    frame = path.frame
    signals = path.signals
    open_returns = path.open_returns
    primary_returns = path.strategy_returns
    secondary_returns = path.fixed_returns
    vol_target_position = path.position
    fixed_position = path.fixed_position
    periods = annualization_factor(candidate.interval)
    trade_count = int((vol_target_position.diff().abs() > 1e-12).sum())
    pnl = float(primary_returns.sum() * settings.position_sizing.fixed_notional_usd)
    metrics_primary = _metrics_from_returns(primary_returns, interval=candidate.interval, trade_count=trade_count, pnl=pnl)
    metrics_secondary = _metrics_from_returns(
        secondary_returns,
        interval=candidate.interval,
        trade_count=int((fixed_position.diff().abs() > 1e-12).sum()),
        pnl=float(secondary_returns.sum() * settings.position_sizing.fixed_notional_usd),
    )
    gross_returns = (
        vol_target_position * open_returns.fillna(0.0)
        + _apply_funding(vol_target_position, frame, funding)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    metrics_gross = _metrics_from_returns(
        gross_returns,
        interval=candidate.interval,
        trade_count=trade_count,
        pnl=float(gross_returns.sum() * settings.position_sizing.fixed_notional_usd),
    )

    forward_returns = open_returns.fillna(0.0)
    # Use the executable signal (lagged one bar) for IC so it matches actual
    # trade timing.  ``signals`` here is already clipped but not shifted;
    # the position uses ``signals.shift(1)`` (line 107), so IC must too.
    executable_sig = signals.shift(1).fillna(0.0)
    sig_arr = executable_sig.to_numpy(dtype=float)
    fwd_arr = forward_returns.to_numpy(dtype=float)
    # G1 (P2-2): a bar whose next bar sits across a data outage has no 5-minute
    # forward return — drop those bars from the IC inputs (compacted, then
    # re-aligned with NaN so downstream .dropna() consumers stay positional).
    ic_keep = ~_pre_gap_flags(frame, candidate.interval)
    ic_vals = np.full(sig_arr.size, np.nan)
    rankic_vals = np.full(sig_arr.size, np.nan)
    ic_vals[ic_keep] = rolling_pearson_ic(sig_arr[ic_keep], fwd_arr[ic_keep], window=_IC_WINDOW_BARS, min_periods=10)
    rankic_vals[ic_keep] = rolling_rank_ic(sig_arr[ic_keep], fwd_arr[ic_keep], window=_IC_WINDOW_BARS, min_periods=10)
    ic_series = pd.Series(ic_vals, index=signals.index)
    rankic_series = pd.Series(rankic_vals, index=signals.index)
    avg_hold = _avg_holding_period(vol_target_position)
    block_len = settings.bootstrap.block_length_bars(avg_hold)
    ret_arr = primary_returns.to_numpy(dtype=float)
    ret_arr = ret_arr[np.isfinite(ret_arr)]
    if len(ret_arr) >= 3:
        boot_sharpes = _block_bootstrap_sharpes(ret_arr, periods, settings.bootstrap.n_resamples, block_len, seed=42)
        ci = (
            float(np.quantile(boot_sharpes, settings.bootstrap.ci_levels[0])),
            float(np.quantile(boot_sharpes, settings.bootstrap.ci_levels[1])),
        )
    else:
        ci = (0.0, 0.0)
    permutation_p = permutation_test_mean_ic(
        sig_arr[ic_keep],
        fwd_arr[ic_keep],
        n_permutations=settings.permutation_test.n_permutations,
    )
    if trial_ledger is not None:
        trial = TrialRecord(
            trial_id=str(uuid.uuid4()),
            candidate_id=candidate.candidate_id,
            experiment_id=None,
            hypothesis_family=candidate.hypothesis_family,
            method_id=candidate.method_id,
        )
        trial_ledger.record(trial)
        counts = trial_ledger.counts_for(candidate.hypothesis_family)
    elif trial_counts is not None:
        counts = _normalize_trial_counts(trial_counts)
    else:
        counts = {
            "effective_trials_count": 1,
            "global_cumulative_trials_count": 1,
        }
    observed_sr = metrics_primary.sharpe
    periods_per_year = annualization_factor(candidate.interval)
    dsr = deflated_sharpe_ratio(primary_returns, observed_sr=observed_sr, trials_count=counts["effective_trials_count"], periods_per_year=periods_per_year)
    _ = haircut_sharpe(observed_sr, trials_count=counts["effective_trials_count"], observations=max(len(primary_returns), 1), periods_per_year=periods_per_year)
    n_obs, ret_skew, ret_kurt = return_moments(primary_returns)
    dsr_prob = deflated_sharpe_probability(
        observed_sr=observed_sr,
        trials_count=counts["effective_trials_count"],
        periods_per_year=periods_per_year,
        observations=max(n_obs, 2),
        skew=ret_skew,
        kurtosis=ret_kurt,
    )
    if regime_labels is not None:
        # F3 (P1-5): labels computed on the full frame and sliced by the
        # caller, so the slice's early bars carry real trailing regimes
        # instead of the labeler's in-slice warm-up default.
        regimes = pd.Series(np.asarray(regime_labels, dtype=object), index=frame.index).astype(str)
    else:
        regimes = label_btc_regime(btc_regime_frame if btc_regime_frame is not None else frame, settings.regime)
        if len(regimes) == len(frame):
            regimes = pd.Series(regimes.to_numpy(), index=frame.index).astype(str)
        else:
            regimes = regimes.reindex(frame.index).fillna("sideways").astype(str)
    trade_mask = vol_target_position.diff().abs().fillna(vol_target_position.abs()) > 1e-12
    regime_metrics = {}
    for regime in sorted(set(regimes)):
        regime_mask = regimes == regime
        regime_returns = primary_returns.loc[regime_mask]
        regime_metrics[regime] = _metrics_from_returns(
            regime_returns,
            interval=candidate.interval,
            trade_count=int((trade_mask & regime_mask).sum()),
            pnl=float(regime_returns.sum() * settings.position_sizing.fixed_notional_usd),
        )
    adv = float(frame["quote_volume"].tail(min(len(frame), 288 * 30)).sum() / max(1, min(len(frame), 288 * 30) / 288))
    estimated_capacity = adv * settings.capacity.adv_participation
    break_even_cost_bps = _break_even_cost_bps(primary_returns, vol_target_position)
    expected_ic_mid = abs(float(candidate.params.get("expected_ic_mid", 0.02))) or 0.02
    observed_ic = abs(float(ic_series.mean(skipna=True))) if not ic_series.dropna().empty else 0.0
    window_stability = compute_oos_window_diagnostics(
        primary_returns,
        vol_target_position,
        candidate.interval,
        n_windows=4,
        ic_series=ic_series,
    )
    trial_diagnostics = {
        "candidate_type": candidate.candidate_type,
        "parent_candidate_id": candidate.parent_candidate_id,
        "effective_trials_at_eval": counts["effective_trials_count"],
        "global_trials_at_eval": counts["global_cumulative_trials_count"],
        "complexity_score": int(candidate.params.get("complexity_score", 1)),
        "dsr": float(dsr),
    }
    return BacktestResult(
        experiment_id=str(uuid.uuid4()),
        candidate_id=candidate.candidate_id,
        hypothesis_family=candidate.hypothesis_family,
        method_id=candidate.method_id,
        symbol=candidate.symbol,
        market=candidate.market,
        interval=candidate.interval,
        metrics_primary=metrics_primary,
        metrics_secondary=metrics_secondary,
        metrics_gross=metrics_gross,
        equity_curve=_bounded_equity_curve(primary_returns),
        ic_tstat_nw=newey_west_tstat(ic_series.dropna(), lag=_IC_WINDOW_BARS),
        rankic_tstat_nw=newey_west_tstat(rankic_series.dropna(), lag=_IC_WINDOW_BARS),
        sharpe_ci_5_95=ci,
        probabilistic_sharpe=probabilistic_sharpe_ratio(primary_returns, observed_sr=observed_sr, periods_per_year=periods_per_year),
        deflated_sharpe=dsr,
        deflated_sharpe_prob=dsr_prob,
        return_skew=ret_skew,
        return_kurtosis=ret_kurt,
        effective_trials_at_eval=counts["effective_trials_count"],
        global_trials_at_eval=counts["global_cumulative_trials_count"],
        pbo=None,
        permutation_test_pvalue=permutation_p,
        regime_conditional_metrics=regime_metrics,
        avg_participation_rate=path.avg_participation,
        estimated_capacity_usd=estimated_capacity,
        factor_turnover=float(vol_target_position.diff().abs().mean()),
        break_even_cost_bps=break_even_cost_bps,
        avg_holding_period_bars=avg_hold,
        return_autocorr_lag1=return_autocorrelation_lag1(primary_returns),
        data_quality_notes=data_quality_notes or [],
        # F1 (P1-4): since the Q3 switch the evaluated frame IS the
        # out-of-sample slice (validation in rounds, the untouched holdout at
        # the terminal), so every trade in it is OOS relative to discovery.
        # The old fold mask degenerated to "last 25% of the slice" on frames
        # shorter than the configured windows — an arbitrary subset.
        oos_trade_count=int(trade_mask.sum()),
        actual_cost_bps=path.avg_cost_bps,
        prior_posterior_ic_ratio=observed_ic / expected_ic_mid if expected_ic_mid > 0 else 1.0,
        window_stability=window_stability,
        trial_diagnostics=trial_diagnostics,
    )


def _strategy_returns(
    frame: pd.DataFrame,
    open_returns: pd.Series,
    position: pd.Series,
    *,
    settings: Settings,
    funding: pd.DataFrame | None,
) -> tuple[pd.Series, float, float]:
    order_notional = settings.position_sizing.fixed_notional_usd * position.diff().abs().fillna(position.abs())
    participation = (order_notional / frame["quote_volume"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    slippage_bps = settings.costs.slippage_base_bps + settings.costs.slippage_k * (participation.clip(lower=0.0) ** settings.costs.slippage_gamma)
    turnover = position.diff().abs().fillna(position.abs())
    avg_cost_bps = float((settings.costs.taker_bps + slippage_bps).replace([np.inf, -np.inf], np.nan).fillna(0).mean())
    cost_returns = turnover * (settings.costs.taker_bps + slippage_bps) / 10_000.0
    funding_returns = _apply_funding(position, frame, funding)
    strategy_returns = (position * open_returns.fillna(0.0) + funding_returns - cost_returns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return strategy_returns, avg_cost_bps, float(participation.mean())


def _position_buffer_threshold(candidate: CandidateStrategySpec) -> float:
    try:
        value = float(candidate.params.get("position_buffer", 0.05))
    except (TypeError, ValueError):
        value = 0.05
    return max(0.0, min(value, 1.0))


def _apply_side_mode(signals: pd.Series, candidate: CandidateStrategySpec) -> pd.Series:
    mode = str(candidate.params.get("side_mode", "both")).lower()
    if mode == "long_only":
        return signals.clip(lower=0.0)
    if mode == "short_only":
        return signals.clip(upper=0.0)
    return signals


def _break_even_cost_bps(returns: pd.Series, position: pd.Series) -> float:
    turnover = float(position.diff().abs().sum())
    if turnover <= 0:
        return 0.0
    return float(max(0.0, returns.sum()) / turnover * 10_000.0)


def _normalize_trial_counts(counts: dict[str, int]) -> dict[str, int]:
    normalized = {
        "family_trials_count": 1,
        "rolling_90d_trials_count": 1,
        "effective_trials_count": 1,
        "global_cumulative_trials_count": 1,
    }
    normalized.update({key: max(1, int(value)) for key, value in counts.items()})
    return normalized


def _short_allowed(hypothesis_family: str) -> bool:
    """Check boundary conditions: is short-side trading allowed for this family?"""
    b = boundary_conditions(hypothesis_family)
    return bool(b.get("short_allowed", True))


def build_backtest_detail(
    frame: pd.DataFrame,
    signals: pd.Series,
    candidate: CandidateStrategySpec,
    settings: Settings,
    result: BacktestResult,
    *,
    funding: pd.DataFrame | None = None,
    max_chart_rows: int = 2_000,
    max_trades: int = 500,
) -> dict:
    """Build a chart-ready experiment detail payload without changing summary metrics."""
    path = evaluate_strategy_path(frame, signals, candidate, settings, funding=funding)
    frame = path.frame
    signals = path.signals
    position = path.position
    strategy_returns = path.strategy_returns
    equity = (1.0 + strategy_returns).cumprod().replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    drawdown = equity / equity.cummax().replace(0, np.nan) - 1.0

    detail_frame = frame.copy()
    detail_frame["signal"] = signals
    detail_frame["position"] = position
    detail_frame["strategy_return"] = strategy_returns
    detail_frame["equity"] = equity
    detail_frame["drawdown"] = drawdown.fillna(0.0)
    detail_frame["participation"] = path.avg_participation

    trades = _trade_events(detail_frame, max_trades=max_trades)
    if max_chart_rows > 0 and len(detail_frame) > max_chart_rows:
        detail_frame = detail_frame.iloc[-max_chart_rows:].reset_index(drop=True)

    return {
        "experiment_id": result.experiment_id,
        "candidate_id": candidate.candidate_id,
        "hypothesis_id": candidate.hypothesis_id,
        "symbol": candidate.symbol,
        "market": candidate.market,
        "interval": candidate.interval,
        "params": candidate.params,
        "summary": result.model_dump(mode="json"),
        "ohlcv": _records(detail_frame, ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]),
        "series": _records(detail_frame, ["open_time", "signal", "position", "strategy_return", "equity", "drawdown", "participation"]),
        "trades": trades,
        "chart_rows": len(detail_frame),
        "total_rows": len(frame),
    }


def _trade_events(frame: pd.DataFrame, *, max_trades: int) -> list[dict]:
    delta = frame["position"].diff().fillna(frame["position"])
    trades = frame.loc[delta.abs() > 1e-12, ["open_time", "open", "position", "signal", "equity"]].copy()
    trades["delta"] = delta.loc[trades.index]
    trades["side"] = np.where(trades["delta"] > 0, "buy", "sell")
    trades = trades.tail(max_trades)
    out = []
    for row in trades.itertuples(index=False):
        out.append({
            "open_time": int(row.open_time),
            "side": str(row.side),
            "price": float(row.open),
            "position": float(row.position),
            "delta": float(row.delta),
            "signal": float(row.signal),
            "equity": float(row.equity),
        })
    return out


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    cleaned = frame[columns].replace([np.inf, -np.inf], np.nan)
    records = cleaned.where(pd.notnull(cleaned), None).to_dict(orient="records")
    for record in records:
        if "open_time" in record and record["open_time"] is not None:
            record["open_time"] = int(record["open_time"])
    return records
