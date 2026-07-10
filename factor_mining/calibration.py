"""FAR calibration harness — measure the statistical gates' false-acceptance rate.

The 16-gate stack mixes two axes: statistical validity (which should behave like a
calibrated false-acceptance-rate control — feed it noise, it accepts at ≤ q) and
economic deployability (cost/capacity — a quality judgment). This harness measures the
FAR of the STATISTICAL gates only, under a circular-rotation null, and isolates whether
lineage-deduped trial counts (bias-audit finding A) inflate it.

Null: circular-rotate a real candidate signal relative to the frame (`np.roll` by a large
random offset). This is the same null the pipeline already trusts in
`permutation_test_mean_ic` (stats/metrics.py:311), generalized from IC-only to the full
gate stack. It preserves each series' own structure and destroys the signal→return
alignment, so under it every acceptance is a *false* acceptance.

Finding-A isolation: the DSR expected-max penalty and the FDR multiplicity both grow with
the trial count N. Running the null twice — once with the deduped N the pipeline computes,
once with the raw config count — measures whether the dedup inflates the FAR. DSR is
monotone in N, so FAR(raw N) ≤ FAR(deduped N) always; the question is whether FAR(deduped)
exceeds the nominal rate.

Nothing here writes to any store — it is pure computation over an in-memory frame.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor_mining.backtest.engine import run_backtest
from factor_mining.config import Settings
from factor_mining.mining import factor_signal
from factor_mining.models import CandidateStrategySpec
from factor_mining.registry import get_method
from factor_mining.stats.metrics import benjamini_hochberg, combined_ic_tstat_pvalue
from factor_mining.validation.gatecheck import run_gatecheck

# FAR-controlling gates only. G8 (cost margin) and G10 (capacity) are the economic axis:
# they pass/fail on real data regardless of predictive nullity, so including them would
# conflate the two axes and corrupt the false-acceptance-rate number. Excluded by design
# from the FAR arms; the POWER harness measures both axes — separating them is its point.
STATISTICAL_GATES: tuple[str, ...] = ("G1", "G2", "G3", "G5", "G7")
ECONOMIC_GATES: tuple[str, ...] = ("G8", "G10")
# G2/PBO is batch-relative (CSCV over a candidate pool; standalone run_backtest returns
# pbo=None and the gate fails closed), so no single-signal harness trial can ever pass it.
# It stays in per-gate reports — visibly fail-closed — but composites AND only the gates
# measurable standalone; for FAR this makes the composite a conservative upper bound.
STANDALONE_AND_GATES: tuple[str, ...] = ("G1", "G3", "G5", "G7")
_ALL = "ALL_STAT"


def rotate_signal(signal, rng: np.random.Generator, min_gap: int) -> np.ndarray:
    """The null: circular-shift ``signal`` by a random offset ≥ ``min_gap``.

    ``min_gap`` should be ≥ the signal's feature lookback so the rotation cannot leave a
    trivially-small, still-aligned overlap that would understate the FAR."""
    arr = np.asarray(signal, dtype=float)
    n = arr.size
    if n <= 2 * max(1, min_gap) + 1:
        return np.roll(arr, max(1, n // 2))
    k = int(rng.integers(min_gap, n - min_gap))
    return np.roll(arr, k)


def plant_signal_with_horizon(
    frame: pd.DataFrame, rng: np.random.Generator, *, alpha: float, noise: float, horizon: int
) -> np.ndarray:
    """A signal with genuine (clairvoyant) edge whose payoff accrues over ``horizon`` bars.

    The engine fills ``signals.shift(1)`` at ``open[t+1]``, so the signal at ``t`` predicts
    the path ``open[t+1] → open[t+1+horizon]``. Overlapping targets make the signal
    ~``horizon``-bar autocorrelated, so positions persist and turnover falls ≈ 1/horizon —
    the realistic profile for slow alphas, and what lets the cost gates be measured fairly
    (a per-bar edge churns every bar and dies on costs regardless of statistical strength).
    Deliberately look-ahead: a controlled-strength positive control, not a strategy.
    With ``noise=1``, ``alpha`` ≈ the signal↔target correlation for small values.

    The noise shares the horizon's timescale (variance-preserving rolling smooth):
    per-bar iid noise would make even a slow alpha churn its position every bar, so
    turnover — not evidence — would dominate every gate and the cost axis could never
    be measured fairly (the first smoke run showed exactly that: netSR ≈ −160 at α=0)."""
    n = len(frame)
    open_ = pd.Series(frame["open"].to_numpy(dtype=float))
    target = open_.shift(-(1 + int(horizon))) / open_.shift(-1) - 1.0
    std = float(target.std()) or 1.0
    z = ((target - float(target.mean())) / std).fillna(0.0).to_numpy()
    eps = rng.standard_normal(n)
    if int(horizon) > 1:
        eps = pd.Series(eps).rolling(int(horizon), min_periods=1).mean().to_numpy() * math.sqrt(float(horizon))
    return np.tanh(alpha * z + noise * eps)


def plant_signal(frame: pd.DataFrame, rng: np.random.Generator, *, alpha: float, noise: float) -> np.ndarray:
    """FAR power arm: next-bar (horizon=1) planted edge."""
    return plant_signal_with_horizon(frame, rng, alpha=alpha, noise=noise, horizon=1)


def plant_regime_concentrated_signal(
    frame: pd.DataFrame,
    regimes,
    kept_labels,
    rng: np.random.Generator,
    *,
    alpha: float,
    noise: float,
    horizon: int,
) -> np.ndarray:
    """Clairvoyant edge ONLY inside ``kept_labels`` regimes; pure noise elsewhere.

    This is the correct positive control for the regime-filter comparison: a *uniform*
    clairvoyant signal has edge in every regime, so filtering can only remove good bars
    and would make regime-conditioning look strictly harmful. A real regime-conditioned
    signal has predictive power concentrated in some regimes and none in others — that is
    exactly the shape where a regime filter should help, by zeroing the noise-regime
    exposure that otherwise churns cost and dilutes IC."""
    edge = plant_signal_with_horizon(frame, rng, alpha=alpha, noise=noise, horizon=horizon)
    labels = np.asarray(regimes, dtype=object)
    in_kept = np.isin(labels, list(kept_labels))
    pure_noise = np.tanh(noise * rng.standard_normal(len(frame)))
    return np.where(in_kept, edge, pure_noise)


def calibration_settings(settings: Settings, *, n_resamples: int = 250) -> Settings:
    """Cheaper bootstrap + no permutation test. Neither changes *which* gates are being
    calibrated, and both dominate per-backtest cost, so this keeps a Monte-Carlo sweep
    tractable without altering the FAR being measured."""
    return settings.model_copy(update={
        "bootstrap": settings.bootstrap.model_copy(update={"n_resamples": int(n_resamples)}),
        "permutation_test": settings.permutation_test.model_copy(update={"n_permutations": 0}),
    })


def default_candidates(
    *,
    families: tuple[str, ...] = ("momentum", "mean_reversion", "volatility"),
    lookbacks: tuple[int, ...] = (6, 12, 24, 48, 96, 144),
    symbol: str = "BTCUSDT",
    market: str = "spot",
    interval: str = "5m",
) -> list[CandidateStrategySpec]:
    """A representative factor_signal candidate set (real signals the pipeline produces)."""
    out: list[CandidateStrategySpec] = []
    for family in families:
        for lookback in lookbacks:
            out.append(CandidateStrategySpec(
                candidate_id=f"cal_{family}_{lookback}",
                hypothesis_id="cal",
                method_id="factor_scoring",
                hypothesis_family=family,
                symbol=symbol,
                market=market,
                interval=interval,
                params={"signal_source": "factor_signal", "factor_family": family, "lookback": lookback},
                max_feature_lookback_bars=lookback * 4,
            ))
    return out


def _evaluate(
    frame: pd.DataFrame,
    signal,
    candidate: CandidateStrategySpec,
    settings: Settings,
    *,
    effective_trials: int,
):
    """One backtest → gatecheck under trial count ``effective_trials``.

    ``effective_trials`` feeds both N-dependent FAR gates: G1 via the DSR expected-max
    penalty inside ``run_backtest``, and G3 via the BH-FDR multiplicity here."""
    sig = pd.Series(np.asarray(signal, dtype=float))
    trial_counts = {
        "effective_trials_count": int(effective_trials),
        "global_cumulative_trials_count": int(effective_trials),
    }
    result = run_backtest(frame, sig, candidate, settings, trial_counts=trial_counts)
    raw_p = combined_ic_tstat_pvalue(result.ic_tstat_nw, result.rankic_tstat_nw)
    fdr_p = benjamini_hochberg([raw_p], n_tests=int(effective_trials))[0]
    gate = run_gatecheck(result, settings, method=get_method(candidate.method_id), fdr_adjusted_pvalue=fdr_p)
    return gate, result


def gate_trial(
    frame: pd.DataFrame,
    signal,
    candidate: CandidateStrategySpec,
    settings: Settings,
    *,
    effective_trials: int,
) -> dict[str, bool]:
    """FAR-arm view: per-statistical-gate pass booleans plus ``ALL_STAT`` — the AND of the
    standalone-measurable gates (G2 is pool-relative and fails closed here, so including it
    would make the composite trivially zero instead of a measured rate)."""
    gate, _ = _evaluate(frame, signal, candidate, settings, effective_trials=effective_trials)
    status = {item.rule_id: item.status for item in gate.items}
    passed = {g: (status.get(g) == "pass") for g in STATISTICAL_GATES}
    passed[_ALL] = all(passed[g] for g in STANDALONE_AND_GATES)
    return passed


def wilson_ci(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a pass rate k/n — honest about Monte-Carlo error."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class ArmResult:
    n_trials: int
    passes: dict[str, int]

    def rate(self, gate: str) -> float:
        return self.passes.get(gate, 0) / self.n_trials if self.n_trials else 0.0

    def ci(self, gate: str) -> tuple[float, float]:
        return wilson_ci(self.passes.get(gate, 0), self.n_trials)


@dataclass
class CalibrationReport:
    null_dedup: ArmResult      # FAR under the deduped N the pipeline actually uses
    null_raw: ArmResult        # FAR under the honest raw config count
    power: ArmResult           # acceptance of a planted real edge (sanity / power)
    n_dedup: int
    n_raw: int


def _empty_counts() -> dict[str, int]:
    return {g: 0 for g in (*STATISTICAL_GATES, _ALL)}


def accumulate(
    frame: pd.DataFrame,
    candidates: list[CandidateStrategySpec],
    settings: Settings,
    *,
    n_surrogates: int,
    n_dedup: int,
    n_raw: int,
    min_gap: int = 288,
    power_draws: int = 20,
    alpha: float = 6.0,
    noise: float = 1.0,
    resamples: int = 250,
    seed: int = 0,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], int, int]:
    """Core Monte-Carlo loop (kept separate from ``calibrate`` so the CLI can shard
    candidates across processes and merge the returned counts). Returns
    (null_dedup, null_raw, power, n_null, n_power)."""
    cs = calibration_settings(settings, n_resamples=resamples)
    rng = np.random.default_rng(seed)
    null_d, null_r, power = _empty_counts(), _empty_counts(), _empty_counts()
    n_null = n_power = 0
    gates = (*STATISTICAL_GATES, _ALL)
    for candidate in candidates:
        lookback = int(candidate.params["lookback"])
        family = str(candidate.params["factor_family"])
        base = np.asarray(factor_signal(frame, family=family, lookback=lookback), dtype=float)
        gap = max(int(min_gap), int(candidate.max_feature_lookback_bars))
        for _ in range(n_surrogates):
            rotated = rotate_signal(base, rng, gap)
            pd_arm = gate_trial(frame, rotated, candidate, cs, effective_trials=n_dedup)
            pr_arm = gate_trial(frame, rotated, candidate, cs, effective_trials=n_raw)
            for g in gates:
                null_d[g] += int(pd_arm[g])
                null_r[g] += int(pr_arm[g])
            n_null += 1
        for _ in range(power_draws):
            planted = plant_signal(frame, rng, alpha=alpha, noise=noise)
            pp = gate_trial(frame, planted, candidate, cs, effective_trials=n_dedup)
            for g in gates:
                power[g] += int(pp[g])
            n_power += 1
    return null_d, null_r, power, n_null, n_power


def calibrate(
    frame: pd.DataFrame,
    candidates: list[CandidateStrategySpec],
    settings: Settings,
    *,
    n_surrogates: int = 100,
    n_dedup: int | None = None,
    n_raw: int | None = None,
    **kwargs,
) -> CalibrationReport:
    """Single-process convenience wrapper over :func:`accumulate`.

    ``n_dedup`` defaults to the number of candidates (each is one lineage — the honest
    deduped count for this batch); ``n_raw`` defaults to 12× that (the ledger's observed
    distinct-configs-per-lineage ratio), standing in for the hidden grid intensity."""
    n_dedup = n_dedup if n_dedup is not None else max(1, len(candidates))
    n_raw = n_raw if n_raw is not None else n_dedup * 12
    null_d, null_r, power, n_null, n_power = accumulate(
        frame, candidates, settings, n_surrogates=n_surrogates, n_dedup=n_dedup, n_raw=n_raw, **kwargs
    )
    return CalibrationReport(
        null_dedup=ArmResult(n_null, null_d),
        null_raw=ArmResult(n_null, null_r),
        power=ArmResult(n_power, power),
        n_dedup=n_dedup,
        n_raw=n_raw,
    )


# ── Power calibration: how strong must a real alpha be? ─────────────────────
#
# The FAR harness measures whether noise gets through (it shouldn't); this measures
# how strong truth must be to get through. Composites separate the two axes:
#   ALL_STAT   — standalone-measurable statistical gates (G1, G3, G5, G7)
#   ALL_ECON   — economic gates (G8 cost margin, G10 capacity)
#   PROD_X_PBO — the production blocking verdict modulo G2 (G2 is pool-relative and
#                fails closed on standalone trials, so the raw verdict is trivially
#                False here; "no blocking failure other than G2" is the honest proxy).

POWER_COMPOSITES: tuple[str, ...] = ("ALL_STAT", "ALL_ECON", "PROD_X_PBO")


@dataclass
class PowerTrial:
    alpha: float
    horizon: int
    gross_sharpe: float
    net_sharpe: float
    break_even_cost_bps: float
    actual_cost_bps: float
    factor_turnover: float
    avg_holding_period_bars: float
    oos_trade_count: int
    passes: dict[str, bool]
    failed_blocking: tuple[str, ...]


def power_trial(
    frame: pd.DataFrame,
    signal,
    candidate: CandidateStrategySpec,
    settings: Settings,
    *,
    effective_trials: int,
    alpha: float,
    horizon: int,
) -> PowerTrial:
    """One planted-edge trial through the FULL gate stack, reporting achieved market-unit
    metrics so 'how strong' reads in Sharpe, not in the alpha knob. Blocking failures are
    identified structurally: blocking rules emit status=='fail', advisory rules 'warn'."""
    gate, result = _evaluate(frame, signal, candidate, settings, effective_trials=effective_trials)
    status = {item.rule_id: item.status for item in gate.items}
    passes = {rule_id: (value == "pass") for rule_id, value in status.items()}
    passes["ALL_STAT"] = all(passes.get(g, False) for g in STANDALONE_AND_GATES)
    passes["ALL_ECON"] = all(passes.get(g, False) for g in ECONOMIC_GATES)
    failed_blocking = tuple(sorted(item.rule_id for item in gate.items if item.status == "fail"))
    passes["PROD_X_PBO"] = set(failed_blocking) <= {"G2"}
    return PowerTrial(
        alpha=float(alpha),
        horizon=int(horizon),
        gross_sharpe=float(result.metrics_gross.sharpe) if result.metrics_gross else 0.0,
        net_sharpe=float(result.metrics_primary.sharpe),
        break_even_cost_bps=float(result.break_even_cost_bps),
        actual_cost_bps=float(result.actual_cost_bps),
        factor_turnover=float(result.factor_turnover),
        avg_holding_period_bars=float(result.avg_holding_period_bars),
        oos_trade_count=int(result.oos_trade_count),
        passes=passes,
        failed_blocking=failed_blocking,
    )


def power_sweep(
    frame: pd.DataFrame,
    candidate: CandidateStrategySpec,
    settings: Settings,
    *,
    alphas,
    horizon: int,
    draws: int,
    effective_trials: int = 400,
    noise: float = 1.0,
    resamples: int = 250,
    seed: int = 0,
    regimes=None,
    regime_keep=None,
    regime_filter: bool = False,
) -> list[PowerTrial]:
    """Sweep planted-edge strength at one payoff horizon. One candidate spec suffices:
    the planted signal replaces the candidate's own signal — the spec only supplies
    interval/method/cost context — so alphas × draws control the Monte Carlo.

    When ``regimes``/``regime_keep`` are given, the edge is regime-concentrated (see
    :func:`plant_regime_concentrated_signal`); ``regime_filter=True`` additionally zeroes
    the signal outside the kept regimes (models ``regime_filter`` ON). Comparing the two
    isolates whether filtering recovers a regime-concentrated edge."""
    cs = calibration_settings(settings, n_resamples=resamples)
    rng = np.random.default_rng(seed)
    keep = list(regime_keep) if regime_keep else []
    mask = np.isin(np.asarray(regimes, dtype=object), keep) if (regimes is not None and keep) else None
    trials: list[PowerTrial] = []
    for alpha in alphas:
        for _ in range(draws):
            if mask is not None:
                sig = plant_regime_concentrated_signal(
                    frame, regimes, keep, rng, alpha=float(alpha), noise=noise, horizon=horizon
                )
                if regime_filter:
                    sig = np.where(mask, sig, 0.0)
            else:
                sig = plant_signal_with_horizon(frame, rng, alpha=float(alpha), noise=noise, horizon=horizon)
            trials.append(
                power_trial(frame, sig, candidate, cs, effective_trials=effective_trials, alpha=float(alpha), horizon=horizon)
            )
    return trials


def summarize_power(trials: list[PowerTrial], gates: tuple[str, ...]) -> list[dict]:
    """Per (horizon, alpha): mean achieved metrics, pass rates, blocking-failure shares
    (the failure-share ranking is what names the binding constraint at each strength)."""
    grouped: dict[tuple[int, float], list[PowerTrial]] = {}
    for trial in trials:
        grouped.setdefault((trial.horizon, trial.alpha), []).append(trial)
    rows: list[dict] = []
    for horizon, alpha in sorted(grouped):
        batch = grouped[(horizon, alpha)]
        n = len(batch)
        failure_share: Counter[str] = Counter()
        for trial in batch:
            failure_share.update(trial.failed_blocking)
        rows.append({
            "horizon": horizon,
            "alpha": alpha,
            "n": n,
            "gross_sharpe": sum(t.gross_sharpe for t in batch) / n,
            "net_sharpe": sum(t.net_sharpe for t in batch) / n,
            "break_even_cost_bps": sum(t.break_even_cost_bps for t in batch) / n,
            "turnover": sum(t.factor_turnover for t in batch) / n,
            "holding_bars": sum(t.avg_holding_period_bars for t in batch) / n,
            "oos_trades": sum(t.oos_trade_count for t in batch) / n,
            "rates": {g: sum(1 for t in batch if t.passes.get(g, False)) / n for g in gates},
            "blocking_failure_share": {g: c / n for g, c in failure_share.most_common()},
        })
    return rows


def minimum_detectable_alpha(rows: list[dict], gate: str, *, level: float = 0.5) -> dict | None:
    """Smallest swept alpha whose ``gate`` pass rate reaches ``level``, with the achieved
    gross/net Sharpe at that strength — so the answer reads in market units. None if the
    sweep never reaches ``level`` (the gate is beyond the swept range)."""
    for row in sorted(rows, key=lambda r: r["alpha"]):
        if row["rates"].get(gate, 0.0) >= level:
            return {
                "alpha": row["alpha"],
                "gross_sharpe": row["gross_sharpe"],
                "net_sharpe": row["net_sharpe"],
                "rate": row["rates"][gate],
            }
    return None
