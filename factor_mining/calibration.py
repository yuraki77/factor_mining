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
# conflate the two axes and corrupt the false-acceptance-rate number. Excluded by design.
STATISTICAL_GATES: tuple[str, ...] = ("G1", "G2", "G3", "G5", "G7")
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


def plant_signal(frame: pd.DataFrame, rng: np.random.Generator, *, alpha: float, noise: float) -> np.ndarray:
    """Power arm: a signal with genuine (clairvoyant) edge over the engine's next-bar open
    return. The engine trades ``signals.shift(1)`` against ``open[t+1]/open[t]-1``, so the
    signal at ``t`` must predict the return realised over ``[t+1, t+2]`` — i.e. ``fwd`` twice
    shifted. Deliberately look-ahead: it is a positive control proving the gates *can*
    accept a real edge, not a strategy."""
    open_ = pd.Series(frame["open"].to_numpy(dtype=float))
    fwd = open_.shift(-1) / open_ - 1.0
    target = fwd.shift(-1)
    std = float(target.std()) or 1.0
    z = ((target - float(target.mean())) / std).fillna(0.0).to_numpy()
    return np.tanh(alpha * z + noise * rng.standard_normal(len(frame)))


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


def gate_trial(
    frame: pd.DataFrame,
    signal,
    candidate: CandidateStrategySpec,
    settings: Settings,
    *,
    effective_trials: int,
) -> dict[str, bool]:
    """One backtest → gatecheck under trial count ``effective_trials``. Returns per-gate
    pass booleans (condition held) plus ``ALL_STAT`` = all statistical gates passed.

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
    status = {item.rule_id: item.status for item in gate.items}
    passed = {g: (status.get(g) == "pass") for g in STATISTICAL_GATES}
    passed[_ALL] = all(passed[g] for g in STATISTICAL_GATES)
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
