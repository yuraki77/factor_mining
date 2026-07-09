"""Measure the statistical gates' false-acceptance rate under a circular-rotation null.

Answers two questions with numbers:
  1. Is the gate stack's statistical axis a calibrated FAR control (accepts noise at ≤ q)?
  2. Does the lineage-deduped trial count (bias-audit finding A) inflate that FAR?

Reads the real parquet warehouse READ-ONLY; writes nothing to any store or DB.

  .venv/bin/python scripts/far_calibration.py --tail 80000 --surrogates 200 --workers 4

Precision scales with (candidates × surrogates); the printed Wilson CIs show it. The
default is a quick look — a tight estimate at the nominal 5% wants thousands of null trials.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

from factor_mining.calibration import (
    STATISTICAL_GATES,
    ArmResult,
    CalibrationReport,
    accumulate,
    default_candidates,
)
from factor_mining.config import load_settings
from factor_mining.data.loader import load_frame

_ALL = "ALL_STAT"
_GATE_LABELS = {
    "G1": "G1 DSR-prob≥.95", "G2": "G2 PBO<thr", "G3": "G3 FDR-p≤.05",
    "G5": "G5 Sharpe-CI>0", "G7": "G7 trades≥100", _ALL: "ALL statistical",
}


def _merge(parts: list[tuple[dict, dict, dict, int, int]]) -> tuple[dict, dict, dict, int, int]:
    keys = (*STATISTICAL_GATES, _ALL)
    nd = {g: 0 for g in keys}
    nr = {g: 0 for g in keys}
    pw = {g: 0 for g in keys}
    n_null = n_power = 0
    for d, r, p, nn, np_ in parts:
        for g in keys:
            nd[g] += d[g]; nr[g] += r[g]; pw[g] += p[g]
        n_null += nn; n_power += np_
    return nd, nr, pw, n_null, n_power


def _print_report(report: CalibrationReport, *, nominal: float) -> None:
    print(f"\nStatistical-gate false-acceptance rate (null = circular-rotated signal)")
    print(f"  null trials/arm: {report.null_dedup.n_trials:,}   power trials: {report.power.n_trials:,}")
    print(f"  N (deduped, pipeline): {report.n_dedup:,}   N (raw config count): {report.n_raw:,}")
    print(f"  nominal target: {nominal:.0%}\n")
    header = f"  {'gate':<18} {'FAR deduped-N':>22} {'FAR raw-N':>22} {'power(accept)':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for g in (*STATISTICAL_GATES, _ALL):
        d_lo, d_hi = report.null_dedup.ci(g)
        r_lo, r_hi = report.null_raw.ci(g)
        d = f"{report.null_dedup.rate(g):6.1%} [{d_lo:.1%},{d_hi:.1%}]"
        r = f"{report.null_raw.rate(g):6.1%} [{r_lo:.1%},{r_hi:.1%}]"
        flag = "  <-- exceeds nominal" if (g in ("G1", "G3", _ALL) and report.null_dedup.ci(g)[0] > nominal) else ""
        print(f"  {_GATE_LABELS[g]:<18} {d:>22} {r:>22} {report.power.rate(g):>15.1%}{flag}")
    print()
    print("  Reading it: FAR deduped-N is the rate the live pipeline runs at. If G1/G3/ALL")
    print("  deduped-N sit at or below nominal, the FAR control holds and finding A is benign.")
    print("  If deduped-N exceeds nominal while raw-N does not, the dedup inflates the FAR.")
    print("  Power should be high — a harness that rejects everything is broken, not calibrated.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail", type=int, default=50_000, help="Use the last N bars of data.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--market", default="spot")
    ap.add_argument("--surrogates", type=int, default=100, help="Null rotations per candidate.")
    ap.add_argument("--power-draws", type=int, default=20, help="Planted-signal draws per candidate.")
    ap.add_argument("--workers", type=int, default=1, help="Processes to shard candidates across.")
    ap.add_argument("--n-raw-mult", type=int, default=12, help="raw N = deduped N × this (ledger ratio).")
    ap.add_argument("--nominal", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    settings = load_settings()
    frame = load_frame(settings, symbol=args.symbol, market=args.market, tail=args.tail)
    candidates = default_candidates(symbol=args.symbol, market=args.market)
    n_dedup = max(1, len(candidates))
    n_raw = n_dedup * args.n_raw_mult
    print(f"Loaded {len(frame):,} bars for {args.symbol}/{args.market}; "
          f"{len(candidates)} candidates × {args.surrogates} surrogates × 2 N-arms "
          f"= {len(candidates) * args.surrogates * 2:,} null backtests")

    common = dict(settings=settings, n_surrogates=args.surrogates, n_dedup=n_dedup, n_raw=n_raw,
                  power_draws=args.power_draws)
    if args.workers <= 1:
        parts = [accumulate(frame, candidates, seed=args.seed, **common)]
    else:
        # Shard candidates; each worker gets its own seed so the null draws differ.
        shards = [candidates[i::args.workers] for i in range(args.workers)]
        shards = [s for s in shards if s]
        with ProcessPoolExecutor(max_workers=len(shards)) as ex:
            futures = [ex.submit(accumulate, frame, shard, seed=args.seed + i, **common)
                       for i, shard in enumerate(shards)]
            parts = [f.result() for f in futures]

    nd, nr, pw, n_null, n_power = _merge(parts)
    report = CalibrationReport(
        null_dedup=ArmResult(n_null, nd), null_raw=ArmResult(n_null, nr),
        power=ArmResult(n_power, pw), n_dedup=n_dedup, n_raw=n_raw,
    )
    _print_report(report, nominal=args.nominal)


if __name__ == "__main__":
    main()
