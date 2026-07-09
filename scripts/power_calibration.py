"""How strong must a REAL alpha be to survive the gate system?

Plants clairvoyant edges of controlled strength (alpha) and payoff horizon into real
market data and runs them through the full gate stack. Reports, per (horizon, alpha):
achieved gross/net Sharpe (market units, not the knob), per-gate pass rates, the two
axis composites (statistical vs economic), and which blocking gate binds. The minimum
detectable alpha (MDA) is the smallest strength where a gate's pass rate clears 50%/80%.

Reads the parquet warehouse READ-ONLY; writes nothing to any store.

  .venv/bin/python scripts/power_calibration.py --tail 50000 --draws 20 --workers 4

Note: G2/PBO is pool-relative (CSCV over a candidate batch) and fails closed on
standalone trials, so the production verdict is reported as PROD_X_PBO — "no blocking
failure other than G2". G7 (trade count) depends on horizon and window length, which is
exactly why it appears as a binding constraint for slow signals.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

from factor_mining.calibration import (
    POWER_COMPOSITES,
    default_candidates,
    minimum_detectable_alpha,
    power_sweep,
    summarize_power,
)
from factor_mining.config import load_settings
from factor_mining.data.loader import load_frame

REPORT_GATES: tuple[str, ...] = ("G1", "G3", "G5", "G7", "G8", "G10", *POWER_COMPOSITES)
REALISTIC_GROSS_SR = (1.0, 4.0)  # live near-miss diagnostics put real candidates here


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _print_horizon(rows: list[dict], horizon: int) -> None:
    hrows = [r for r in rows if r["horizon"] == horizon]
    print(f"\n== horizon {horizon} bars ==")
    header = (
        f"  {'alpha':>6} {'grossSR':>8} {'netSR':>8} {'brkeven':>8} {'hold':>6} {'oosTrd':>7}"
        f"  {'G1':>5} {'G3':>5} {'G5':>5} {'G7':>5} {'STAT':>5} | {'G8':>5} {'G10':>5} {'ECON':>5} | {'PROD*':>6}  binding"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in hrows:
        rates = r["rates"]
        top = next(iter(r["blocking_failure_share"].items()), None)
        binding = f"{top[0]} ({top[1]:.0%})" if top else "—"
        band = " *" if REALISTIC_GROSS_SR[0] <= r["gross_sharpe"] <= REALISTIC_GROSS_SR[1] else "  "
        print(
            f"  {r['alpha']:>6.3f} {r['gross_sharpe']:>8.2f} {r['net_sharpe']:>8.2f}"
            f" {r['break_even_cost_bps']:>8.1f} {r['holding_bars']:>6.0f} {r['oos_trades']:>7.0f}"
            f"  {rates['G1']:>5.0%} {rates['G3']:>5.0%} {rates['G5']:>5.0%} {rates['G7']:>5.0%}"
            f" {rates['ALL_STAT']:>5.0%} | {rates['G8']:>5.0%} {rates['G10']:>5.0%}"
            f" {rates['ALL_ECON']:>5.0%} | {rates['PROD_X_PBO']:>6.0%}{band}{binding}"
        )
    print("  (* = achieved gross Sharpe in the realistic band "
          f"{REALISTIC_GROSS_SR[0]:.0f}–{REALISTIC_GROSS_SR[1]:.0f}; "
          "binding = most-failed blocking gate at that strength)")

    for gate in ("G1", "ALL_STAT", "ALL_ECON", "PROD_X_PBO"):
        for level in (0.5, 0.8):
            mda = minimum_detectable_alpha(hrows, gate, level=level)
            if mda is None:
                print(f"  MDA {gate:>10} @{level:.0%}: not reached in swept range")
            else:
                print(
                    f"  MDA {gate:>10} @{level:.0%}: alpha={mda['alpha']:.3f} "
                    f"(achieved grossSR≈{mda['gross_sharpe']:.2f}, netSR≈{mda['net_sharpe']:.2f})"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail", type=int, default=50_000)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--market", default="spot")
    ap.add_argument("--alphas", default="0,0.01,0.02,0.05,0.1,0.2,0.5,1.0")
    ap.add_argument("--horizons", default="1,48")
    ap.add_argument("--draws", type=int, default=20, help="Planted draws per (horizon, alpha) cell.")
    ap.add_argument("--n-trials", type=int, default=400, help="Effective trial count N fed to G1/G3 (a round's budget).")
    ap.add_argument("--entry-band", type=float, default=0.0, help="Hysteresis entry band on the planted signal (0 = off).")
    ap.add_argument("--exit-band", type=float, default=0.0, help="Hysteresis exit band (< entry).")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    settings = load_settings()
    frame = load_frame(settings, symbol=args.symbol, market=args.market, tail=args.tail)
    candidate = default_candidates(
        families=("momentum",), lookbacks=(12,), symbol=args.symbol, market=args.market
    )[0]
    if args.entry_band > 0.0 or args.exit_band > 0.0:
        candidate.params["entry_band"] = args.entry_band
        candidate.params["exit_band"] = args.exit_band
        print(f"Hysteresis band on planted signal: entry={args.entry_band}, exit={args.exit_band}")
    alphas = _parse_floats(args.alphas)
    horizons = _parse_ints(args.horizons)
    cells = [(h, a) for h in horizons for a in alphas]
    print(
        f"Loaded {len(frame):,} bars {args.symbol}/{args.market}; "
        f"{len(cells)} cells × {args.draws} draws = {len(cells) * args.draws:,} planted backtests "
        f"(N={args.n_trials} fed to G1/G3)"
    )

    common = dict(settings=settings, draws=args.draws, effective_trials=args.n_trials)
    if args.workers <= 1:
        trials = []
        for idx, (horizon, alpha) in enumerate(cells):
            trials.extend(power_sweep(frame, candidate, alphas=[alpha], horizon=horizon, seed=args.seed + idx, **common))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(power_sweep, frame, candidate, alphas=[alpha], horizon=horizon, seed=args.seed + idx, **common)
                for idx, (horizon, alpha) in enumerate(cells)
            ]
            trials = [t for f in futures for t in f.result()]

    rows = summarize_power(trials, REPORT_GATES)
    for horizon in horizons:
        _print_horizon(rows, horizon)
    print(
        "\n  Reading it: STAT is the statistical axis (G1·G3·G5·G7), ECON the economic axis"
        "\n  (G8·G10), PROD* the production blocking verdict ex-PBO. Where STAT rises with"
        "\n  strength but ECON stays flat, the gate system is cost-bound, not evidence-bound.\n"
    )


if __name__ == "__main__":
    main()
