# First Full-History Power Sweep (post cost-refocus) — 2026-07-10

Measurement-only run to answer: now that the cost meters are honest and the hysteresis
band exists, can the gate system pass realistic-strength signals, and what is the binding
constraint? No mining logic was changed — only the calibration harness
(`scripts/power_calibration.py`, `factor_mining/calibration.py`).

**Setup:** BTCUSDT **um_futures**, conservative default cost (5bps taker), **200,000 bars**
(2024-08-06 → 2026-07-01), N=400 multiplicity, 12 draws/cell, clairvoyant planted edge.
Reported netSR is what a *perfect* predictor of the given strength achieves after costs — an
upper bound on a real same-IC signal; the alpha where a gate reaches 50% is that gate's MDA.

## Results

| Config | Stat MDA (G1≥50%) | G8 / cost | Full gate (PROD*) | turnover | hold (bars) |
|---|---|---|---|---|---|
| A. Band OFF (h24/48/96) | α=0.75 · netSR≈5–6 | **0% everywhere** | 0% | 0.014–0.044 | ~always-in |
| B. Band ON, h24 | α=0.75 · netSR 7.0 | 0% | 0% | 0.034 | 18 |
| B. Band ON, h48 | α=0.75 · netSR 7.9 | 0% (break-even 10.8) | 0% | 0.021 | 27 |
| B. Band ON, h96 | α=0.50 · netSR 4.4 | **100% @α=1** (netSR 9.7, be 13.0) | **100% @α=1** | 0.013 | 42 |
| C. Regime-conc, filter OFF | — | 0% | 0% | **0.21** · netSR **−127** | 9 |
| D. Regime-conc, filter ON | 33% @α=1 (h48) | 0% | 0% | **0.007** · netSR **+3** | 10 |

Band = entry 0.4 / exit 0.15. Regime-concentrated = clairvoyant edge in the HMM `bull`
regime (13% of bars), pure noise elsewhere; filter ON zeroes the noise-regime bars.

## Findings

**Cost is no longer absolute, but still binds most of the space.** Band OFF: G8 = 0% at every
horizon and strength — the always-in signal churns (12–25k trades) and fees crush gross SR
25→net 1. Band ON: G8 unblocks only at **h96, α=1** (netSR 9.7, break-even 13 > the ~11 bar);
h24/h48 stay cost-blocked because a *fixed* 0.4/0.15 band under-holds at short horizons.

**Statistics binds first (lower), cost binds second (higher).** G1 fires at **netSR≈5–6** (the
N=400 multiplicity floor); G8 needs **netSR≈10**. In the netSR 5–10 band a signal is
statistically real but cost-blocked — so cost, not statistics, is the *final* binding
constraint. Both clear only above netSR≈10 at long horizons.

**The regime filter is a validated, essential lever for regime-concentrated edges.** Without
it, a bull-concentrated edge is destroyed by noise-regime churn (netSR −127, turnover 0.21).
With it, the edge is recovered (netSR +3, turnover 0.007 — a 30× cut, STAT 0%→33%). A uniform
clairvoyant edge can't show this (filtering only removes good bars); the concentrated plant is
the honest test.

**The sobering ceiling.** The realistic-strength region (gross SR 1–4) passes *nothing* in any
config — single-shot validation needs netSR≈10, far above real factor Sharpes (~1–2). This is
the N=400 multiplicity tax on one 139-day holdout, not a cost problem. The right mechanism for
real signals is the **Provisional ladder** (accumulate OOS confidence over calendar time),
which the factory already has — expect discoveries as time-surviving Provisionals, rarely as
instant Validated.

## Answers

1. **Cost still a bottleneck?** Partially — an absolute wall without the band, unblocked only at
   long horizons with the band. Still the binding economic constraint for h24/h48 and moderate
   strength.
2. **Statistical power the binding constraint?** No — cost binds at a higher strength (netSR≈10)
   than statistics (netSR≈5–6). Cost is the final gate for the netSR 5–10 band.
3. **Proceed or adjust?** Proceed with: (a) bias toward long horizons (h96+); (b) rely on the
   optimizer's per-signal band search, not a fixed band, for short horizons; (c) keep regime
   conditioning — the filter is essential for concentrated edges.

## Caveats

- Clairvoyant plants: netSR is an upper bound on a real same-IC signal.
- HMM labeled **0% bear** on this window (convergence warning), so the regime test used
  `bull` only. The filter *mechanism* result is robust to label quality.
- N=400 directly sets the statistical bar; a smaller per-round budget lowers it (the earlier
  FAR audit showed N barely affects the false-acceptance rate, so the budget is a power/throughput
  knob, not a safety knob).

## Deeper re-runs (2026-07-10)

**Deep 1 — h96 band-ON cost crossing (fine alphas, 24 draws).** Statistics clears at
**netSR 5.7** (α=0.6, 100%); cost (G8) transitions sharply α=0.8 (38%) → **α=0.9 (96%,
netSR 9.0, break-even 12.7)** → α=1.0 (100%, netSR 10.0). Full production gate (PROD*) clears
at α=0.9. So the stat→cost gap at the working horizon is **netSR 5.7 → 9.0**, tighter than the
coarse estimate.

**Deep 2 — h48 band-tightness sweep at α=1.0 (the decision experiment).** A tighter band
UNBLOCKS cost at the short horizon the fixed band failed:

| band (entry/exit) | break-even | G8 | netSR | turnover | oosTrd |
|---|---|---|---|---|---|
| 0.4 / 0.15 | 10.9 | **0%** | 25.3 | 0.025 | 15.9k |
| 0.5 / 0.20 | 11.9 | 12% | 25.3 | 0.023 | 13.8k |
| **0.6 / 0.25** | 13.3 | **100%** | 25.2 | 0.020 | 11.5k |
| 0.7 / 0.30 | 15.3 | 100% | 25.1 | 0.017 | 9.2k |
| 0.8 / 0.40 | 18.3 | 100% | 24.8 | 0.014 | 6.5k |

Break-even rises monotonically (10.9 → 18.3) while netSR is essentially flat (25.3 → 24.8,
−2%). **The h48 cost block was a fixed-band artifact, not structural** — tightening removes
low-conviction churn almost for free. Band tightness is a strong, monotone, near-costless lever
on cost margin.

**Implication for the cost-objective decision.** Optimizing the band per signal is clearly
worthwhile: cost-blocked short horizons are recoverable by searching tighter bands. The current
WS2 optimizer offers band pairs up to only 0.45/0.22 and gates them on a turnover diagnostic —
it should (a) extend the band range to ~0.6–0.8 entry, and (b) use a **cost-margin acquisition
objective** (push break-even past 2× realized cost) rather than only reacting to turnover. That
would move the factory from "only long horizons clear cost" to "cost-clearing bands found per
signal at any horizon."
