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

## Regime repair shapes (2026-07-10, after the G7×G8 frontier finding)

Motivation: 0 of 1,820 `regime_mixing` near-misses held both positive margin and ≥100 OOS
trades — the hard filter's shape trades the gates against each other. Four repair shapes
measured at h48, band 0.4/0.15, bull-concentrated edge (13% of bars; "signed" plants an
*inverting* edge instead), α=1.0, um_futures:

| mode | netSR | grossSR | oosTrd | G1/STAT |
|---|---|---|---|---|
| hard (baseline repair) | 3.03 | 11.9 | 5,158 | 17% |
| **entry_only** | **3.15** | 12.1 | **5,639** | 17% |
| soft (w=0.25) | 2.61 | 11.8 | 5,588 | 8% |
| **signed** (inverting edge) | **11.16** | 25.5 | **15,896** | **100%** |

Findings: **entry_only weakly dominates hard** — same margin, ~9% more trades — strictly
better on the G7 axis at no G8 cost. **signed is the big unlock for inverting edges**: it
recovers the FULL uniform-edge performance (netSR 11.2 ≈ the uniform-plant 10.8) with the
full trade count, because sign-flipping harvests the edge in every regime instead of
discarding 87% of bars; G1 fires at 67% already at α=0.5 where hard shows 0%. **soft
underperformed here by construction** — the plant makes outside-regime bars *pure* noise
(soft's worst case); real regime_mixing candidates have weak-but-nonzero outside edge,
which is soft's target population. The decisive test for all shapes is the live metric:
the joint population (`cost_margin>0 AND oos_trades≥100` among regime_mixing), 0/1,820 at
baseline, must become >0 in subsequent rounds. G7 unchanged at 100.

**Implication for the cost-objective decision.** Optimizing the band per signal is clearly
worthwhile: cost-blocked short horizons are recoverable by searching tighter bands. The current
WS2 optimizer offers band pairs up to only 0.45/0.22 and gates them on a turnover diagnostic —
it should (a) extend the band range to ~0.6–0.8 entry, and (b) use a **cost-margin acquisition
objective** (push break-even past 2× realized cost) rather than only reacting to turnover. That
would move the factory from "only long horizons clear cost" to "cost-clearing bands found per
signal at any horizon."

## Factory post cost-objective measurement (2026-07-10)

The cost-margin band objective landed in commit `416b908` (extends `_LOCAL_TUNING_BANDS` to
0.60/0.25 and 0.80/0.40; offers bands on G8 deficit; ranks tighter bands first for
cost-blocked parents). Question for the live factory: **does this produce more Provisional
candidates, and do they clear cost more honestly?**

### Baseline (pre cost-objective; last discovery 2026-07-08)

Last factory discovery `cli_20260708_215506_b0a9f222` (~4.4h, trial_budget=400, workers=4,
LLM brief, full history). Note: that round predates WS0–WS2 meters/band *and* the
cost-objective — it is the pre-stack control, not an apples-to-apples band-only A/B.

| metric | value |
|---|---|
| Active Provisionals | **896** (all created 2026-07-08) |
| With `entry_band > 0` | **0** |
| Positive `cost_margin_bps` | **155 / 896 (17%)** |
| Near-misses that round | 899 |
| `cost_destroyed_edge` share | **255 / 899 (28.4%)** |
| Dominant near-miss | `regime_mixing` 491 (55%) |
| Live independent lineages | 580 (26 435 dev lineages archived) |
| Validated | 0 |

### Run protocol

- Command: `fm worker run-once --mode discovery` (factory args: `--trial-budget 400 --llm
  --brief <standing> --workers 4`).
- Code path verified editable: `_LOCAL_TUNING_BANDS = (0.30/0.15, 0.45/0.22, 0.60/0.25, 0.80/0.40)`.
- Primary run: `cli_20260710_102232_5ab94aa1` started 2026-07-10T02:22Z.
- Compare **this run's new Provisionals** and **this run_id's near-miss mix** to the 07-08
  baseline; also report post-round shelf totals (rechecks may demote).
- Success criteria (directional, not a hard gate): (1) new Provisionals with `entry_band>0`
  appear; (2) `cost_destroyed_edge` share falls vs 28%; (3) positive cost-margin share among
  new Provisionals rises vs 17%. "More Provisionals" is secondary — power_sweep already says
  real signals land as time-surviving Provisionals, not volume.

### Results — round `cli_20260710_102232_5ab94aa1` (completed 2026-07-10)

Wall time **5.37h** (02:22–07:44 UTC). Factory args: trial_budget=400, LLM brief, workers=4,
full history (~684k bars). Discovery backtests 546+640; pre-gate produced **352×2** repair
candidates of which **256×2 = 512** were `local_grid_tuning` (the cost-objective band path);
merged repairs 100+85. Terminal holdout: **0** round-gate production survivors (as the power
sweep ceiling predicts). Live lineages 580 → **870**.

| metric | baseline 07-08 evening | this round (cost-objective stack) | Δ |
|---|---|---|---|
| New Provisionals (created that day) | ~442–454 / round | **475** | +~5–7% |
| Active shelf after | 896 | **1371** (896 old + 475 new) | +475 |
| With `entry_band > 0` (new) | **0** | **24** (all `local_grid_tuning`) | new channel |
| Band pairs used | — | 0.80/0.40 ×14, 0.60/0.25 ×10 | measured region |
| Positive `cost_margin_bps` (new) | 17% shelf-wide | **102/475 (21.5%)** | +4.5pp |
| **Banded** new: pos cost margin | — | **18/24 (75%)**, med **+24 bps** | quality |
| Unbanded new: pos cost margin | — | 84/451 (18.6%), med **−12 bps** | control |
| Near-misses | 899 | 1371 | more pool |
| `cost_destroyed_edge` share | **28.4%** | **20.1%** | **−8.3pp** |
| `regime_mixing` share | 54.6% | 60.0% | +5.4pp (now binds first) |
| Validated / production_passed | 0 | 0 | unchanged |

Optimizer selected 8 research survivors per symbol group for combos; several carry
`cost_margin` in their evidence reasons (netSR≈0.7–1.5). HardScore stayed 0-positive —
Provisional admission, not production validation.

### Interpretation

1. **Cost-objective is working as a *quality* lever, not a volume flood.** New Provisional
   count is only slightly above the pre-stack baseline (+~5–7%). The signal is that **banded
   local-grid children land with honest positive cost margin 4× as often as unbanded peers**
   (75% vs 19%), and exclusively in the power-sweep-cleared pairs (0.60/0.25, 0.80/0.40).

2. **`cost_destroyed_edge` share fell 28% → 20%.** Cost is less often the *primary* near-miss
   reason; `regime_mixing` is now clearly first (60%). That matches the power-sweep finding:
   once bands can clear G8, the binding constraint moves to regime-conditional structure
   (exactly what the standing brief already targets).

3. **No instant Validated** — consistent with the sobering ceiling (need netSR≈10 for
   single-shot production on this multiplicity). Discoveries should still accrue as
   time-surviving Provisionals on the paper clock.

4. **Caveat on attribution.** The 07-08 baseline predated WS0–WS2 (honest meters + band) as
   well as `416b908`. The **24 banded Provisionals** isolate the cost-objective contribution;
   the shelf-wide cost-share drop compounds meters + band + objective. Not a pure A/B of the
   acquisition term alone.

### Follow-ups

- Factory `discovery_due` will not re-fire until ≥`min_new_days` of new bars (extents now
  recorded to 2026-07 data tip). For another budgeted search without waiting on data, run
  `fm mine run --trial-budget 400 --llm --brief … --workers 4` directly.
- Next rounds: watch whether banded Provisionals recheck-survive, and whether
  `regime_mixing` repair share falls after the brief continues to push regime gating.
- Optional: update the standing brief to note that **tight bands (entry 0.6–0.8) are now
  searched by the optimizer** so LLM hypotheses need not re-invent fee-clearing hysteresis.

## Regime matched-parent selection (blocked — empty eligible set)

Attempted protocol on the 823 `regime_mixing` near-misses from
`cli_20260710_102232_5ab94aa1` (script: `scripts/regime_matched_parent_select.py`, report:
`docs/artifacts/regime_matched_parent_select.json`).

**Selection criteria (all required, gates unrelaxed):** positive cost margin; original OOS
trades ≥100 (`oos_trade_count`, G7); fixed-horizon IC stability; discovery/validation regime
label + sign consistency. Then deterministic matched-parent replay arms: parent / hard
filter / signed hard / signed entry-only / signed soft weight. Success would require no-op
repair = 0, real Δ net Sharpe + Δ cost margin on repair-validation, final OOS trades ≥100,
regime/sign stable discovery→validation — **without** relaxing DSR, FDR, or 2× cost.

### Funnel (strict)

| stage | n |
|---|---|
| pool `regime_mixing` | **823** |
| cost_margin_bps > 0 | 191 |
| oos_trade_count ≥ 100 | 328 |
| **cost AND trades** | **0** |
| IC stable / regime-sign (downstream) | 0 |
| **eligible for replay** | **0** |

### Why empty (bipartite structure — fail loud)

The 823 do **not** form a continuum; they split into two non-overlapping arms:

| arm | n | cost margin | OOS trades |
|---|---|---|---|
| pos cost margin | 191 | >0 (some huge, sparse-trade artifacts) | **max 30, median 0** |
| oos trades ≥100 | 328 | **all negative** (max ≈ −9.4 bps, med ≈ −12.2) | ≥100 |

So **no parent simultaneously has (a) room above the 2× cost bar and (b) enough OOS trades
to re-test a regime filter without immediately failing G7.** That is exactly the regime-
filter tradeoff: hard filtering concentrated regimes cuts churn *and* sample size; the
high-margin survivors here are already trade-starved, and the trade-rich ones are still
cost-underwater.

Replay was **not** started — inventing parents by relaxing OOS trades or cost margin would
violate the stated success criteria. When a future round produces a non-empty eligible set,
the same script encodes the five arm overlays; signed entry-only / soft-weight still need
signal-path support beyond today's hard `regime_filter` zeroing.
