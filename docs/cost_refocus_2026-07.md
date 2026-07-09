# Cost-Model Refocus — factor_mining (2026-07-09)

The power-calibration harness measured the pipeline as **cost-bound, not evidence-bound**:
the statistical axis showed a clean dose-response while the economic gate (G8) never passed
at any planted strength. This note records the refocus — fix the meters, ground the costs,
shape the execution — with before/after numbers. Commits: WS0 `5553971`, WS1 `af66b39`,
WS2 `d7003da`, WS3 (this note).

## WS0 — the meters were mis-defined (the biggest single lever)

Two definitional bugs sat inside G8's own inputs, so part of the "bottleneck" was
measurement artifact.

| quantity | before | after |
|---|---|---|
| `break_even_cost_bps` (engine.py:679) | `max(0, NET_pnl) / turnover` — margin *above* current costs; a strategy paying exactly its edge read 0 | `max(0, NET_pnl + cost_paid) / turnover` — the true **gross** break-even level |
| `actual_cost_bps` (engine.py:655) | per-bar mean of the cost *rate*, idle bars included → ~6bps floor regardless of trading | **turnover-weighted**: total cost paid / total turnover |

Net effect of the break-even bug: G8 (`break_even > 2× actual`) effectively demanded gross
break-even **> ~3×** cost — a full multiple harsher than its stated intent. First direct
unit tests of the cost arithmetic (`tests/test_cost_metrics.py`) now pin the anchor
identity: a strategy whose gross PnL equals its cost paid has `break_even ==` its realized
cost rate.

**Cut-over caveat:** `break_even_cost_bps` and `actual_cost_bps` (and the `cost_margin_bps`
derived from them in `research_gate`/`near_miss`) change meaning as of 2026-07-09. Experiments
archived before then carry old-semantics values; do not compare pre- and post-cut-over cost
margins. No schema change; `backtest_master` is unaffected (it reproduces only
return/sharpe/drawdown).

## WS1 — per-market cost truth

Costs were global: spot and USDT-M futures shared `taker_bps = 5.0`, so spot was priced at
the futures fee — wrong in the unfavorable direction. `CostConfig.per_market` now carries
partial overrides, resolved per backtest by `costs_for_market(settings, market)` on
`candidate.market`. `default.yaml` ships Binance **regular-tier** fees (spot ~10bps taker,
um_futures ~5bps, 2026-07) as a **user-editable placeholder** — replace with your real
VIP/BNB-discount tier and every backtest re-prices. `apply_trade_overrides` still layers a
global scenario knob on top (it propagates into per-market entries so "what if taker were X"
reaches every market).

## WS2 — hysteresis conviction band

A continuous tanh signal is always-in-market and spreads its edge over thousands of tiny
trades that costs eat. `_apply_hysteresis_band` (engine.py) enters only when
`|signal| ≥ entry_band`, holds (sign flips allowed) until `|signal| ≤ exit_band`, then goes
flat — fewer, larger trades that can clear the round-trip cost. Read from
`candidate.params`; `entry=exit=0` is a pass-through (identity), so it is off by default and
existing backtests are byte-identical. The optimizer aims at it: the local-grid tuner offers
coupled band pairs only to turnover-heavy / cost-dragged parents (low-turnover grids
unchanged), and the combo turnover-control path emits bands under the same diagnostic. The
LLM research brief now asks for explicit conviction bands.

## WS3 — re-measure (power sweep, tail 20k, um_futures 5bps, N=400)

The economic axis, previously flat at every strength, now responds. Horizon 48, by planted
strength (achieved netSR in parens):

| α (netSR) | break-even | G8 off→on | ECON off→on | PROD* off→on |
|---|---|---|---|---|
| 0.5 (≈4) | 7.8 → 8.9 | 0% → 0% | 0% → 0% | 0% → 0% |
| 1.0 (≈15) | 11.7 → **13.0** | 12% → **100%** | 12% → **100%** | 12% → **100%** |

At α=1 the band drops the holding period from always-in to ~27 bars and pushes break-even
past the 2×-cost bar, so a strong signal now clears the **full production gate**. Horizon 144
shows the same direction (G8 94%→100% at α=1; 6%→19% at α=0.5).

**What is now the binding constraint.** With honest meters and the band available, a
signal that reaches netSR≈15 passes both axes. Below that, the binding gate is G1/G3
(statistical power on the 20k-bar window), not cost — the pre-refocus ordering is inverted
for banded strong signals. The remaining cost-bound regime is sub-~50-bar profiles at
moderate strength.

**MDA is still high on this window** (~α=1) because 20k bars ≈ 3% of history; statistical
power at fixed Sharpe grows ~√T, so the definitive full-history sweep will lower the
statistical MDA. The cost axis, however, is demonstrably no longer the wall.

**FAR unaffected** (verified): null-arm statistical FAR stays G1 0% / G3 1.1% / G5 0.4% /
ALL 0% after the cost changes — expected, since G8 is not in the FAR composite and costs only
hurt noise.

## Environment gotcha (not in the tree)

The venv held a stale **non-editable** copy of `factor_mining` in site-packages, so
`scripts/` entry points ran old code while pytest ran source — the first "post-fix"
checkpoint showed no change until `uv pip install -e . --no-deps` was re-run. Re-check this
after any `uv sync`; a faithful script-mode import must resolve to the source tree.

## Follow-ups

- Run the definitive full-history power sweep (`--tail 200000`) to get the real statistical
  MDA now that costs are unblocked.
- Set the real fee tier in `default.yaml` `costs.per_market`.
- Watch the next factory rounds: `cost_destroyed_edge` near-miss share should fall and the
  first survivors with honestly-positive cost margin should appear.
