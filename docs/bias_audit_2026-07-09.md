# Bias Audit — factor_mining (2026-07-09)

Scope: read-only audit for the six classic backtest-validity failures — look-ahead bias,
overfitting, multiple testing, train/test pollution, factor decay, survivorship bias. No
code changes. Line numbers refer to the tree at commit `6e2ff82`. Ledger figures are from
the production store on 2026-07-09 (post go-live reset).

Headline verdict: the pipeline has already been through a serious quant-correctness
remediation (the `Q*`/`A*`/`B*`/`C*` commit series and the many "prior code leaked X, now
fixed" comments). Five of the six lenses are verifiably clean. The one residual —
within-lineage selection intensity hidden from the deflation count `N` — is real but
concentrated, largely a dev-era artifact already curtailed by the trial budget, and does
**not** touch the certifying gate. It is documented here as a known limitation rather than
patched.

---

## 1. Verified clean (traced to code, not assumed)

**Look-ahead — signals & features.** Every indicator in `factors/engineering.py` and every
signal in `mining.py:factor_signal` (mining.py:558-592) is causal (`rolling`/`ewm`/positive
`shift`/`diff`/`cumsum`); no global `.mean()/.std()` applied historically, no `shift(-n)`,
no full-sample `.rank(pct=True)`. This is what makes the "compute features once on the full
frame, then slice into splits" pattern safe — it was checked specifically because that
pattern leaks if *any* feature is non-causal.

**Look-ahead — execution/IC alignment.** The signal is lagged (`signals.shift(1)`) and earns
next-bar open-to-open returns (`backtest/engine.py:396-397`); IC is computed on the *same
lagged* signal (engine.py:514-515), so it is not inflated by contemporaneous alignment.

**Look-ahead — regime labels.** The HMM is fit on an initial **prefix only**
(`frame.iloc[:fit_rows]`), predicted with the causal `rolling_forward_regime`, the fit prefix
is masked to `"unknown"`, and the series is `.shift(1)` lagged (`pipeline.py:5244-5261`). A
prior full-sample rank look-ahead was already fixed to `expanding().rank()`
(`stats/regime.py:15-18`).

**Look-ahead — funding.** `merge_funding_to_frame` (loader.py:389) uses `ffill` only; a prior
`.bfill()` that leaked the first future funding rate backward was removed (loader.py:401-405).
Pre-first-event bars are 0.0.

**Train/test pollution.** Splits are chronological 60/20/20 (discovery / repair-validation /
final-OOS; `pipeline.py:272-274`) with purge+embargo gaps carved between them (Q10,
`pipeline.py:2507-2528`). The final-OOS window is always the chronological tail — regime-aware
shifting of the test window was explicitly rejected as look-ahead (Q7, `pipeline.py:2491`).
The holdout is a **single terminal evaluation** of validation-selected survivors, untouched
in-round (`pipeline.py:1560-1577`).

**Multiple testing.** The DSR expected-max penalty is applied with the annualization bug fixed
(the prior code under-stated the haircut by `sqrt(periods_per_year)` ≈ 324× on 5m bars;
`stats/metrics.py:244-249`). FDR is family-stratified Benjamini-Hochberg with the cumulative
cross-round family trial count as `n_tests` (Q15). `N` is the distinct-lineage count, not raw
evaluations — see §2 for the tradeoff this makes.

**Survivorship.** `data/panel.py` encodes a point-in-time universe: NaN = "not listed / delisted
at t", with a `listing_dates` filter whose docstring cites avoiding survivorship bias
(panel.py:8-10, 70-72). Note the live universe is BTC+ETH only, so this machinery is not
currently exercised (see §3).

**Factor decay.** An IC decay-curve diagnostic feeds the evidence gate (`evidence.py:259
_decay_quality`, gatecheck `decay_curve_supported`), and the live recheck/demotion loop retires
survivors whose walk-forward edge decays over calendar time.

---

## 2. Residual finding — within-lineage selection intensity is hidden from `N`

**Mechanism.** Grid variants (`c_grid_*`), pre-gate repairs (`c_pre_*`), and optimizer offspring
inherit their root's `lineage_id`, so the deflation count `N` (DSR expected-max trials and FDR
family count) does not grow with how hard a single hypothesis is tuned. Selecting the best of
50 correlated grid configs contributes 1 to `N`, not 50.

**Quantified exposure (production ledger, live `trials` + `trials_archive`, 2026-07-09):**

- 432,762 raw evaluations → 27,015 distinct lineages (**16.0×** collapse); 337,123 distinct
  configs actually built.
- Composition of raw evaluations: 56.0% grid, 27.6% root re-eval, 15.9% pre-gate repair, 0.5%
  optimizer/repair.
- The collapse is **pathologically concentrated, not broad**: median and p99 configs-per-lineage
  are both **1** — i.e. >99% of lineages are a single config evaluated once, for which dedup is
  exactly honest. The 16× average is driven by a handful of monster lineages (max **295,058**
  configs collapsed into one), the unbudgeted dev-era mean_reversion runs. In the archive, the
  **top 1% of lineages hold 92%** of all configs built.

**Why it is bounded going forward.** The per-round trial budget (WS2) can't stop grid search
within a lineage but it killed the runaway: in the live post-reset table the max configs/lineage
is **205** (down from 295k), p99 is 73, and top-1% concentration falls from 92% to 36%.

**Why it does not touch the certifying gate.** Grid intensity inflates the *validation*
estimate (in-sample selection), but the single terminal holdout re-tests validation-winners on
fresh, purged data with `N` raised to the run-wide effective count. Only validation *winners*
reach the holdout, so its multiplicity counts holdout-looks honestly; the grid losers never
touch it. The deduped `N` therefore under-penalizes only the **in-round research-survivor
(Provisional) admission** step — a shelf-size/noise effect — not the Validated gate. Terminal
holdout survival plus the 90-day OOS clock remain the real filters.

**Why finding B (feed real trial-SR variance into `expected_max_sharpe`) was not pursued.** The
principled correction for correlated trials is López de Prado's effective-`N` via the across-trial
SR dispersion. It is infeasible cheaply here: the `trials` table has no SR column and
`experiment_id` is NULL for 100% of trial rows (recorded before the backtest assigns one), so
there is no join path to SRs. Only 166 archived `experiment_detail` artifacts carry SR (~0.6% of
lineages, a biased subset). Recovering per-trial SR dispersion would require a new write on the
hot mining loop; manufacturing a heuristic `N`-inflation instead would add an unprincipled fudge
to an otherwise statistically careful codebase.

**Disposition: accept as documented limitation.** The exposure that made this look moderate is a
dev-era artifact, already archived out and structurally prevented by the budget, and the
certifying gate was never under-penalized. If defense-in-depth against the residual live tail is
ever wanted, the one principled, SR-free move is a **per-lineage grid cap per round** (a "grid
budget" mirroring the fresh-lineage budget) that bounds tuning intensity at the source. The data
(live max 205, holdout terminal) says it is not urgent.

---

## 3. Scope caveats (not bugs)

- **Universe = BTC + ETH.** The two mega-caps are effectively the survivors of crypto; edges
  mined only on them have a generalization ceiling and the search sees no delisted-symbol
  representation. The point-in-time survivorship machinery (§1) exists but is not exercised by
  the 2-symbol time-series path. A universe-selection bias to keep in mind as the factory keeps
  mining this narrow set.
- **Purge gap uses the floor, not the feature-scaled value.** `_build_data_split_plan` is called
  with `walk_forward.purge_bars_floor` (288) rather than
  `walk_forward.purge_bars(max_feature_lookback)` = `max(288, 2×lookback)` (config.py:36-37).
  Low leakage impact — features are causal, so a test-window feature reaching back into training
  data is not itself leakage, and the label horizon is short — but a purpose-built method is
  bypassed; reconcile or document why the floor suffices.
- **`open_returns.fillna(0.0)` tail** (engine.py:511) pairs the last signal with a 0 forward
  return: one-bar dilution of IC toward zero. Negligible, not leakage.

---

## 4. One-line verdict

Of the six lenses, five are verifiably clean and the sixth (multiple testing) is honest across
distinct hypotheses and blind only within them — a residual that is concentrated in archived
dev-era runs, curtailed by the trial budget going forward, and absent from the terminal holdout
that actually certifies alpha.
