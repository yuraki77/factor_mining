# Research-System Review — factor_mining (2026-07-02)

Scope: review-only pass over the full pipeline (no code changes). Every finding cites the
code it is based on. Line numbers refer to the tree at commit `84053cb`.

---

## 1. Current System Map

**Data layer**
- `factor_mining/data/binance.py` + `scripts/backfill_data.py` sync Binance zips into a
  partitioned parquet warehouse (`data/parquet/market=…/dataset=…`).
- `data/loader.py:load_frame` loads klines (sorted, deduped, clipped to
  `data.start_date`, optionally pinned to `end_ms` for reproduction).
  `load_funding` + `funding_event_zscore_to_frame` (loader.py:365) build a causal,
  event-window funding z-score (ffill only; pre-first-event = 0).
  `load_supplemental_features` (loader.py:202) ffills USD-M supplemental datasets
  (mark/index/premium klines, OI, long/short ratios, basis) onto the 5m grid.
- `data/quality.py:kline_quality_notes` emits degraded-ratio notes (duplicates,
  OHLC invariants, zero-volume, jumps, gaps); gaps are flagged but not repaired.

**Feature/signal layer**
- `factors/engineering.py:generate_features` — ~70 trailing indicators (rolling/EWM only;
  verified causal).
- `mining.py:factor_signal` — per-family trailing signals (momentum/mean-rev/vol/funding).
- `dsl/` — factor DSL; `dsl/evaluator.py` ops are all trailing (`ts_*`, `delay`, `delta`).
- `regime/hmm.py:MarkovRegimeDetector` — 5-state GaussianHMM fit on a prefix
  (≤ 50k bars, `pipeline.py:_fit_regime_model:4682`), causal filtered probabilities,
  transition-matrix forward prediction, prefix masked "unknown", shifted 1 bar.
- `pipeline.py:_build_signal_for_uncached:4309` dispatches DSL → feature → factor_signal →
  composite → legacy, then applies smoothing/threshold controls, regime modulation and
  regime/funding filters.

**Evaluation layer**
- `pipeline.py:_build_data_split_plan:2107` — single chronological 60/20/20 split
  (discovery / repair-validation / final-OOS) of whatever frame was loaded.
- `pipeline.py:_run_mining_round:1313` — per round: build signals on the full frame,
  backtest discovery slice → generate pre-gate repairs + local grid tuning
  (`_build_local_grid_tuning_candidates:2357`) → backtest validation slice, CSCV PBO on
  validation (`_apply_batch_pbo:3330`) → merge pool (`_select_repair_merge_pool:3016`) →
  backtest **final-OOS slice** → merge-pool DSR trial penalty
  (`_apply_merge_pool_trial_penalty:3293`) → FDR + 16-rule GateCheck
  (`validation/gatecheck.py`) → ResearchGate (`research_gate.py`) → NearMiss → HardScore →
  traditional optimizer + evolutionary operators generate next-round candidates.
- `backtest/engine.py:run_backtest:422` — vol-targeted position from 1-bar-lagged signal,
  open→open returns, taker fees + participation slippage, funding transfers, exit rules;
  emits IC/RankIC NW t-stats, block-bootstrap Sharpe CI, PSR, DSR, permutation p,
  regime-conditional metrics, window-stability diagnostics.
- Statistics in `stats/metrics.py`; trial counting in `trial_ledger.py` + SQLite
  (`storage.py`).

**Promotion path**
A candidate is promoted by: (a) passing GateCheck (blocking rules G1 DSR>0, G2 PBO,
G5 bootstrap-CI, G8 cost margin, G10 capacity, G11 leakage flags, G14 data quality,
G15 schedulable) with evidence-tier stratification, or (b) becoming a persistent
"research survivor" (`research_gate.py:build_research_survivor_records:134`) that is
re-seeded into later runs and promoted when `fdr_pvalue < 0.10 AND trades ≥ 100`
(`pipeline.py:_update_research_survivor_store:3689`). Top scorers are archived with
git SHA, config hash, and data-extent manifest (`archive.py`); `reproduce_candidate`
(pipeline.py:591) re-runs an archived candidate pinned to the archived data extent.

**What is genuinely good** (worth stating, because the fix plan should not undo it):
prior look-ahead bugs were found and fixed with regression tests
(`stats/regime.py:14` trailing vol rank, `loader.py:356` funding ffill-only,
`pipeline.py:2135` final OOS always the chronological tail, DSR/PSR annualization fixes
in `stats/metrics.py:104,134`); signals are lagged one bar; costs/funding are modeled;
trial counts persist across runs; archives carry provenance; 283 tests pass.

---

## 2. Overall Assessment

**Classification: a factor exploration tool with substantial validation scaffolding —
not yet a research validation system.** It sits between the first two categories, and
the scaffolding (DSR, PBO, FDR, bootstrap, permutation, survivor store) is real, but
three structural problems currently make headline results unreliable:

1. **The "final OOS" is not out-of-sample after round 1.** Every mining round, every
   optimizer iteration, every re-run, and every survivor recheck re-evaluates on the same
   chronological last-20% window, and the optimizer's search is steered by metrics
   measured on that window. The DSR trial penalty compensates for breadth of search, not
   for adaptive reuse of the test window.
2. **The primary significance statistic is inflated.** IC/RankIC t-stats are Newey-West
   t-stats over a 288-bar *rolling* IC series sampled every bar, with NW bandwidth
   ≈ n^(1/3) ≪ 288; consecutive observations share 287/288 of their data. G3/G4, the FDR
   layer, and survivor promotion all consume these t-stats.
3. **Promotion can happen without new data.** The survivor store's 90-day OOS requirement
   is stored but never enforced, and its trade-count fallback can substitute the full
   backtest trade count.

Fixing those three (plus the placebo G11) would move the system solidly into the
"research validation system with some robustness" category. The execution model, cost
model, and reproducibility layer are already unusually good for a project at this stage.

---

## 3. Findings

### P0-1 — Final-OOS window is reused across rounds, runs, and survivor rechecks, and steers the optimizer

- **Category:** statistical validation / data snooping
- **Location:** `pipeline.py:1628-1651` (final backtests), `pipeline.py:1906-1947`
  (`build_optimization_context(round_candidates, round_backtests, …)` — `round_backtests`
  *are* the final-OOS results), `pipeline.py:1040` (`current_candidates = new_candidates`
  loops back), `optimizers/traditional_optimizer.py:603-620` (hill-climb keyed on
  `delta_sharpe` measured on final OOS), `pipeline.py:713-728` + `verify_research_survivors`
  (survivors re-seeded and re-verified on the same last-20% split),
  `_build_data_split_plan:2107` (split is fractional, so it is identical for every run on
  the same data extent).
- **Current behavior:** GateCheck, ResearchGate, HardScore and the optimizer context are
  all computed on the final-OOS slice. Next-round candidates (repairs, grid tuning
  survivors, optimizer adjustments, evolutionary children) are selected/tuned using those
  final-OOS metrics, then evaluated on the same slice again. Across CLI runs the window
  only moves when new data is synced.
- **Risk:** classic adaptive-overfitting loop — the test window degrades into a validation
  window; reported OOS Sharpe/IC of anything produced after round 1 (or after the first
  run) is biased upward, and the FDR/DSR numbers no longer mean what they claim.
  `_apply_merge_pool_trial_penalty` (pipeline.py:3293) raises the DSR trial count, which
  penalizes breadth but not the *direction* of adaptation toward the test window.
- **Remediation direction:**
  1. Move all optimizer/evolutionary/repair *selection* inputs to validation-slice
     metrics only; final-OOS metrics should be write-once per candidate lineage.
  2. Make the final-OOS evaluation a scheduled, rate-limited event: candidates queue on
     validation results and touch the holdout once (embargoed "lockbox" pattern), or
     implement true rolling walk-forward so each evaluation consumes previously unseen
     bars (the `walk_forward` config already describes this — see P1-3).
  3. Record per-candidate-lineage holdout-touch counts in `trial_diagnostics` and gate on
     them (e.g., fail G-new if the lineage has seen the holdout > k times).
- **Acceptance criteria:** a test proves the optimizer context contains no final-OOS
  metrics during discovery rounds; a counter of final-OOS evaluations per lineage exists
  in artifacts; re-running `run_pipeline` twice on identical data does not re-score the
  same lineage on the holdout twice without an explicit flag.

### P0-2 — IC/RankIC Newey-West t-stats are computed on overlapping rolling-IC series → inflated significance

- **Category:** statistical validation
- **Location:** `backtest/engine.py:470-471` (`rolling_pearson_ic/rolling_rank_ic` with
  `window=288`, evaluated at every bar), `engine.py:557-558`
  (`newey_west_tstat(ic_series.dropna())`), `stats/metrics.py:45-64` (bandwidth
  `lag = n**(1/3)`), consumed by `validation/gatecheck.py:66-68` (G3/G4/G4R),
  `apply_fdr` (gatecheck.py:22), `research_gate.py:161` and
  `pipeline.py:3713-3717` (survivor promotion p-value).
- **Current behavior:** the "IC series" is a 288-bar rolling correlation sampled every
  bar; adjacent values share 287/288 observations, so its autocorrelation extends to
  ~288 lags, while the NW bandwidth is ~40–50 for typical n. The variance of the mean is
  underestimated; t-stats (and hence the p-values fed to BH-FDR and survivor promotion)
  are inflated by roughly √(288/lag) ≈ 2–3×.
- **Risk:** G3/G4 thresholds (2.0) and the promotion threshold (`fdr_p < 0.10`) are far
  easier to clear than intended → false discoveries pass the exact layer that is supposed
  to control them.
- **Remediation direction:** compute the IC t-stat from *non-overlapping* per-bar ICs —
  either (a) per-bar signal·return products (the mean of which is the covariance;
  NW with lag ~ signal autocorrelation is then appropriate), (b) rolling IC sampled every
  `window` bars, or (c) keep the rolling series but set the NW bandwidth ≥ window.
  Add a regression test with a known-null signal: the empirical rejection rate of G4 at
  t>2 must be ≈ 2.3% on simulated white noise, not the current (much higher) rate.
- **Acceptance criteria:** simulation test in `tests/test_stats.py` shows null rejection
  rate within 2× of nominal; archived artifacts record the estimator variant used.

### P0-3 — Research-survivor promotion never enforces the 90-day OOS requirement and can double-count trades

- **Category:** statistical validation / experiment management
- **Location:** `pipeline.py:_update_research_survivor_store:3707-3719` (promotes on
  `fdr_pvalue < promotion_fdr and current_trades >= min_trades` — no elapsed-days check),
  `config.py:146` (`research_survivor_min_oos_days: 90` — stored into records at
  `research_gate.py:173` but read nowhere), `research_gate.py:159` and `pipeline.py:3712`
  (`current_trades = int(result.oos_trade_count or result.metrics_primary.trade_count)` —
  when `oos_trade_count == 0` the *full-slice* trade count silently substitutes),
  `storage.py:398` (`paper_trade_start_date` carried forward but never compared to now).
- **Current behavior:** a survivor re-seeded into the next run is re-evaluated on the same
  final-OOS window (see P0-1) and can be promoted immediately if the (inflated, P0-2)
  p-value clears 0.10 — zero genuinely new observations required. The promotion criteria
  string (`"NW FDR P < 0.10 AND trades >= 100"`) is what gets displayed, so the artifact
  claims a paper-trading bar that was never applied.
- **Risk:** premature promotion of high-Sharpe/high-IC candidates — precisely the failure
  mode the survivor store exists to prevent; artifacts misstate the criteria actually
  enforced.
- **Remediation direction:** in `_update_research_survivor_store`, require
  `now - paper_trade_start_date >= required_oos_days` **and** that the recheck window
  contains data beyond the record's original `data_end` (store the final-OOS start/end ms
  in the record); remove the `or metrics_primary.trade_count` fallback (treat
  `oos_trade_count == 0` as 0).
- **Acceptance criteria:** unit test: a survivor rechecked on an identical data extent is
  not promoted; a survivor with `oos_trade_count=0` and large full-sample trade count is
  not promoted; promotion requires ≥ `required_oos_days` of post-record data.

### P0-4 — G11 "leakage checks passed" is a placebo

- **Category:** testing / false research claims
- **Location:** `models.py:119-120` (`leakage_checks_passed: bool = True`,
  `split_overlap_detected: bool = False` — defaults, never written anywhere else),
  `validation/gatecheck.py:83` (blocking rule G11 evaluates those defaults).
- **Current behavior:** every gate report asserts "Lookahead and split leakage checks
  passed" although no check is ever executed.
- **Risk:** false research claim in every archived artifact; a future regression that
  introduces leakage will still stamp "passed".
- **Remediation direction:** either implement the checks (e.g., shifted-signal
  equivalence: re-run IC with signal shifted +1 and assert degradation; assert
  `final_oos_start_time` > max timestamp used in candidate selection metadata) and set
  the fields from the pipeline, or delete the rule so the report stops claiming it.
- **Acceptance criteria:** G11 value is computed (test asserts it flips on a synthetic
  leaked signal), or the rule and fields are removed.

---

### P1-1 — FDR control is per-family, per-round, per-symbol batch only; G3 is non-blocking

- **Category:** statistical validation
- **Location:** `validation/gatecheck.py:22-44` (`apply_fdr` groups the current round's
  `round_backtests` by `hypothesis_family`, BH with `n_tests ≥ 10` floor),
  `gatecheck.py:66` (G3 is `_warn_item` — not in `_BLOCKING_RULES:18`), callers at
  `pipeline.py:1719` and `verify` path 1247.
- **Current behavior:** the BH correction sees only one round × one symbol/market ×
  one family of p-values (often < 20 tests). Cross-round and cross-run multiplicity is
  handled only through the DSR trials penalty; the FDR-adjusted p-value that drives
  survivor promotion is never corrected for the thousands of cumulative trials the
  `TrialLedger` records. G3 failing blocks nothing at gate level.
- **Risk:** "FDR-controlled" is overstated; family-level FDR resets every round, so
  repeated rounds re-roll the dice.
- **Remediation direction:** feed BH with `n_tests = max(batch, family_trials_count)` from
  the ledger (the count is already computed at `pipeline.py:2230`), or apply BH across
  the accumulated family p-value history; decide explicitly whether G3 should block for
  promotion-bound candidates (survivor promotion already uses it as if it were binding).
- **Acceptance criteria:** BH `n_tests` reflects ledger counts (test with a mocked ledger);
  documented decision on G3 blocking semantics.

### P1-2 — Permutation test ignores serial dependence and its config threshold is unused

- **Category:** statistical validation
- **Location:** `stats/metrics.py:204-233` (i.i.d. shuffle of factor values; also returns
  a *normal-approximation* p-value instead of the empirical one whenever `null_std > 0`),
  `config.py:74` (`permutation_test.rejection_threshold: 0.05` referenced nowhere),
  only consumer is a +1 score bonus at `optimizers/traditional_optimizer.py:962`.
- **Current behavior:** permuting a heavily autocorrelated signal against
  vol-clustered returns understates the null variance of the correlation → p-values are
  anti-conservative; the configured rejection threshold is dead config; the statistic
  never gates anything.
- **Risk:** the one robustness check advertised as a permutation test provides little
  protection and can mislead the optimizer's scoring.
- **Remediation direction:** use block permutation / circular shifts of the signal
  (preserves both marginals and autocorrelation), return the empirical p (the
  `(exceed+1)/(n+1)` estimator already computed), and either wire
  `rejection_threshold` into GateCheck or delete it.
- **Acceptance criteria:** null-simulation test with AR(1) signal and GARCH-like returns
  shows ≈ nominal rejection; config field is used or removed.

### P1-3 — `walk_forward` / purge / embargo config describes an evaluation that does not exist

- **Category:** time alignment / experiment management
- **Location:** `configs/default.yaml:15-22` (train 12m / validation 3m / test 3m,
  `purge_bars_floor: 288`, `embargo_bars: 288`, `min_folds: 4`); only consumer is
  `backtest/engine.py:642-674` (`walk_forward_oos_mask`, used solely for
  `_oos_trade_count`); the real split is `_build_data_split_plan` (pipeline.py:2107) —
  a single 60/20/20 cut with **no purge or embargo bars** between segments; `min_folds`
  is read nowhere.
- **Current behavior:** there is no walk-forward evaluation and no purge/embargo around
  the discovery/validation/final boundaries. (Leakage impact is small because the
  forward horizon is 1 bar and signals are trailing, but validation-selected candidates'
  rolling state crosses the boundary warm rather than embargoed.)
- **Risk:** config misrepresents methodology; anyone reading `configs/default.yaml` or
  the registry's `walk_forward_analysis` method believes fold-based OOS exists; also
  blocks the natural fix for P0-1.
- **Remediation direction:** implement rolling walk-forward as the final evaluation
  (reusing `walk_forward_oos_mask`'s window math, aggregating per-fold metrics with
  `min_folds`), apply `purge/embargo` at the 60/20/20 boundaries, or delete the unused
  keys.
- **Acceptance criteria:** either per-fold OOS metrics appear in `BacktestResult` /
  artifacts with a test over a synthetic frame, or the dead config is removed and the
  registry method marked unimplemented.

### P1-4 — `oos_trade_count` on the final slice degenerates to a "last 25%" fallback

- **Category:** factor logic / statistical validation
- **Location:** `backtest/engine.py:667-674` (train+val bars exceed the ~14-month final
  slice → mask empty → `mask.iloc[int(n*0.75):] = True`), `_oos_trade_count:677`;
  consumers: G7 (`gatecheck.py:72`), survivor records (`research_gate.py:159`),
  promotion (`pipeline.py:3712`).
- **Current behavior:** during pipeline runs, `run_backtest` receives the final-OOS slice,
  so "OOS trades" silently means "trades in the last quarter of the final slice"
  (~3.5 months), not the configured walk-forward windows.
- **Risk:** `min_oos_trades: 100` measures an arbitrary sub-window; semantics differ
  between standalone backtests (full frame) and pipeline backtests (slice), making G7 and
  promotion counts incomparable across contexts.
- **Remediation direction:** when the input frame *is* the final OOS slice, count trades
  over the whole slice (the pipeline knows this — pass an explicit flag or count in
  `_run_mining_round`); reserve the walk-forward mask for full-frame runs.
- **Acceptance criteria:** test: for a final-slice backtest, `oos_trade_count` equals the
  slice's trade count; G7 evaluates the intended window.

### P1-5 — Vol-target warm-up zeroes the first ~30 days of every evaluation slice

- **Category:** execution realism / time alignment
- **Location:** `backtest/engine.py:369-374` (`realized_vol` needs
  `vol_window = 30d × 288` bars; `leverage = (target/vol).fillna(0.0)`), applied per
  slice because signals are sliced and re-backtested per segment
  (`pipeline.py:_slice_tasks:2177`).
- **Current behavior:** in each discovery/validation/final backtest, the first
  `vol_window` bars have leverage 0 → position 0 → returns 0. On the ~14-month final OOS
  that discards ~7% of the window; on shorter `tail=50_000` runs (see P2-4) it discards
  ~17% of each slice. Regime labels in `label_btc_regime` similarly need a 60-day warm-up
  within the slice (engine.py:510-514), pushing early final-OOS bars into "sideways".
- **Risk:** metrics (Sharpe, trade counts, window-stability quartiles) are computed over
  a silently shorter effective window; the first stability window
  (`compute_oos_window_diagnostics`, engine.py:98) is structurally quieter — distorts
  stability scores and cross-candidate comparability (uniformly, but nontrivially).
- **Remediation direction:** compute vol/regime state on the full frame and slice the
  *state* along with the signal (the pipeline already slices signals from full-frame
  computations; do the same for realized vol and regime labels via task payload or
  context), or start each slice's metrics after the warm-up index.
- **Acceptance criteria:** test: final-slice backtest position is non-zero from bar 0
  given an active signal; regime-conditional metrics on the final slice match labels
  computed on the full frame.

### P1-6 — PBO definition is inconsistent between the mining and survivor-verify paths

- **Category:** statistical validation
- **Location:** mining path: `_apply_batch_pbo(repair_validation_frame, …)`
  (pipeline.py:1573) then copied onto final results (pipeline.py:1646-1648); verify path:
  `_apply_batch_pbo(final_frame, …)` (pipeline.py:1226); singleton batches get
  `pbo = 1.0` (pipeline.py:3356-3359).
- **Current behavior:** G2 evaluates a CSCV-PBO computed on the validation window during
  mining but on the final-OOS window during survivor verification; PBO also depends on
  what else happened to be in the batch (selection among concurrent candidates), so the
  same candidate gets different PBO across runs; small batches are auto-failed (1.0).
- **Risk:** G2 pass/fail is not a stable property of a candidate; survivors can flip
  between fail/pass on recheck purely from batch composition. (The conservative 1.0
  default is good; the inconsistency is the problem.)
- **Remediation direction:** fix the PBO window (validation) for both paths and persist
  the batch composition (candidate ids) in `trial_diagnostics` so a PBO value is
  interpretable; consider per-lineage PBO vs. a frozen reference pool.
- **Acceptance criteria:** both call sites use the same window; artifacts record the
  comparison-pool ids; test asserts verify-path PBO window equals mining-path window.

### P1-7 — Evolutionary correlation filter reads final-OOS data before evaluation

- **Category:** data snooping
- **Location:** `pipeline.py:1991-1999` and 2026-2035
  (`_filter_evolutionary_output_correlation(children, …, final_frame, …, final_regimes,
  final_funding_rate, …)`).
- **Current behavior:** newly generated evolutionary children are accepted/rejected based
  on signal correlations computed on the final-OOS slice.
- **Risk:** mild but direct use of holdout data in candidate construction; combined with
  P0-1 it further couples the search to the test window.
- **Remediation direction:** compute the redundancy filter on the discovery or validation
  slice.
- **Acceptance criteria:** the filter call sites receive discovery/validation frames; a
  test pins this.

### P1-8 — Reproducibility gaps: LLM nondeterminism, hash-only archive verification, mutable warehouse

- **Category:** experiment management
- **Location:** `archive.py:67-79` (`verify_archive` re-hashes the stored JSON — it
  verifies file integrity, not reproducibility), `mining.py`
  `generate_hypotheses_with_deepseek` and `llm/mutation.py` (no persisted
  prompt/response/seed → a resumed or repeated run cannot regenerate the same candidates
  unless checkpoints exist), parquet warehouse files are overwritten in place with no
  content hash in run artifacts (only `data_extent` rows/min/max in archives,
  `loader.py:309`), run checkpoints fingerprint settings+args+row-extent but not data
  content (`pipeline.py:402-417`).
- **Current behavior:** good bones — fixed seeds (bootstrap/HMM/pipeline = 42), settings
  hash, git SHA, `reproduce_candidate` with `data_end_ms` pinning — but a "verified"
  archive has never been re-run, and identical row counts with silently re-synced data
  pass the fingerprint.
- **Risk:** silent drift between archived claims and what a re-run would produce;
  LLM-dependent runs are unrepeatable once checkpoints are pruned.
- **Remediation direction:** add a content hash (e.g., per-partition parquet sha256) to
  `data_extent` and checkpoint fingerprints; extend `fm exp reproduce` to compare
  reproduced `metrics_primary` against the archive within tolerance and store that
  verdict; persist LLM raw responses alongside `latest_hypotheses`.
- **Acceptance criteria:** `verify_archive` (or a new `fm exp verify`) re-runs and
  compares metrics; archives include data content hashes; LLM outputs are persisted per
  run id.

---

### P2-1 — Funding transfer requires exact `calc_time == open_time` match and is O(n) Python

- **Category:** execution realism / performance
- **Location:** `backtest/engine.py:181-190` (`_apply_funding` dict lookup per bar in a
  Python loop).
- **Risk:** any misalignment (funding settlement not on a 5m boundary, gap bar at the
  settlement time) silently drops the funding cash flow; the loop is also a hot-path cost
  (runs 2× per `evaluate_strategy_path`). Tests cover the aligned case
  (`tests/test_funding_alignment.py`) but not gap/misaligned cases.
- **Remediation:** vectorize via `searchsorted` on `open_time` mapping each funding event
  to the first bar at-or-after `calc_time`; add a gap-bar test.
- **Acceptance criteria:** funding applied when the settlement bar is missing/offset;
  runtime of `_strategy_returns` path drops measurably.

### P2-2 — Gaps are flagged but features/returns treat gap-adjacent bars as contiguous

- **Category:** data
- **Location:** `data/quality.py:89-93` (gap notes only), `loader.py:load_frame` (no
  reindex to a complete grid), all rolling features and `open.shift(-1)` returns.
- **Risk:** returns and indicators across outage gaps are treated as 5-minute quantities;
  G14 only reacts when degraded ratio > 10/20%.
- **Remediation:** either reindex to the full grid with explicit NaN bars (and make
  position/cost logic gap-aware), or mark gap-adjacent bars and exclude them from IC and
  trade entry.
- **Acceptance criteria:** a synthetic frame with a 6-hour gap produces no cross-gap
  1-bar return in IC inputs.

### P2-3 — `pipeline.py` is a 4,874-line god-module

- **Category:** architecture
- **Location:** `factor_mining/pipeline.py` (orchestration, checkpointing, split logic,
  signal construction, repair search, PBO, survivor store, diagnostics all in one file).
- **Risk:** the P0/P1 fixes above all land in this file; review and test isolation are
  already strained (e.g., the split plan, signal builder and survivor store can only be
  imported with the whole module).
- **Remediation:** mechanical extraction along existing seams: `splits.py`
  (`_build_data_split_plan`, `_slice_tasks`, masks), `signals.py`
  (`_build_signal_for*`, transforms, filters), `survivors.py`, `repairs.py`; no behavior
  change, keep names.
- **Acceptance criteria:** module ≤ ~1.5k lines; imports unchanged for public entry
  points; test suite green.

### P2-4 — Convenience runner evaluates on a ~6-month tail but its artifacts look like full-history results

- **Category:** experiment management
- **Location:** `run_full_pipeline.py:26` (`tail=50_000` ≈ 174 days → final OOS ≈ 35 days,
  further shortened by P1-5 warm-up), artifacts saved under the same keys
  (`latest_backtests`, …) as full runs.
- **Risk:** results from the quick path are indistinguishable from full-history results in
  the dashboard/artifacts; 35-day OOS windows with 100-trade warn gates invite
  cherry-picking.
- **Remediation:** stamp `tail`/`sample_bars` and the split timestamps into every artifact
  row (split info exists at `pipeline.py:1687-1694`; propagate to backtest payloads), and
  have the UI label short-window runs.
- **Acceptance criteria:** every stored backtest artifact carries the evaluated window
  extent; UI shows it.

### P2-5 — Supplemental-dataset timestamp semantics unverified

- **Category:** time alignment
- **Location:** `loader.py:_aligned_dataset_series:394` (ffill by `timestamp`/`open_time`
  for OI, long/short ratios, basis, mark/index/premium klines).
- **Risk:** if a provider timestamp marks a period *start* while the value summarizes the
  period (Binance OI/ratio snapshots are period-end, klines are period-start), a feature
  could be up to one dataset-period early. The 1-bar execution lag mitigates only
  intra-5m effects, not 5-minute-dataset off-by-one.
- **Remediation:** document/verify each dataset's timestamp convention against the
  downloader (`data/binance.py`), add a one-period safety shift where the convention is
  period-start summarizing forward.
- **Acceptance criteria:** per-dataset convention table in code comments + alignment test
  with synthetic data.

### P2-6 — Dead/misleading knobs and minor metric quirks

- **Category:** experiment management / factor logic
- **Location & behavior:**
  - `configs/default.yaml:47-51` `permute_target`/`test_statistic` unused (see P1-2);
  - `cpcv.n_groups/test_groups` drive plain CSCV, not CPCV (`pipeline.py:3424`
    `_cpcv_splits` is an alias for `_cscv_splits`); naming overstates the method;
  - `_apply_transform` hardcodes a 12-bar EWM smoothing inside `tanh_zscore`
    (pipeline.py:4135) independent of `smooth_span`, so "smoothing off" never is;
  - `metrics_primary.trade_count` counts every position delta including vol-target
    rebalancing (engine.py:443), inflating "trades" vs. G7's mask-based count;
  - `_metrics_from_returns` drops non-finite returns then compounds — fine, but
    `calmar=0` when `mdd == 0` hides infinite-Calmar cases (engine.py:70).
- **Remediation:** delete or wire dead config; rename `cpcv` → `cscv`; make the 12-bar
  EWM part of `smooth_span`; document trade-count semantics.
- **Acceptance criteria:** config keys all reachable from code; naming matches method.

---

## 4. Prioritized Remediation Plan (suggested order)

| Order | Finding | Effort | Why this order |
|---|---|---|---|
| 1 | P0-2 IC t-stat estimator | S | Small, self-contained, unblocks trust in G3/G4/FDR and everything downstream. |
| 2 | P0-3 survivor promotion guards | S | Few lines + tests; stops premature promotion immediately. |
| 3 | P0-4 G11 placebo | S | Implement or remove; restores artifact honesty. |
| 4 | P0-1 holdout discipline | M–L | Requires moving optimizer/evolution selection to validation metrics + holdout-touch accounting; do after 1–3 so the re-measured baseline is meaningful. |
| 5 | P1-3 walk-forward or config cleanup | M | Natural companion to P0-1 (rolling OOS is the sustainable fix). |
| 6 | P1-1 FDR scope, P1-2 permutation | S–M | Statistical layer coherence. |
| 7 | P1-4/P1-5/P1-6/P1-7 | S–M | Window/warm-up/PBO consistency. |
| 8 | P1-8 reproducibility | M | Content hashes + verified reproduce. |
| 9 | P2-* | S each | Opportunistic; P2-3 extraction ideally before P0-1's larger edits. |

**Expectation to set:** after fixes 1–4, measured pass rates and survivor counts will drop
— that is the system becoming honest, not regressing.
