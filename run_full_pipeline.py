"""
Full pipeline: DeepSeek/default hypothesis → backtest → gatecheck → traditional optimize.

Uses existing btc_5m_5y.parquet from parent project, adapted to factor_mining format.
"""
import os, sys, json, uuid, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Ensure factor_mining is importable
sys.path.insert(0, ".")

from factor_mining.config import load_settings
from factor_mining.models import (
    HypothesisSpec, CandidateStrategySpec, BacktestResult,
    GateCheckResult, HardScoreReport, MetricsBlock, DataQualityNote,
)
from factor_mining.registry import METHOD_REGISTRY, schedulable_methods
from factor_mining.storage import MetadataStore
from factor_mining.mining import default_hypotheses, build_v1_candidates, generate_hypotheses_with_deepseek
from factor_mining.factors.engineering import generate_features, INDICATOR_META
from factor_mining.validation.gatecheck import apply_fdr
from factor_mining.factors.returns import forward_returns
from factor_mining.backtest.engine import run_backtest
from factor_mining.validation.gatecheck import run_gatecheck
from factor_mining.hardscore import hardscore
from factor_mining.optimizers.traditional_optimizer import (
    build_optimization_context, optimize_traditionally,
)
from factor_mining.hypotheses.discovered import boundary_conditions, should_continue_mining


def load_data():
    """Load BTCUSDT 5m data from parent project parquet, convert to factor_mining format."""
    parquet_path = "../btc_5m_5y.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    # Convert to factor_mining kline format
    frame = pd.DataFrame({
        "open_time": (df.index.astype("int64") // 10**6).astype(int),
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "volume": df["volume"].values,
        "close_time": (df.index.astype("int64") // 10**6).astype(int),
        "quote_volume": (df["volume"] * df["close"]).values,
        "trade_count": 0,
        "taker_buy_volume": (df["volume"] * 0.5).values,
        "taker_buy_quote_volume": (df["volume"] * df["close"] * 0.5).values,
        "market": "spot",
    })
    frame["data_quality_degraded"] = False
    return frame


_worker_frame = None
_worker_settings = None


def _init_worker(frame: pd.DataFrame, settings: "Settings") -> None:
    global _worker_frame, _worker_settings
    _worker_frame = frame
    _worker_settings = settings


def _run_one_backtest(args: tuple) -> "BacktestResult | Exception":
    global _worker_frame, _worker_settings
    signal_arr, candidate_dict = args
    from factor_mining.models import CandidateStrategySpec
    from factor_mining.backtest.engine import run_backtest

    candidate = CandidateStrategySpec.model_validate(candidate_dict)
    signal = pd.Series(signal_arr, index=_worker_frame.index)
    try:
        return run_backtest(_worker_frame, signal, candidate, _worker_settings, data_quality_notes=[])
    except Exception as exc:
        return exc


def main():
    settings = load_settings()
    store = MetadataStore(settings.data.sqlite_path)
    print("=" * 70)
    print("FULL PIPELINE: DeepSeek → Backtest → GateCheck → HardScore → Traditional Optimizer")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════
    # STEP 1: Hypothesis Generation (DeepSeek)
    # ═══════════════════════════════════════════════════════
    print("\n[1/6] Generating hypotheses via DeepSeek...")
    t0 = time.time()

    try:
        hypotheses = generate_hypotheses_with_deepseek(
            settings,
            count=5,
            research_brief=(
                "Generate rigorous first-principles BTC/ETH factor hypotheses for Binance spot "
                "and USD-M perpetual 5m data. Focus on time-series momentum, mean-reversion, "
                "volatility regime, volume confirmation, and funding basis factors. "
                "Only produce hypotheses suitable for N=2 symbols (BTCUSDT, ETHUSDT). "
                "Each hypothesis must include economic mechanism, testable prediction, "
                "null hypothesis, expected IC range, and expected decay halflife in bars."
            ),
        )
        print(f"  DeepSeek generated {len(hypotheses)} hypotheses in {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"  DeepSeek failed: {e}")
        print(f"  Falling back to default + discovered hypotheses")
        hypotheses = default_hypotheses()
        print(f"  Using {len(hypotheses)} built-in hypotheses")

    for h in hypotheses:
        print(f"  {h.hypothesis_id}: [{h.hypothesis_family}] {h.economic_mechanism[:80]}...")

    # ═══════════════════════════════════════════════════════
    # STEP 2: Build Candidates + Load Data
    # ═══════════════════════════════════════════════════════
    print(f"\n[2/6] Building candidates and loading data...")

    # Use limited methods for speed (not all 22 schedulable methods)
    from factor_mining.registry import get_method
    fast_methods = [
        get_method("factor_scoring"),
        get_method("parameter_sweep"),
        get_method("ic_analysis"),
        get_method("rank_ic_analysis"),
    ]
    candidates = []
    for hypothesis in hypotheses:
        for symbol in settings.data.symbols[:1]:
            for method in fast_methods:
                if method.is_ml and hypothesis.hypothesis_family not in {"momentum", "mean_reversion", "volatility"}:
                    continue
                candidates.append(CandidateStrategySpec(
                    candidate_id=f"c_{uuid.uuid4().hex[:12]}",
                    hypothesis_id=hypothesis.hypothesis_id,
                    method_id=method.method_id,
                    hypothesis_family=hypothesis.hypothesis_family,
                    symbol=symbol,
                    interval=settings.data.default_interval,
                    params={"expected_ic_mid": 0.01, "oos_trade_count": 50},
                    is_ml=method.is_ml,
                ))
    print(f"  {len(candidates)} candidates ({len(hypotheses)} hypotheses × {len(fast_methods)} methods)")

    frame = load_data()
    # Use last 50K rows for speed
    frame = frame.iloc[-50000:].reset_index(drop=True)
    print(f"  Data: {len(frame):,} rows (last 50K), {frame['open_time'].min()} → {frame['open_time'].max()}")

    # Reduce bootstrap/permutation for speed
    settings.bootstrap.n_resamples = 200
    settings.permutation_test.n_permutations = 200

    # Generate features
    features_df, feature_meta = generate_features(frame)
    fwd = forward_returns(frame["close"], horizons=[1, 12, 48])
    print(f"  Features: {len(features_df.columns)} cols")

    # ═══════════════════════════════════════════════════════
    # STEP 3: Run Backtests
    # ═══════════════════════════════════════════════════════
    print(f"\n[3/6] Running backtests on {len(candidates)} candidates...")
    t0 = time.time()

    # Pre-compute signal arrays (avoid serializing full Series to workers)
    tasks = []
    for i, c in enumerate(candidates):
        family_features = [
            col for col, m in feature_meta.items()
            if m.get("family", "") == c.hypothesis_family
        ]
        if not family_features:
            family_features = [
                col for col, m in feature_meta.items()
                if m.get("family", "") in ("trend_following", "mean_reversion", "volatility_regime", "volume_confirmation")
            ]
        feature_col = family_features[abs(i) % len(family_features)]
        signal_arr = features_df[feature_col].fillna(0).clip(-3, 3).to_numpy(dtype=float)
        tasks.append((signal_arr, c.model_dump()))

    result_by_idx = {}
    max_workers = min(os.cpu_count() or 4, len(tasks))
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(frame, settings)) as executor:
        future_to_idx = {executor.submit(_run_one_backtest, task): idx for idx, task in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            c = candidates[idx]
            result = future.result()
            if isinstance(result, Exception):
                print(f"  [{idx+1}/{len(candidates)}] {c.candidate_id[:16]}... SKIP: {result}")
            else:
                result_by_idx[idx] = result

    # Preserve order, filter failed candidates
    valid_indices = sorted(result_by_idx.keys())
    results = [result_by_idx[i] for i in valid_indices]
    candidates = [candidates[i] for i in valid_indices]

    print(f"  {len(results)} backtests completed in {time.time()-t0:.0f}s ({max_workers} workers)")

    # ═══════════════════════════════════════════════════════
    # STEP 4: GateCheck
    # ═══════════════════════════════════════════════════════
    print(f"\n[4/6] Running GateCheck...")

    gatechecks = []
    methods_map = {m.method_id: m for m in METHOD_REGISTRY}
    fdr_map = apply_fdr(results, settings)
    for r in results:
        method = methods_map.get(r.method_id)
        if method is None:
            method = schedulable_methods(2)[0]
        gc = run_gatecheck(r, settings, method=method, fdr_adjusted_pvalue=fdr_map.get(r.experiment_id, r.permutation_test_pvalue))
        gatechecks.append(gc)

    passed = sum(1 for g in gatechecks if g.passed)
    print(f"  Passed: {passed}/{len(gatechecks)}")

    # Show failures
    for r, g in zip(results, gatechecks):
        if not g.passed:
            fail_ids = [item.rule_id for item in g.failures]
            print(f"  FAIL {r.candidate_id[:16]}...: {fail_ids} "
                  f"(SR={r.metrics_primary.sharpe:+.2f}, DSR={r.deflated_sharpe:+.3f})")
        else:
            print(f"  PASS {r.candidate_id[:16]}...: "
                  f"SR={r.metrics_primary.sharpe:+.2f}, DSR={r.deflated_sharpe:+.3f}, "
                  f"IC_t={r.ic_tstat_nw:+.1f}")

    # ═══════════════════════════════════════════════════════
    # STEP 5: HardScore
    # ═══════════════════════════════════════════════════════
    print(f"\n[5/6] Computing HardScore...")

    scores = []
    for r, g in zip(results, gatechecks):
        hs = hardscore(r, g, fdr_adjusted_pvalue=fdr_map.get(r.experiment_id, r.permutation_test_pvalue))
        scores.append(hs)
        if hs.score > 0:
            print(f"  {r.candidate_id[:16]}...: score={hs.score:.1f}  "
                  f"haircut={hs.haircut_sharpe:+.3f}  fdr_p={hs.fdr_adjusted_pvalue:.4f}")

    top_scored = sorted(scores, key=lambda s: s.score, reverse=True)[:5]
    if top_scored:
        print(f"  Top score: {top_scored[0].score:.1f}")

    # ═══════════════════════════════════════════════════════
    # STEP 6: Traditional Optimization
    # ═══════════════════════════════════════════════════════
    print(f"\n[6/6] Traditional optimization...")

    # Filter to valid candidates/results/gatechecks
    valid_triples = [(c, r, g) for c, r, g in zip(candidates, results, gatechecks)]
    c_list, r_list, g_list = zip(*valid_triples) if valid_triples else ([], [], [])
    c_list, r_list, g_list = list(c_list), list(r_list), list(g_list)

    ctx = build_optimization_context(c_list, r_list, g_list, iteration=0)
    print(f"  Context: {ctx['num_candidates']} candidates, {ctx['num_gatecheck_passed']} passed")

    opt = optimize_traditionally(ctx, "full")
    print(f"  Optimizer action: {opt['action']}")

    # Show combinations
    for combo in opt.get("combinations", [])[:3]:
        print(f"  Combo: {combo.get('factor_ids', [])} weights={combo.get('weights', [])}")
        rationale = combo.get("rationale", "")
        print(f"    {rationale[:150]}...")

    # Show new hypotheses
    for h in opt.get("next_hypotheses", [])[:3]:
        print(f"  Next hypothesis: [{h.get('family', '?')}] {h.get('mechanism', '')[:120]}...")

    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Hypotheses:    {len(hypotheses)}")
    print(f"  Candidates:    {len(candidates)}")
    print(f"  Backtests:     {len(results)}")
    print(f"  GateCheck OK:  {passed}/{len(gatechecks)}")
    print(f"  HardScore >0:  {sum(1 for s in scores if s.score > 0)}")
    print(f"  Optimizer combos:{len(opt.get('combinations', []))}")
    print(f"  New hypotheses:{len(opt.get('next_hypotheses', []))}")


if __name__ == "__main__":
    main()
