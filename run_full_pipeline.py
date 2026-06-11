"""Full factor-mining pipeline runner.

Thin convenience wrapper around :func:`factor_mining.pipeline.run_pipeline` — the
single canonical engine (hypotheses → candidates → backtest → gatecheck →
hardscore → optimize). This script used to reimplement that entire flow with its
own ProcessPool and per-stage logic, which silently drifted from the library
(missing its fixes, fees, gates). It now just configures a run and prints a
summary, so there is exactly one pipeline implementation to maintain (Q5).
"""

from __future__ import annotations

from factor_mining.config import load_settings
from factor_mining.pipeline import run_pipeline
from factor_mining.storage import MetadataStore


def main() -> None:
    settings = load_settings()
    store = MetadataStore(settings.data.sqlite_path)

    # DeepSeek hypotheses (run_pipeline falls back to built-ins if the LLM is
    # unavailable); last 50k bars keeps this convenience run fast.
    result = run_pipeline(
        settings,
        use_llm=True,
        tail=50_000,
        hypothesis_count=5,
        store=store,
    )

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Rounds:        {result.total_rounds}")
    print(f"  Hypotheses:    {len(result.hypotheses)}")
    print(f"  Candidates:    {len(result.candidates)}")
    print(f"  Backtests:     {len(result.backtests)}")
    print(f"  GateCheck OK:  {result.n_gatecheck_passed}/{len(result.gatechecks)}")
    print(f"  HardScore >0:  {sum(1 for s in result.hardscores if s.score > 0)}")
    print(f"  Elapsed:       {result.elapsed_s:.0f}s")
    if result.errors:
        print(f"  Errors:        {len(result.errors)}")


if __name__ == "__main__":
    main()
