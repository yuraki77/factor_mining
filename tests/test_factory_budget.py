"""The per-round trial budget makes a continuously-running factory
statistically sustainable: every admitted fresh lineage permanently raises
the expected-max-Sharpe deflation for all future candidates, so the cap must
bind exactly at ledger entry — and must never tax derived candidates or
survivor rechecks, which do not raise the independent-trial count N.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from inspect import signature

from typer.testing import CliRunner

from factor_mining.cli import app
from factor_mining.config import DataConfig, Settings
from factor_mining.factory import RoundBudget
from factor_mining.models import CandidateStrategySpec
from factor_mining.pipeline import PipelineResult, _lineage_root_id, run_pipeline


def _candidate(
    cid: str,
    family: str = "momentum",
    parent: str | None = None,
    lineage: str | None = None,
) -> CandidateStrategySpec:
    return CandidateStrategySpec(
        candidate_id=cid,
        hypothesis_id="h1",
        method_id="m1",
        hypothesis_family=family,
        symbol="BTCUSDT",
        parent_candidate_id=parent,
        lineage_id=lineage,
    )


def test_fresh_lineages_capped_per_family() -> None:
    """The cap is stratified: one prolific family (mean_reversion reached 403k
    raw trials in dev) must not spend the whole round's budget and poison the
    FDR denominator for everyone else."""
    budget = RoundBudget(4, ["momentum", "mean_reversion"], family_floor=0)
    candidates = [_candidate(f"m{i}", "momentum") for i in range(3)] + [
        _candidate(f"r{i}", "mean_reversion") for i in range(3)
    ]
    admitted, dropped = budget.admit(candidates, lineage_root=_lineage_root_id)
    assert [c.candidate_id for c in admitted] == ["m0", "m1", "r0", "r1"]
    assert dropped == {"momentum": 1, "mean_reversion": 1}


def test_derived_and_survivor_candidates_are_free() -> None:
    """Grid variants/repairs inherit their root's lineage and survivor
    rechecks re-evaluate an already-counted lineage: neither raises N, so an
    exhausted budget must still admit them or the factory could never tune or
    recheck once the round's budget was spent."""
    budget = RoundBudget(0, ["momentum"], family_floor=0)
    derived_by_parent = _candidate("d1", parent="root1")
    derived_by_lineage = _candidate("d2", lineage="root1")
    survivor = _candidate("s1")
    fresh = _candidate("f1")
    admitted, dropped = budget.admit(
        [derived_by_parent, fresh, derived_by_lineage, survivor],
        lineage_root=_lineage_root_id,
        survivor_candidate_ids={"s1"},
    )
    # order preserved; only the fresh lineage is dropped
    assert [c.candidate_id for c in admitted] == ["d1", "d2", "s1"]
    assert dropped == {"momentum": 1}


def test_family_floor_guarantees_minimum_share() -> None:
    """A round with many families must not starve each family below the floor
    — an underpowered family can never accumulate the evidence to graduate."""
    budget = RoundBudget(2, ["momentum", "mean_reversion"], family_floor=5)
    candidates = [_candidate(f"m{i}", "momentum") for i in range(5)]
    admitted, dropped = budget.admit(candidates, lineage_root=_lineage_root_id)
    assert len(admitted) == 5
    assert dropped == {}


def test_unknown_family_gets_default_quota_lazily() -> None:
    """Next-hypothesis candidates can mint families unseen at round start;
    they get the same quota rather than a free pass or a hard zero."""
    budget = RoundBudget(2, ["momentum"], family_floor=0)
    candidates = [_candidate(f"v{i}", "volatility") for i in range(3)]
    admitted, dropped = budget.admit(candidates, lineage_root=_lineage_root_id)
    assert len(admitted) == 2
    assert dropped == {"volatility": 1}


def test_admit_is_thread_safe_across_symbol_groups() -> None:
    """Symbol groups admit concurrently from worker threads; overspending
    under a race would silently understate the DSR penalty."""
    budget = RoundBudget(100, ["momentum"], family_floor=0)

    def admit_batch(batch_idx: int) -> int:
        batch = [_candidate(f"c{batch_idx}_{i}") for i in range(50)]
        admitted, _ = budget.admit(batch, lineage_root=_lineage_root_id)
        return len(admitted)

    with ThreadPoolExecutor(max_workers=8) as executor:
        totals = list(executor.map(admit_batch, range(8)))
    assert sum(totals) == 100


def test_run_pipeline_trial_budget_defaults_to_none() -> None:
    """trial_budget=None must mean 'today's behavior' so existing callers
    (backtest_master bridge, fm mine run) are untouched."""
    parameter = signature(run_pipeline).parameters["trial_budget"]
    assert parameter.default is None
    assert parameter.kind is parameter.KEYWORD_ONLY


def test_cli_run_passes_trial_budget_through(tmp_path, monkeypatch) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    seen: dict = {}

    def fake_run_pipeline(settings_arg, **kwargs):
        seen.update(kwargs)
        return PipelineResult(elapsed_s=0.1)

    monkeypatch.setattr("factor_mining.cli.load_settings", lambda: settings)
    monkeypatch.setattr("factor_mining.pipeline.run_pipeline", fake_run_pipeline)

    result = CliRunner().invoke(app, ["mine", "run", "--tail", "5", "--trial-budget", "7"])

    assert result.exit_code == 0
    assert seen["trial_budget"] == 7
