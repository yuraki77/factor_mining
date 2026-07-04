"""The Watch stratum only exists if near-misses survive across runs: the
per-round artifact is insert-or-replace on artifact scope and is overwritten
by the next run, so the table is the durable record of "interesting if X
changes" candidates and their repair paths.
"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta

from factor_mining.models import NearMissAnalysis
from factor_mining.storage import MetadataStore


def _near_miss(candidate_id: str, *, reason: str = "cost_destroyed_edge", actionable: bool = True) -> NearMissAnalysis:
    return NearMissAnalysis(
        experiment_id=f"exp-{candidate_id}",
        candidate_id=candidate_id,
        primary_reason=reason,
        actionable=actionable,
        repair_actions=["widen_exit"] if actionable else [],
    )


def test_near_misses_accumulate_across_runs_and_dedupe_per_experiment(tmp_path) -> None:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.save_near_misses([_near_miss("c1"), _near_miss("c2")], run_id="run_a")
    # re-analysis of the same experiment updates in place instead of duplicating
    store.save_near_misses([_near_miss("c1", reason="excess_turnover")], run_id="run_b")

    items = store.list_near_misses()
    assert len(items) == 2
    by_candidate = {item.candidate_id: item for item in items}
    assert by_candidate["c1"].primary_reason == "excess_turnover"
    assert by_candidate["c2"].repair_actions == ["widen_exit"]


def test_actionable_filter_keeps_watch_to_repairable_items(tmp_path) -> None:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.save_near_misses([
        _near_miss("c1", actionable=True),
        _near_miss("c2", reason="no_evidence", actionable=False),
    ])
    actionable = store.list_near_misses(actionable_only=True)
    assert [item.candidate_id for item in actionable] == ["c1"]


def test_watch_window_and_prune_age_items_out(tmp_path) -> None:
    """Watch is 'interesting if X changes', not an archive — stale entries
    must age out of the view and be prunable from the table."""
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.save_near_misses([_near_miss("old"), _near_miss("new")])
    backdated = (datetime.now(UTC) - timedelta(days=40)).isoformat()
    with closing(store.connect()) as conn:
        with conn:
            conn.execute("update near_misses set created_at = ? where candidate_id = ?", (backdated, "old"))

    recent = store.list_near_misses(since_days=30)
    assert [item.candidate_id for item in recent] == ["new"]

    removed = store.prune_near_misses(before_days=30)
    assert removed == 1
    assert [item.candidate_id for item in store.list_near_misses()] == ["new"]
