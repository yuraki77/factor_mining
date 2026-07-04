from datetime import datetime, timedelta, timezone

from factor_mining.config import DataConfig, Settings
from factor_mining.models import TrialRecord
from factor_mining.storage import MetadataStore
from factor_mining.trial_ledger import TrialLedger


def test_trial_ledger_uses_stricter_family_or_rolling_count(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    ledger = TrialLedger(store, settings)
    now = datetime.now(timezone.utc)
    for idx in range(3):
        ledger.record(
            TrialRecord(
                trial_id=f"old-{idx}",
                candidate_id=f"c-old-{idx}",
                hypothesis_family="momentum",
                method_id="rule_mining",
                evaluated_at=now - timedelta(days=120),
            )
        )
    for idx in range(5):
        ledger.record(
            TrialRecord(
                trial_id=f"recent-{idx}",
                candidate_id=f"c-recent-{idx}",
                hypothesis_family="funding_basis",
                method_id="funding_rate_event_mining",
                evaluated_at=now - timedelta(days=idx),
            )
        )
    counts = ledger.counts_for("momentum", now=now)
    assert counts["family_trials_count"] == 3
    assert counts["rolling_90d_trials_count"] == 5
    assert counts["effective_trials_count"] == 5
    assert counts["global_cumulative_trials_count"] == 8


def test_trial_ledger_counts_lineages_not_variant_evaluations(tmp_path) -> None:
    # WHY: the E[max SR] deflation needs N = independent search paths. A grid
    # sweep around one idea is one path, and re-evaluating the same candidate
    # (new round, other symbol) is not a new discovery attempt — counting raw
    # rows compounds N until no strategy can ever clear the gate.
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    ledger = TrialLedger(store, settings)
    now = datetime.now(timezone.utc)
    ledger.record(
        TrialRecord(trial_id="root", candidate_id="c-root", hypothesis_family="momentum", method_id="rule_mining")
    )
    for idx in range(3):
        ledger.record(
            TrialRecord(
                trial_id=f"grid-{idx}",
                candidate_id=f"c-grid-{idx}",
                hypothesis_family="momentum",
                method_id="rule_mining",
                lineage_id="c-root",
            )
        )
    # Re-evaluation of the root under a fresh trial id.
    ledger.record(
        TrialRecord(trial_id="root-again", candidate_id="c-root", hypothesis_family="momentum", method_id="rule_mining")
    )
    counts = ledger.counts_for("momentum", now=now)
    assert counts["family_trials_count"] == 1
    assert counts["effective_trials_count"] == 1
    assert counts["global_cumulative_trials_count"] == 1


def test_trial_store_migrates_legacy_rows_to_lineages(tmp_path) -> None:
    # WHY: 400k+ pre-lineage rows are dominated by grid/repair variants whose
    # parents cannot be recovered; without a backfill they keep every future
    # run's trial count at "nothing can ever pass". Variants collapse into a
    # per-family bucket, originals keep their candidate_id, duplicates dedupe.
    import sqlite3

    db_path = tmp_path / "meta.sqlite3"
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            create table trials (
                trial_id text primary key,
                candidate_id text not null,
                experiment_id text,
                hypothesis_family text not null,
                method_id text not null,
                evaluated_at text not null
            )
            """
        )
        rows = [
            ("t1", "c_ab12", "momentum"),
            ("t2", "c_ab12", "momentum"),  # duplicate evaluation of an original
            ("t3", "c_cd34", "momentum"),
            ("t4", "c_grid_x1", "momentum"),
            ("t5", "c_grid_x2", "momentum"),
            ("t6", "c_pre_y1", "momentum"),
            ("t7", "c_grid_z1", "volatility"),
        ]
        conn.executemany(
            "insert into trials values (?, ?, null, ?, 'rule_mining', '2026-06-01T00:00:00+00:00')",
            rows,
        )
    conn.close()

    store = MetadataStore(db_path)
    settings = Settings(data=DataConfig(sqlite_path=db_path))
    ledger = TrialLedger(store, settings)
    now = datetime(2026, 6, 15, tzinfo=timezone.utc)

    counts = ledger.counts_for("momentum", now=now)
    # two originals + one legacy_derived:momentum bucket
    assert counts["family_trials_count"] == 3
    # + one legacy_derived:volatility bucket
    assert counts["global_cumulative_trials_count"] == 4


def test_derived_candidates_share_their_roots_lineage(tmp_path) -> None:
    # WHY: repairs and grid variants are parameter-local moves around one
    # idea; recording each as an independent trial multiplied N by ~18x in
    # practice and made the DSR gate structurally impossible.
    from factor_mining.models import CandidateStrategySpec
    from factor_mining.pipeline import _record_candidate_trials

    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    original = CandidateStrategySpec(
        candidate_id="c-root",
        hypothesis_id="h1",
        method_id="rule_mining",
        hypothesis_family="momentum",
        symbol="BTCUSDT",
    )
    repair = original.model_copy(deep=True)
    repair.candidate_id = "c-pre-1"
    repair.candidate_type = "repair"
    repair.parent_candidate_id = original.candidate_id
    repair.lineage_id = original.candidate_id

    cumulative: dict[str, int] = {}
    _record_candidate_trials([original, repair], store, settings, cumulative)

    assert cumulative["momentum"] == 1
    ledger = TrialLedger(store, settings)
    assert ledger.counts_for("momentum")["family_trials_count"] == 1

