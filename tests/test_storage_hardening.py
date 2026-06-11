"""A5: SQLite hardening for the cross-process MetadataStore.

The mining daemon writes this DB while the gRPC server / UI read it, so the
connection must use WAL + a busy_timeout (concurrent readers, writers wait
rather than erroring) and pipeline-event sequence numbers must be assigned
atomically so two writers on the same run never collide.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

from factor_mining.models import TrialRecord
from factor_mining.storage import MetadataStore


def _trial(trial_id: str, family: str) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id, candidate_id=trial_id, hypothesis_family=family, method_id="m"
    )


def test_trial_counts_partition_on_canonical_family(tmp_path: Path) -> None:
    """I7: family variants must share one multiplicity count. 'momentum' and
    'trend_following' canonicalize to the same family, so a trial under each gives
    a family count of 2 by either spelling — otherwise each variant would look like
    a fresh, unpenalized family and understate the deflated-Sharpe trial penalty."""
    store = MetadataStore(tmp_path / "m.sqlite")
    store.record_trial(_trial("t1", "momentum"))
    store.record_trial(_trial("t2", "trend_following"))
    assert store.trial_counts("momentum")[0] == 2
    assert store.trial_counts("trend_following")[0] == 2
    # An unrecognized family partitions on its own raw name.
    store.record_trial(_trial("t3", "totally_unknown_family"))
    assert store.trial_counts("totally_unknown_family")[0] == 1
    assert store.trial_counts("momentum")[0] == 2  # unaffected by the unknown family


def test_connect_enables_wal_and_busy_timeout(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "m.sqlite")
    with closing(store.connect()) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert int(conn.execute("PRAGMA busy_timeout").fetchone()[0]) >= 30000


def test_pipeline_event_seq_is_monotonic_and_gapless(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "m.sqlite")
    for i in range(5):
        store.append_pipeline_event("run-1", phase="p", message=f"m{i}")
    seqs = [event["seq"] for event in store.load_pipeline_events("run-1")]
    assert seqs == [1, 2, 3, 4, 5]


def test_event_seq_is_atomic_across_connections(tmp_path: Path) -> None:
    """WHY: the daemon and gRPC server are separate processes sharing this DB.
    seq is computed inside the INSERT (not read-then-insert), so interleaved
    writers to the same run produce a gapless, duplicate-free sequence."""
    path = tmp_path / "m.sqlite"
    daemon = MetadataStore(path)
    server = MetadataStore(path)
    daemon.append_pipeline_event("run-x", phase="p", message="a1")
    server.append_pipeline_event("run-x", phase="p", message="b1")
    daemon.append_pipeline_event("run-x", phase="p", message="a2")
    server.append_pipeline_event("run-x", phase="p", message="b2")
    seqs = [event["seq"] for event in daemon.load_pipeline_events("run-x")]
    assert seqs == [1, 2, 3, 4]
    assert len(set(seqs)) == 4  # no duplicate sequence numbers
