from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from factor_mining.models import NearMissAnalysis, ResearchSurvivorRecord, TrajectoryRecord, TrialRecord, UTC


def _canonical_family(hypothesis_family: str) -> str:
    """Canonical signal-construction family used to partition the trial ledger.

    Family variants ("mean reversion" vs "mean_reversion", repairs, etc.) must
    share one multiplicity count (I7) — otherwise each spelling looks like a
    fresh, unpenalized family and the deflated-Sharpe trial penalty is understated.
    Lazy import keeps storage free of an import cycle through mining.
    """
    from factor_mining.mining import normalize_family

    return normalize_family(hypothesis_family) or hypothesis_family


class MetadataStore:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        # ``timeout`` becomes SQLite's busy_timeout: a writer waits for the lock
        # to clear instead of immediately raising "database is locked". This DB
        # is shared cross-process (the mining daemon writes while the gRPC
        # server / UI read), so contention is expected, not exceptional.
        conn = sqlite3.connect(self.sqlite_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # WAL lets readers run concurrently with the single writer instead of
        # blocking each other; NORMAL is the WAL-safe durability setting; the
        # explicit busy_timeout guards against the C-level ``timeout`` being
        # bypassed. PRAGMAs run outside any transaction here (fresh connection).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with closing(self.connect()) as conn:
            with conn:
                conn.executescript(
                    """
                    create table if not exists trials (
                        trial_id text primary key,
                        candidate_id text not null,
                        experiment_id text,
                        hypothesis_family text not null,
                        method_id text not null,
                        evaluated_at text not null,
                        lineage_id text
                    );
    
                    create index if not exists idx_trials_family_time
                        on trials(hypothesis_family, evaluated_at);
    
                    create table if not exists artifacts (
                        artifact_id text primary key,
                        kind text not null,
                        payload_json text not null,
                        created_at text not null
                    );
    
                    create table if not exists data_coverage (
                        coverage_id text primary key,
                        payload_json text not null,
                        created_at text not null
                    );
    
                    create table if not exists pipeline_runs (
                        run_id text primary key,
                        status text not null,
                        args_json text not null,
                        started_at text not null,
                        ended_at text,
                        error text
                    );
    
                    create table if not exists pipeline_events (
                        event_id text primary key,
                        run_id text not null,
                        seq integer not null,
                        phase text not null,
                        level text not null,
                        message text not null,
                        payload_json text not null,
                        created_at text not null
                    );
    
                    create index if not exists idx_pipeline_events_run_seq
                        on pipeline_events(run_id, seq);

                    create table if not exists research_survivors (
                        candidate_id text primary key,
                        experiment_id text not null,
                        status text not null,
                        payload_json text not null,
                        created_at text not null,
                        updated_at text not null
                    );

                    create index if not exists idx_research_survivors_status_updated
                        on research_survivors(status, updated_at);

                    create table if not exists near_misses (
                        id text primary key,
                        candidate_id text not null,
                        experiment_id text not null,
                        run_id text,
                        primary_reason text not null,
                        actionable integer not null,
                        payload_json text not null,
                        created_at text not null
                    );

                    create index if not exists idx_near_misses_created
                        on near_misses(created_at);

                    create table if not exists trajectories (
                        trajectory_id text primary key,
                        candidate_id text not null,
                        operator text not null,
                        payload_json text not null,
                        created_at text not null
                    );

                    create index if not exists idx_trajectories_candidate
                        on trajectories(candidate_id);

                    create index if not exists idx_trajectories_operator
                        on trajectories(operator);
                    """
                )
                self._migrate_trial_lineage(conn)

    @staticmethod
    def _migrate_trial_lineage(conn: sqlite3.Connection) -> None:
        """Backfill ``trials.lineage_id`` so trial counts measure independent
        search paths rather than raw evaluations.

        Legacy rows predate lineage tracking. Parameter-local variants
        (grid tuning, pre-gate repairs, optimizer adjust/repair) cannot be
        re-linked to their parents, so they collapse into one per-family
        bucket — their parents carry the lineage count through their own
        rows. Everything else keeps its candidate_id as its own lineage.
        GLOB keeps '_' literal (LIKE would treat it as a wildcard)."""
        columns = {row[1] for row in conn.execute("pragma table_info(trials)")}
        if "lineage_id" not in columns:
            conn.execute("alter table trials add column lineage_id text")
        conn.execute(
            """
            update trials set lineage_id = 'legacy_derived:' || hypothesis_family
            where lineage_id is null
              and (candidate_id glob 'c_grid_*' or candidate_id glob 'c_pre_*'
                   or candidate_id glob 'c_adj_*' or candidate_id glob 'c_rep_*')
            """
        )
        conn.execute("update trials set lineage_id = candidate_id where lineage_id is null")
        conn.execute(
            "create index if not exists idx_trials_family_lineage on trials(hypothesis_family, lineage_id)"
        )

    def record_trial(self, record: TrialRecord) -> None:
        with closing(self.connect()) as conn:
            with conn:
                conn.execute(
                    """
                    insert or ignore into trials
                        (trial_id, candidate_id, experiment_id, hypothesis_family, method_id, evaluated_at, lineage_id)
                    values (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.trial_id,
                        record.candidate_id,
                        record.experiment_id,
                        _canonical_family(record.hypothesis_family),
                        record.method_id,
                        record.evaluated_at.astimezone(UTC).isoformat(),
                        record.lineage_id or record.candidate_id,
                    ),
                )

    def trial_counts(self, hypothesis_family: str, now: datetime | None = None, window_days: int = 90) -> tuple[int, int, int]:
        """Distinct-lineage trial counts: N for the E[max SR] deflation is the
        number of independent search paths, so grid/repair variants and
        re-evaluations of the same candidate must not compound it.
        ``lineage_id`` is non-null by migration + record_trial invariant."""
        now = (now or datetime.now(UTC)).astimezone(UTC)
        cutoff = now - timedelta(days=window_days)
        with closing(self.connect()) as conn:
            family_count = conn.execute(
                "select count(distinct lineage_id) from trials where hypothesis_family = ?",
                (_canonical_family(hypothesis_family),),
            ).fetchone()[0]
            rolling_count = conn.execute(
                "select count(distinct lineage_id) from trials where evaluated_at >= ?",
                (cutoff.isoformat(),),
            ).fetchone()[0]
            global_count = conn.execute("select count(distinct lineage_id) from trials").fetchone()[0]
        return int(family_count), int(rolling_count), int(global_count)

    def save_artifact(self, artifact_id: str, kind: str, payload: dict) -> None:
        with closing(self.connect()) as conn:
            with conn:
                conn.execute(
                    """
                    insert or replace into artifacts (artifact_id, kind, payload_json, created_at)
                    values (?, ?, ?, ?)
                    """,
                    (artifact_id, kind, json.dumps(payload, default=str, sort_keys=True), datetime.now(UTC).isoformat()),
                )

    def load_artifact(self, artifact_id: str) -> dict | None:
        with closing(self.connect()) as conn:
            row = conn.execute(
                "select payload_json from artifacts where artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def delete_artifacts_with_prefix(self, prefix: str) -> int:
        """Remove artifacts by id prefix (an abandoned run's checkpoints).
        ``_`` and ``%`` are escaped — run ids contain underscores, which are
        LIKE wildcards and would otherwise over-match."""
        like = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with closing(self.connect()) as conn:
            with conn:
                before = conn.total_changes
                conn.execute("delete from artifacts where artifact_id like ? escape '\\'", (like,))
                return conn.total_changes - before

    def delete_artifacts(self, artifact_ids: set[str]) -> int:
        if not artifact_ids:
            return 0
        with closing(self.connect()) as conn:
            with conn:
                before = conn.total_changes
                conn.executemany(
                    "delete from artifacts where artifact_id = ?",
                    [(artifact_id,) for artifact_id in sorted(artifact_ids)],
                )
                return conn.total_changes - before

    def save_coverage(self, coverage_id: str, payload: dict) -> None:
        with closing(self.connect()) as conn:
            with conn:
                conn.execute(
                    """
                    insert or replace into data_coverage (coverage_id, payload_json, created_at)
                    values (?, ?, ?)
                    """,
                    (coverage_id, json.dumps(payload, default=str, sort_keys=True), datetime.now(UTC).isoformat()),
                )

    def list_coverage(self, limit: int = 500) -> list[dict[str, Any]]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                """
                select coverage_id, payload_json, created_at
                from data_coverage
                order by created_at desc
                limit ?
                """,
                (limit,),
            ).fetchall()
        return [_coverage_row_to_dict(row) for row in rows]

    def create_pipeline_run(self, run_id: str, args: dict[str, Any]) -> None:
        with closing(self.connect()) as conn:
            with conn:
                conn.execute(
                    """
                    insert into pipeline_runs (run_id, status, args_json, started_at)
                    values (?, ?, ?, ?)
                    """,
                    (run_id, "running", json.dumps(args, default=str, sort_keys=True), datetime.now(UTC).isoformat()),
                )

    def update_pipeline_run(self, run_id: str, status: str, error: str | None = None) -> None:
        ended_at = datetime.now(UTC).isoformat() if status in {"completed", "failed", "cancelled", "stopped"} else None
        with closing(self.connect()) as conn:
            with conn:
                if ended_at is None:
                    conn.execute(
                        "update pipeline_runs set status = ?, error = ? where run_id = ?",
                        (status, error, run_id),
                    )
                else:
                    conn.execute(
                        "update pipeline_runs set status = ?, ended_at = ?, error = ? where run_id = ?",
                        (status, ended_at, error, run_id),
                    )

    def pipeline_run_status(self, run_id: str) -> str | None:
        with closing(self.connect()) as conn:
            row = conn.execute(
                "select status from pipeline_runs where run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else str(row["status"])

    def pipeline_run(self, run_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as conn:
            row = conn.execute(
                "select * from pipeline_runs where run_id = ?",
                (run_id,),
            ).fetchone()
        return _run_row_to_dict(row)

    def request_pipeline_stop(self, run_id: str) -> None:
        self.update_pipeline_run(run_id, "stopping")

    def active_pipeline_run(self) -> dict[str, Any] | None:
        with closing(self.connect()) as conn:
            row = conn.execute(
                """
                select * from pipeline_runs
                where status in ('running', 'stopping')
                order by started_at desc
                limit 1
                """
            ).fetchone()
        return _run_row_to_dict(row)

    def list_pipeline_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "select * from pipeline_runs order by started_at desc limit ?",
                (limit,),
            ).fetchall()
        return [_run_row_to_dict(row) for row in rows if row is not None]

    def append_pipeline_event(
        self,
        run_id: str,
        *,
        phase: str,
        message: str,
        level: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        with closing(self.connect()) as conn:
            with conn:
                # Compute seq inside the INSERT so the max(seq) read and the write
                # are one atomic statement under SQLite's single-writer lock. The
                # prior read-then-insert let two concurrent writers observe the
                # same max(seq) and emit duplicate sequence numbers for a run.
                conn.execute(
                    """
                    insert into pipeline_events
                        (event_id, run_id, seq, phase, level, message, payload_json, created_at)
                    values (
                        ?, ?,
                        (select coalesce(max(seq), 0) + 1 from pipeline_events where run_id = ?),
                        ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        str(uuid.uuid4()),
                        run_id,
                        run_id,
                        phase,
                        level,
                        message,
                        json.dumps(payload or {}, default=str, sort_keys=True),
                        datetime.now(UTC).isoformat(),
                    ),
                )

    def load_pipeline_events(self, run_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with closing(self.connect()) as conn:
            rows = conn.execute(
                """
                select * from pipeline_events
                where run_id = ?
                order by seq asc
                limit ?
                """,
                (run_id, limit),
            ).fetchall()
        return [_event_row_to_dict(row) for row in rows]

    def prune_artifacts(
        self,
        *,
        kind: str,
        keep_artifact_ids: set[str] | None = None,
        max_unprotected_rows: int = 0,
    ) -> int:
        """Delete old artifacts of a kind while preserving explicit keepers."""
        keep_artifact_ids = keep_artifact_ids or set()
        with closing(self.connect()) as conn:
            rows = conn.execute(
                """
                select artifact_id
                from artifacts
                where kind = ?
                order by created_at desc
                """,
                (kind,),
            ).fetchall()

            to_delete: list[str] = []
            retained_unprotected = 0
            for row in rows:
                artifact_id = str(row["artifact_id"])
                if artifact_id in keep_artifact_ids:
                    continue
                if retained_unprotected < max_unprotected_rows:
                    retained_unprotected += 1
                    continue
                to_delete.append(artifact_id)

            if not to_delete:
                return 0

            with conn:
                conn.executemany(
                    "delete from artifacts where artifact_id = ?",
                    [(artifact_id,) for artifact_id in to_delete],
                )
            return len(to_delete)

    def vacuum(self) -> None:
        with closing(self.connect()) as conn:
            conn.execute("vacuum")

    def upsert_research_survivors(self, records: list[ResearchSurvivorRecord]) -> None:
        if not records:
            return
        with closing(self.connect()) as conn:
            existing_rows = conn.execute(
                f"select candidate_id, payload_json from research_survivors where candidate_id in ({','.join(['?'] * len(records))})",
                [record.candidate_id for record in records],
            ).fetchall()
            existing = {
                row["candidate_id"]: ResearchSurvivorRecord.model_validate(json.loads(row["payload_json"]))
                for row in existing_rows
            }
            rows = []
            for record in records:
                previous = existing.get(record.candidate_id)
                if previous is not None:
                    # Freshly built round records always default the recheck
                    # bookkeeping to zero — preserve it like the paper clock,
                    # or every upsert would wipe the demotion failure streak.
                    record = record.model_copy(update={
                        "paper_trade_start_date": previous.paper_trade_start_date,
                        "created_at": previous.created_at,
                        "consecutive_recheck_failures": previous.consecutive_recheck_failures,
                        "last_recheck_at": previous.last_recheck_at,
                    })
                rows.append((
                    record.candidate_id,
                    record.experiment_id,
                    record.status,
                    json.dumps(record.model_dump(mode="json"), default=str, sort_keys=True),
                    record.created_at.astimezone(UTC).isoformat(),
                    record.updated_at.astimezone(UTC).isoformat(),
                ))
            with conn:
                conn.executemany(
                    """
                    insert into research_survivors
                        (candidate_id, experiment_id, status, payload_json, created_at, updated_at)
                    values (?, ?, ?, ?, ?, ?)
                    on conflict(candidate_id) do update set
                        experiment_id = excluded.experiment_id,
                        status = excluded.status,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    rows,
                )

    def list_research_survivors(self, status: str | None = "active", limit: int = 200) -> list[ResearchSurvivorRecord]:
        query = """
            select payload_json
            from research_survivors
        """
        params: list[Any] = []
        if status is not None:
            query += " where status = ?"
            params.append(status)
        query += " order by updated_at desc limit ?"
        params.append(limit)
        with closing(self.connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [ResearchSurvivorRecord.model_validate(json.loads(row["payload_json"])) for row in rows]

    def update_research_survivor_status(self, candidate_id: str, status: str, reason: str | None = None) -> None:
        with closing(self.connect()) as conn:
            row = conn.execute(
                "select payload_json from research_survivors where candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                return
            record = ResearchSurvivorRecord.model_validate(json.loads(row["payload_json"]))
            now = datetime.now(UTC)
            updated = record.model_copy(update={
                "status": status,
                "status_reason": reason,
                "updated_at": now,
            })
            with conn:
                conn.execute(
                    """
                    update research_survivors
                    set status = ?, payload_json = ?, updated_at = ?
                    where candidate_id = ?
                    """,
                    (
                        status,
                        json.dumps(updated.model_dump(mode="json"), default=str, sort_keys=True),
                        now.isoformat(),
                        candidate_id,
                    ),
                )

    def update_research_survivor_fields(self, candidate_id: str, **fields: Any) -> None:
        """Patch payload fields on one survivor without rebuilding the record
        (recheck bookkeeping: failure streak, last_recheck_at)."""
        if not fields:
            return
        with closing(self.connect()) as conn:
            row = conn.execute(
                "select payload_json from research_survivors where candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                return
            record = ResearchSurvivorRecord.model_validate(json.loads(row["payload_json"]))
            now = datetime.now(UTC)
            updated = record.model_copy(update={**fields, "updated_at": now})
            with conn:
                conn.execute(
                    "update research_survivors set payload_json = ?, updated_at = ? where candidate_id = ?",
                    (
                        json.dumps(updated.model_dump(mode="json"), default=str, sort_keys=True),
                        now.isoformat(),
                        candidate_id,
                    ),
                )

    # ── Near-miss methods (durable Watch stratum) ───────────────────

    def save_near_misses(self, items: list[NearMissAnalysis], run_id: str | None = None) -> None:
        """Accumulate near-misses across runs. The per-round artifact is
        insert-or-replace on artifact_scope and vanishes on the next run;
        this table is what makes the Watch stratum survive."""
        if not items:
            return
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                f"{item.experiment_id}:{item.candidate_id}",
                item.candidate_id,
                item.experiment_id,
                run_id,
                item.primary_reason,
                int(item.actionable),
                json.dumps(item.model_dump(mode="json"), default=str, sort_keys=True),
                now,
            )
            for item in items
        ]
        with closing(self.connect()) as conn:
            with conn:
                conn.executemany(
                    """
                    insert into near_misses
                        (id, candidate_id, experiment_id, run_id, primary_reason, actionable, payload_json, created_at)
                    values (?, ?, ?, ?, ?, ?, ?, ?)
                    on conflict(id) do update set
                        run_id = excluded.run_id,
                        primary_reason = excluded.primary_reason,
                        actionable = excluded.actionable,
                        payload_json = excluded.payload_json,
                        created_at = excluded.created_at
                    """,
                    rows,
                )

    def list_near_misses(
        self,
        *,
        since_days: int | None = None,
        actionable_only: bool = False,
        limit: int = 500,
    ) -> list[NearMissAnalysis]:
        query = "select payload_json from near_misses"
        clauses: list[str] = []
        params: list[Any] = []
        if since_days is not None:
            clauses.append("created_at >= ?")
            params.append((datetime.now(UTC) - timedelta(days=since_days)).isoformat())
        if actionable_only:
            clauses.append("actionable = 1")
        if clauses:
            query += " where " + " and ".join(clauses)
        query += " order by created_at desc limit ?"
        params.append(limit)
        with closing(self.connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [NearMissAnalysis.model_validate(json.loads(row["payload_json"])) for row in rows]

    def prune_near_misses(self, *, before_days: int) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=before_days)).isoformat()
        with closing(self.connect()) as conn:
            with conn:
                before = conn.total_changes
                conn.execute("delete from near_misses where created_at < ?", (cutoff,))
                return conn.total_changes - before

    # ── Trajectory methods ──────────────────────────────────────────

    def save_trajectory(self, record: TrajectoryRecord) -> None:
        with closing(self.connect()) as conn:
            with conn:
                conn.execute(
                    """
                    insert or replace into trajectories
                        (trajectory_id, candidate_id, operator, payload_json, created_at)
                    values (?, ?, ?, ?, ?)
                    """,
                    (
                        record.trajectory_id,
                        record.candidate_id,
                        record.operator,
                        json.dumps(record.model_dump(mode="json"), default=str, sort_keys=True),
                        record.created_at.astimezone(UTC).isoformat(),
                    ),
                )

    def load_trajectory(self, trajectory_id: str) -> TrajectoryRecord | None:
        with closing(self.connect()) as conn:
            row = conn.execute(
                "select payload_json from trajectories where trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
        if row is None:
            return None
        return TrajectoryRecord.model_validate(json.loads(row["payload_json"]))

    def list_trajectories(
        self,
        *,
        candidate_id: str | None = None,
        operator: str | None = None,
        limit: int = 200,
    ) -> list[TrajectoryRecord]:
        query = "select payload_json from trajectories"
        params: list[Any] = []
        conditions: list[str] = []
        if candidate_id is not None:
            conditions.append("candidate_id = ?")
            params.append(candidate_id)
        if operator is not None:
            conditions.append("operator = ?")
            params.append(operator)
        if conditions:
            query += " where " + " and ".join(conditions)
        query += " order by created_at desc limit ?"
        params.append(limit)
        with closing(self.connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [TrajectoryRecord.model_validate(json.loads(row["payload_json"])) for row in rows]

    def prune_trajectories(
        self,
        *,
        keep_ids: set[str] | None = None,
        max_unprotected_rows: int = 500,
    ) -> int:
        keep_ids = keep_ids or set()
        with closing(self.connect()) as conn:
            rows = conn.execute(
                "select trajectory_id from trajectories order by created_at desc",
            ).fetchall()
            to_delete: list[str] = []
            retained_unprotected = 0
            for row in rows:
                tid = str(row["trajectory_id"])
                if tid in keep_ids:
                    continue
                if retained_unprotected < max_unprotected_rows:
                    retained_unprotected += 1
                    continue
                to_delete.append(tid)
            if not to_delete:
                return 0
            with conn:
                conn.executemany(
                    "delete from trajectories where trajectory_id = ?",
                    [(tid,) for tid in to_delete],
                )
            return len(to_delete)


def ensure_project_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _run_row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "run_id": row["run_id"],
        "status": row["status"],
        "args": json.loads(row["args_json"]),
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "error": row["error"],
    }


def _event_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "run_id": row["run_id"],
        "seq": row["seq"],
        "phase": row["phase"],
        "level": row["level"],
        "message": row["message"],
        "payload": json.loads(row["payload_json"]),
        "created_at": row["created_at"],
    }


def _coverage_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"])
    payload["coverage_id"] = row["coverage_id"]
    payload["created_at"] = row["created_at"]
    return payload
