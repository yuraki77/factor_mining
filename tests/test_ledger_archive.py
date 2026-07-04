"""WS6(b) — the statistical reset. Dev-phase trials were spent debugging the
machine, not searching for alpha; left in the live ledger they would
permanently haircut every real discovery through the expected-max-Sharpe
penalty. The reset must be explicit (--yes), atomic, and preserving: rows
move to trials_archive as the honest historical record, they are not deleted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

import factor_mining.cli as cli
from factor_mining.config import DataConfig, Settings
from factor_mining.models import TrialRecord
from factor_mining.storage import MetadataStore
from factor_mining.trial_ledger import TrialLedger


def _seed_dev_and_live_trials(store: MetadataStore, settings: Settings) -> datetime:
    ledger = TrialLedger(store, settings)
    now = datetime.now(UTC)
    for idx in range(4):
        ledger.record(
            TrialRecord(
                trial_id=f"dev-{idx}",
                candidate_id=f"c-dev-{idx}",
                hypothesis_family="momentum",
                method_id="factor_scoring",
                evaluated_at=now - timedelta(days=30),
            )
        )
    for idx in range(2):
        ledger.record(
            TrialRecord(
                trial_id=f"live-{idx}",
                candidate_id=f"c-live-{idx}",
                hypothesis_family="momentum",
                method_id="factor_scoring",
                evaluated_at=now,
            )
        )
    return now


def test_archive_resets_live_counts_and_preserves_history(tmp_path) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    now = _seed_dev_and_live_trials(store, settings)
    assert store.trial_counts("momentum")[0] == 6

    cutoff = now - timedelta(days=1)
    hot_rows, hot_lineages, move_rows, move_lineages = store.trial_archive_preview(cutoff)
    assert (hot_rows, hot_lineages) == (6, 6)
    assert (move_rows, move_lineages) == (4, 4)

    moved = store.archive_trials(cutoff)
    assert moved == 4
    family, _rolling, global_count = store.trial_counts("momentum")
    # the reset in one line: G1's N and the FDR denominator see only
    # post-cutoff lineages from here on
    assert family == 2
    assert global_count == 2
    # ...but nothing was destroyed: the archive keeps the full history
    assert store.archived_trial_lineage_count() == 4


def test_cli_refuses_without_yes_then_archives_with_it(tmp_path, monkeypatch) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    store = MetadataStore(settings.data.sqlite_path)
    _seed_dev_and_live_trials(store, settings)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")

    refused = CliRunner().invoke(cli.app, ["ledger", "archive", "--before", tomorrow])
    assert refused.exit_code != 0
    assert store.trial_counts("momentum")[0] == 6, "no rows may move without --yes"

    accepted = CliRunner().invoke(cli.app, ["ledger", "archive", "--before", tomorrow, "--yes"])
    assert accepted.exit_code == 0
    assert store.trial_counts("momentum")[0] == 0
    assert store.archived_trial_lineage_count() == 6


def test_cli_rejects_malformed_cutoff(tmp_path, monkeypatch) -> None:
    settings = Settings(data=DataConfig(sqlite_path=tmp_path / "meta.sqlite3"))
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    result = CliRunner().invoke(cli.app, ["ledger", "archive", "--before", "last tuesday", "--yes"])
    assert result.exit_code != 0
