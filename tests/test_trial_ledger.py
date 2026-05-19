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

