from __future__ import annotations

from datetime import datetime

from factor_mining.config import Settings
from factor_mining.models import TrialRecord
from factor_mining.storage import MetadataStore


class TrialLedger:
    def __init__(self, store: MetadataStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def record(self, record: TrialRecord) -> None:
        self.store.record_trial(record)

    def counts_for(self, hypothesis_family: str, now: datetime | None = None) -> dict[str, int]:
        family_count, rolling_count, global_count = self.store.trial_counts(
            hypothesis_family=hypothesis_family,
            now=now,
            window_days=self.settings.trial_ledger.window_days,
        )
        return {
            "family_trials_count": family_count,
            "rolling_90d_trials_count": rolling_count,
            "effective_trials_count": max(family_count, rolling_count),
            "global_cumulative_trials_count": global_count,
        }

