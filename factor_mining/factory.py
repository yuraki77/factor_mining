"""Continuous discovery factory: arrival detection and factory state.

The factory inverts the batch-mining posture: discovery rounds are triggered
by the arrival of new market data (never by a hot loop), each spends a fixed
trial budget, and between rounds the worker only rechecks existing survivors.
State lives in a single ``factory_state`` artifact so no schema change is
needed and MetadataStore's surface stays untouched.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection, Iterable
from datetime import UTC, datetime
from typing import Any

from factor_mining.config import Settings
from factor_mining.data.loader import data_end_ms
from factor_mining.storage import MetadataStore, _canonical_family

FACTORY_STATE_ARTIFACT_ID = "factory_state"
_MS_PER_DAY = 86_400_000.0


def extent_key(symbol: str, market: str) -> str:
    return f"{symbol}/{market}"


def load_factory_state(store: MetadataStore) -> dict[str, Any]:
    payload = store.load_artifact(FACTORY_STATE_ARTIFACT_ID)
    if not payload:
        return {"schema_version": 1, "last_discovery_extent_ms": {}}
    payload.setdefault("schema_version", 1)
    payload.setdefault("last_discovery_extent_ms", {})
    return payload


def save_factory_state(store: MetadataStore, state: dict[str, Any]) -> None:
    store.save_artifact(FACTORY_STATE_ARTIFACT_ID, "factory_state", state)


def current_extents_ms(settings: Settings) -> dict[str, int]:
    """Cheap max-open_time per tracked symbol/market; 0 means no data on disk."""
    extents: dict[str, int] = {}
    for symbol in settings.data.symbols:
        for market in settings.data.markets:
            extents[extent_key(symbol, market)] = data_end_ms(settings, symbol=symbol, market=market)
    return extents


def new_days_since_discovery(current: dict[str, int], last: dict[str, Any]) -> dict[str, float]:
    """Days of new data per key, only for keys with a prior recorded extent."""
    deltas: dict[str, float] = {}
    for key, end_ms in current.items():
        prior = int(last.get(key, 0) or 0)
        if prior <= 0 or end_ms <= 0:
            continue
        deltas[key] = max(0.0, (end_ms - prior) / _MS_PER_DAY)
    return deltas


def discovery_due(current: dict[str, int], last: dict[str, Any], min_new_days: float) -> bool:
    """Discovery fires only when every tracked key with data has accrued at
    least ``min_new_days`` of unseen bars (min over keys): a budgeted round
    should spend its trials on a coherent increment of fresh out-of-sample
    data, not because one lagging feed dripped a few bars. First run (no
    extent ever recorded) always fires — that is factory go-live.
    """
    tracked = {key: value for key, value in current.items() if value > 0}
    if not tracked:
        return False
    known = [key for key in tracked if int(last.get(key, 0) or 0) > 0]
    if not known:
        return True
    deltas = new_days_since_discovery(tracked, last)
    return bool(deltas) and min(deltas.values()) >= float(min_new_days)


def record_discovery_completed(store: MetadataStore, *, run_id: str, extents_ms: dict[str, int]) -> None:
    state = load_factory_state(store)
    state["last_discovery_extent_ms"] = {key: int(value) for key, value in extents_ms.items() if value > 0}
    state["last_discovery_run_id"] = run_id
    state["last_discovery_at"] = datetime.now(UTC).isoformat()
    save_factory_state(store, state)


def record_recheck_completed(store: MetadataStore, *, run_id: str) -> None:
    state = load_factory_state(store)
    state["last_recheck_run_id"] = run_id
    state["last_recheck_at"] = datetime.now(UTC).isoformat()
    save_factory_state(store, state)


class RoundBudget:
    """Per-discovery-round cap on NEW search lineages.

    Every admitted fresh lineage permanently raises the expected-max-Sharpe
    deflation for all future candidates, so the cap binds exactly where trials
    enter the ledger. Derived candidates (grid variants, repairs, optimizer
    offspring) inherit their root's lineage and are always admitted — they do
    not raise the independent-trial count N. Survivor rechecks are likewise
    free. The cap is split per hypothesis family so one prolific generator
    cannot starve the rest and the family-stratified FDR keeps a balanced
    denominator.
    """

    def __init__(self, total: int, families: Iterable[str], *, family_floor: int = 0) -> None:
        canonical = sorted({_canonical_family(family) for family in families if family})
        if canonical:
            quota = max(int(family_floor), int(total) // len(canonical))
        else:
            quota = max(int(family_floor), int(total))
        self._default_quota = quota
        self._remaining: dict[str, int] = {family: quota for family in canonical}
        self._lock = threading.Lock()

    def admit(
        self,
        candidates: list[Any],
        *,
        lineage_root: Callable[[Any], str],
        survivor_candidate_ids: Collection[str] = frozenset(),
    ) -> tuple[list[Any], dict[str, int]]:
        """Split *candidates* (order preserved) into admitted and the count of
        fresh-lineage drops by family. ``lineage_root`` is passed in rather
        than imported so the pipeline's lineage rule stays the single source
        of truth."""
        admitted: list[Any] = []
        dropped: dict[str, int] = {}
        with self._lock:
            for candidate in candidates:
                is_fresh = lineage_root(candidate) == candidate.candidate_id
                if not is_fresh or candidate.candidate_id in survivor_candidate_ids:
                    admitted.append(candidate)
                    continue
                family = _canonical_family(candidate.hypothesis_family)
                remaining = self._remaining.setdefault(family, self._default_quota)
                if remaining > 0:
                    self._remaining[family] = remaining - 1
                    admitted.append(candidate)
                else:
                    dropped[family] = dropped.get(family, 0) + 1
        return admitted, dropped
