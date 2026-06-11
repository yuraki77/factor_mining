"""DSL fingerprint novelty tracking."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from factor_mining.storage import MetadataStore

logger = logging.getLogger(__name__)


class FingerprintStore:
    """Deduplicate canonical DSL fingerprints across archive/survivor/batch sets."""

    def __init__(self, store: MetadataStore | None = None) -> None:
        self._store = store
        self._batch: set[str] = set()
        self._batch_candidates: dict[str, set[str]] = defaultdict(set)
        self._parent_bank: set[str] = set()
        self._archive_cache: set[str] | None = None
        self._survivor_cache: set[str] | None = None

    def register(self, fingerprint: str, candidate_id: str) -> None:
        self._batch.add(fingerprint)
        self._batch_candidates[fingerprint].add(candidate_id)

    def register_parent(self, fingerprint: str) -> None:
        self._parent_bank.add(fingerprint)

    def is_novel(
        self,
        fingerprint: str,
        *,
        archive_fingerprints: set[str] | None = None,
        survivor_fingerprints: set[str] | None = None,
        batch_fingerprints: set[str] | None = None,
        parent_bank_fingerprints: set[str] | None = None,
    ) -> bool:
        """Return True when *fingerprint* has not already appeared.

        Parent fingerprints are registered for caller diagnostics, but archive,
        survivor, and batch duplicates are rejected uniformly.
        """
        del parent_bank_fingerprints

        archive = set(archive_fingerprints) if archive_fingerprints is not None else self.load_archive_fingerprints()
        survivor = set(survivor_fingerprints) if survivor_fingerprints is not None else self.load_survivor_fingerprints()
        batch = set(batch_fingerprints) if batch_fingerprints is not None else set(self._batch)

        if fingerprint in batch:
            return False
        if fingerprint in archive:
            return False
        if fingerprint in survivor:
            return False
        return True

    def candidate_ids_for(self, fingerprint: str) -> set[str]:
        return set(self._batch_candidates.get(fingerprint, set()))

    def load_archive_fingerprints(self, store: MetadataStore | None = None, limit: int = 500) -> set[str]:
        if self._archive_cache is not None:
            return self._archive_cache
        del limit
        metadata_store = store or self._store
        if metadata_store is None:
            return set()
        fingerprints: set[str] = set()
        try:
            for artifact_name in ("initial_candidates", "latest_candidates"):
                payload = metadata_store.load_artifact(artifact_name)
                if isinstance(payload, dict):
                    for item in payload.get("items", []) or []:
                        fp = _extract_fingerprint(item)
                        if fp:
                            fingerprints.add(fp)
        except Exception:
            logger.warning("failed to load archived DSL fingerprints", exc_info=True)
        self._archive_cache = fingerprints
        return fingerprints

    def load_survivor_fingerprints(self, store: MetadataStore | None = None) -> set[str]:
        if self._survivor_cache is not None:
            return self._survivor_cache
        metadata_store = store or self._store
        if metadata_store is None:
            return set()
        fingerprints: set[str] = set()
        try:
            for survivor in metadata_store.list_research_survivors(status="active", limit=500):
                payload = survivor.candidate_payload
                if isinstance(payload, dict):
                    fp = _extract_fingerprint(payload)
                    if fp:
                        fingerprints.add(fp)
        except Exception:
            logger.warning("failed to load survivor DSL fingerprints", exc_info=True)
        self._survivor_cache = fingerprints
        return fingerprints

    def clear_caches(self) -> None:
        self._archive_cache = None
        self._survivor_cache = None
        self._batch.clear()
        self._batch_candidates.clear()
        self._parent_bank.clear()


def _extract_fingerprint(item: dict[str, Any]) -> str | None:
    fp = item.get("dsl_fingerprint")
    if isinstance(fp, str) and fp:
        return fp
    params = item.get("params")
    if isinstance(params, dict):
        fp = params.get("dsl_fingerprint")
        if isinstance(fp, str) and fp:
            return fp
    return None
