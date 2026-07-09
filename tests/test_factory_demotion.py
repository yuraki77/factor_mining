"""Demotion gives the recheck loop teeth: without it, survivors whose edge
decayed sit on the Provisional shelf forever as zombies. Only holdout-grade
evaluations (allow_promotion=True: terminal final-OOS or verify-survivors)
may move the failure streak — per-round maintenance re-sees the same data and
would otherwise book several "consecutive" failures in a single run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from factor_mining.config import Settings
from factor_mining.models import (
    BacktestResult,
    MetricsBlock,
    ResearchGateResult,
    ResearchSurvivorRecord,
)
from factor_mining.pipeline import _update_research_survivor_store
from factor_mining.storage import MetadataStore

_CID = "cand-decay"
_EID = f"exp-{_CID}"


def _result(oos_trades: int = 10) -> BacktestResult:
    return BacktestResult(
        experiment_id=_EID,
        candidate_id=_CID,
        hypothesis_family="momentum",
        method_id="factor_scoring",
        symbol="BTCUSDT",
        market="um_futures",
        interval="5m",
        metrics_primary=MetricsBlock(sharpe=0.1),
        metrics_gross=MetricsBlock(sharpe=0.2),
        factor_turnover=0.1,
        break_even_cost_bps=1.0,
        actual_cost_bps=2.0,
        oos_trade_count=oos_trades,
    )


def _gate(status: str) -> ResearchGateResult:
    return ResearchGateResult(experiment_id=_EID, candidate_id=_CID, status=status)


def _seeded_store(tmp_path) -> MetadataStore:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.upsert_research_survivors([ResearchSurvivorRecord(candidate_id=_CID, experiment_id=_EID)])
    return store


def _recheck(store: MetadataStore, *, status: str, allow_promotion: bool) -> None:
    _update_research_survivor_store(
        store=store,
        records=[],
        rechecked_candidate_ids={_CID},
        research_gates=[_gate(status)],
        results=[_result()],  # oos_trades < min keeps the hard-retire rule out of play
        fdr_map={_EID: 0.5},  # above promotion threshold: never promotes
        settings=Settings(),
        allow_promotion=allow_promotion,
    )


def test_k_consecutive_holdout_failures_demote_to_retired(tmp_path) -> None:
    store = _seeded_store(tmp_path)
    for expected_streak in (1, 2):
        _recheck(store, status="rejected", allow_promotion=True)
        (record,) = store.list_research_survivors(status=None)
        assert record.status == "active"
        assert record.consecutive_recheck_failures == expected_streak
        assert record.last_recheck_at is not None

    _recheck(store, status="rejected", allow_promotion=True)  # K=3 default
    (record,) = store.list_research_survivors(status=None)
    assert record.status == "retired"
    assert record.status_reason == "demoted_3_consecutive_recheck_failures"


def test_surviving_a_recheck_resets_the_streak(tmp_path) -> None:
    """The rule is CONSECUTIVE failures — edge that recovers on fresh data is
    exactly what the factory exists to keep, so one pass wipes the slate."""
    store = _seeded_store(tmp_path)
    _recheck(store, status="rejected", allow_promotion=True)
    _recheck(store, status="rejected", allow_promotion=True)
    _recheck(store, status="research_survivor", allow_promotion=True)
    (record,) = store.list_research_survivors(status=None)
    assert record.status == "active"
    assert record.consecutive_recheck_failures == 0


def test_per_round_maintenance_never_moves_the_streak(tmp_path) -> None:
    """allow_promotion=False evaluations re-see the same validation slice; if
    they counted, one multi-round run could demote a survivor in an afternoon
    without any new OOS evidence."""
    store = _seeded_store(tmp_path)
    for _ in range(5):
        _recheck(store, status="rejected", allow_promotion=False)
    (record,) = store.list_research_survivors(status=None)
    assert record.status == "active"
    assert record.consecutive_recheck_failures == 0


def test_upsert_preserves_failure_streak_like_the_paper_clock(tmp_path) -> None:
    """Round records are rebuilt fresh (streak defaults to 0); the upsert must
    carry the stored streak forward or every round would grant amnesty."""
    store = _seeded_store(tmp_path)
    _recheck(store, status="rejected", allow_promotion=True)
    _recheck(store, status="rejected", allow_promotion=True)

    rebuilt = ResearchSurvivorRecord(candidate_id=_CID, experiment_id=_EID, current_trades=42)
    assert rebuilt.consecutive_recheck_failures == 0
    store.upsert_research_survivors([rebuilt])

    (record,) = store.list_research_survivors(status=None)
    assert record.current_trades == 42
    assert record.consecutive_recheck_failures == 2


def _aged_store(tmp_path, *, days: int = 120) -> MetadataStore:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.upsert_research_survivors([
        ResearchSurvivorRecord(
            candidate_id=_CID,
            experiment_id=_EID,
            paper_trade_start_date=datetime.now(UTC) - timedelta(days=days),
        )
    ])
    return store


def _ladder_recheck(store: MetadataStore, *, break_even_bps: float, sharpe: float) -> None:
    result = _result(oos_trades=150).model_copy(update={
        "metrics_primary": MetricsBlock(sharpe=sharpe),
        "break_even_cost_bps": break_even_bps,
        "actual_cost_bps": 6.0,
    })
    _update_research_survivor_store(
        store=store,
        records=[],
        rechecked_candidate_ids={_CID},
        research_gates=[_gate("research_survivor")],
        results=[result],
        fdr_map={_EID: 0.01},  # gross signal clearly "significant"
        settings=Settings(),
        allow_promotion=True,
    )


def test_ladder_promotion_requires_positive_cost_margin_and_net_sharpe(tmp_path) -> None:
    """FDR measures GROSS predictive power. Post-reset shelves hold survivors
    with 1000+ trades, tiny FDR p, and cost_margin ~ -12bps — without the net
    terms they would age 90 days and promote into Validated while losing
    money after costs."""
    store = _aged_store(tmp_path)
    _ladder_recheck(store, break_even_bps=1.0, sharpe=1.0)  # margin 1 - 2*6 = -11
    (record,) = store.list_research_survivors(status=None)
    assert record.status == "active", "negative cost margin must block promotion"

    _ladder_recheck(store, break_even_bps=30.0, sharpe=-0.5)  # margin +18, net negative
    (record,) = store.list_research_survivors(status=None)
    assert record.status == "active", "negative net sharpe must block promotion"

    _ladder_recheck(store, break_even_bps=30.0, sharpe=1.0)
    (record,) = store.list_research_survivors(status=None)
    assert record.status == "promoted"
    assert record.status_reason == "promotion_criteria_met"


def test_recheck_finds_paper_clock_behind_a_large_retired_backlog(tmp_path) -> None:
    """After a reset the retired backlog (26k+) dwarfs the active shelf. The
    store-side promotion/demotion must load the ACTIVE shelf, not status=None
    with the default 200-row window — otherwise the retired rows crowd out
    active survivors' paper clocks and freeze their ladder progress. Here an
    aged, cost-positive survivor sitting behind 300 retired rows must still
    promote."""
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.upsert_research_survivors([
        ResearchSurvivorRecord(candidate_id=f"dead-{i}", experiment_id=f"e-{i}", status="retired")
        for i in range(300)
    ])
    store.upsert_research_survivors([
        ResearchSurvivorRecord(
            candidate_id=_CID,
            experiment_id=_EID,
            paper_trade_start_date=datetime.now(UTC) - timedelta(days=120),
        )
    ])
    result = _result(oos_trades=150).model_copy(update={
        "metrics_primary": MetricsBlock(sharpe=1.0),
        "break_even_cost_bps": 30.0,
        "actual_cost_bps": 6.0,
    })
    _update_research_survivor_store(
        store=store,
        records=[],
        rechecked_candidate_ids={_CID},
        research_gates=[_gate("research_survivor")],
        results=[result],
        fdr_map={_EID: 0.01},
        settings=Settings(),
        allow_promotion=True,
    )
    promoted = {r.candidate_id: r for r in store.list_research_survivors(status="promoted", limit=10_000)}
    assert _CID in promoted, "aged survivor behind the retired backlog must still promote"


def test_active_survivor_listing_scales_past_default_limit(tmp_path) -> None:
    """Recheck paths must see the WHOLE shelf: under the old default limit of
    200, survivor #201+ was silently never rechecked — and therefore never
    demoted — so the shelf could only grow."""
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.upsert_research_survivors([
        ResearchSurvivorRecord(candidate_id=f"c{i}", experiment_id=f"e{i}") for i in range(250)
    ])
    assert len(store.list_research_survivors(status="active")) == 200
    assert len(store.list_research_survivors(status="active", limit=10_000)) == 250


def test_pre_factory_payloads_still_parse() -> None:
    """Rows written before the factory fields existed must load unchanged."""
    legacy = ResearchSurvivorRecord(candidate_id=_CID, experiment_id=_EID).model_dump(mode="json")
    legacy.pop("consecutive_recheck_failures")
    legacy.pop("last_recheck_at")
    parsed = ResearchSurvivorRecord.model_validate(legacy)
    assert parsed.consecutive_recheck_failures == 0
    assert parsed.last_recheck_at is None
