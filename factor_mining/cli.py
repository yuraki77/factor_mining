from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer

from factor_mining.archive import reproduce_archive
from factor_mining.config import load_settings
from factor_mining.data.binance import BinanceArchiveClient
from factor_mining.data.universe import resolve_universe
from factor_mining.llm.providers import provider_from_settings
from factor_mining.registry import METHOD_REGISTRY, schedulable_methods
from factor_mining.storage import MetadataStore, ensure_project_dirs
from factor_mining.worker import run_worker


app = typer.Typer(help="Factor Mining Automation v1")
data_app = typer.Typer(help="Data commands")
mine_app = typer.Typer(help="Mining commands")
gate_app = typer.Typer(help="GateCheck commands")
hardscore_app = typer.Typer(help="HardScore commands")
exp_app = typer.Typer(help="Experiment archive commands")
worker_app = typer.Typer(help="Worker commands")
llm_app = typer.Typer(help="LLM provider commands")

app.add_typer(data_app, name="data")
app.add_typer(mine_app, name="mine")
app.add_typer(gate_app, name="gate")
app.add_typer(hardscore_app, name="hardscore")
app.add_typer(exp_app, name="exp")
app.add_typer(worker_app, name="worker")
app.add_typer(llm_app, name="llm")


@app.callback()
def main() -> None:
    """Local-first rigorous factor mining."""


@data_app.command("sync")
def data_sync(
    dry_run: bool = typer.Option(False, help="Plan downloads without writing files."),
    interval: str | None = typer.Option(None, help="Override interval, e.g. 5m or 1m."),
    start: str | None = typer.Option(None, help="Start month date, YYYY-MM-DD."),
    end: str | None = typer.Option(None, help="End month date, YYYY-MM-DD."),
    symbols: str | None = typer.Option(None, help="Comma-separated symbol override, e.g. BTCUSDT,ETHUSDT."),
    universe: str | None = typer.Option(None, help="Universe preset, e.g. um_liquid_30."),
    markets: str | None = typer.Option(None, help="Comma-separated market override: spot,um_futures."),
    source: str = typer.Option("archive", help="Data source: archive or rest."),
    supplemental: bool = typer.Option(False, help="With --source rest, also sync USD-M mark/index/premium/OI/sentiment datasets."),
) -> None:
    settings = load_settings()
    store = MetadataStore(settings.data.sqlite_path)
    ensure_project_dirs([settings.data.raw_dir, settings.data.parquet_dir, settings.data.sqlite_path.parent])
    client = BinanceArchiveClient(settings, store)
    sync_symbols = _resolve_cli_symbols(symbols, universe)
    sync_markets = _parse_csv(markets)
    if source == "archive":
        records = client.sync(symbols=sync_symbols, markets=sync_markets, interval=interval, start=_parse_date(start), end=_parse_date(end), dry_run=dry_run)
    elif source == "rest":
        records = client.sync_rest(symbols=sync_symbols, markets=sync_markets, interval=interval, start=_parse_date(start), end=_parse_date(end), supplemental=supplemental, dry_run=dry_run)
    else:
        raise typer.BadParameter("source must be 'archive' or 'rest'")
    typer.echo(f"Processed {len(records)} {source} assets")
    for record in records[:20]:
        typer.echo(f"{record.status:10s} {record.market}/{record.dataset}/{record.symbol}/{record.interval} {record.year}-{record.month:02d}")
    if len(records) > 20:
        typer.echo(f"... {len(records) - 20} more")


@mine_app.command("run")
def mine_run(
    use_llm: bool = typer.Option(False, "--llm", help="Use DeepSeek to generate first-principles hypotheses."),
    hypothesis_count: int = typer.Option(5, "--hypotheses", help="Number of LLM hypotheses to request."),
    research_brief: str | None = typer.Option(None, "--brief", help="Optional research brief for DeepSeek."),
    symbols: str | None = typer.Option(None, help="Comma-separated symbol override, e.g. BTCUSDT,ETHUSDT."),
    universe: str | None = typer.Option(None, help="Universe preset, e.g. um_liquid_30."),
    max_workers: int | None = typer.Option(None, "--workers", help="Number of parallel backtest workers."),
    tail: int | None = typer.Option(None, "--tail", help="Use only last N rows of data (faster dev runs)."),
    archive_top: int = typer.Option(3, "--archive", help="Number of top experiments to archive."),
    iterations: int = typer.Option(1, "--iterations", help="Max mining rounds (1=single pass, >1=iterative MiniMax optimization)."),
) -> None:
    """Run the full factor mining pipeline: hypotheses → backtest → gatecheck → hardscore → optimize.

    With --iterations > 1, MiniMax suggestions are backtested in subsequent rounds
    until convergence or the iteration limit is reached.
    """
    from factor_mining.pipeline import run_pipeline

    settings = load_settings()
    run_symbols = _resolve_cli_symbols(symbols, universe)
    if run_symbols is not None:
        settings = settings.model_copy(update={"data": settings.data.model_copy(update={"symbols": run_symbols})})
    store = MetadataStore(settings.data.sqlite_path)

    result = run_pipeline(
        settings,
        use_llm=use_llm,
        max_workers=max_workers,
        tail=tail,
        archive_top=archive_top,
        research_brief=research_brief,
        hypothesis_count=hypothesis_count,
        iterations=iterations,
        store=store,
    )

    if result.errors:
        typer.echo(f"\n⚠  {len(result.errors)} error(s) during pipeline run (see above).")
        raise typer.Exit(code=1)

    typer.echo(f"\n✓ Pipeline complete in {result.elapsed_s:.0f}s. "
               f"{result.n_gatecheck_passed}/{len(result.gatechecks)} gatecheck passed. "
               f"Top score: {result.top_candidates[0][2].score:.1f}" if result.top_candidates else "")


@gate_app.command("run")
def gate_run() -> None:
    """Re-run GateCheck on cached backtest results."""
    settings = load_settings()
    store = MetadataStore(settings.data.sqlite_path)

    bt_artifact = store.load_artifact("latest_backtests")
    if bt_artifact is None:
        typer.echo("No cached backtest results found. Run 'fm mine run' first.")
        raise typer.Exit(code=1)

    from factor_mining.models import BacktestResult
    from factor_mining.validation.gatecheck import apply_fdr, run_gatecheck
    from factor_mining.registry import METHOD_REGISTRY, get_method
    from factor_mining.stats.metrics import combined_ic_tstat_pvalue

    results = [BacktestResult.model_validate(r) for r in bt_artifact["items"]]
    methods_map = {m.method_id: m for m in METHOD_REGISTRY}
    fdr_map = apply_fdr(results, settings)
    passed = 0

    for r in results:
        method = methods_map.get(r.method_id) or get_method(r.method_id)
        fdr_p = fdr_map.get(r.experiment_id, combined_ic_tstat_pvalue(r.ic_tstat_nw, r.rankic_tstat_nw))
        gc = run_gatecheck(r, settings, method=method, fdr_adjusted_pvalue=fdr_p)
        if gc.passed:
            passed += 1
        else:
            fail_ids = [item.rule_id for item in gc.failures]
            typer.echo(f"  FAIL {r.candidate_id[:16]}... {fail_ids} "
                       f"SR={r.metrics_primary.sharpe:+.2f} DSR={r.deflated_sharpe:+.3f}")

    typer.echo(f"GateCheck: {passed}/{len(results)} passed")


@hardscore_app.command("run")
def hardscore_run() -> None:
    """Re-run HardScore on cached backtest + gatecheck results."""
    settings = load_settings()
    store = MetadataStore(settings.data.sqlite_path)

    bt_artifact = store.load_artifact("latest_backtests")
    gc_artifact = store.load_artifact("latest_gatechecks")
    if bt_artifact is None or gc_artifact is None:
        typer.echo("No cached results found. Run 'fm mine run' first.")
        raise typer.Exit(code=1)

    from factor_mining.models import BacktestResult, GateCheckResult
    from factor_mining.hardscore import hardscore
    from factor_mining.validation.gatecheck import apply_fdr
    from factor_mining.stats.metrics import combined_ic_tstat_pvalue

    results = [BacktestResult.model_validate(r) for r in bt_artifact["items"]]
    gatechecks = [GateCheckResult.model_validate(g) for g in gc_artifact["items"]]
    fdr_map = apply_fdr(results, settings)

    scores = []
    for r, g in zip(results, gatechecks):
        fdr_p = fdr_map.get(r.experiment_id, combined_ic_tstat_pvalue(r.ic_tstat_nw, r.rankic_tstat_nw))
        hs = hardscore(r, g, fdr_adjusted_pvalue=fdr_p, settings=settings)
        scores.append(hs)

    for hs in sorted(scores, key=lambda s: s.score, reverse=True):
        if hs.score > 0:
            typer.echo(f"  score={hs.score:.1f} haircut={hs.haircut_sharpe:+.3f} fdr_p={hs.fdr_adjusted_pvalue:.4f}")

    top = [s for s in scores if s.score > 0][:5]
    if top:
        typer.echo(f"Top score: {top[0].score:.1f}")


@exp_app.command("reproduce")
def exp_reproduce(experiment_id: str, root: Path = Path("archives")) -> None:
    result = reproduce_archive(experiment_id, root=root)
    typer.echo(result)


@worker_app.command("run")
def worker_run() -> None:
    run_worker()


@llm_app.command("check")
def llm_check() -> None:
    settings = load_settings()
    deepseek = provider_from_settings("deepseek", settings)
    minimax = provider_from_settings("minimax", settings)
    typer.echo(f"DeepSeek: {'configured' if deepseek.is_configured else 'missing'} ({deepseek.api_key_env})")
    typer.echo(f"  base_url={settings.llm.deepseek.base_url}")
    typer.echo(f"  hypothesis_model={settings.llm.deepseek.hypothesis_model}")
    typer.echo(f"  hardscore_model={settings.llm.deepseek.hardscore_model}")
    typer.echo(f"MiniMax: {'configured' if minimax.is_configured else 'missing'} ({minimax.api_key_env})")
    typer.echo(f"  base_url={settings.llm.minimax.base_url}")
    typer.echo(f"  optimizer_model={settings.llm.minimax.optimizer_model}")


@app.command("ui")
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Dashboard host."),
    port: int = typer.Option(8501, "--port", help="Dashboard port. If busy, the next free port is used."),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the dashboard in a browser."),
) -> None:
    from factor_mining.ui import run_dashboard

    run_dashboard(host=host, port=port, open_browser=open_browser)


@app.command("methods")
def methods(all_methods: bool = typer.Option(False, "--all", help="Show all registry methods.")) -> None:
    methods_to_show = METHOD_REGISTRY if all_methods else schedulable_methods(2)
    for method in methods_to_show:
        typer.echo(f"{method.method_id:36s} {method.status:12s} {method.display_name}")


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).date()


def _resolve_cli_symbols(symbols: str | None, universe: str | None) -> list[str] | None:
    if symbols and universe:
        raise typer.BadParameter("Use either --symbols or --universe, not both")
    if symbols:
        return resolve_universe(_parse_csv(symbols))
    if universe:
        try:
            return resolve_universe(preset=universe)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    return None


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    app()
