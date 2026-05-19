# Factor Mining Automation v1

Local-first BTC/ETH factor mining automation with strict statistical gates.

The project is intentionally conservative: rigorous validation is prioritized over the number of archived discoveries.

## Quick Start

```bash
uv sync --extra dev --no-editable
cp .env.example .env
# Fill DEEPSEEK_API_KEY and MINIMAX_API_KEY in .env.
uv run --no-editable fm llm check
uv run --no-editable fm --help
uv run --no-editable fm ui
```

## Cold Start

1. Install dependencies:

```bash
uv sync --extra dev --no-editable
```

2. Configure LLM keys:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
DEEPSEEK_API_KEY=...
MINIMAX_API_KEY=...
```

Model names and base URLs live in `configs/default.yaml` under `llm`.
Secrets should stay in `.env` or shell environment variables.

3. Verify local setup:

```bash
uv run --no-editable fm llm check
uv run pytest
```

4. Smoke-test the data downloader without writing files:

```bash
uv run --no-editable fm data sync --dry-run --start 2026-04-01 --end 2026-04-01
```

5. Generate the initial built-in hypothesis/candidate pool:

```bash
uv run --no-editable fm mine run
```

Use DeepSeek once keys are configured:

```bash
uv run --no-editable fm mine run --use-llm --hypothesis-count 5
```

6. Launch the local dashboard:

```bash
uv run --no-editable fm ui
```

## Main Commands

```bash
uv run --no-editable fm data sync --help
uv run --no-editable fm mine run
uv run --no-editable fm gate run
uv run --no-editable fm hardscore run
uv run --no-editable fm exp reproduce EXP_ID
uv run --no-editable fm worker run
uv run --no-editable fm llm check
uv run --no-editable fm ui
```

## Storage Layout

- `data/raw/` downloaded Binance zip/checksum files
- `data/parquet/` normalized Parquet warehouse
- `data/factor_mining.sqlite3` task, experiment, trial, and archive metadata
- `archives/` reproducible experiment bundles
