from __future__ import annotations

from collections.abc import Sequence


DEFAULT_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")

# Liquid USD-M perpetual starter universe. This is intentionally a stable,
# manually curated research preset, not a live open-interest ranking.
BINANCE_UM_LIQUID_30: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "UNIUSDT",
    "NEARUSDT",
    "AAVEUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "SUIUSDT",
    "FILUSDT",
    "ATOMUSDT",
    "ETCUSDT",
    "INJUSDT",
    "TIAUSDT",
    "WIFUSDT",
    "PEPEUSDT",
    "ORDIUSDT",
    "SEIUSDT",
    "1000SHIBUSDT",
)

UNIVERSE_PRESETS: dict[str, tuple[str, ...]] = {
    "default": DEFAULT_SYMBOLS,
    "um_liquid_30": BINANCE_UM_LIQUID_30,
}


def resolve_universe(symbols: Sequence[str] | None = None, preset: str | None = None) -> list[str]:
    if symbols is not None:
        return _normalize_symbols(symbols)
    key = preset or "default"
    if key not in UNIVERSE_PRESETS:
        valid = ", ".join(sorted(UNIVERSE_PRESETS))
        raise ValueError(f"Unknown universe preset {key!r}. Expected one of: {valid}")
    return list(UNIVERSE_PRESETS[key])


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = raw_symbol.strip().upper()
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    if not normalized:
        raise ValueError("At least one symbol is required")
    return normalized
