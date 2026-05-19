import pytest

from factor_mining.data.universe import BINANCE_UM_LIQUID_30, resolve_universe


def test_resolve_universe_deduplicates_explicit_symbols() -> None:
    assert resolve_universe(["btcusdt", "ETHUSDT", "btcusdt"]) == ["BTCUSDT", "ETHUSDT"]


def test_resolve_universe_expands_liquid_um_preset() -> None:
    symbols = resolve_universe(preset="um_liquid_30")

    assert symbols[0:2] == ["BTCUSDT", "ETHUSDT"]
    assert len(symbols) == 30
    assert symbols == list(BINANCE_UM_LIQUID_30)


def test_resolve_universe_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown universe preset"):
        resolve_universe(preset="not_real")


def test_resolve_universe_rejects_empty_explicit_symbols() -> None:
    with pytest.raises(ValueError, match="At least one symbol"):
        resolve_universe([])
