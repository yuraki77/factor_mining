import os

from factor_mining.config import BootstrapConfig, PositionSizingConfig, Settings, apply_trade_overrides, load_dotenv
from factor_mining.mining import _extract_chat_json
from factor_mining.registry import METHOD_REGISTRY, get_method, schedulable_methods


def test_config_defaults_encode_final_plan() -> None:
    settings = Settings()
    assert settings.data.default_interval == "5m"
    assert settings.data.on_demand_intervals == ["1m"]
    assert settings.trial_ledger.partition == "family_and_rolling_window"
    assert settings.bootstrap.block_length_bars(1000) == 2000
    assert BootstrapConfig().block_length_bars(10) == 288


def test_trade_overrides_are_run_scoped_and_symbol_specific() -> None:
    settings = Settings(position_sizing=PositionSizingConfig(max_leverage=2.0))

    updated = apply_trade_overrides(
        settings,
        btc_leverage=5.0,
        eth_leverage=4.0,
        taker_bps=6.5,
        slippage_base_bps=1.25,
    )

    assert settings.position_sizing.symbol_max_leverage == {}
    assert updated.position_sizing.max_leverage_for("BTCUSDT") == 5.0
    assert updated.position_sizing.max_leverage_for("ETH") == 4.0
    assert updated.position_sizing.max_leverage_for("SOLUSDT") == 2.0
    assert updated.costs.taker_bps == 6.5
    assert updated.costs.slippage_base_bps == 1.25


def test_registry_keeps_84_method_blueprint_and_blocks_cross_section_for_n2() -> None:
    assert len(METHOD_REGISTRY) == 84
    blocked = get_method("cross_symbol_validation")
    assert blocked.status == "blocked_v1"
    assert blocked.requires_cross_section
    schedulable_ids = {method.method_id for method in schedulable_methods(universe_size=2)}
    assert "template_constrained_search" in schedulable_ids
    assert "cross_symbol_validation" not in schedulable_ids


def test_dotenv_loader_does_not_require_committing_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=secret-test-key\n", encoding="utf-8")
    load_dotenv(env_file)
    assert os.environ["DEEPSEEK_API_KEY"] == "secret-test-key"


def test_llm_chat_json_extractor_reads_openai_compatible_content() -> None:
    response = {"choices": [{"message": {"content": '{"hypotheses": []}'}}]}
    assert _extract_chat_json(response) == {"hypotheses": []}
