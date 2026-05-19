import os

from factor_mining.config import BootstrapConfig, Settings, load_dotenv
from factor_mining.mining import _extract_chat_json
from factor_mining.registry import METHOD_REGISTRY, get_method, schedulable_methods


def test_config_defaults_encode_final_plan() -> None:
    settings = Settings()
    assert settings.data.default_interval == "5m"
    assert settings.data.on_demand_intervals == ["1m"]
    assert settings.trial_ledger.partition == "family_and_rolling_window"
    assert settings.bootstrap.block_length_bars(1000) == 2000
    assert BootstrapConfig().block_length_bars(10) == 288


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
