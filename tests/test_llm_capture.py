"""I6: LLM raw exchanges must be capturable for per-run persistence.

Checkpoints are prunable, so without a durable record of the prompt and
response an LLM-seeded run can never be audited or regenerated. The capture
must be populated as soon as the API answers — a response that fails
hypothesis validation is exactly the one worth inspecting later.
"""

from __future__ import annotations

import json

import pytest

import factor_mining.mining as mining
from factor_mining.config import Settings
from factor_mining.mining import generate_hypotheses_with_deepseek


class _FakeProvider:
    is_configured = True
    api_key_env = "FAKE_KEY"

    def __init__(self, content: str) -> None:
        self._content = content
        self.seen_messages: list[dict] | None = None

    def chat_json(self, *, model: str, messages: list[dict]) -> dict:
        self.seen_messages = messages
        return {"choices": [{"message": {"content": self._content}}]}


def _valid_content() -> str:
    return json.dumps({
        "hypotheses": [
            {
                "hypothesis_id": "h_test",
                "hypothesis_family": "momentum",
                "economic_mechanism": "m",
                "testable_prediction": "p",
                "null_hypothesis": "n",
                "expected_ic_range": [0.01, 0.03],
                "expected_decay_halflife_bars": 12,
            }
        ]
    })


def test_capture_records_prompt_model_and_response(monkeypatch) -> None:
    provider = _FakeProvider(_valid_content())
    monkeypatch.setattr(mining, "provider_from_settings", lambda name, settings: provider)

    capture: dict = {}
    hypotheses = generate_hypotheses_with_deepseek(Settings(), count=1, capture=capture)

    assert len(hypotheses) == 1
    assert capture["provider"] == "deepseek"
    assert capture["model"]
    assert capture["messages"] == provider.seen_messages
    assert capture["response"]["choices"][0]["message"]["content"] == _valid_content()


def test_capture_survives_unparseable_response(monkeypatch) -> None:
    """The raw exchange must be in the capture even when validation raises —
    persisting only successful parses would hide the failures that need audit."""
    provider = _FakeProvider("this is not json")
    monkeypatch.setattr(mining, "provider_from_settings", lambda name, settings: provider)

    capture: dict = {}
    with pytest.raises(RuntimeError):
        generate_hypotheses_with_deepseek(Settings(), count=1, capture=capture)

    assert capture["response"]["choices"][0]["message"]["content"] == "this is not json"
    assert capture["messages"] == provider.seen_messages
