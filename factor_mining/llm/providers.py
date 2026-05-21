from __future__ import annotations

import os
from typing import Any

import httpx

from factor_mining.config import Settings


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self.provider = provider
        self.api_key_env = api_key_env or self._default_env_key(provider)
        self.api_key = api_key or os.getenv(self.api_key_env)
        self.base_url = base_url or "https://api.deepseek.com"

    def chat_json(self, *, model: str, messages: list[dict[str, str]], schema_hint: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(f"{self.provider} API key is not configured")
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        if schema_hint:
            payload["schema_hint"] = schema_hint
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            data = _clean_reasoning_content(data)
            return data

    @staticmethod
    def _default_env_key(provider: str) -> str:
        return "DEEPSEEK_API_KEY"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


def provider_from_settings(provider: str, settings: Settings) -> OpenAICompatibleProvider:
    if provider != "deepseek":
        raise ValueError(f"Unsupported LLM provider: {provider}")
    cfg = settings.llm.deepseek
    return OpenAICompatibleProvider(provider=provider, base_url=cfg.base_url, api_key_env=cfg.api_key_env)


def _clean_reasoning_content(response: dict) -> dict:
    """Strip <think> reasoning tags and ```json fences from model output."""
    import re
    choices = response.get("choices", [])
    for choice in choices:
        msg = choice.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*$", "", content)
            msg["content"] = content.strip()
    return response


def hypothesis_system_prompt() -> str:
    return (
        "Generate only BTC/ETH time-series or funding/basis hypotheses. "
        "Do not generate cross-sectional factors. Every hypothesis must include an economic mechanism, "
        "testable prediction, null hypothesis, expected IC range, and expected decay halflife in bars. "
        "Use hypothesis_family exactly as one of: momentum, mean_reversion, volatility, "
        "funding_basis, volume_confirmation."
    )
