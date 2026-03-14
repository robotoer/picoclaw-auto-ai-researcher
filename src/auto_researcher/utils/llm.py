"""LLM client abstraction for the research system."""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel

from auto_researcher.config import LLMConfig, LLMProvider


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict[str, int] = {}


class LLMClient:
    """Unified LLM client supporting Anthropic and OpenAI APIs."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.provider == LLMProvider.ANTHROPIC:
            return await self._anthropic_generate(prompt, system, temp, tokens)
        else:
            return await self._openai_generate(prompt, system, temp, tokens)

    async def generate_structured(
        self,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate a response and parse it as JSON."""
        response = await self.generate(
            prompt=prompt + "\n\nRespond with valid JSON only.",
            system=system,
            temperature=temperature,
        )
        text = response.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return dict(json.loads(text))

    async def _anthropic_generate(
        self, prompt: str, system: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            body["system"] = system

        resp = await self._client.post(
            "https://api.anthropic.com/v1/messages",
            json=body,
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            content=data["content"][0]["text"],
            model=data["model"],
            usage={
                "input_tokens": data.get("usage", {}).get("input_tokens", 0),
                "output_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
        )

    async def _openai_generate(
        self, prompt: str, system: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self._client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": self.config.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            },
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=data["model"],
            usage=dict(data.get("usage", {})),
        )

    async def close(self) -> None:
        await self._client.aclose()
