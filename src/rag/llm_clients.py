from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from src.core.config import get_settings
from src.core.logging import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str


class BaseLLMClient:
    async def generate(self, prompt: str, timeout_s: float = 60.0) -> str:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    """
    Родной Ollama API: POST /api/generate
    (Надёжнее, чем /v1/chat/completions, потому что это "родной" контракт Ollama)
    """

    def __init__(self) -> None:
        s = get_settings()
        self.base_url = s.ollama_base_url.rstrip("/")
        self.model = s.ollama_model
        self.keep_alive = s.ollama_keep_alive
        self.num_predict = s.ollama_num_predict
        self.temperature = s.ollama_temperature

    async def generate(self, prompt: str, timeout_s: float = 60.0) -> str:
        url = f"{self.base_url}/api/generate"

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": self.num_predict,
                "temperature": self.temperature,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama error: ReadTimeout({repr(e)})") from e
        except httpx.HTTPStatusError as e:
            body = e.response.text[:1000] if e.response is not None else ""
            raise LLMError(f"Ollama HTTP {e.response.status_code}: {body}") from e
        except Exception as e:
            raise LLMError(f"Ollama error: {repr(e)}") from e

        text = data.get("response")
        if not isinstance(text, str):
            raise LLMError("Ollama returned invalid response format.")
        return text.strip()


class OpenAICompatClient(BaseLLMClient):
    """
    Любой OpenAI-compatible endpoint: POST /v1/chat/completions
    (может быть OpenAI, или другой провайдер с тем же контрактом)
    """

    def __init__(self) -> None:
        s = get_settings()
        self.base_url = s.openai_base_url.rstrip("/")
        self.api_key = s.openai_api_key
        self.model = s.openai_model
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set for openai mode.")

    async def generate(self, prompt: str, timeout_s: float = 60.0) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Отвечай на русском языке."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as e:
            raise LLMError(f"OpenAI-compatible API error: ReadTimeout({repr(e)})") from e
        except httpx.HTTPStatusError as e:
            body = e.response.text[:1200] if e.response is not None else ""
            raise LLMError(f"OpenAI-compatible API error: HTTP {e.response.status_code}: {body}") from e
        except Exception as e:
            raise LLMError(f"OpenAI-compatible API error: {repr(e)}") from e

        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            raise LLMError("OpenAI-compatible API returned unexpected format.")

        if not isinstance(text, str):
            raise LLMError("OpenAI-compatible API returned non-text content.")
        return text.strip()
