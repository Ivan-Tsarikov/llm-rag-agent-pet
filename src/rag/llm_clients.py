from __future__ import annotations

import os
import httpx


class LLMError(RuntimeError):
    pass


class OllamaClient:
    """
    Родной Ollama API: POST /api/generate
    Требует запущенный Ollama на localhost:11434 и скачанную модель.
    """
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    async def generate(self, prompt: str, timeout_s: float = 60.0) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}

        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(url, json=payload)
                if r.status_code >= 400:
                    raise LLMError(f"Ollama {r.status_code}: {r.text}")
                data = r.json()
                return (data.get("response") or "").strip()
        except Exception as e:
            raise LLMError(f"Ollama error: {repr(e)}") from e


class OpenAICompatClient:
    """
    Оставляем как опцию (внешний API или OpenAI-совместимые серверы).
    """
    def __init__(self) -> None:
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set")

    async def generate(self, prompt: str, timeout_s: float = 30.0) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer using the provided context."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(url, headers=headers, json=payload)
                if r.status_code >= 400:
                    raise LLMError(f"OpenAI {r.status_code}: {r.text}")
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise LLMError(f"OpenAI-compatible API error: {repr(e)}") from e