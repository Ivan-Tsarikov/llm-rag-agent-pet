from __future__ import annotations

import os
from typing import List

from src.core.config import get_settings
from src.core.logging import get_logger
from src.rag.llm_clients import LLMError, OllamaClient, OpenAICompatClient

log = get_logger(__name__)


def _build_context(hits, max_chars: int = 3500) -> str:
    """
    Собираем контекст из топ-хитов с ограничением по символам.
    """
    parts: list[str] = []
    total = 0

    for h in hits:
        src = h.record.source_path
        cid = h.record.chunk_id
        snippet = h.record.text.strip()

        block = f"[SOURCE: {src} | chunk={cid}]\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n".join(parts).strip()


def _build_prompt(question: str, context: str) -> str:
    # Коротко и жёстко: отвечай по источникам, не выдумывай
    return (
        "Ты помощник службы поддержки маркетплейса. Отвечай на русском.\n"
        "Используй ТОЛЬКО предоставленный контекст.\n"
        "Если ответа нет в контексте — так и скажи и уточни, какой информации не хватает.\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Контекст:\n{context}\n\n"
        "Ответ:"
    )


async def generate_answer(question: str, hits, llm_mode: str | None = None) -> str:
    """
    Генерирует ответ по вопросу и найденным фрагментам.
    llm_mode:
      - "ollama" или "openai" (если None — возьмём из settings/env)
    """
    settings = get_settings()

    mode = (llm_mode or os.getenv("LLM_MODE") or settings.llm_mode).lower().strip()
    if mode not in ("ollama", "openai"):
        mode = "ollama"

    context = _build_context(hits, max_chars=3500)
    prompt = _build_prompt(question, context)

    try:
        if mode == "openai":
            client = OpenAICompatClient()
        else:
            client = OllamaClient()

        # timeout берём разумный: Ollama может быть медленным на первом запросе
        text = await client.generate(prompt, timeout_s=90.0)
        return text.strip()
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(str(e)) from e
