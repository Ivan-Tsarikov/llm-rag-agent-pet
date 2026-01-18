"""Run the LangChain demo pipeline from the CLI."""

from __future__ import annotations

import asyncio
import os
import sys

from src.rag.llm_clients import LLMError, OllamaClient, OpenAICompatClient
from src.rag.retriever import Retriever
from src.langchain_demo.pipeline import run_langchain_rag


async def _main() -> int:
    """Execute a single LangChain RAG request."""
    question = sys.argv[1] if len(sys.argv) > 1 else "Как восстановить доступ к аккаунту?"
    retriever = Retriever()

    llm_mode = os.getenv("LLM_MODE", "ollama").lower()
    try:
        if llm_mode == "openai":
            llm_client = OpenAICompatClient()
        else:
            llm_client = OllamaClient()
    except LLMError as exc:
        print(f"LLM client error: {exc}")
        return 1

    try:
        answer, sources = await run_langchain_rag(
            question=question,
            retriever=retriever,
            llm_client=llm_client,
            top_k=5,
        )
    except LLMError as exc:
        print(f"LLM error: {exc}")
        return 1

    print(f"Q: {question}\n")
    print("Answer:")
    print(answer)
    print("\nSources:")
    for idx, src in enumerate(sources, start=1):
        print(
            f"{idx}) score={src.score:.4f} | {src.source_path} | chunk_id={src.chunk_id}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
