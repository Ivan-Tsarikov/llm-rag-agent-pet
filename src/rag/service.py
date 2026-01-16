from __future__ import annotations

from typing import List

from src.core.logging import get_logger
from src.index.faiss_store import SearchHit
from src.rag.llm_clients import OllamaClient, OpenAICompatClient, LLMError

log = get_logger(__name__)


def build_prompt(question: str, hits: List[SearchHit]) -> str:
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        context_blocks.append(
            f"[Source {i}] file={h.record.source_path} chunk_id={h.record.chunk_id}\n{h.record.text}"
        )

    context = "\n\n".join(context_blocks) if context_blocks else "(no context)"

    return f"""Answer the question using ONLY the context below.
            If the context is insufficient, say that you don't have enough information.

            Question:
            {question}

            Context:
            {context}

            Answer:
            """


async def generate_answer(question: str, hits: List[SearchHit], llm_mode: str) -> str:
    prompt = build_prompt(question, hits)

    # llm_mode: "ollama" | "openai"
    if llm_mode == "openai":
        client = OpenAICompatClient()
    else:
        client = OllamaClient()

    try:
        return await client.generate(prompt)
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(str(e)) from e
