from __future__ import annotations

from typing import Iterable, List

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence

from src.rag.llm_clients import BaseLLMClient, LLMError
from src.rag.retriever import Retriever
from src.rag.schemas import SourceItem


def _build_context(hits: Iterable, max_chars: int = 3500) -> str:
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


def _prompt_template() -> PromptTemplate:
    return PromptTemplate.from_template(
        "Ты помощник службы поддержки маркетплейса. Отвечай на русском.\n"
        "Используй ТОЛЬКО предоставленный контекст.\n"
        "Если ответа нет в контексте — так и скажи и уточни, какой информации не хватает.\n\n"
        "Вопрос:\n{question}\n\n"
        "Контекст:\n{context}\n\n"
        "Ответ:"
    )


async def run_langchain_rag(
    question: str,
    retriever: Retriever,
    llm_client: BaseLLMClient,
    top_k: int,
) -> tuple[str, List[SourceItem]]:
    hits = retriever.search(question, top_k=top_k)

    build_inputs = RunnableLambda(
        lambda q: {"question": q, "context": _build_context(hits, max_chars=3500)}
    )
    prompt = _prompt_template()
    chain = RunnableSequence(build_inputs, prompt)

    prompt_value = chain.invoke(question)
    try:
        answer = await llm_client.generate(prompt_value.to_string(), timeout_s=90.0)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(str(exc)) from exc

    sources = [
        SourceItem(
            source_path=h.record.source_path,
            chunk_id=h.record.chunk_id,
            score=h.score,
            text=h.record.text[:800],
        )
        for h in hits
    ]

    return answer.strip(), sources
