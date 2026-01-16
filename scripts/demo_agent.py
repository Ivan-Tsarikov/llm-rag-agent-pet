import asyncio
import json

from src.core.config import get_settings
from src.rag.retriever import Retriever
from src.agent.tools import ToolRegistry, ToolSpec
from src.agent.calc import safe_calc, CalcError
from src.agent.agent import run_agent

from src.rag.llm_clients import OllamaClient


async def main():
    settings = get_settings()
    retriever = Retriever()

    llm = OllamaClient()

    tools = ToolRegistry()

    async def tool_search_docs(args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        top_k = int(args.get("top_k", 5))
        top_k = max(1, min(top_k, 10))
        hits = retriever.search(query, top_k=top_k)
        return {
            "hits": [
                {
                    "source_path": h.record.source_path,
                    "chunk_id": h.record.chunk_id,
                    "score": float(h.score),
                    "text": h.record.text[:800],
                }
                for h in hits
            ]
        }

    async def tool_calc(args: dict) -> dict:
        expr = str(args.get("expression", "")).strip()
        try:
            r = safe_calc(expr)
            return {"value": r.value, "formatted": r.formatted}
        except CalcError as e:
            return {"error": str(e)}

    tools.register(
        ToolSpec(
            name="search_docs",
            description="Ищет релевантные фрагменты в базе документов.",
            args_schema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}},
            handler=tool_search_docs,
            timeout_s=2.0,
        )
    )
    tools.register(
        ToolSpec(
            name="calc",
            description="Считает арифметическое выражение (безопасно).",
            args_schema={"type": "object", "properties": {"expression": {"type": "string"}}},
            handler=tool_calc,
            timeout_s=1.0,
        )
    )

    async def llm_generate(prompt: str, timeout_s: float):
        return await llm.generate(prompt, timeout_s=timeout_s)

    questions = [
        "Как восстановить доступ к аккаунту?",
        "Какие признаки подозрительной активности?",
        "Посчитай 3.5% от 12000",
        "Какие правила у пунктов выдачи (ПВЗ)?",
    ]

    for q in questions:
        print("\n" + "=" * 80)
        print("Q:", q)
        ans, steps = await run_agent(
            llm_generate=llm_generate,
            question=q,
            tools=tools,
            max_steps=4,
            llm_timeout_s=90.0,
            retry_once=True,
        )
        print("tool:", steps[0].tool)
        if steps[0].tool == "search_docs":
            hits = steps[0].tool_result.get("hits", [])
            if hits:
                print("top_source:", hits[0].get("source_path"))
        print("A:", ans)


if __name__ == "__main__":
    asyncio.run(main())