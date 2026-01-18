"""Run the local tool-calling agent against a few sample questions.

Example:
    python -m scripts.demo_agent
"""

import asyncio
import json

from src.core.config import get_settings
from src.rag.retriever import Retriever
from src.agent.tool_backend import build_tool_registry
from src.agent.agent import run_agent

from src.rag.llm_clients import OllamaClient


async def main():
    """Run a short agent demo using local tools and Ollama."""
    settings = get_settings()
    retriever = Retriever()

    llm = OllamaClient()

    tools = build_tool_registry(backend="local", retriever=retriever)

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
