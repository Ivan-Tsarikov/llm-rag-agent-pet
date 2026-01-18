"""Run the agent with tools served via MCP.

Example:
    python -m scripts.demo_agent_mcp
"""

import asyncio
import os

from src.agent.agent import run_agent
from src.agent.tool_backend import build_tool_registry
from src.core.config import get_settings
from src.mcp.client import MCPClient
from src.rag.llm_clients import OllamaClient


async def main() -> None:
    """Run a short MCP-backed agent demo."""
    settings = get_settings()
    _ = settings

    # TOOL_BACKEND toggles MCP usage in the main app, kept for parity in demos.
    os.environ["TOOL_BACKEND"] = "mcp"

    mcp_url = os.getenv("MCP_URL", "http://localhost:9001")
    mcp_client = MCPClient(mcp_url)
    tools = build_tool_registry(backend="mcp", mcp_client=mcp_client)

    llm = OllamaClient()

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
