from __future__ import annotations

from typing import Any, Dict, Optional

from src.agent.tool_impl import search_docs_impl, calc_impl
from src.agent.tools import ToolRegistry, ToolSpec
from src.core.logging import get_logger
from src.mcp.client import MCPClient
from src.rag.retriever import Retriever


log = get_logger(__name__)


def build_tool_registry(
    *,
    backend: str,
    retriever: Optional[Retriever] = None,
    mcp_client: Optional[MCPClient] = None,
) -> ToolRegistry:
    tools = ToolRegistry()
    backend_norm = (backend or "local").lower()

    if backend_norm == "mcp":
        if mcp_client is None:
            raise ValueError("MCP client is required for MCP backend.")

        async def tool_search_docs(args: Dict[str, Any]) -> Dict[str, Any]:
            return mcp_client.call_tool("search_docs", args)

        async def tool_calc(args: Dict[str, Any]) -> Dict[str, Any]:
            return mcp_client.call_tool("calc", args)

        log.info("Tool backend configured: mcp")
    else:
        async def tool_search_docs(args: Dict[str, Any]) -> Dict[str, Any]:
            if retriever is None:
                return {"error": "Index is not ready. Run: python scripts/build_index.py"}
            query = str(args.get("query", ""))
            top_k = args.get("top_k", 5)
            return search_docs_impl(retriever, query=query, top_k=top_k)

        async def tool_calc(args: Dict[str, Any]) -> Dict[str, Any]:
            expression = str(args.get("expression", ""))
            return calc_impl(expression)

        log.info("Tool backend configured: local")

    tools.register(
        ToolSpec(
            name="search_docs",
            description="Ищет релевантные фрагменты в базе документов.",
            args_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=tool_search_docs,
            timeout_s=2.0,
        )
    )

    tools.register(
        ToolSpec(
            name="calc",
            description="Считает арифметическое выражение (безопасно).",
            args_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=tool_calc,
            timeout_s=1.0,
        )
    )

    return tools
