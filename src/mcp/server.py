from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.agent.tool_impl import search_docs_impl, calc_impl
from src.core.logging import get_logger
from src.rag.retriever import Retriever


log = get_logger(__name__)

ALLOWED_TOOLS = {"search_docs", "calc"}


def create_mcp_app() -> FastAPI:
    app = FastAPI(title="MCP Tools Server", version="0.1.0")

    @app.on_event("startup")
    async def startup_event() -> None:
        try:
            app.state.retriever = Retriever()
            log.info("MCP retriever ready.")
        except Exception as exc:
            app.state.retriever = None
            log.warning("MCP retriever not ready: %s", exc)

    @app.post("/tools/{tool_name}")
    async def call_tool(tool_name: str, payload: Dict[str, Any]) -> JSONResponse:
        if tool_name not in ALLOWED_TOOLS:
            raise HTTPException(status_code=404, detail="Tool not found.")

        if tool_name == "search_docs":
            retriever = getattr(app.state, "retriever", None)
            if retriever is None:
                return JSONResponse({"error": "Index is not ready. Run: python scripts/build_index.py"})
            query = str(payload.get("query", ""))
            top_k = payload.get("top_k", 5)
            log.info("tool_backend=mcp tool=search_docs")
            return JSONResponse(search_docs_impl(retriever, query=query, top_k=top_k))

        if tool_name == "calc":
            expression = str(payload.get("expression", ""))
            log.info("tool_backend=mcp tool=calc")
            return JSONResponse(calc_impl(expression))

        return JSONResponse({"error": "Tool not allowed."}, status_code=400)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "env": os.getenv("APP_ENV", "dev")}

    return app
