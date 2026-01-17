from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.errors import unhandled_exception_handler
from src.core.logging import setup_logging, get_logger
from src.core.middleware import RequestIdMiddleware, SimpleAccessLogMiddleware

from src.rag.schemas import AskRequest, AskResponse, SourceItem
from src.rag.retriever import Retriever
from src.rag.service import generate_answer
from src.rag.llm_clients import LLMError, OllamaClient, OpenAICompatClient

from src.agent.tools import ToolRegistry
from src.agent.tool_backend import build_tool_registry
from src.agent.agent import run_agent, AgentError
from src.mcp.client import MCPClient


# ---------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------
settings = get_settings()
setup_logging(settings.log_level)
log = get_logger(__name__)

app = FastAPI(title="RAG + Agent + MCP (MVP)", version="0.1.0")

app.add_middleware(RequestIdMiddleware)
app.add_middleware(SimpleAccessLogMiddleware)
app.add_exception_handler(Exception, unhandled_exception_handler)


# ---------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------
def _get_retriever() -> Retriever:
    r = getattr(app.state, "retriever", None)
    if r is None:
        raise HTTPException(
            status_code=503,
            detail="Index is not ready. Run: python scripts/build_index.py",
        )
    return r


def _get_llm_client():
    c = getattr(app.state, "llm_client", None)
    if c is None:
        raise HTTPException(
            status_code=503,
            detail="LLM client is not ready. Check LLM_MODE / Ollama/OpenAI settings.",
        )
    return c


# ---------------------------------------------------------------------
# Startup init: retriever + llm client + agent tools
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    # Retriever
    try:
        app.state.retriever = Retriever()
        log.info("Retriever ready.")
    except Exception as e:
        app.state.retriever = None
        log.warning("Retriever not ready (index missing?): %s", e)

    # LLM client (for agent prompt-generation)
    llm_mode = os.getenv("LLM_MODE", "ollama").lower()  # ollama|openai
    try:
        if llm_mode == "openai":
            app.state.llm_client = OpenAICompatClient()
        else:
            app.state.llm_client = OllamaClient()
        log.info("LLM client ready: mode=%s", llm_mode)
    except Exception as e:
        app.state.llm_client = None
        log.warning("LLM client not ready: %s", e)

        
    tool_backend = settings.tool_backend
    mcp_url = settings.mcp_url
    retriever = getattr(app.state, "retriever", None)
    mcp_client = MCPClient(mcp_url) if tool_backend.lower() == "mcp" else None

    tools = build_tool_registry(
        backend=tool_backend,
        retriever=retriever,
        mcp_client=mcp_client,
        )

    app.state.agent_tools = tools
    log.info("Agent tools registered: %s", sorted(tools.allowlist()))


# ---------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "env": settings.app_env}


@app.get("/")
def root():
    return JSONResponse(
        {
            "service": "rag-agent-mcp",
            "endpoints": [
                "/health (GET)",
                "/ask (POST)",
                "/agent/ask (POST)",
                "/debug/index (GET)",
                "/debug/search (POST)",
            ],
        }
    )


# ---------------------------------------------------------------------
# RAG endpoint
# ---------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    retriever = _get_retriever()

    top_k = req.top_k or settings.top_k
    hits = retriever.search(req.question, top_k=top_k)

    llm_mode = os.getenv("LLM_MODE", "ollama").lower()  # ollama|openai

    # 1 ретрай при LLM ошибке
    try:
        answer = await generate_answer(req.question, hits, llm_mode=llm_mode)
    except LLMError as e:
        log.warning("LLM failed, retry once: %s", e)
        try:
            answer = await generate_answer(req.question, hits, llm_mode=llm_mode)
        except LLMError as e2:
            raise HTTPException(status_code=502, detail=str(e2)) from e2

    sources = [
        SourceItem(
            source_path=h.record.source_path,
            chunk_id=h.record.chunk_id,
            score=h.score,
            text=h.record.text[:800],
        )
        for h in hits
    ]
    return AskResponse(answer=answer, sources=sources)


# ---------------------------------------------------------------------
# Debug endpoints (retrieval only)
# ---------------------------------------------------------------------
class DebugSearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)


@app.post("/debug/search")
def debug_search(req: DebugSearchRequest):
    retriever = _get_retriever()

    hits = retriever.search(req.question, top_k=req.top_k)

    query_norm: Optional[float] = None
    if hasattr(retriever, "query_vector_norm"):
        try:
            query_norm = float(retriever.query_vector_norm(req.question))
        except Exception:
            query_norm = None

    return {
        "question_received": req.question,
        "embedding_model_name": getattr(retriever, "embedding_model_name", None),
        "query_vector_l2_norm": query_norm,
        "top": [
            {
                "source_path": h.record.source_path,
                "chunk_id": h.record.chunk_id,
                "score": float(h.score),
                "text_preview": h.record.text[:120],
            }
            for h in hits[: min(5, len(hits))]
        ],
    }


@app.get("/debug/index")
def debug_index():
    index_dir = Path(settings.index_dir).resolve()
    info: dict[str, Any] = {
        "cwd": os.getcwd(),
        "index_dir": settings.index_dir,
        "index_dir_resolved": str(index_dir),
        "index_files_exist": {
            "faiss.index": (index_dir / "faiss.index").exists(),
            "chunks.jsonl": (index_dir / "chunks.jsonl").exists(),
            "index_meta.json": (index_dir / "index_meta.json").exists(),
        },
        "retriever_ready": getattr(app.state, "retriever", None) is not None,
    }
    retriever = getattr(app.state, "retriever", None)
    if retriever is not None and hasattr(retriever, "store") and hasattr(retriever.store, "records"):
        info["chunks_loaded"] = len(retriever.store.records)
        info["embedding_model_name"] = getattr(retriever, "embedding_model_name", None)
    return info


# гарантируем UTF-8 в JSON-ответах (ответы, не запросы)
@app.middleware("http")
async def force_json_utf8(request, call_next):
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if ct.startswith("application/json") and "charset=" not in ct:
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


# ---------------------------------------------------------------------
# Agent endpoint (JSON tool-calling)
# ---------------------------------------------------------------------
class AgentAskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: Optional[int] = Field(default=5, ge=1, le=10)
    debug: Optional[bool] = False


class AgentAskResponse(BaseModel):
    answer: str
    sources: list[dict]
    trace: Optional[list[dict]] = None


@app.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(req: AgentAskRequest):
    tools: ToolRegistry = getattr(app.state, "agent_tools", None)
    if tools is None:
        raise HTTPException(status_code=503, detail="Agent tools are not ready.")

    llm_client = _get_llm_client()

    async def llm_generate(prompt: str, timeout_s: float):
        # llm_client должен поддерживать generate(prompt, timeout_s=...)
        return await llm_client.generate(prompt, timeout_s=timeout_s)

    try:
        answer, steps = await run_agent(
            llm_generate=llm_generate,
            question=req.question,
            tools=tools,
            max_steps=4,
            llm_timeout_s=90.0,
            retry_once=True,
        )
    except AgentError as e:
        return AgentAskResponse(
            answer=f"Не смог завершить агентный ответ: {e}",
            sources=[],
            trace=None,
        )

    # Источники: берём из первого успешного search_docs (если агент его вызывал)
    sources: list[dict] = []
    for st in steps:
        if st.tool == "search_docs" and st.tool_result and isinstance(st.tool_result, dict):
            hits = st.tool_result.get("hits")
            if isinstance(hits, list):
                sources = hits[: (req.top_k or 5)]
                break

    trace = None
    if req.debug:
        trace = [
            {
                "step": s.step,
                "action": s.action,
                "tool": s.tool,
                "tool_args": s.tool_args,
                "tool_result": s.tool_result if isinstance(s.tool_result, dict) else None,
            }
            for s in steps
        ]

    return AgentAskResponse(answer=answer, sources=sources, trace=trace)
