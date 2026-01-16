from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.core.errors import unhandled_exception_handler
from src.core.logging import setup_logging, get_logger
from src.core.middleware import RequestIdMiddleware, SimpleAccessLogMiddleware

import os
from fastapi import HTTPException
from src.rag.schemas import AskRequest, AskResponse, SourceItem
from src.rag.retriever import Retriever
from src.rag.service import generate_answer
from src.rag.llm_clients import LLMError

from pathlib import Path
from pydantic import BaseModel, Field

settings = get_settings()
setup_logging(settings.log_level)
log = get_logger(__name__)

app = FastAPI(title="RAG + Agent + MCP (MVP)", version="0.1.0")

app.add_middleware(RequestIdMiddleware)
app.add_middleware(SimpleAccessLogMiddleware)

app.add_exception_handler(Exception, unhandled_exception_handler)

# Инициализируем Retriever один раз на процесс
# Если индекс не найден — обработаем в /ask
retriever = None
try:
    retriever = Retriever()
except Exception as e:
    log.warning("Retriever not ready (index missing?): %s", e)

@app.get("/health")
def health():
    # Минимум информации: достаточно, чтобы при мониторинг понять “жив/не жив”
    return {"status": "ok", "env": settings.app_env}


@app.get("/")
def root():
    return JSONResponse(
        {
            "service": "rag-agent-mcp",
            "endpoints": ["/health (GET)", "/ask (POST) - soon"],
        }
    )

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Index is not ready. Run: python scripts/build_index.py",
        )

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
            raise HTTPException(status_code=502, detail=str(e2))

    sources = [
        SourceItem(
            source_path=h.record.source_path,
            chunk_id=h.record.chunk_id,
            score=h.score,
            text=h.record.text[:800],  # ограничим размер в API
        )
        for h in hits
    ]

    return AskResponse(answer=answer, sources=sources)

class DebugSearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)

@app.post("/debug/search")
def debug_search(req: DebugSearchRequest):
    hits = retriever.search(req.question, top_k=req.top_k)

    # важное: покажем вопрос, и первые 5 результатов
    return {
        "question_received": req.question,
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
    info = {
        "cwd": os.getcwd(),
        "index_dir": settings.index_dir,
        "index_dir_resolved": str(index_dir),
        "index_files_exist": {
            "faiss.index": (index_dir / "faiss.index").exists(),
            "chunks.jsonl": (index_dir / "chunks.jsonl").exists(),
        },
        "retriever_ready": retriever is not None,
    }
    if retriever is not None:
        info["chunks_loaded"] = len(retriever.store.records)
    return info

@app.middleware("http")
async def force_json_utf8(request, call_next):
    response = await call_next(request)
    ct = response.headers.get("content-type", "")
    if ct.startswith("application/json") and "charset=" not in ct:
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response
