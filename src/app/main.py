from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.core.errors import unhandled_exception_handler
from src.core.logging import setup_logging, get_logger
from src.core.middleware import RequestIdMiddleware, SimpleAccessLogMiddleware

settings = get_settings()
setup_logging(settings.log_level)
log = get_logger(__name__)

app = FastAPI(title="RAG + Agent + MCP (MVP)", version="0.1.0")

app.add_middleware(RequestIdMiddleware)
app.add_middleware(SimpleAccessLogMiddleware)

app.add_exception_handler(Exception, unhandled_exception_handler)


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
