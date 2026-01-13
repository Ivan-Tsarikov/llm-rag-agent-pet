from fastapi import Request
from fastapi.responses import JSONResponse

from .context import request_id_var
from .logging import get_logger

log = get_logger(__name__)


def error_payload(message: str):
    return {
        "error": {
            "message": message,
            "request_id": request_id_var.get("-"),
        }
    }


async def unhandled_exception_handler(request: Request, exc: Exception):
    # В проде лучше не отдавать детали исключения наружу
    log.exception("Unhandled error: %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content=error_payload("Internal server error"))
