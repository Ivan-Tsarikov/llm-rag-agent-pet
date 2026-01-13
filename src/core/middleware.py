import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .context import request_id_var
from .logging import get_logger

log = get_logger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    - Берёт X-Request-ID если пришёл
    - Иначе генерирует
    - Проставляет в контекст и в заголовок ответа
    """

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id")
        rid = (incoming or uuid.uuid4().hex[:12]).strip()
        token = request_id_var.set(rid)
        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_var.reset(token)


class SimpleAccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        try:
            response: Response = await call_next(request)
            return response
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000
            log.info("%s %s -> %.1fms", request.method, request.url.path, dt_ms)
