import logging
from typing import Optional

from .context import request_id_var


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True


def setup_logging(level: str = "INFO", *, quiet_uvicorn: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | rid=%(request_id)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(RequestIdFilter())

    # убираем дефолтные хендлеры, чтобы не было дублей
    root.handlers = []
    root.addHandler(handler)

    if quiet_uvicorn:
        # Uvicorn может шуметь в INFO. Оставим, но чуть приглушим.
        logging.getLogger("uvicorn.error").setLevel(level.upper())
        logging.getLogger("uvicorn.access").setLevel("WARNING")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "app")
