from __future__ import annotations

import json
from typing import Any, Dict
from urllib import request, error

from src.core.logging import get_logger


log = get_logger(__name__)

ALLOWED_TOOLS = {"search_docs", "calc"}


class MCPClient:
    def __init__(self, base_url: str, timeout_s: float = 4.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            try:
                raw = exc.read().decode("utf-8")
            except Exception:
                return {"error": f"HTTP error: {exc.code}"}
        except error.URLError as exc:
            return {"error": f"Connection error: {exc.reason}"}

        try:
            data_obj = json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "invalid response"}

        if not isinstance(data_obj, dict):
            return {"error": "invalid response"}

        return data_obj

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in ALLOWED_TOOLS:
            return {"error": "Tool not allowed."}
        log.info("tool_backend=mcp tool=%s", name)
        return self._post(f"/tools/{name}", args)
