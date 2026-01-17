from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error for {url}: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {body}") from exc


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    print("[1/3] API health check...")
    health = request_json("GET", "http://localhost:8000/health")
    assert_true(health.get("status") == "ok", f"Unexpected health response: {health}")
    print("    OK")

    print("[2/3] MCP calc tool...")
    calc = request_json(
        "POST",
        "http://localhost:9001/tools/calc",
        {"expression": "3.5% * 12000"},
    )
    value = calc.get("value")
    assert_true(value is not None, f"Missing value in calc response: {calc}")
    assert_true(abs(float(value) - 420.0) < 1e-6, f"Unexpected calc value: {value}")
    print("    OK")

    print("[3/3] Agent ask...")
    agent = request_json(
        "POST",
        "http://localhost:8000/agent/ask",
        {"question": "Как восстановить доступ к аккаунту?"},
    )
    sources = agent.get("sources")
    assert_true(isinstance(sources, list) and sources, f"Missing sources: {agent}")
    first_source = sources[0]
    source_path = first_source.get("source_path", "") if isinstance(first_source, dict) else ""
    assert_true(
        "account_security" in source_path,
        f"Unexpected source_path: {source_path}",
    )
    print("    OK")

    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        sys.exit(1)
