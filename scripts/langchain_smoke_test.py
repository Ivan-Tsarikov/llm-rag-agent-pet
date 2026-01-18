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
    print("[1/1] LangChain ask...")
    response = request_json(
        "POST",
        "http://localhost:8000/ask_langchain",
        {"question": "Как восстановить доступ к аккаунту?"},
    )
    sources = response.get("sources")
    assert_true(isinstance(sources, list) and sources, f"Missing sources: {response}")
    first_source = sources[0]
    source_path = first_source.get("source_path", "") if isinstance(first_source, dict) else ""
    assert_true(
        "account_security" in source_path,
        f"Unexpected source_path: {source_path}",
    )
    print("    OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        sys.exit(1)
