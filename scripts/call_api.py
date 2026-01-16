import argparse
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return json.loads(text)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise SystemExit(f"HTTP {e.code} {e.reason}\n{body}") from e
    except URLError as e:
        raise SystemExit(f"Connection error: {e}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in response: {e}") from e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", help="User question (UTF-8).")
    p.add_argument("--url", default="http://localhost:8000/ask", help="API endpoint URL.")
    p.add_argument("--top-k", type=int, default=5, help="Top K sources.")
    p.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout seconds.")
    p.add_argument("--debug", action="store_true", help="Print compact sources list.")
    args = p.parse_args()

    payload = {"question": args.question, "top_k": args.top_k}
    res = post_json(args.url, payload, timeout=args.timeout)

    # Pretty-print full response (keeps Cyrillic)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.debug and isinstance(res, dict) and "sources" in res:
        print("\nSOURCES (compact):")
        for i, s in enumerate(res["sources"][: args.top_k], 1):
            print(f"{i}) {s.get('source_path')} | chunk={s.get('chunk_id')} | score={s.get('score')}")


if __name__ == "__main__":
    main()
