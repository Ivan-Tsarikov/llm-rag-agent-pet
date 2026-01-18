"""Directly call MCP tools for quick manual verification.

Example:
    python -m scripts.demo_mcp_tools
"""

import os

from src.mcp.client import MCPClient


def main() -> None:
    """Call MCP tools and print raw responses."""
    base_url = os.getenv("MCP_URL", "http://localhost:9001")
    client = MCPClient(base_url)

    print("MCP URL:", base_url)

    print("\ncalc('3.5% * 12000')")
    calc_res = client.call_tool("calc", {"expression": "3.5% * 12000"})
    print(calc_res)

    print("\nsearch_docs('как восстановить доступ к аккаунту?')")
    search_res = client.call_tool(
        "search_docs",
        {"query": "как восстановить доступ к аккаунту?", "top_k": 5},
    )
    print(search_res)

    hits = search_res.get("hits") if isinstance(search_res, dict) else None
    if isinstance(hits, list) and hits:
        print("top_source:", hits[0].get("source_path"))
    
    query = "a"*2501
    expression = "1+"*300
    print("\nПроверка на лимиты (calc)")
    calc_res = client.call_tool("calc", {"expression": "1+"*300})
    print(calc_res)

    print("\nПроверка на лимиты (search_docs)")
    search_res = client.call_tool(
        "search_docs",
        {"query": "a"*2501},
    )
    print(search_res)



if __name__ == "__main__":
    main()
