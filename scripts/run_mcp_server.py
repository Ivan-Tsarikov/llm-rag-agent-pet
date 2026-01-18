"""Run the MCP tools server with Uvicorn."""

import os

import uvicorn

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.mcp.server import create_mcp_app


def main() -> None:
    """Start the MCP FastAPI app."""
    settings = get_settings()
    setup_logging(settings.log_level)

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "9001"))

    app = create_mcp_app()
    uvicorn.run(app, host=host, port=port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
