#!/usr/bin/env python
"""Start the reservations MCP server (settings from .env: MCP_HOST, MCP_PORT, MCP_API_KEY, ...)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mcp_server.server import main  # noqa: E402

if __name__ == "__main__":
    main()
