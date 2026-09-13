"""Local stdio MCP server used by lifecycle integration tests."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("comet-lifecycle-fixture", log_level="ERROR")


@server.tool()
def echo(value: str) -> str:
    """Echo one value so the client can prove the opened session is usable."""
    return f"echo:{value}"


if __name__ == "__main__":
    server.run(transport="stdio")
