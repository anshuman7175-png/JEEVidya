"""Local MCP server for the JEEVidya workspace.

Run this server locally, then expose port 8000 through a tunnel such as ngrok.
Tasklet can then connect to the public HTTPS URL for the MCP endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


WORKSPACE_ROOT = Path(__file__).resolve().parent
IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
}

mcp = FastMCP("JEEVidya Workspace", json_response=True)


def _resolve_path(relative_path: str) -> Path:
    candidate = (WORKSPACE_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError("Path escapes the workspace root") from exc
    return candidate


def _is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.relative_to(WORKSPACE_ROOT).parts)


@mcp.tool()
def list_workspace(relative_path: str = ".", max_items: int = 300) -> dict[str, Any]:
    """List files and folders beneath a workspace path."""
    root = _resolve_path(relative_path)
    if not root.exists():
        raise FileNotFoundError(relative_path)

    items: list[dict[str, Any]] = []
    if root.is_file():
        rel = root.relative_to(WORKSPACE_ROOT).as_posix()
        return {"root": rel, "items": [{"path": rel, "type": "file"}]}

    for entry in sorted(root.rglob("*")):
        if len(items) >= max_items:
            break
        if _is_ignored(entry):
            continue
        items.append(
            {
                "path": entry.relative_to(WORKSPACE_ROOT).as_posix(),
                "type": "directory" if entry.is_dir() else "file",
            }
        )

    return {
        "root": root.relative_to(WORKSPACE_ROOT).as_posix(),
        "items": items,
        "truncated": len(items) >= max_items,
    }


@mcp.tool()
def read_workspace_file(relative_path: str, max_chars: int = 50000) -> dict[str, Any]:
    """Read a UTF-8 text file from the workspace."""
    path = _resolve_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(relative_path)

    text = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > max_chars
    return {
        "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
        "content": text[:max_chars],
        "truncated": truncated,
    }


@mcp.tool()
def search_workspace(query: str, relative_path: str = ".", max_results: int = 50) -> dict[str, Any]:
    """Search text files in the workspace for a literal string."""
    root = _resolve_path(relative_path)
    if not root.exists():
        raise FileNotFoundError(relative_path)

    results: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if len(results) >= max_results:
            break
        if not path.is_file() or _is_ignored(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if query in content:
            results.append({"path": path.relative_to(WORKSPACE_ROOT).as_posix()})

    return {"query": query, "results": results, "truncated": len(results) >= max_results}


@mcp.tool()
def workspace_root() -> dict[str, str]:
    """Return the absolute path to the workspace root."""
    return {"path": str(WORKSPACE_ROOT)}


if __name__ == "__main__":
    print(f"Starting MCP server for {WORKSPACE_ROOT}")
    print("Connect through your tunnel at http://127.0.0.1:8000/mcp")
    mcp.run(transport="streamable-http")