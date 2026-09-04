"""
Pharosyn dev-utilities MCP server.

An MCP server exposes three kinds of thing to a client (Claude Code, your own
agent, Claude Desktop, ...):

  * tools     - functions the model may CALL (verbs; may have side effects)
  * resources - data the client may READ into context (nouns; addressed by URI)
  * prompts   - reusable prompt templates the user may INVOKE (in Claude Code
                these show up as /mcp__<server>__<prompt> slash commands)

Everything below talks over stdio: the client launches this file as a
subprocess and speaks JSON-RPC on stdin/stdout. That is why nothing here may
ever print to stdout - use stderr for logging.

SDK note: this is the `mcp` Python SDK v2. In v1 the class was FastMCP
(mcp.server.fastmcp); in v2 it is MCPServer (mcp.server.mcpserver).
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# The directory this server is allowed to touch. Everything is sandboxed to it.
ROOT = Path(os.environ.get("PHAROSYN_ROOT", Path.cwd())).resolve()
DB_PATH = ROOT / ".pharosyn" / "notes.db"

# Directories that are never worth searching.
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".pharosyn",
}

MAX_MATCH_LINE = 300  # truncate long lines so one minified file cannot flood context

server = MCPServer(
    name="pharosyn-devtools",
    version="0.1.0",
    # `instructions` is shown to the model as guidance about the whole server.
    # Treat it as a very short system prompt for your toolset.
    instructions=(
        "Local development utilities for the Pharosyn project. Use list_files and "
        "search_code to explore the tree, scan_todos to find outstanding work, and "
        "the note tools to record and recall decisions across sessions. All paths are "
        "relative to the project root."
    ),
)


# --------------------------------------------------------------------------
# Helpers (not exposed over MCP)
# --------------------------------------------------------------------------

def _log(msg: str) -> None:
    """stdout belongs to the protocol; logs go to stderr."""
    print(f"[pharosyn-devtools] {msg}", file=sys.stderr, flush=True)


def _is_skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)


def _iter_files(pattern: str):
    """Yield files under ROOT matching a glob, skipping noise directories."""
    for path in ROOT.glob(pattern):
        if path.is_file() and not _is_skipped(path):
            yield path


def _safe_path(relative: str) -> Path:
    """Resolve a user-supplied path and refuse to escape ROOT.

    Any tool that accepts a path needs a check like this. An MCP server runs
    with your full user privileges - the model's input is untrusted.
    """
    candidate = (ROOT / relative).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f"path escapes project root: {relative!r}")
    return candidate


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            text       TEXT NOT NULL,
            tags       TEXT NOT NULL DEFAULT ''
        )
        """
    )
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
#
# The decorator turns the function signature into a JSON Schema the model sees.
# Type hints become the schema; the docstring becomes the tool description;
# Field(description=...) documents individual parameters. Write these for a
# reader who has never seen your codebase - they are prompt engineering, not
# just documentation.

@server.tool(
    # Annotations are hints to the *client* about how to treat the tool.
    # readOnlyHint lets a host auto-approve it instead of prompting every time.
    annotations=ToolAnnotations(title="List project files", readOnlyHint=True),
)
def list_files(
    pattern: Annotated[str, Field(description="Glob relative to the project root, e.g. '**/*.py'")] = "**/*",
    limit: Annotated[int, Field(description="Maximum number of paths to return", ge=1, le=1000)] = 200,
) -> list[str]:
    """List files in the project matching a glob pattern.

    Skips .git, .venv, node_modules and other build/cache directories.
    """
    out = [str(p.relative_to(ROOT)).replace("\\", "/") for p in _iter_files(pattern)]
    out.sort()
    return out[:limit]


@server.tool(annotations=ToolAnnotations(title="Search code", readOnlyHint=True))
def search_code(
    pattern: Annotated[str, Field(description="Python regular expression to search for")],
    file_glob: Annotated[str, Field(description="Which files to search, as a glob")] = "**/*.py",
    ignore_case: Annotated[bool, Field(description="Case-insensitive match")] = False,
    limit: Annotated[int, Field(description="Maximum matches to return", ge=1, le=500)] = 50,
) -> list[dict]:
    """Search file contents for a regular expression.

    Returns one entry per match with the file path, 1-indexed line number, and
    the matching line.
    """
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        # Two very different error paths, and the choice matters:
        #
        #   raise ToolError(...)  -> "a failure I saw coming". Your message is
        #       sent back to the model as an is_error result, so it can read it
        #       and retry with a corrected argument. No traceback is logged.
        #
        #   raise anything else   -> the SDK wraps it in UnexpectedToolError,
        #       logs the traceback server-side, and sends the model only a
        #       generic "Error executing tool search_code". Your message is
        #       deliberately withheld, so internals cannot leak to the model.
        #
        # A bad regex is the model's fault and the model can fix it: ToolError.
        raise ToolError(f"invalid regular expression {pattern!r}: {exc}") from exc

    matches: list[dict] = []
    for path in _iter_files(file_glob):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                matches.append({
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "line": lineno,
                    "text": line.strip()[:MAX_MATCH_LINE],
                })
                if len(matches) >= limit:
                    return matches
    return matches


@server.tool(annotations=ToolAnnotations(title="Read a file", readOnlyHint=True))
def read_file(
    path: Annotated[str, Field(description="File path relative to the project root, e.g. 'README.md'")],
    start_line: Annotated[int, Field(description="First line to return, 1-indexed", ge=1)] = 1,
    max_lines: Annotated[int, Field(description="How many lines to return", ge=1, le=2000)] = 150,
) -> dict:
    """Read the contents of a text file in the project.

    Returns the requested line range with 1-indexed line numbers. Use this after
    list_files or search_code to actually see what a file says.
    """
    # _safe_path is the reason this tool is not a security hole: without it,
    # path="../../../../Users/drysd/.ssh/id_rsa" would just work.
    target = _safe_path(path)
    if not target.is_file():
        raise ToolError(f"no such file: {path}")
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ToolError(f"could not read {path}: {exc}") from exc

    chunk = lines[start_line - 1 : start_line - 1 + max_lines]
    return {
        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "total_lines": len(lines),
        "start_line": start_line,
        "truncated": start_line - 1 + len(chunk) < len(lines),
        "content": "\n".join(f"{start_line + i:>5}  {ln}" for i, ln in enumerate(chunk)),
    }


@server.tool(annotations=ToolAnnotations(title="Scan for TODOs", readOnlyHint=True))
def scan_todos(
    file_glob: Annotated[str, Field(description="Which files to scan, as a glob")] = "**/*.py",
    limit: Annotated[int, Field(description="Maximum items to return", ge=1, le=500)] = 100,
) -> list[dict]:
    """Find outstanding TODO / FIXME / HACK / XXX markers in the project."""
    return search_code(
        pattern=r"\b(TODO|FIXME|HACK|XXX)\b",
        file_glob=file_glob,
        ignore_case=False,
        limit=limit,
    )


@server.tool(
    annotations=ToolAnnotations(
        title="Record a note",
        readOnlyHint=False,      # it writes
        destructiveHint=False,   # ...but only ever appends
        idempotentHint=False,    # calling twice creates two notes
    ),
)
def note_add(
    text: Annotated[str, Field(description="The note body, in plain prose")],
    tags: Annotated[str, Field(description="Optional comma-separated tags, e.g. 'infra,decision'")] = "",
) -> dict:
    """Record a durable note about this project.

    Use for decisions, gotchas, and context worth surviving past the current
    session. Notes persist in .pharosyn/notes.db and are searchable with
    note_search.
    """
    normalized = ",".join(t.strip() for t in tags.split(",") if t.strip())
    created = _now()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO notes (created_at, text, tags) VALUES (?, ?, ?)",
            (created, text.strip(), normalized),
        )
        note_id = cur.lastrowid
    _log(f"note #{note_id} recorded")
    return {"id": note_id, "created_at": created, "text": text.strip(), "tags": normalized}


@server.tool(annotations=ToolAnnotations(title="Search notes", readOnlyHint=True))
def note_search(
    query: Annotated[str, Field(description="Substring to look for in note text or tags; empty returns the most recent")] = "",
    limit: Annotated[int, Field(description="Maximum notes to return", ge=1, le=200)] = 20,
) -> list[dict]:
    """Search previously recorded project notes, newest first."""
    with _db() as conn:
        if query.strip():
            rows = conn.execute(
                "SELECT * FROM notes WHERE text LIKE ? OR tags LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------
#
# A resource is addressed by URI and is READ, not called. The distinction that
# matters in practice: tools are chosen by the model, resources are attached by
# the user or the client application. Use resources for "here is the context",
# tools for "go do something".

@server.resource(
    "notes://recent",
    name="Recent project notes",
    description="The 20 most recently recorded Pharosyn notes, newest first.",
    mime_type="text/markdown",
)
def recent_notes() -> str:
    rows = note_search(query="", limit=20)
    if not rows:
        return "_No notes recorded yet._"
    lines = ["# Recent Pharosyn notes", ""]
    for r in rows:
        tags = f"  [{r['tags']}]" if r["tags"] else ""
        lines.append(f"- **#{r['id']}** ({r['created_at']}){tags}\n  {r['text']}")
    return "\n".join(lines)


@server.resource(
    "notes://tag/{tag}",     # {tag} makes this a resource *template*:
    name="Notes by tag",     # the client can fill in any tag at read time.
    description="All notes carrying a given tag.",
    mime_type="text/markdown",
)
def notes_by_tag(tag: str) -> str:
    rows = note_search(query=tag, limit=100)
    rows = [r for r in rows if tag in r["tags"].split(",")]
    if not rows:
        # Resources have the same split: ResourceError / ResourceNotFoundError
        # for anticipated failures, anything else becomes a logged crash.
        raise ResourceNotFoundError(f"no notes tagged {tag!r}")
    return "\n".join(f"- **#{r['id']}** ({r['created_at']}) {r['text']}" for r in rows)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
#
# In Claude Code these appear as /mcp__pharosyn-devtools__standup

@server.prompt(
    name="standup",
    description="Draft a standup update from recent notes and outstanding TODOs.",
)
def standup_prompt(focus: str = "") -> str:
    notes = recent_notes()
    todos = scan_todos(limit=20)
    todo_text = "\n".join(f"- {t['path']}:{t['line']}  {t['text']}" for t in todos) or "_none found_"
    focus_line = f"\nFocus especially on: {focus}\n" if focus.strip() else ""
    return (
        "Write a short standup update for the Pharosyn project: what moved, what is "
        "in flight, and what is blocked. Be concrete and skip filler.\n"
        f"{focus_line}\n"
        f"## Recent notes\n{notes}\n\n"
        f"## Outstanding TODOs\n{todo_text}\n"
    )


# --------------------------------------------------------------------------

if __name__ == "__main__":
    _log(f"serving over stdio, root={ROOT}")
    # transport="stdio" is the default; "streamable-http" would serve over HTTP
    # instead, which is what you want for a remote/shared server.
    server.run(transport="stdio")
