"""
Exercise the MCP server the way a real client would - but in-process.

`Client(server_object)` connects directly to an MCPServer instance with no
subprocess and no JSON-RPC over pipes. That makes this a fast unit test you can
run on every change. Swap the argument for StdioServerParameters(...) to test
the real subprocess path instead (see agent/ollama_agent.py).

Run:  .venv/Scripts/python.exe tests/smoke_test.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp import Client
from mcp_server.devtools import server


def _text(result) -> str:
    """CallToolResult.content is a list of content blocks; grab the text ones."""
    return "\n".join(b.text for b in result.content if getattr(b, "type", None) == "text")


async def main() -> None:
    async with Client(server) as client:
        print(f"connected to: {client.server_info.name} v{client.server_info.version}\n")

        # --- discovery: exactly what a model is shown -----------------------
        tools = await client.list_tools()
        print("TOOLS")
        for t in tools.tools:
            params = ", ".join(t.input_schema.get("properties", {}))
            print(f"  {t.name}({params})")
            print(f"      {(t.description or '').splitlines()[0]}")

        resources = await client.list_resources()
        templates = await client.list_resource_templates()
        print("\nRESOURCES")
        for r in resources.resources:
            print(f"  {r.uri}  - {r.name}")
        for r in templates.resource_templates:
            print(f"  {r.uri_template}  - {r.name} (template)")

        prompts = await client.list_prompts()
        print("\nPROMPTS")
        for p in prompts.prompts:
            args = ", ".join(a.name for a in (p.arguments or []))
            print(f"  {p.name}({args})")

        # --- calling tools --------------------------------------------------
        print("\n--- list_files(pattern='**/*.py') ---")
        print(_text(await client.call_tool("list_files", {"pattern": "**/*.py"})))

        print("\n--- note_add ---")
        print(_text(await client.call_tool("note_add", {
            "text": "Chose Ollama over vLLM for v1: single binary, good enough throughput at one user.",
            "tags": "infra,decision",
        })))

        print("\n--- note_search(query='ollama') ---")
        print(_text(await client.call_tool("note_search", {"query": "Ollama"})))

        print("\n--- search_code(pattern='def ') ---")
        print(_text(await client.call_tool("search_code", {"pattern": r"^def ", "limit": 5})))

        # --- error path: the server should report, not crash ----------------
        print("\n--- search_code with a broken regex (expect isError) ---")
        bad = await client.call_tool("search_code", {"pattern": "(unclosed"})
        print(f"is_error={bad.is_error}: {_text(bad)}")

        # --- reading a resource --------------------------------------------
        print("\n--- read_resource('notes://recent') ---")
        res = await client.read_resource("notes://recent")
        print(res.contents[0].text)

        print("\n--- read_resource('notes://tag/infra') ---")
        res = await client.read_resource("notes://tag/infra")
        print(res.contents[0].text)

        # --- rendering a prompt --------------------------------------------
        print("\n--- get_prompt('standup') ---")
        got = await client.get_prompt("standup", {"focus": "the Hetzner box"})
        for msg in got.messages:
            print(f"[{msg.role}] {msg.content.text[:400]}...")

    print("\nOK - all surfaces responded.")


if __name__ == "__main__":
    asyncio.run(main())
