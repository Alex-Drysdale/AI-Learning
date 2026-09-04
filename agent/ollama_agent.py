"""
A tool-calling agent running on a self-hosted open-weight LLM via Ollama,
using the same MCP server Claude Code uses.

This is the payoff of MCP: `mcp_server/devtools.py` was never written for any
particular model. Claude Code speaks to it, and so does this 150-line agent
pointed at a Qwen or Llama model on your own hardware. One server, many clients.

An "agent" here is not a framework - it is this loop:

    1. send the conversation + the tool schemas to the model
    2. if the model replied with tool calls, execute them and append the results
    3. go to 1, until the model answers with prose instead of a tool call

Everything else (memory, planning, retries) is an elaboration of those steps.

Usage:
    # against a local ollama
    .venv/Scripts/python.exe agent/ollama_agent.py "What TODOs are outstanding?"

    # against your Hetzner box (see deploy/hetzner-ollama.md)
    OLLAMA_HOST=http://10.0.0.2:11434 OLLAMA_MODEL=qwen3:32b \
        .venv/Scripts/python.exe agent/ollama_agent.py "Summarise the project notes"
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from ollama import AsyncClient, ResponseError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Pick a model that is actually trained for tool calling - most are not.
# qwen3 is the reliable open-weight default; llama3.1+ and mistral-nemo also work.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
MAX_ROUNDS = 8  # a hard stop, so a confused model cannot loop forever

SYSTEM_PROMPT = """You are a development assistant for the Pharosyn project.

You have tools for exploring the codebase and for reading and writing durable
project notes. Prefer calling a tool over guessing: if you are asked about
files, TODOs, or past decisions, look them up.

When you have the answer, reply in plain prose. Be concise and concrete."""


# --------------------------------------------------------------------------
# Bridging MCP <-> Ollama
# --------------------------------------------------------------------------

def mcp_tools_to_ollama(mcp_tools: list[Any]) -> list[dict]:
    """Convert MCP tool definitions into Ollama's tool format.

    This is almost an identity function - both sides use JSON Schema - which is
    exactly why MCP is worth using. The adapter for OpenAI, Anthropic, or any
    other tool-calling API is this same ten lines.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }
        for tool in mcp_tools
    ]


def result_to_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into a string for the model to read."""
    parts = [b.text for b in result.content if getattr(b, "type", None) == "text"]
    text = "\n".join(parts) if parts else "(no output)"
    if result.is_error:
        # Hand the failure back rather than raising. A good model reads the
        # message and retries with corrected arguments - that self-correction
        # is most of what makes an agent feel capable.
        return f"ERROR: {text}"
    return text


# --------------------------------------------------------------------------
# The agent loop
# --------------------------------------------------------------------------

async def run_agent(task: str) -> str:
    # Launch the MCP server as a subprocess and talk JSON-RPC over its pipes.
    # This is the real transport - the same one Claude Code uses - unlike the
    # in-process shortcut in tests/smoke_test.py.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server" / "devtools.py")],
        env={**os.environ, "PHAROSYN_ROOT": str(PROJECT_ROOT)},
    )

    llm = AsyncClient(host=OLLAMA_HOST)

    async with Client(server_params) as mcp:
        listed = await mcp.list_tools()
        tools = mcp_tools_to_ollama(listed.tools)
        print(f"[mcp]    {len(tools)} tools: {', '.join(t['function']['name'] for t in tools)}")
        print(f"[llm]    {OLLAMA_MODEL} @ {OLLAMA_HOST}\n")

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for round_no in range(1, MAX_ROUNDS + 1):
            try:
                response = await llm.chat(model=OLLAMA_MODEL, messages=messages, tools=tools)
            except ResponseError as exc:
                if "not found" in str(exc).lower():
                    return (f"Model {OLLAMA_MODEL!r} is not pulled on {OLLAMA_HOST}.\n"
                            f"Run:  ollama pull {OLLAMA_MODEL}")
                raise

            msg = response.message
            # Append the assistant turn verbatim - including its tool_calls.
            # Dropping them breaks the model's view of its own history.
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or "(empty response)"

            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                if isinstance(args, str):  # some models emit a JSON string
                    args = json.loads(args)

                print(f"[round {round_no}] -> {name}({json.dumps(args)})")
                try:
                    result = await mcp.call_tool(name, args)
                    output = result_to_text(result)
                except Exception as exc:
                    # An unknown tool name or a transport failure. Still feed it
                    # back: the loop must never die on a bad model output.
                    output = f"ERROR: tool call failed: {exc}"

                preview = output.replace("\n", " ")[:160]
                print(f"[round {round_no}] <- {preview}{'...' if len(output) > 160 else ''}")

                messages.append({
                    "role": "tool",
                    "tool_name": name,   # Ollama matches results to calls by name
                    "content": output,
                })

        return f"(stopped after {MAX_ROUNDS} rounds without a final answer)"


def main() -> None:
    task = " ".join(sys.argv[1:]).strip()
    if not task:
        print(__doc__)
        sys.exit(1)
    answer = asyncio.run(run_agent(task))
    print("\n" + "=" * 70 + f"\n{answer}")


if __name__ == "__main__":
    main()
