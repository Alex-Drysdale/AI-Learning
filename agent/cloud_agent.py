"""
The same agent, against any OpenAI-compatible endpoint.

Groq, Cerebras, OpenRouter, Together, and local Ollama all speak the OpenAI
chat-completions shape, so one file covers every free tier *and* your laptop.
Only two environment variables change between them.

    # Groq - fastest free tier, no credit card
    LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_API_KEY=gsk_...  LLM_MODEL=llama-3.3-70b-versatile

    # Cerebras - highest free daily volume
    LLM_BASE_URL=https://api.cerebras.ai/v1      LLM_API_KEY=csk-...  LLM_MODEL=llama-3.3-70b

    # OpenRouter - widest model choice
    LLM_BASE_URL=https://openrouter.ai/api/v1    LLM_API_KEY=sk-or-... LLM_MODEL=qwen/qwen3-32b

    # Local Ollama - free, offline, no rate limits
    LLM_BASE_URL=http://localhost:11434/v1       LLM_API_KEY=ollama   LLM_MODEL=qwen3:8b

Usage:
    .venv/Scripts/python.exe agent/cloud_agent.py "What TODOs are outstanding?"

Compare with ollama_agent.py: the loop is identical, but the message shapes
differ. OpenAI-style tool calls carry an `id` and JSON-*string* arguments, and
results are matched back by `tool_call_id`. Ollama's native API matches by name
and passes arguments as a dict. This is the kind of per-provider detail MCP
saves you from having to repeat for every *tool* - but not for every *model API*.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, BadRequestError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Windows consoles default to cp1252, which cannot encode most of what an LLM
# writes - a single en dash or non-breaking hyphen in the answer crashes the
# print() with UnicodeEncodeError after the whole run has succeeded.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load .env from the project root. Real environment variables always win over
# the file, so you can still override per-command:
#     LLM_MODEL=qwen3:32b python agent/cloud_agent.py "..."
load_dotenv(PROJECT_ROOT / ".env")

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")  # local Ollama ignores the value
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
MAX_ROUNDS = 8

# Cap each tool result before it enters the conversation. Without this, one
# read_file on a large file can single-handedly exceed a free tier's
# tokens-per-minute limit - and it costs you on every subsequent round too,
# because the whole history is resent each time.
MAX_TOOL_RESULT_CHARS = 4000

SYSTEM_PROMPT = """You are a development assistant for the Pharosyn project.

You have tools for exploring the codebase and for reading and writing durable
project notes. Prefer calling a tool over guessing: if you are asked about
files, TODOs, or past decisions, look them up.

When you have the answer, reply in plain prose. Be concise and concrete."""


def mcp_tools_to_openai(mcp_tools: list[Any]) -> list[dict]:
    """MCP tool definitions -> OpenAI tool format. Again, nearly identity."""
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
    parts = [b.text for b in result.content if getattr(b, "type", None) == "text"]
    text = "\n".join(parts) if parts else "(no output)"
    if result.is_error:
        return f"ERROR: {text}"
    if len(text) > MAX_TOOL_RESULT_CHARS:
        # Tell the model it was truncated rather than silently lying to it -
        # otherwise it will conclude the file simply ends here.
        omitted = len(text) - MAX_TOOL_RESULT_CHARS
        text = (text[:MAX_TOOL_RESULT_CHARS]
                + f"\n\n[...truncated, {omitted} more characters. "
                  "Narrow the request - a line range or a tighter pattern - to see the rest.]")
    return text


def check_config() -> str | None:
    """Catch the common setup mistakes before spending a request on them."""
    if not API_KEY or "replace_me" in API_KEY:
        return (
            "No API key configured.\n\n"
            "  1. cp .env.example .env\n"
            "  2. put your key in .env as LLM_API_KEY=...\n\n"
            "Free key, no credit card: https://console.groq.com\n"
            ".env is gitignored, so it will not be committed."
        )
    if "localhost" not in BASE_URL and not BASE_URL.startswith("https://"):
        return f"Refusing to send an API key over plaintext HTTP to {BASE_URL}."
    return None


async def run_agent(task: str) -> str:
    if problem := check_config():
        return problem

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server" / "devtools.py")],
        env={**os.environ, "PHAROSYN_ROOT": str(PROJECT_ROOT)},
    )
    llm = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    async with Client(server_params) as mcp:
        listed = await mcp.list_tools()
        tools = mcp_tools_to_openai(listed.tools)
        print(f"[mcp] {len(tools)} tools: {', '.join(t['function']['name'] for t in tools)}")
        print(f"[llm] {MODEL} @ {BASE_URL}\n")

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for round_no in range(1, MAX_ROUNDS + 1):
            try:
                response = await llm.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools,
                )
            except BadRequestError as exc:
                # The model tried to call a tool that does not exist. Some
                # providers (Groq) validate server-side and reject the whole
                # request, so there is no assistant message to attach a result
                # to - the correction has to go in as a new user turn.
                #
                # This is usually a signal, not noise: the model reached for a
                # capability your server does not expose. Read the tool name it
                # invented and consider whether you should have built it.
                if "tool_use_failed" not in str(exc):
                    raise
                available = ", ".join(t["function"]["name"] for t in tools)
                print(f"[round {round_no}] !! model invented a tool; correcting")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last tool call named a tool that does not exist, so it was "
                        f"rejected. The only tools available are: {available}. "
                        "Retry using one of those, or answer directly if none fit."
                    ),
                })
                continue
            except APIStatusError as exc:
                # Groq signals a too-large request as 413, not 429, but the
                # error code is still rate_limit_exceeded. Check both.
                if exc.status_code in (413, 429) or "rate_limit" in str(exc):
                    # The most common free-tier failure. Agent loops are
                    # token-hungry: every round resends the whole history plus
                    # all tool schemas, so tokens-per-minute bites long before
                    # requests-per-minute does.
                    return (
                        "Hit the provider's token-per-minute limit.\n\n"
                        "Every round resends the full history plus all tool schemas, so "
                        "usage grows quadratically over a run. Options:\n"
                        "  - wait a minute and retry\n"
                        "  - lower MAX_TOOL_RESULT_CHARS (currently "
                        f"{MAX_TOOL_RESULT_CHARS})\n"
                        "  - ask the model for a narrower line range\n"
                        "  - switch LLM_MODEL to a model with a higher TPM cap\n\n"
                        f"Provider said: {exc.message if hasattr(exc, 'message') else exc}"
                    )
                if exc.status_code in (401, 403):
                    return f"Auth rejected by {BASE_URL}. Check LLM_API_KEY."
                if exc.status_code == 404:
                    return f"Model {MODEL!r} not available at {BASE_URL}. Check LLM_MODEL."
                raise
            except APIConnectionError:
                # Nothing listening. Overwhelmingly this is "no .env, so it fell
                # back to a local Ollama that isn't running".
                hint = ""
                if "localhost" in BASE_URL:
                    hint = ("\n\nThat is the default local Ollama endpoint. Either start "
                            "Ollama, or create a .env (cp .env.example .env) with a free "
                            "Groq key to use a hosted model instead.")
                return f"Could not reach {BASE_URL}.{hint}"

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or "(empty response)"

            for call in msg.tool_calls:
                name = call.function.name
                # OpenAI-style arguments are a JSON *string*, not a dict.
                # Small models sometimes emit malformed JSON here - handle it.
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    output = f"ERROR: your arguments were not valid JSON: {exc}"
                    print(f"[round {round_no}] -> {name}(<malformed>)")
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                    continue

                print(f"[round {round_no}] -> {name}({json.dumps(args)})")
                try:
                    output = result_to_text(await mcp.call_tool(name, args))
                except Exception as exc:
                    output = f"ERROR: tool call failed: {exc}"

                preview = output.replace("\n", " ")[:160]
                print(f"[round {round_no}] <- {preview}{'...' if len(output) > 160 else ''}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,   # matched by id, not by name
                    "content": output,
                })

        return f"(stopped after {MAX_ROUNDS} rounds without a final answer)"


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]).strip()
    if not task:
        print(__doc__)
        sys.exit(1)
    answer = asyncio.run(run_agent(task))
    print("\n" + "=" * 70 + f"\n{answer}")
