"""
Ask questions about UK Parliament in the terminal.

The same agent loop as First MCP/agent/cloud_agent.py, pointed at the Hansard
MCP server. Works against any OpenAI-compatible endpoint (Groq, Cerebras,
OpenRouter, local Ollama) - see .env.example.

    .venv/Scripts/python.exe agent/ask.py "What debates happened last week?"
    .venv/Scripts/python.exe agent/ask.py "Who spoke most on 7 July 2026?"
    .venv/Scripts/python.exe agent/ask.py "What did Ed Miliband say in July?"

The loop is three steps: send the conversation plus the tool schemas, execute
any tool calls and append the results, repeat until the model answers in prose.
Everything else here is guard rails learned by hitting real failures.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, BadRequestError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Windows consoles are cp1252 and cannot encode most of what an LLM writes -
# one en dash in the answer would crash print() after a successful run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(PROJECT_ROOT / ".env")

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("LLM_API_KEY", "ollama")
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")

MAX_ROUNDS = 8
# A week of Hansard is ~5,000 contributions. Even with server-side snippets,
# one over-broad call can swamp a free tier's tokens-per-minute budget.
MAX_TOOL_RESULT_CHARS = 6000


def system_prompt() -> str:
    """Built fresh each run so the date is never stale.

    Models do not reliably know today's date, and every question here is
    relative to it ("last week"). Telling it explicitly removes a whole class
    of silently-wrong answers, and means no date-lookup tool is needed.
    """
    today = date.today()
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_sunday = last_monday + timedelta(days=6)
    return f"""You answer questions about the UK Parliament using the Hansard tools.

Today is {today:%A %d %B %Y} ({today:%Y-%m-%d}).
"Last week" means {last_monday:%Y-%m-%d} to {last_sunday:%Y-%m-%d}.
All tool dates are ISO YYYY-MM-DD.

How to work:
- A question that names a person: call find_member FIRST to get their member_id,
  then pass that id to list_contributions. The tools take ids, not names.
- "Who spoke": use who_spoke. It counts on the server. Never try to list
  thousands of contributions and count them yourself.
- Quoting someone: use list_contributions with full_text=true and a SMALL limit.
  Full speeches run to thousands of characters.
- Check `total_count` and `truncated` in every result. If you saw a subset, say so.

Parliament does not sit every day. An empty result usually means recess, not an
error - say that plainly rather than implying nothing was discussed.

Answer in plain prose. Be concrete, cite dates, and do not invent anything the
tools did not return."""


def mcp_tools_to_openai(mcp_tools: list[Any]) -> list[dict]:
    """MCP tool definitions -> OpenAI tool format. Both are JSON Schema."""
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.input_schema,
        },
    } for t in mcp_tools]


def result_to_text(result: Any) -> str:
    parts = [b.text for b in result.content if getattr(b, "type", None) == "text"]
    text = "\n".join(parts) if parts else "(no output)"
    if result.is_error:
        # Hand failures back rather than raising. A good model reads the message
        # and retries with corrected arguments.
        return f"ERROR: {text}"
    if len(text) > MAX_TOOL_RESULT_CHARS:
        omitted = len(text) - MAX_TOOL_RESULT_CHARS
        text = (text[:MAX_TOOL_RESULT_CHARS]
                + f"\n\n[...truncated, {omitted} more characters. Narrow the request - "
                  "a shorter date range, a smaller limit, or full_text=false.]")
    return text


def check_config() -> str | None:
    if not API_KEY or "replace_me" in API_KEY:
        return ("No API key configured.\n\n"
                "  1. cp .env.example .env\n"
                "  2. put your key in .env as LLM_API_KEY=...\n\n"
                "Free key, no credit card: https://console.groq.com")
    if "localhost" not in BASE_URL and not BASE_URL.startswith("https://"):
        return f"Refusing to send an API key over plaintext HTTP to {BASE_URL}."
    return None


async def run_agent(question: str) -> str:
    if problem := check_config():
        return problem

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server" / "hansard.py")],
        env={**os.environ},
    )
    llm = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    async with Client(server_params) as mcp:
        tools = mcp_tools_to_openai((await mcp.list_tools()).tools)
        print(f"[mcp] {len(tools)} tools: {', '.join(t['function']['name'] for t in tools)}")
        print(f"[llm] {MODEL} @ {BASE_URL}\n")

        messages: list[dict] = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": question},
        ]

        for round_no in range(1, MAX_ROUNDS + 1):
            try:
                response = await llm.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools,
                )
            except BadRequestError as exc:
                # The model called a tool that does not exist. Groq validates
                # server-side and rejects the whole request, so there is no
                # assistant message to attach a result to - the correction has
                # to go in as a new user turn.
                if "tool_use_failed" not in str(exc):
                    raise
                available = ", ".join(t["function"]["name"] for t in tools)
                print(f"[round {round_no}] !! model invented a tool; correcting")
                messages.append({"role": "user", "content": (
                    "Your last tool call named a tool that does not exist. The only tools "
                    f"available are: {available}. Retry with one of those.")})
                continue
            except APIStatusError as exc:
                # Groq reports an over-large request as 413, not 429.
                if exc.status_code in (413, 429) or "rate_limit" in str(exc):
                    return ("Hit the provider's tokens-per-minute limit.\n\n"
                            "Every round resends the full history plus all tool schemas. Try a "
                            "narrower date range, or wait a minute and retry.\n\n"
                            f"Provider said: {exc}")
                if exc.status_code in (401, 403):
                    return f"Auth rejected by {BASE_URL}. Check LLM_API_KEY in .env."
                if exc.status_code == 404:
                    return f"Model {MODEL!r} not available at {BASE_URL}. Check LLM_MODEL."
                raise
            except APIConnectionError:
                hint = ("\n\nThat is the default local Ollama endpoint. Create a .env "
                        "(cp .env.example .env) with a Groq key to use a hosted model."
                        if "localhost" in BASE_URL else "")
                return f"Could not reach {BASE_URL}.{hint}"

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or "(empty response)"

            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    print(f"[round {round_no}] -> {name}(<malformed json>)")
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": f"ERROR: arguments were not valid JSON: {exc}"})
                    continue

                print(f"[round {round_no}] -> {name}({json.dumps(args)})")
                try:
                    output = result_to_text(await mcp.call_tool(name, args))
                except Exception as exc:
                    output = f"ERROR: tool call failed: {exc}"

                preview = output.replace("\n", " ")[:150]
                print(f"[round {round_no}] <- {preview}{'...' if len(output) > 150 else ''}")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

        return f"(stopped after {MAX_ROUNDS} rounds without a final answer)"


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print(__doc__)
        sys.exit(1)
    answer = asyncio.run(run_agent(question))
    print("\n" + "=" * 70 + f"\n{answer}")
