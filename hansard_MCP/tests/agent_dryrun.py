"""
Exercise the full agent loop with a scripted fake model.

Everything is real except the LLM: the MCP server runs as a genuine subprocess
over stdio, the Hansard API is really called, and the loop is the same code that
runs against Groq. Only chat.completions.create is replaced by a canned script.

Worth keeping. It means you can work on the agent's plumbing with no API key and
no rate limit, it is deterministic (real models vary run to run), and it fails
loudly if an SDK upgrade changes the message shapes.

Run:  .venv/Scripts/python.exe tests/agent_dryrun.py
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as ToolCall,
    Function,
)

from agent import ask


def assistant_tool_call(call_id: str, tool: str, /, **arguments) -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=None, tool_calls=[
        ToolCall(id=call_id, type="function",
                 function=Function(name=tool, arguments=json.dumps(arguments))),
    ])


# Round 1 resolves a name. Round 2 sends a deliberately malformed date, which
# the server must reject with a readable message rather than crashing. Round 3
# recovers with a valid call. Round 4 answers in prose, ending the loop.
SCRIPT = [
    assistant_tool_call("c1", "find_member", name="Ed Miliband"),
    assistant_tool_call("c2", "list_contributions",
                        start_date="last Tuesday", end_date="2026-07-31", member_id=1510),
    assistant_tool_call("c3", "list_contributions",
                        start_date="2026-07-01", end_date="2026-07-31", member_id=1510, limit=5),
    ChatCompletionMessage(role="assistant",
                          content="Resolved the member, recovered from a bad date, and read "
                                  "his contributions. Loop works."),
]

seen: dict = {"calls": 0, "tool_names": [], "errors": 0}


async def fake_create(*, model, messages, tools=None, **kwargs):
    idx = seen["calls"]
    seen["calls"] += 1
    assert tools, "the agent must pass tool schemas to the model"

    if idx == 0:
        names = [t["function"]["name"] for t in tools]
        print(f"[fake-llm] received {len(tools)} tool schemas: {', '.join(names)}")
        blob = json.dumps(tools)
        print(f"[fake-llm] schemas cost ~{len(blob) // 4} tokens, resent every round")
        assert messages[0]["role"] == "system"
        assert "Today is" in messages[0]["content"], "system prompt must state today's date"
        print("[fake-llm] system prompt carries today's date  OK")

    # Count the tool results the agent fed back that were errors.
    for m in messages:
        if m.get("role") == "tool" and str(m.get("content", "")).startswith("ERROR:"):
            seen["errors"] = max(seen["errors"], 1)

    return SimpleNamespace(choices=[SimpleNamespace(message=SCRIPT[idx])])


class FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=fake_create))


async def main() -> None:
    ask.AsyncOpenAI = FakeClient
    answer = await ask.run_agent("dry run")
    print("\n" + "=" * 70 + f"\n{answer}")

    problems = []
    if seen["calls"] != 4:
        problems.append(f"expected 4 model turns, got {seen['calls']}")
    if not seen["errors"]:
        problems.append("the bad-date tool error was never fed back to the model")
    if "Loop works" not in answer:
        problems.append("loop did not terminate on the scripted prose answer")

    if problems:
        print("\nFAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("\nOK - loop called real tools, fed an error back, recovered, and ended on prose.")


if __name__ == "__main__":
    asyncio.run(main())
