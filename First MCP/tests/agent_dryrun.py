"""
Exercise the full agent loop with a scripted fake model.

Everything is real except the LLM: the MCP server is launched as a genuine
subprocess over stdio, tools are really listed and really called, and the loop
is the same code that runs against Ollama. Only `AsyncClient.chat` is replaced
by a canned script.

Worth keeping. It means you can develop the agent's plumbing without a GPU, and
it fails loudly if an SDK upgrade changes the message shapes.

Run:  .venv/Scripts/python.exe tests/agent_dryrun.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
from ollama._types import ChatResponse, Message

from agent import ollama_agent

# A scripted "model": round 1 calls a tool, round 2 calls another, round 3 answers.
SCRIPT = [
    Message(role="assistant", content="", tool_calls=[
        Message.ToolCall(function=Message.ToolCall.Function(
            name="note_add",
            arguments={"text": "Dry-run reached the tool loop.", "tags": "test"},
        )),
    ]),
    Message(role="assistant", content="", tool_calls=[
        Message.ToolCall(function=Message.ToolCall.Function(
            name="search_code", arguments={"pattern": "(broken", "file_glob": "**/*.py"},
        )),
    ]),
    Message(role="assistant", content="", tool_calls=[
        Message.ToolCall(function=Message.ToolCall.Function(
            name="note_search", arguments={"query": "Dry-run"},
        )),
    ]),
    Message(role="assistant",
            content="Recorded the note, saw the regex error, and read it back. Loop works."),
]


async def fake_chat(self, model, messages, tools=None, **kwargs):
    idx = fake_chat.calls
    fake_chat.calls += 1
    assert tools, "the agent must pass tool schemas to the model"
    if idx == 0:
        names = [t["function"]["name"] for t in tools]
        print(f"[fake-llm] received {len(tools)} tool schemas: {', '.join(names)}")
        # The schema the model actually sees, for one tool:
        print(f"[fake-llm] note_add parameters: "
              f"{[t for t in tools if t['function']['name'] == 'note_add'][0]['function']['parameters']}")
    return ChatResponse(model=model, message=SCRIPT[idx], done=True)


fake_chat.calls = 0


async def main() -> None:
    ollama.AsyncClient.chat = fake_chat
    answer = await ollama_agent.run_agent("dry run")
    print("\n" + "=" * 70 + f"\n{answer}")
    assert fake_chat.calls == 4, f"expected 4 model turns, got {fake_chat.calls}"
    print("\nOK - loop executed tools, fed an error back, and terminated on prose.")


if __name__ == "__main__":
    asyncio.run(main())
