# Pharosyn — MCP servers and self-hosted agents

A working, annotated example of the two things:

1. **An MCP server** (`mcp_server/devtools.py`) — dev utilities for this project,
   consumed by Claude Code.
2. **An agent** (`agent/cloud_agent.py`) — a tool-calling loop on an open-weight
   model, consuming *the same* MCP server.

That overlap is the whole point. You write a capability once; every MCP-speaking
client can use it.

```
                       ┌─────────────────────┐
   Claude Code ───────▶│                     │
                       │  pharosyn-devtools  │──▶ your files, notes.db
   cloud_agent.py ────▶│    (MCP server)     │
        │              └─────────────────────┘
        └──▶ Llama/Qwen on a free tier, or Ollama locally
             (the model brings no tools of its own)
```

---

## Part 1 — MCP

### What it actually is

MCP is a JSON-RPC protocol for handing capabilities to a model. A **client**
(Claude Code) launches a **server** (a program you write) and asks it what it
can do. That's it. The value is that it's a standard: the same server works in
Claude Code, Claude Desktop, and your own agent, with no per-client adapters.

A server exposes three things, and the distinction matters:

| | What it is | Who decides to use it | Use it for |
|---|---|---|---|
| **Tool** | a function the model can call | the **model**, mid-task | actions: search, write, query an API |
| **Resource** | data at a URI the client can read | the **user/client** | context: docs, logs, records |
| **Prompt** | a parameterised prompt template | the **user** | repeatable workflows (`/mcp__server__name`) |

Most servers are 90% tools. Reach for resources when the user should choose
what gets attached; reach for prompts when you keep retyping the same request.

### You're already using them

You have seven remote MCP servers connected. `claude mcp list` shows them.
PubMed, ChEMBL and bioRxiv are MCP servers someone else hosts — when Claude
searches PubMed for you, it's calling tools over this exact protocol.

Adding one is a single command:

```bash
claude mcp add --transport http linear https://mcp.linear.app/mcp   # remote
claude mcp add my-server -- python /path/to/server.py               # local stdio
claude mcp list                                                     # health check
```

Three scopes decide who sees it: `-s local` (you, this project — the default),
`-s user` (you, everywhere), `-s project` (committed to `.mcp.json`, shared
with your team). Project-scoped servers require explicit approval on first run,
which is why `claude mcp list` currently shows this one as *Pending approval* —
a `.mcp.json` arriving via `git pull` should never silently execute.

### Build one

Setup:

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`mcp_server/devtools.py` is a complete, commented server with five tools, two
resources (one a URI template), and a prompt. The core of it:

```python
from mcp.server.mcpserver import MCPServer

server = MCPServer(name="pharosyn-devtools", version="0.1.0")

@server.tool()
def list_files(pattern: str = "**/*", limit: int = 200) -> list[str]:
    """List files in the project matching a glob pattern."""
    ...

server.run(transport="stdio")
```

Your type hints become the JSON Schema the model sees; your docstring becomes
the tool description. There is no schema to hand-write.

**Version note:** this is the `mcp` Python SDK **v2**. If you find a tutorial
using `FastMCP` from `mcp.server.fastmcp`, that's v1 — the class was renamed to
`MCPServer` in `mcp.server.mcpserver`, and the wire types moved from camelCase
to snake_case (`tool.input_schema`, `result.is_error`). Either pin `mcp<2` or
follow the code here.

Test it without wiring it into anything:

```bash
.venv/Scripts/python.exe tests/smoke_test.py
```

`Client(server_object)` connects in-process — no subprocess, no pipes — so this
runs in under a second and belongs in CI.

Then use it in Claude Code: `.mcp.json` already registers it. Run `claude`,
approve the server, and `/mcp` will list it.

### Four things that will bite you

1. **Never print to stdout.** stdout *is* the protocol on a stdio server. One
   stray `print()` corrupts the JSON-RPC stream and the server dies with an
   unhelpful parse error. Log to stderr.
2. **Tool descriptions are prompt engineering.** The model picks tools using
   only the name, the docstring, and the parameter descriptions. Vague docstring,
   unused tool.
3. **Raise the right error.** In SDK v2, `ToolError` means "a failure I saw
   coming" — your message goes back to the model, which can correct itself. Any
   other exception becomes `UnexpectedToolError`: traceback logged server-side,
   model gets a generic message. That's deliberate — it stops internals leaking
   into the context — but it also means a `ValueError` you *wanted* the model to
   read gets swallowed.
4. **The model's input is untrusted.** Your server runs with your privileges.
   Every path parameter needs the `_safe_path()` treatment; every shell-out
   needs an allowlist. Assume the arguments are adversarial, because a prompt
   injection in a file the model read can make them so.

---

## Part 2 — The agent

### An agent is a loop, not a framework

```
1. send conversation + tool schemas to the model
2. model replies with tool calls → execute them, append results
3. repeat until it replies with prose instead
```

`agent/cloud_agent.py` is that, in about 150 lines, with a round cap so a
confused model can't spin forever. Memory, planning, and retries are all
elaborations on those three steps — worth understanding before adopting a
framework that hides them.

The MCP→provider bridge is nearly an identity function, because both sides use
JSON Schema:

```python
{"type": "function", "function": {
    "name": tool.name,
    "description": tool.description,
    "parameters": tool.input_schema,   # straight through
}}
```

The adapter for any other tool-calling API is the same ten lines. That's what
you're buying with MCP.

### Run it without a GPU

```bash
.venv/Scripts/python.exe tests/agent_dryrun.py
```

Everything is real except the model: the server is a genuine subprocess, tools
really execute, a deliberate error really gets fed back. Only `chat()` is
scripted. Develop the plumbing on your laptop, swap in the real model at the end.

### Run it for real — for free

Use `agent/cloud_agent.py`. It's the same loop against any OpenAI-compatible
endpoint, which means every free tier *and* local Ollama, switched by two env
vars. Free tiers are good enough in 2026 that buying hardware to practice on
would be a mistake.

**Where the API key goes:**

```bash
cp .env.example .env     # then edit .env and paste your key
```

`.env` is gitignored; `.env.example` is committed and must never hold a real
key. The agent loads `.env` automatically, and real environment variables still
override it, so `LLM_MODEL=openai/gpt-oss-20b python agent/cloud_agent.py "..."`
works for one-off changes.

Never paste a key directly into a `.py` file. Anything committed to git is
effectively public forever — providers scan GitHub for leaked keys, and if one
does leak, rotate it in the console rather than deleting the commit.

| Provider | Free tier | Models | Best for |
|---|---|---|---|
| **Groq** | 30 RPM, ~14.4K req/day, 8–12K TPM. No card. | GPT-OSS 120B/20B, Qwen3.8 27B, Compound (check `/models` — the lineup changes) | **Start here.** Fastest inference anywhere; agent loops feel instant |
| **Cerebras** | ~1M tokens/day, ~30K TPM | Llama 3.3 70B and others | Highest free volume; long runs |
| **OpenRouter** | 20 RPM, 50 req/day (→1,000 after a one-time $10) | Almost everything, one key | Model comparison; that $10 is the best value here |
| **Local Ollama** | Unlimited, offline | Whatever fits your RAM | Privacy, no rate limits, no network |

```bash
# Groq
LLM_BASE_URL=https://api.groq.com/openai/v1 \
LLM_API_KEY=gsk_... \
LLM_MODEL=openai/gpt-oss-120b \
  .venv/Scripts/python.exe agent/cloud_agent.py "What TODOs are outstanding?"

# Local Ollama (defaults — no env vars needed)
.venv/Scripts/python.exe agent/cloud_agent.py "Summarise the project notes"
```

**Tokens-per-minute is what actually limits you**, not requests-per-minute. Each
round resends the entire history plus all tool schemas, so a 5-round agent run
can burn 15K+ tokens. Groq's 6K TPM on some models will 429 you mid-loop; the
agent catches this and says so instead of crashing.

**Pick a tool-trained model.** Most open-weight models can't reliably emit tool
calls — they'll write JSON into their prose and your loop will never fire.
Qwen3 and Llama 3.3 are reliable; Llama 3.1+ and Mistral-Nemo also work. If tool
calls never appear, suspect the model before your code.

### Running locally on this laptop

The 780M is an iGPU sharing system RAM, so budget against your 16 GB total, not
against VRAM. `qwen3:4b` is comfortable; `qwen3:8b` (~5 GB) works; 14B+ will
thrash. Note that **Ollama's ROCm runner crashes on the 780M** (gfx1103) — use
the bundled Vulkan backend instead (`OLLAMA_VULKAN=1`).

### Self-hosting on a rented GPU

See **[deploy/hetzner-ollama.md](deploy/hetzner-ollama.md)** — machine
selection, VRAM sizing, and not exposing an unauthenticated Ollama port to the
internet. Short version: Hetzner sells GPUs only in the dedicated line (GEX44
20 GB €184/mo, GEX130 48 GB €838/mo, GEX131 96 GB €889/mo), billed monthly with
no scale-to-zero. **Don't do this to practice** — do it when you need data
locality or freedom from rate limits.

In every remote setup the MCP server still runs **on your laptop** — only prompt
text crosses the network. Your files never reach the provider.

---

## Layout

```
mcp_server/devtools.py    the MCP server — tools, resources, prompts
agent/cloud_agent.py      the agent, any OpenAI-compatible endpoint (use this)
agent/ollama_agent.py     same loop against Ollama's native API
tests/smoke_test.py       in-process test of every MCP surface
tests/agent_dryrun.py     full agent loop against a scripted model
deploy/hetzner-ollama.md  self-hosting guide
.mcp.json                 registers the server with Claude Code
```

## Next steps

- Add a tool that touches something you actually care about — ChEMBL lookups,
  an assay CSV, your lab notebook — and watch it appear in both clients at once.
- Switch the server to `transport="streamable-http"` and run it next to the
  model instead of on your laptop.
- Give the agent a memory: it already has `note_add` / `note_search`, so seed
  the conversation from `note_search` on startup.
# AI-Learning
