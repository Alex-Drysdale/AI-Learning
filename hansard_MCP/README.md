# Hansard MCP

An MCP server over the UK Parliament APIs, plus a terminal agent that uses it.
Second project in this repo, after [`First MCP`](../First%20MCP) — same patterns,
but pointed at a real external API instead of a local SQLite file.

It answers three questions:

1. **What debates happened last week?** → `list_debates`
2. **Who spoke last week?** → `who_spoke`
3. **What did person X say last week?** → `find_member` + `list_contributions`

```
                     ┌──────────────────┐      hansard-api.parliament.uk
   Claude Code ─────▶│                  │─────▶
                     │  hansard (MCP)   │      members-api.parliament.uk
   agent/ask.py ────▶│                  │─────▶
        │            └──────────────────┘      (no auth, Open Parliament Licence)
        └──▶ gpt-oss-120b on Groq
```

## Quick start

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
cp .env.example .env          # then paste a Groq key (free, no card)

.venv/Scripts/python.exe agent/ask.py "What debates happened last week?"
.venv/Scripts/python.exe agent/ask.py "Who spoke most in the Commons on 7 July 2026?"
.venv/Scripts/python.exe agent/ask.py "What did Ed Miliband say in July 2026?"
```

In Claude Code: `.mcp.json` already registers the server — run `claude` from this
folder, approve it, and ask the same questions there. One server, two clients.

## The tools

| Tool | Answers |
|---|---|
| `find_member(name, limit)` | Name → `member_id`. **Call this first** for any question naming a person; the other tools take ids, not names. |
| `list_debates(start_date, end_date, house, limit)` | What was debated in a date range. |
| `list_contributions(start_date, end_date, member_id, search_term, house, full_text, limit)` | What was actually said — by one member, or matching a topic. |
| `who_spoke(start_date, end_date, house, top)` | Ranked speakers with contribution counts. Counts server-side. |
| `get_debate(debate_id, start_index, max_items)` | Drill into one debate's transcript, paginated. |

Dates are ISO `YYYY-MM-DD`. There is deliberately **no date tool** — `agent/ask.py`
puts today's date in the system prompt and lets the model do the arithmetic, and
Claude Code already knows the date.

## The design constraint that shaped everything

A single sitting day carries **942 spoken contributions** (756 Commons + 186 Lords
on 2026-07-07). A week is roughly 5,000 — about a novel. Measured, not guessed.

So no tool returns bulk text:

- **List tools return ~220-character snippets.** The longest speech measured was
  **8,961 characters** — 40× its snippet. `full_text=True` exists for quoting, and
  the tool description warns to use it with a small limit.
- **`who_spoke` aggregates on the server** and returns names and counts only, no
  speech text at all. Asking a model to count 5,000 rows is not an option.
- **Every list result carries `total_count` and `truncated`.** This is not
  decoration: in a live run the model saw `truncated: true` on 20 of 24 results
  and re-called with a higher limit on its own.
- **`who_spoke` caps its scan at 1,000 contributions** and says so in a `note`
  field when it does, rather than presenting a sample as a complete count.

## Notes worth keeping

**Silent wrong answers are the real risk.** A malformed date makes the Hansard API
return zero results rather than an error, so the model would confidently report
"nothing was debated that week". `_check_dates` rejects non-ISO dates and reversed
ranges up front for exactly this reason.

**Recess is not an error.** Parliament does not sit every day. Both the server
instructions and the agent's system prompt say so, so an empty week is reported as
recess rather than as a failure. Verified: asking "what debates happened last week"
in September correctly returned "24–30 August, no debates — Parliament in recess".

**`ToolError` vs a crash.** Anticipated failures (bad date, unknown member, unknown
house) raise `ToolError`, so the message reaches the model and it can retry. Any
other exception becomes `UnexpectedToolError` — logged server-side, generic message
to the model. `tests/agent_dryrun.py` sends a deliberately bad date to prove the
recovery path works.

**Hansard's query convention** is `?queryParameters.startDate=...`, not plain
`?startDate=`. Easy to get wrong; `_search_params()` centralises it.

## Tests

```bash
.venv/Scripts/python.exe tests/smoke_test.py     # 20 checks against the live API
.venv/Scripts/python.exe tests/agent_dryrun.py   # the agent loop, no LLM needed
```

`smoke_test.py` connects in-process (`Client(server)`) so there's no subprocess,
but the HTTP calls are real. `agent_dryrun.py` runs the genuine loop — real
subprocess, real Hansard calls — with only `chat.completions.create` scripted, so
you can work on the agent with no API key and no rate limit.

## Files

```
mcp_server/hansard.py    the server — 5 tools, shared HTTP helper with retry
agent/ask.py             the terminal agent (any OpenAI-compatible endpoint)
tests/smoke_test.py      every tool, against the live API
tests/agent_dryrun.py    the agent loop, scripted model
.mcp.json                registers the server with Claude Code
```

## Next steps

- Add `search_term` questions ("who talked about floating wind?") — the tool already
  supports it, it just isn't exercised yet.
- Add the other Parliament APIs: Bills, Commons/Lords Votes, Written Questions,
  Committees, Register of Interests. All unauthenticated, all the same shape.
- Cache repeat calls — `who_spoke` for a busy week costs 10 HTTP round trips.
