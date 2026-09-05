"""
Exercise every Hansard tool against the real Parliament API.

Connects in-process with Client(server) - no subprocess, no JSON-RPC over pipes -
so this is fast, but the HTTP calls are genuine. Uses 2026-07-06..2026-07-10,
a week known to contain sitting days (2026-07-07 alone has 942 contributions).

Run:  .venv/Scripts/python.exe tests/smoke_test.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mcp import Client
from mcp_server.hansard import server

WEEK_START, WEEK_END = "2026-07-06", "2026-07-10"


def payload(result) -> dict:
    text = "\n".join(b.text for b in result.content if getattr(b, "type", None) == "text")
    if result.is_error:
        return {"_error": text}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text}


async def main() -> None:
    failures = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
        if not condition:
            failures.append(label)

    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        print(f"\n{len(tools)} tools: {', '.join(t.name for t in tools)}\n")

        # --- find_member ------------------------------------------------
        print("find_member('Ed Miliband')")
        r = payload(await client.call_tool("find_member", {"name": "Ed Miliband"}))
        top = (r.get("results") or [{}])[0]
        check("resolves to member_id 1510", top.get("member_id") == 1510, f"got {top.get('member_id')}")
        check("carries party and constituency",
              bool(top.get("party") and top.get("constituency")),
              f"{top.get('party')} / {top.get('constituency')}")

        # --- question 1 -------------------------------------------------
        print("\nlist_debates  (Q1: what debates happened last week)")
        r = payload(await client.call_tool("list_debates", {
            "start_date": WEEK_START, "end_date": WEEK_END, "house": "Commons", "limit": 5,
        }))
        check("returns debates", len(r.get("results") or []) > 0, f"{r.get('total_count')} total")
        check("reports truncation honestly", r.get("truncated") is True)
        first = (r.get("results") or [{}])[0]
        check("rows have a debate_id to drill into", bool(first.get("debate_id")))
        print(f"        e.g. {first.get('date')} {first.get('house')} - {first.get('title')}")

        # --- question 2 -------------------------------------------------
        print("\nwho_spoke  (Q2: who spoke last week)")
        r = payload(await client.call_tool("who_spoke", {
            "start_date": "2026-07-07", "end_date": "2026-07-07", "house": "Commons", "top": 5,
        }))
        check("ranks speakers", len(r.get("results") or []) > 0,
              f"{r.get('distinct_speakers')} distinct speakers")
        check("counts descend",
              all(a["count"] >= b["count"] for a, b in zip(r["results"], r["results"][1:])))
        check("returns no speech text", all("text" not in row for row in r.get("results") or []))
        print(f"        scanned {r.get('contributions_scanned')} of {r.get('total_contributions')}")
        print(f"        note: {r.get('note')}")
        for row in (r.get("results") or [])[:3]:
            print(f"        {row['count']:>4}  {row['name']}")

        # --- question 3 -------------------------------------------------
        print("\nlist_contributions  (Q3: what did Ed Miliband say)")
        r = payload(await client.call_tool("list_contributions", {
            "start_date": "2026-07-01", "end_date": "2026-07-31", "member_id": 1510, "limit": 3,
        }))
        rows = r.get("results") or []
        check("returns his contributions", len(rows) > 0, f"{r.get('total_count')} in July")
        check("all attributed to him", all(x.get("member_id") == 1510 for x in rows))
        check("snippets are short by default",
              all(len(x.get("text") or "") <= 230 for x in rows))
        if rows:
            print(f"        {rows[0]['date']} | {rows[0]['debate']}")
            print(f"        {rows[0]['text'][:120]}...")

        # full_text opt-in. Comparing a single contribution proves nothing -
        # many are genuinely short (Miliband's Topical Questions answers run
        # 92-239 chars). Fetch the same batch both ways and compare pairwise.
        print("\nlist_contributions  (snippet vs full_text)")
        args = {"start_date": "2026-07-07", "end_date": "2026-07-07",
                "house": "Commons", "limit": 40}
        brief = payload(await client.call_tool("list_contributions", args))["results"]
        full = payload(await client.call_tool("list_contributions", {**args, "full_text": True}))["results"]

        check("both modes return the same rows",
              [x["member"] for x in brief] == [x["member"] for x in full])
        check("every snippet is capped",
              all(len(x["text"]) <= 226 for x in brief),
              f"longest snippet {max(len(x['text']) for x in brief)}")
        longer = [(len(b["text"]), len(f["text"])) for b, f in zip(brief, full)
                  if len(f["text"]) > len(b["text"])]
        check("full_text returns more where the speech is long",
              len(longer) > 0, f"{len(longer)}/{len(brief)} rows longer")
        if longer:
            b_len, f_len = max(longer, key=lambda p: p[1])
            check("truncation is marked with an ellipsis",
                  any(x["text"].endswith("...") for x in brief))
            print(f"        longest: {b_len} chars snippet -> {f_len} chars full")

        # --- drill-down -------------------------------------------------
        print("\nget_debate  (drill-down)")
        debate_id = first.get("debate_id")
        r = payload(await client.call_tool("get_debate", {"debate_id": debate_id, "max_items": 5}))
        check("returns a titled transcript", bool(r.get("title")), r.get("title", "")[:50])
        check("reports total_items", isinstance(r.get("total_items"), int),
              f"{r.get('total_items')} items")

        # --- error paths ------------------------------------------------
        print("\nerror handling (all should be clean errors, not crashes)")
        r = payload(await client.call_tool("list_debates", {
            "start_date": "last Tuesday", "end_date": WEEK_END}))
        check("rejects a non-ISO date", "_error" in r, r.get("_error", "")[:70])

        r = payload(await client.call_tool("list_debates", {
            "start_date": "2026-07-10", "end_date": "2026-07-06"}))
        check("rejects reversed dates", "_error" in r, r.get("_error", "")[:70])

        r = payload(await client.call_tool("list_debates", {
            "start_date": WEEK_START, "end_date": WEEK_END, "house": "Senate"}))
        check("rejects an invalid house", "_error" in r, r.get("_error", "")[:70])

        r = payload(await client.call_tool("find_member", {"name": "Zzzqqx"}))
        check("unknown member is a readable error", "_error" in r, r.get("_error", "")[:70])

        # --- recess is not an error -------------------------------------
        print("\nrecess behaviour")
        r = payload(await client.call_tool("list_debates", {
            "start_date": "2026-08-15", "end_date": "2026-08-16"}))
        check("a quiet range returns 0, not an error",
              "_error" not in r, f"total_count={r.get('total_count')}")

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
