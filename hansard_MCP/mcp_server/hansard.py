"""
Hansard MCP server - tools for querying the UK Parliament APIs.

Answers three questions:
  1. What debates happened last week?      -> list_debates
  2. Who spoke last week?                  -> who_spoke
  3. What did person X say last week?       -> find_member + list_contributions

Two upstream hosts, both unauthenticated (Open Parliament Licence):
  - hansard-api.parliament.uk   debates and spoken contributions
  - members-api.parliament.uk   MPs and Lords, for name -> id lookup

THE GOVERNING CONSTRAINT: a single sitting day carries ~942 spoken contributions
(756 Commons + 186 Lords on 2026-07-07), so a week is roughly 5,000 - about a
novel's worth of text. Nothing here may return raw contributions in bulk. List
tools return short snippets, who_spoke aggregates server-side and returns no
speech text at all, and every list result states the true total so the model
knows when it is looking at a subset.

Runs over stdio: stdout is the JSON-RPC channel, so logging goes to stderr.
"""

from __future__ import annotations

import re
import sys
from typing import Annotated, Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HANSARD = "https://hansard-api.parliament.uk"
MEMBERS = "https://members-api.parliament.uk/api"

TIMEOUT = 20.0
SNIPPET_CHARS = 220        # per-contribution preview length
PAGE_SIZE = 100            # upstream max per request
WHO_SPOKE_MAX_PAGES = 10   # hard ceiling => at most 1,000 contributions scanned

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Identify ourselves to Parliament's servers rather than sending a bare default.
USER_AGENT = "hansard-mcp/0.1 (learning project; +https://github.com/Alex-Drysdale/AI-Learning)"

server = MCPServer(
    name="hansard",
    version="0.1.0",
    instructions=(
        "Query UK Parliament's Hansard record. All dates are ISO YYYY-MM-DD.\n"
        "To answer questions about a named person, ALWAYS call find_member first to get their "
        "member_id - the contribution tools take an id, not a name.\n"
        "To answer 'who spoke', use who_spoke, which counts server-side. Do not try to list "
        "thousands of contributions and count them yourself.\n"
        "Parliament does not sit every day; an empty result for a date range usually means "
        "recess, not an error."
    ),
)


def _log(msg: str) -> None:
    """stdout belongs to the protocol."""
    print(f"[hansard] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Shared HTTP helper
# --------------------------------------------------------------------------

def _get(base: str, path: str, params: dict[str, Any]) -> dict:
    """GET JSON from a Parliament API, with one retry on transient failure.

    Every tool goes through here so timeouts, the User-Agent and retry policy
    are defined once. Params with a value of None are dropped rather than sent
    as the string "None".
    """
    clean = {k: v for k, v in params.items() if v is not None}
    url = f"{base}{path}"

    last_error = ""
    for attempt in (1, 2):
        try:
            with httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
                response = client.get(url, params=clean)
            if response.status_code == 200:
                return response.json()
            # 429 and 5xx are worth one retry; 4xx will not improve.
            if response.status_code < 500 and response.status_code != 429:
                raise ToolError(
                    f"Parliament API returned {response.status_code} for {path}. "
                    "Check the parameters - a malformed date or an unknown id is the usual cause."
                )
            last_error = f"HTTP {response.status_code}"
        except httpx.RequestError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt == 1:
            _log(f"retrying {path} after {last_error}")

    raise ToolError(f"Could not reach the Parliament API ({last_error}). It may be down; try again.")


def _check_dates(start_date: str, end_date: str) -> None:
    """Reject bad dates loudly.

    Without this the upstream API quietly returns zero results for a malformed
    date, and the model reports 'nothing happened that week' - a wrong answer
    that looks like a right one.
    """
    for label, value in (("start_date", start_date), ("end_date", end_date)):
        if not DATE_RE.match(value):
            raise ToolError(f"{label} must be in YYYY-MM-DD format, got {value!r}")
    if end_date < start_date:
        raise ToolError(f"end_date ({end_date}) is before start_date ({start_date})")


def _check_house(house: str | None) -> str | None:
    if house is None:
        return None
    normalised = house.strip().capitalize()
    if normalised not in ("Commons", "Lords"):
        raise ToolError(f"house must be 'Commons' or 'Lords' (or omitted for both), got {house!r}")
    return normalised


def _snippet(text: str | None, limit: int = SNIPPET_CHARS) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "..."


def _search_params(start_date, end_date, house, take, skip=0, **extra) -> dict:
    """Hansard uses a distinctive `queryParameters.*` query-string convention."""
    params = {
        "queryParameters.startDate": start_date,
        "queryParameters.endDate": end_date,
        "queryParameters.house": house,
        "queryParameters.take": take,
        "queryParameters.skip": skip,
    }
    params.update({f"queryParameters.{k}": v for k, v in extra.items()})
    return params


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

@server.tool(annotations=ToolAnnotations(title="Find an MP or Lord", readOnlyHint=True))
def find_member(
    name: Annotated[str, Field(description="Full or partial name, e.g. 'Ed Miliband' or 'Miliband'")],
    limit: Annotated[int, Field(description="Maximum matches to return", ge=1, le=20)] = 5,
) -> dict:
    """Look up an MP or Lord by name to get their member_id.

    Call this FIRST whenever a question names a person: the contribution tools
    take a numeric member_id, not a name. Returns several matches where the name
    is ambiguous so you can pick the right one rather than guessing.
    """
    if not name.strip():
        raise ToolError("name must not be empty")

    data = _get(MEMBERS, "/Members/Search", {
        "Name": name.strip(),
        "take": limit,
        "IsCurrentMember": "true",
    })

    results = []
    for item in data.get("items") or []:
        value = item.get("value") or {}
        membership = value.get("latestHouseMembership") or {}
        results.append({
            "member_id": value.get("id"),
            "name": value.get("nameDisplayAs"),
            "party": (value.get("latestParty") or {}).get("name"),
            # house is 1 = Commons, 2 = Lords in the Members API
            "house": {1: "Commons", 2: "Lords"}.get(membership.get("house")),
            "constituency": membership.get("membershipFrom"),
        })

    if not results:
        raise ToolError(
            f"No current MP or Lord found matching {name!r}. Try a surname alone, or check the "
            "spelling - former members are not searched."
        )
    return {"total_count": data.get("totalResults", len(results)), "results": results}


@server.tool(annotations=ToolAnnotations(title="List debates in a date range", readOnlyHint=True))
def list_debates(
    start_date: Annotated[str, Field(description="First day to include, YYYY-MM-DD")],
    end_date: Annotated[str, Field(description="Last day to include, YYYY-MM-DD")],
    house: Annotated[str | None, Field(description="'Commons' or 'Lords'; omit for both")] = None,
    limit: Annotated[int, Field(description="Maximum debates to return", ge=1, le=100)] = 50,
) -> dict:
    """List the debates that took place between two dates.

    Answers 'what was debated last week'. Returns titles with their date, house
    and a debate_id you can pass to get_debate for the full transcript.
    """
    _check_dates(start_date, end_date)
    house = _check_house(house)

    data = _get(HANSARD, "/search/debates.json",
                _search_params(start_date, end_date, house, limit))

    results = [{
        "date": (row.get("SittingDate") or "")[:10],
        "house": row.get("House"),
        "title": row.get("Title"),
        "section": row.get("DebateSection"),
        "debate_id": row.get("DebateSectionExtId"),
    } for row in (data.get("Results") or [])]

    total = data.get("TotalResultCount", len(results))
    return {
        "total_count": total,
        "returned": len(results),
        "truncated": total > len(results),
        "results": results,
    }


@server.tool(annotations=ToolAnnotations(title="List spoken contributions", readOnlyHint=True))
def list_contributions(
    start_date: Annotated[str, Field(description="First day to include, YYYY-MM-DD")],
    end_date: Annotated[str, Field(description="Last day to include, YYYY-MM-DD")],
    member_id: Annotated[int | None, Field(description="Restrict to one member (from find_member)")] = None,
    search_term: Annotated[str | None, Field(description="Only contributions containing this text")] = None,
    house: Annotated[str | None, Field(description="'Commons' or 'Lords'; omit for both")] = None,
    full_text: Annotated[bool, Field(description="Return whole speeches instead of short snippets. Expensive - use only when quoting someone, and with a small limit")] = False,
    limit: Annotated[int, Field(description="Maximum contributions to return", ge=1, le=100)] = 50,
) -> dict:
    """Find what was actually said in Parliament, optionally by one person.

    With member_id, this answers 'what did X say'. With search_term, it finds
    who discussed a topic. Snippets are returned by default because a single
    sitting day contains around 942 contributions; set full_text only when you
    need to quote someone precisely.
    """
    _check_dates(start_date, end_date)
    house = _check_house(house)

    extra: dict[str, Any] = {}
    if member_id is not None:
        extra["memberId"] = member_id
    if search_term:
        extra["searchTerm"] = search_term

    data = _get(HANSARD, "/search/contributions/Spoken.json",
                _search_params(start_date, end_date, house, limit, **extra))

    results = []
    for row in data.get("Results") or []:
        text = row.get("ContributionTextFull") or row.get("ContributionText") or ""
        results.append({
            "date": (row.get("SittingDate") or "")[:10],
            "member": row.get("MemberName"),
            "member_id": row.get("MemberId"),
            "house": row.get("House"),
            "debate": row.get("DebateSection"),
            "debate_id": row.get("DebateSectionExtId"),
            "text": " ".join(text.split()) if full_text else _snippet(text),
        })

    total = data.get("TotalResultCount", len(results))
    return {
        "total_count": total,
        "returned": len(results),
        "truncated": total > len(results),
        "results": results,
    }


@server.tool(annotations=ToolAnnotations(title="Rank who spoke most", readOnlyHint=True))
def who_spoke(
    start_date: Annotated[str, Field(description="First day to include, YYYY-MM-DD")],
    end_date: Annotated[str, Field(description="Last day to include, YYYY-MM-DD")],
    house: Annotated[str | None, Field(description="'Commons' or 'Lords'; omit for both")] = None,
    top: Annotated[int, Field(description="How many speakers to return", ge=1, le=100)] = 25,
) -> dict:
    """Rank the people who spoke in a date range by number of contributions.

    Answers 'who spoke last week'. The counting happens here, on the server,
    because a week holds thousands of contributions - far too many to list and
    count in a conversation. Returns names and counts only, no speech text.
    """
    _check_dates(start_date, end_date)
    house = _check_house(house)

    counts: dict[int | str, dict] = {}
    scanned = 0
    total = 0

    for page in range(WHO_SPOKE_MAX_PAGES):
        data = _get(HANSARD, "/search/contributions/Spoken.json",
                    _search_params(start_date, end_date, house, PAGE_SIZE, skip=page * PAGE_SIZE))
        rows = data.get("Results") or []
        total = data.get("TotalResultCount", 0)

        for row in rows:
            # Fall back to the attributed name for anyone without a member id
            # (tellers, officials) rather than silently dropping them.
            key = row.get("MemberId") or row.get("MemberName") or "Unknown"
            entry = counts.setdefault(key, {
                "member_id": row.get("MemberId"),
                "name": row.get("MemberName") or row.get("AttributedTo") or "Unknown",
                "house": row.get("House"),
                "count": 0,
            })
            entry["count"] += 1

        scanned += len(rows)
        if len(rows) < PAGE_SIZE or scanned >= total:
            break

    ranked = sorted(counts.values(), key=lambda e: e["count"], reverse=True)
    truncated = scanned < total

    return {
        "total_contributions": total,
        "contributions_scanned": scanned,
        # Say so plainly when the ranking is based on a sample, so the answer
        # is not presented as complete when it is not.
        "truncated": truncated,
        "note": (
            f"Ranking is based on the first {scanned} of {total} contributions "
            f"(scan capped at {WHO_SPOKE_MAX_PAGES * PAGE_SIZE}). Narrow the date range for a "
            "complete count."
        ) if truncated else "Complete - every contribution in this range was counted.",
        "distinct_speakers": len(ranked),
        "results": ranked[:top],
    }


@server.tool(annotations=ToolAnnotations(title="Read one debate", readOnlyHint=True))
def get_debate(
    debate_id: Annotated[str, Field(description="DebateSectionExtId from list_debates or list_contributions")],
    start_index: Annotated[int, Field(description="Item to start from, for paging through long debates", ge=0)] = 0,
    max_items: Annotated[int, Field(description="How many items to return", ge=1, le=200)] = 60,
) -> dict:
    """Read the transcript of a single debate.

    Use after list_debates to drill into one debate. Debates can run to
    thousands of items, so this pages - check `truncated` and call again with a
    higher start_index if you need more.
    """
    if not debate_id.strip():
        raise ToolError("debate_id must not be empty")

    data = _get(HANSARD, f"/debates/debate/{debate_id.strip()}.json", {})
    overview = data.get("Overview") or {}
    items = data.get("Items") or []

    window = items[start_index:start_index + max_items]
    contributions = [{
        "member": item.get("AttributedTo"),
        "member_id": item.get("MemberId"),
        "party": item.get("MemberParty"),
        "text": _snippet(item.get("Value"), 400),
    } for item in window if item.get("ItemType") == "Contribution"]

    return {
        "title": overview.get("Title"),
        "date": (overview.get("Date") or "")[:10],
        "house": overview.get("House"),
        "total_items": len(items),
        "showing": f"{start_index}-{start_index + len(window)}",
        "truncated": start_index + len(window) < len(items),
        "contributions": contributions,
    }


# --------------------------------------------------------------------------

if __name__ == "__main__":
    _log("serving over stdio")
    server.run(transport="stdio")
