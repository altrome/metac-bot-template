"""
Metaculus forecasting bot — evidence gathering module.

Pipeline:
    1. A cheap LLM call (gpt-5.4-mini via llm_calls.call_openAI) decomposes the
       Metaculus question into 6-8 angled sub-queries (recent news, base rates,
       evidence-for, evidence-against, etc.).
    2. Each sub-query is run through Exa with type="deep" for max retrieval
       quality (leaderboard-oriented, not latency-sensitive).
    3. Results are bundled into an EvidenceBundle whose .to_prompt() method
       produces a ready-to-reason context block for the final LLM forecast.

The reasoning / probability call is NOT done here — this module only gathers
evidence. Feed `bundle.to_prompt()` into your reasoner of choice.

Env vars required:
    EXA_API_KEY (the LLM call reuses the OpenAI configuration)
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from exa_py import Exa

from config import EXA_API_KEY
from llm_calls import call_openAI

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class MetaculusQuestion:
    """A Metaculus question, in the shape the API/UI exposes."""
    title: str
    background: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""
    resolution_date: str | None = None     # ISO "YYYY-MM-DD"
    question_type: str = "binary"             # binary | numeric | multiple_choice

    def as_context(self) -> str:
        parts = [f"QUESTION: {self.title}"]
        if self.background:
            parts.append(f"BACKGROUND:\n{self.background}")
        if self.resolution_criteria:
            parts.append(f"RESOLUTION CRITERIA:\n{self.resolution_criteria}")
        if self.fine_print:
            parts.append(f"FINE PRINT:\n{self.fine_print}")
        if self.resolution_date:
            parts.append(f"RESOLUTION DATE: {self.resolution_date}")
        parts.append(f"QUESTION TYPE: {self.question_type}")
        return "\n\n".join(parts)

    @classmethod
    def from_details(cls, question_details: dict) -> MetaculusQuestion:
        """Build from a Metaculus API question dict (post_details["question"])."""
        resolve_time = str(question_details.get("scheduled_resolve_time") or "")
        return cls(
            title=str(question_details.get("title") or ""),
            background=str(question_details.get("description") or ""),
            resolution_criteria=str(question_details.get("resolution_criteria") or ""),
            fine_print=str(question_details.get("fine_print") or ""),
            resolution_date=resolve_time[:10] or None,
            question_type=str(question_details.get("type") or "binary"),
        )


@dataclass
class SubQuery:
    query: str
    purpose: str           # see PURPOSES below
    recency: str = "any"   # "recent" | "any" | "historical"


@dataclass
class EvidenceItem:
    sub_query: str
    purpose: str
    title: str
    url: str
    published_date: str | None
    highlights: list[str] = field(default_factory=list)
    summary: str | None = None


# The rendered bundle goes verbatim into every forecast prompt (NUM_RUNS_PER_QUESTION
# runs per question), so one chatty page must not bloat the whole context.
MAX_HIGHLIGHTS = 3
SUMMARY_MAX_CHARS = 800
HIGHLIGHT_MAX_CHARS = 300


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


@dataclass
class EvidenceBundle:
    question: MetaculusQuestion
    sub_queries: list[SubQuery]
    evidence: list[EvidenceItem]

    def to_prompt(self) -> str:
        """Render the bundle as a context block ready for a reasoner prompt."""
        lines: list[str] = ["# Question", "", self.question.as_context(), "", "# Search plan", ""]
        for i, sq in enumerate(self.sub_queries, 1):
            lines.append(f"{i}. [{sq.purpose} | {sq.recency}] {sq.query}")
        lines += ["", "# Evidence gathered", ""]

        by_purpose: dict[str, list[EvidenceItem]] = {}
        for item in self.evidence:
            by_purpose.setdefault(item.purpose, []).append(item)

        for purpose, items in by_purpose.items():
            lines.append(f"## {purpose}")
            lines.append("")
            for i, e in enumerate(items, 1):
                lines.append(f"### [{purpose} #{i}] {e.title or '(untitled)'}")
                lines.append(f"- URL: {e.url}")
                if e.published_date:
                    lines.append(f"- Published: {e.published_date}")
                lines.append(f"- Surfaced by: \"{e.sub_query}\"")
                if e.summary:
                    lines.append(f"- Summary: {_truncate(e.summary, SUMMARY_MAX_CHARS)}")
                if e.highlights:
                    lines.append("- Highlights:")
                    for h in e.highlights[:MAX_HIGHLIGHTS]:
                        lines.append(f"    - {_truncate(h, HIGHLIGHT_MAX_CHARS)}")
                lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decomposition (Claude)
# ---------------------------------------------------------------------------

PURPOSES = [
    "recent_news",        # what's happening right now around the key entities
    "base_rate",          # historical reference class / frequency
    "expert_analysis",    # domain experts, analyst reports, op-eds
    "indicators",         # quantitative data, official stats, market signals
    "evidence_for",       # strongest case for YES resolution
    "evidence_against",   # strongest case for NO resolution (steelman)
    "resolution_source",  # how the question would actually be measured/sourced
]

DECOMPOSITION_PROMPT = """You are preparing search queries for a Metaculus forecasting question.
Decompose the question into 6-8 sub-queries that, together, surface the evidence a
careful superforecaster would want before estimating a probability.

Cover these angles where applicable (skip any that genuinely don't apply):
- recent_news: status of the key entities right now
- base_rate: historical reference class / how often things like this happen
- expert_analysis: domain experts, analyst reports
- indicators: quantitative data, official statistics, market signals
- evidence_for: strongest case the question resolves YES
- evidence_against: strongest case it resolves NO (steelman the other side)
- resolution_source: how the question would actually be measured/sourced

Return ONLY a JSON array. No prose, no code fences. Each entry must have:
  "query":    short search query, 3-10 words, no quotation marks
  "purpose":  one of [recent_news, base_rate, expert_analysis, indicators,
                       evidence_for, evidence_against, resolution_source]
  "recency":  one of [recent, any, historical]
              - "recent"     → news/events from the last week or so
              - "any"        → no recency constraint
              - "historical" → prefer older/established sources

QUESTION CONTEXT:
{context}
"""


async def decompose_question(question: MetaculusQuestion) -> list[SubQuery]:
    """Use a cheap LLM call to decompose a Metaculus question into angled sub-queries."""
    text = (
        await call_openAI(
            DECOMPOSITION_PROMPT.format(context=question.as_context()), temperature=0.2
        )
    ).strip()

    # Tolerate accidental code fences.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().startswith("json"):
            text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    data = json.loads(text.strip())
    sub_queries: list[SubQuery] = []
    for d in data:
        # Defensive: clamp purpose/recency to known values.
        purpose = d.get("purpose", "expert_analysis")
        if purpose not in PURPOSES:
            purpose = "expert_analysis"
        recency = d.get("recency", "any")
        if recency not in ("recent", "any", "historical"):
            recency = "any"
        sub_queries.append(SubQuery(query=d["query"], purpose=purpose, recency=recency))
    return sub_queries


# ---------------------------------------------------------------------------
# Search (Exa)
# ---------------------------------------------------------------------------

def _max_age_hours_for(recency: str) -> int | None:
    """Map our recency label to Exa's max_age_hours knob."""
    if recency == "recent":
        return 24 * 7      # livecrawl anything older than ~1 week
    if recency == "historical":
        return -1          # cache only, no livecrawl
    return None            # default behavior


def search_one(exa: Exa, sq: SubQuery, num_results: int = 5) -> list[EvidenceItem]:
    """Run a single sub-query through Exa deep search and map to EvidenceItems."""
    contents: dict = {
        "highlights": True,
        "summary": {"query": sq.query},
    }
    max_age = _max_age_hours_for(sq.recency)
    if max_age is not None:
        contents["max_age_hours"] = max_age   # Python SDK uses snake_case inside dicts

    res = exa.search(
        sq.query,
        type="deep",           # highest-quality retrieval; leaderboard > latency
        num_results=num_results,
        contents=contents,
    )

    items: list[EvidenceItem] = []
    for r in res.results:
        items.append(EvidenceItem(
            sub_query=sq.query,
            purpose=sq.purpose,
            title=getattr(r, "title", "") or "",
            url=r.url,
            published_date=getattr(r, "published_date", None),
            highlights=list(getattr(r, "highlights", []) or []),
            summary=getattr(r, "summary", None),
        ))
    return items


def search_all(
    exa: Exa,
    sub_queries: list[SubQuery],
    num_results: int = 5,
    max_workers: int = 4,
) -> list[EvidenceItem]:
    """Run all sub-queries in parallel. Failures on one query don't sink the rest."""
    all_items: list[EvidenceItem] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(search_one, exa, sq, num_results): sq for sq in sub_queries}
        for fut in as_completed(futures):
            sq = futures[fut]
            try:
                all_items.extend(fut.result())
            except Exception as e:  # noqa: BLE001 -- partial evidence beats none
                # Log and continue — partial evidence is better than no evidence.
                print(f"[warn] search failed for {sq.query!r} ({sq.purpose}): {e}")
    return all_items


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

async def gather_evidence(
    question: MetaculusQuestion,
    num_results_per_query: int = 5,
    exa: Exa | None = None,
) -> EvidenceBundle:
    """Decompose → search in parallel → return an EvidenceBundle.

    Pass the result's `.to_prompt()` to your reasoner for the final
    probability estimate.
    """
    exa = exa or Exa(api_key=EXA_API_KEY)

    sub_queries = await decompose_question(question)
    evidence = await asyncio.to_thread(search_all, exa, sub_queries, num_results_per_query)
    return EvidenceBundle(question=question, sub_queries=sub_queries, evidence=evidence)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    q = MetaculusQuestion(
        title="Will SpaceX successfully complete a Starship orbital flight before January 1, 2027?",
        background=(
            "Starship is SpaceX's fully-reusable super-heavy launch system. "
            "As of mid-2026, SpaceX has performed multiple suborbital test flights "
            "but no confirmed full-orbit mission."
        ),
        resolution_criteria=(
            "Resolves YES if SpaceX completes at least one Starship mission that "
            "achieves a full Earth orbit (defined as crossing the launch meridian "
            "at orbital altitude after launch) before 2027-01-01 UTC, as reported "
            "by SpaceX and at least one independent space-tracking source."
        ),
        fine_print=(
            "Partial or failed test flights that do not achieve orbit do not count. "
            "Planned missions that are delayed past the resolution date do not count."
        ),
        resolution_date="2027-01-01",
        question_type="binary",
    )

    bundle = asyncio.run(gather_evidence(q, num_results_per_query=5))

    print(bundle.to_prompt())
    print(f"\n--- Gathered {len(bundle.evidence)} evidence items "
          f"across {len(bundle.sub_queries)} sub-queries ---")
