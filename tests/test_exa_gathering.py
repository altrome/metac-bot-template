"""Tests for the Claude+Exa evidence-gathering pipeline and its wiring.

All Anthropic and Exa calls are faked; the live path is covered by the
E2E script, never here.
"""

import asyncio
import json
from types import SimpleNamespace

import exa_gathering as eg
import main_with_no_framework as mnw


def _fake_llm(payload_text):
    """An async fake for call_openAI that also captures the prompt."""
    captured = []

    async def call(prompt, **kwargs):
        captured.append(prompt)
        return payload_text

    return call, captured


def _fake_exa(fail_queries=()):
    """A minimal Exa client; each query returns one result with a stable URL."""

    def search(query, **kwargs):
        if query in fail_queries:
            raise RuntimeError("boom")
        result = SimpleNamespace(
            title=f"About {query}",
            url=f"https://example.com/{query.replace(' ', '-')}",
            published_date="2026-09-01",
            highlights=["key fact"],
            summary="useful summary",
        )
        return SimpleNamespace(results=[result])

    return SimpleNamespace(search=search)


def _question_dict():
    return {
        "title": "Will X happen?",
        "description": "background",
        "resolution_criteria": "criteria",
        "fine_print": "fine",
        "scheduled_resolve_time": "2026-12-31T12:00:00Z",
        "type": "numeric",
    }


# --- exa_gathering module ---


def test_from_details_maps_api_question():
    q = eg.MetaculusQuestion.from_details(_question_dict())
    assert q.title == "Will X happen?"
    assert q.background == "background"
    assert q.resolution_criteria == "criteria"
    assert q.fine_print == "fine"
    assert q.resolution_date == "2026-12-31"
    assert q.question_type == "numeric"


def test_from_details_handles_missing_fields():
    q = eg.MetaculusQuestion.from_details({})
    assert q.title == ""
    assert q.resolution_date is None
    assert q.question_type == "binary"


def test_decompose_parses_plain_json(monkeypatch):
    payload = json.dumps(
        [
            {"query": "current status of X", "purpose": "recent_news", "recency": "recent"},
            {"query": "how often X happens", "purpose": "base_rate", "recency": "historical"},
        ]
    )
    fake, captured = _fake_llm(payload)
    monkeypatch.setattr(eg, "call_openAI", fake)
    sub_queries = asyncio.run(eg.decompose_question(eg.MetaculusQuestion(title="X?")))
    assert [(sq.query, sq.purpose, sq.recency) for sq in sub_queries] == [
        ("current status of X", "recent_news", "recent"),
        ("how often X happens", "base_rate", "historical"),
    ]
    assert "X?" in captured[0]


def test_decompose_tolerates_code_fences_and_clamps_values(monkeypatch):
    payload = "```json\n" + json.dumps(
        [{"query": "q1", "purpose": "nonsense", "recency": "whenever"}]
    ) + "\n```"
    fake, _captured = _fake_llm(payload)
    monkeypatch.setattr(eg, "call_openAI", fake)
    sub_queries = asyncio.run(eg.decompose_question(eg.MetaculusQuestion(title="X?")))
    assert len(sub_queries) == 1
    assert sub_queries[0].purpose == "expert_analysis", "unknown purpose is clamped"
    assert sub_queries[0].recency == "any", "unknown recency is clamped"


def test_search_all_survives_partial_failures():
    sub_queries = [
        eg.SubQuery("good query", "recent_news"),
        eg.SubQuery("bad query", "base_rate"),
        eg.SubQuery("another good", "indicators"),
    ]
    items = eg.search_all(_fake_exa(fail_queries=("bad query",)), sub_queries)
    assert len(items) == 2
    assert {item.sub_query for item in items} == {"good query", "another good"}


def test_to_prompt_renders_question_and_evidence():
    question = eg.MetaculusQuestion(title="Will X?")
    item = eg.EvidenceItem("q", "base_rate", "Reference class", "https://a", "2026-01-01", ["h"], "s")
    bundle = eg.EvidenceBundle(question=question, sub_queries=[eg.SubQuery("q", "base_rate")], evidence=[item])
    prompt = bundle.to_prompt()
    assert "# Question" in prompt and "Will X?" in prompt
    assert "# Search plan" in prompt and "base_rate" in prompt
    assert "# Evidence gathered" in prompt and "https://a" in prompt
    assert "- Highlights:" in prompt and "- Summary: s" in prompt


def test_to_prompt_truncates_chatty_evidence():
    long_summary = "x" * (eg.SUMMARY_MAX_CHARS + 500)
    highlights = [str(i) * (eg.HIGHLIGHT_MAX_CHARS + 100) for i in range(5)]
    item = eg.EvidenceItem("q", "recent_news", "Big page", "https://a", None, highlights, long_summary)
    bundle = eg.EvidenceBundle(
        question=eg.MetaculusQuestion(title="Will X?"),
        sub_queries=[eg.SubQuery("q", "recent_news")],
        evidence=[item],
    )
    prompt = bundle.to_prompt()
    assert f"{'0' * eg.HIGHLIGHT_MAX_CHARS}…" in prompt, "highlight truncated at the cap"
    assert f"{'x' * eg.SUMMARY_MAX_CHARS}…" in prompt, "summary truncated at the cap"
    assert "1" * (eg.HIGHLIGHT_MAX_CHARS + 100) not in prompt, "at most MAX_HIGHLIGHTS kept"


# --- wiring into run_research ---


def test_evidence_research_returns_prompt_and_urls(monkeypatch):
    seen = []

    def fake_gather_sync(question):
        seen.append(question)
        items = [
            eg.EvidenceItem("q", "recent_news", "One", "https://a", None, [], "s"),
            eg.EvidenceItem("q", "recent_news", "Two", "https://a", None, [], None),
            eg.EvidenceItem("q", "indicators", "Three", "https://b", None, [], None),
        ]
        return eg.EvidenceBundle(
            question=question, sub_queries=[eg.SubQuery("q", "recent_news")], evidence=items
        )

    async def fake_gather(question, **kwargs):
        return fake_gather_sync(question)

    monkeypatch.setattr(mnw, "gather_evidence", fake_gather)
    research, urls = asyncio.run(mnw.run_exa_evidence_research(_question_dict()))
    assert isinstance(seen[0], eg.MetaculusQuestion), "the API dict must be converted first"
    assert seen[0].title == "Will X happen?"
    assert "# Evidence gathered" in research
    assert urls == ["https://a", "https://b"], "URLs must be unique and in evidence order"


def test_evidence_research_falls_back_when_bundle_empty(monkeypatch):
    async def fake_gather(question, **kwargs):
        return eg.EvidenceBundle(question=question, sub_queries=[], evidence=[])

    async def fake_legacy(question):
        return "Research text with URL: https://legacy.example/page inside"

    monkeypatch.setattr(mnw, "gather_evidence", fake_gather)
    monkeypatch.setattr(mnw, "run_exa_research", fake_legacy)
    research, urls = asyncio.run(mnw.run_exa_evidence_research(_question_dict()))
    assert urls == ["https://legacy.example/page"]
    assert "legacy" in research


def test_run_research_uses_evidence_path_with_exa_key(monkeypatch):
    monkeypatch.setattr(mnw, "ASKNEWS_CLIENT_ID", None)
    monkeypatch.setattr(mnw, "ASKNEWS_SECRET", None)
    monkeypatch.setattr(mnw, "EXA_API_KEY", "x")

    async def fake_evidence(question):
        return "EVIDENCE RESEARCH", ["https://a"]

    monkeypatch.setattr(mnw, "run_exa_evidence_research", fake_evidence)
    research, urls = asyncio.run(mnw.run_research(_question_dict()))
    assert research == "EVIDENCE RESEARCH"
    assert urls == ["https://a"]


def test_run_research_without_keys_returns_nothing(monkeypatch):
    monkeypatch.setattr(mnw, "ASKNEWS_CLIENT_ID", None)
    monkeypatch.setattr(mnw, "ASKNEWS_SECRET", None)
    monkeypatch.setattr(mnw, "EXA_API_KEY", None)
    monkeypatch.setattr(mnw, "PERPLEXITY_API_KEY", None)
    research, urls = asyncio.run(mnw.run_research(_question_dict()))
    assert research == "No research done"
    assert urls == []