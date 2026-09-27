"""Tests for the official indicator-data fetch (indicator_data.py).

All HTTP and LLM calls are faked: the FRED endpoints are probed live
elsewhere (the E2E script), never here.
"""

import asyncio

import indicator_data as ind

PAYEMS_CSV = (
    "observation_date,PAYEMS\n"
    "2026-06-01,162100\n"
    "2026-07-01,.\n"
    "2026-08-01,162455\n"
)
PAYEMS_PAGE = (
    "<html><head><title>All Employees, Total Nonfarm (PAYEMS) | FRED | "
    "St. Louis Fed</title></head></html>"
)


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _patch_fred(monkeypatch, csv_text, page_text=PAYEMS_PAGE, error=None):
    def fake_get(url, **kwargs):
        if error is not None:
            raise error
        if "fredgraph.csv" in url:
            return _FakeResponse(csv_text)
        return _FakeResponse(page_text)

    monkeypatch.setattr(ind.requests, "get", fake_get)


def test_extract_series_from_series_page_url():
    text = "Resolves per https://fred.stlouisfed.org/series/CPIAUCSL monthly."
    assert ind.extract_fred_series_from_text(text) == "CPIAUCSL"


def test_extract_series_from_graph_url():
    text = "see https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    assert ind.extract_fred_series_from_text(text) == "CPIAUCSL"


def test_extract_series_absent_returns_none():
    assert ind.extract_fred_series_from_text("No FRED link here") is None


def test_parse_fred_csv_drops_missing_values():
    observations = ind.parse_fred_csv(PAYEMS_CSV)
    assert [date for date, _ in observations] == ["2026-06-01", "2026-08-01"]
    assert observations[1] == ("2026-08-01", 162455.0)


def test_parse_fred_csv_rejects_html_error_page():
    # Bad series IDs return an HTML error page, not a CSV.
    assert ind.parse_fred_csv("<!DOCTYPE html><html></html>") == []


def test_fetch_formats_block_with_commas_and_title(monkeypatch):
    _patch_fred(monkeypatch, PAYEMS_CSV)
    block = ind._fetch_and_format_series("PAYEMS", None)
    assert "FRED series PAYEMS" in block
    assert "All Employees, Total Nonfarm" in block
    assert "2026-06-01: 162,100" in block
    assert "2026-08-01: 162,455" in block
    assert "2026-07-01" not in block, "missing observations must be dropped"
    assert "https://fred.stlouisfed.org/series/PAYEMS" in block


def test_fetch_bad_series_returns_empty(monkeypatch):
    _patch_fred(monkeypatch, "<!DOCTYPE html><html>Series not found</html>")
    assert ind._fetch_and_format_series("NOTASERIES", None) == ""


def test_fetch_title_failure_falls_back_to_label(monkeypatch):
    _patch_fred(monkeypatch, PAYEMS_CSV, page_text="<html></html>")
    block = ind._fetch_and_format_series("PAYEMS", "Nonfarm payrolls, thousands")
    assert "Nonfarm payrolls, thousands" in block
    assert "162,455" in block


def test_gather_prefers_fred_url_over_llm(monkeypatch):
    called = []

    async def fail_llm(prompt):
        called.append(prompt)
        return "SERIES: SHOULDNOTBEUSED\nDESCR: wrong"

    monkeypatch.setattr(ind, "call_openAI", fail_llm)
    _patch_fred(monkeypatch, PAYEMS_CSV)
    question = {
        "title": "Jobs in August",
        "resolution_criteria": "per https://fred.stlouisfed.org/series/PAYEMS",
        "description": "",
        "fine_print": "",
    }
    block = asyncio.run(ind.gather_indicator_data(question))
    assert "FRED series PAYEMS" in block
    assert called == [], "a FRED link in the question must skip the LLM call"


def test_gather_uses_llm_identification(monkeypatch):
    prompts = []

    async def fake_llm(prompt):
        prompts.append(prompt)
        return "SERIES: PAYEMS\nDESCR: All Employees, Total Nonfarm, thousands"

    monkeypatch.setattr(ind, "call_openAI", fake_llm)
    _patch_fred(monkeypatch, PAYEMS_CSV)
    question = {
        "title": "How many jobs will the US add?",
        "resolution_criteria": "Change in nonfarm payrolls per BLS.",
        "description": "",
        "fine_print": "",
    }
    block = asyncio.run(ind.gather_indicator_data(question))
    assert "FRED series PAYEMS" in block
    assert "How many jobs will the US add?" in prompts[0]


def test_gather_llm_none_returns_empty(monkeypatch):
    async def fake_llm(prompt):
        return "NONE"

    monkeypatch.setattr(ind, "call_openAI", fake_llm)
    _patch_fred(monkeypatch, PAYEMS_CSV)
    question = {"title": "Will the president be re-elected?"}
    assert asyncio.run(ind.gather_indicator_data(question)) == ""


def test_gather_unparseable_llm_returns_empty(monkeypatch):
    async def fake_llm(prompt):
        return "I suppose PAYEMS might work."

    monkeypatch.setattr(ind, "call_openAI", fake_llm)
    _patch_fred(monkeypatch, PAYEMS_CSV)
    question = {"title": "How many jobs will the US add?"}
    assert asyncio.run(ind.gather_indicator_data(question)) == ""


def test_gather_fail_soft_on_http_error(monkeypatch):
    async def fake_llm(prompt):
        return "SERIES: PAYEMS\nDESCR: payrolls"

    monkeypatch.setattr(ind, "call_openAI", fake_llm)
    _patch_fred(monkeypatch, None, error=ConnectionError("network down"))
    question = {"title": "How many jobs will the US add?"}
    assert asyncio.run(ind.gather_indicator_data(question)) == ""


def test_gather_fail_soft_on_llm_error(monkeypatch):
    async def fake_llm(prompt):
        raise RuntimeError("api down")

    monkeypatch.setattr(ind, "call_openAI", fake_llm)
    _patch_fred(monkeypatch, PAYEMS_CSV)
    question = {"title": "How many jobs will the US add?"}
    assert asyncio.run(ind.gather_indicator_data(question)) == ""