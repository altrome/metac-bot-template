"""Latest official readings for indicator questions.

Numeric questions that resolve against an official data series (jobs
reports, CPI, GDP, interest rates, inventories) are exactly where Summer
2026 lost its points: news coverage of these series lags, rounds, or
misstates the base value. This module fetches the latest published
observations of the matching FRED series (fred.stlouisfed.org mirrors
most official US statistics and its CSV endpoint needs no API key) so
the forecaster prompt is anchored to ground truth.

Everything here fails soft: on any error the block is an empty string
and forecasting proceeds exactly as before.
"""

import datetime
import re

import requests

from llm_calls import call_openAI

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES_URL = "https://fred.stlouisfed.org/series/"
# Roughly one year back: ~13 monthly or ~52 weekly observations before the cap.
LOOKBACK_DAYS = 370
MAX_OBSERVATIONS = 13
HTTP_TIMEOUT_SECONDS = 15

_SERIES_FROM_URL_PATTERN = re.compile(
    r"fred\.stlouisfed\.org/(?:series/|graph/\?id=|graph/fredgraph\.csv\?id=)([A-Za-z0-9]+)"
)
_SERIES_LINE_PATTERN = re.compile(r"(?im)^\s*SERIES:\s*([A-Za-z0-9]+)\s*$")
_DESCR_LINE_PATTERN = re.compile(r"(?im)^\s*DESCR:\s*(.+)$")

IDENTIFIER_PROMPT = """
You pick the official data series that a forecasting question resolves against.

Task
Decide whether the question's outcome is determined by a specific numeric series of official data (US government statistics or similar: employment, inflation, GDP, interest rates, trade, housing, energy prices, population).
- If it is: name the FRED series (fred.stlouisfed.org, Federal Reserve Economic Data) that mirrors or matches that official series. Only name a series ID you are confident exists on FRED.
- If it is not (an election, an event, a company decision, a poll, a count of announced items, a person's action), or no matching FRED series exists: answer NONE.

Output format (strict, nothing else)
SERIES: <FRED series id>
DESCR: <what the series measures, with its unit>
or exactly:
NONE

Context
Title: {title}
Resolution criteria: {resolution_criteria}
Background: {background}
Fine print: {fine_print}
"""


def extract_fred_series_from_text(text: str) -> str | None:
    """Return the FRED series ID linked in the text (e.g.
    fred.stlouisfed.org/series/CPIAUCSL), or None when no FRED link exists."""
    match = _SERIES_FROM_URL_PATTERN.search(text or "")
    return match.group(1) if match else None


def parse_fred_csv(csv_text: str) -> list[tuple[str, float]]:
    """Parse a fredgraph.csv body into (date, value) observations.

    Bad or nonexistent series IDs return an HTML error page instead of a
    CSV, so anything without the expected header is treated as no data.
    Missing observations are encoded as "." and dropped.
    """
    lines = [line for line in (csv_text or "").split("\n") if line.strip()]
    if not lines or "observation_date" not in lines[0]:
        return []
    observations = []
    for line in lines[1:]:
        date, _, value = line.partition(",")
        try:
            observations.append((date, float(value)))
        except ValueError:
            continue
    return observations


def _format_value(value: float) -> str:
    if value.is_integer():
        return f"{int(value):,}"
    return str(value)


def _fetch_series_title(series_id: str) -> str | None:
    """Fetch the human-readable series title from the FRED series page."""
    try:
        response = requests.get(
            FRED_SERIES_URL + series_id, timeout=HTTP_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        match = re.search(r"<title>([^<]*)</title>", response.text)
        if not match:
            return None
        return match.group(1).replace(" | FRED | St. Louis Fed", "").strip() or None
    except requests.RequestException:
        return None


def _fetch_and_format_series(series_id: str, label: str | None) -> str:
    """Fetch the latest observations and format the prompt block.

    Returns "" when the series cannot be fetched or has no observations.
    """
    start = (
        datetime.datetime.now(datetime.UTC).date()
        - datetime.timedelta(days=LOOKBACK_DAYS)
    ).isoformat()
    response = requests.get(
        FRED_CSV_URL,
        params={"id": series_id, "cosd": start},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    observations = parse_fred_csv(response.text)
    if not observations:
        return ""
    title = _fetch_series_title(series_id) or label or series_id
    lines = [
        f"{date}: {_format_value(value)}"
        for date, value in observations[-MAX_OBSERVATIONS:]
    ]
    return (
        "Official data (latest published values of the series this question resolves against):\n"
        f"FRED series {series_id} - {title}\n"
        + "\n".join(lines)
        + f"\nSource: {FRED_SERIES_URL}{series_id}"
    )


async def _identify_series_with_llm(question_details: dict) -> tuple[str | None, str | None]:
    """Ask a cheap model to name the FRED series that resolves this question.

    Returns (series_id, label) or (None, None) when the question does not
    resolve against an official series (or the answer is unparseable).
    """
    prompt = IDENTIFIER_PROMPT.format(
        title=question_details.get("title") or "",
        resolution_criteria=question_details.get("resolution_criteria") or "",
        background=question_details.get("description") or "",
        fine_print=question_details.get("fine_print") or "",
    )
    response = await call_openAI(prompt)
    if re.search(r"(?im)^\s*NONE\s*$", response):
        return None, None
    series_match = _SERIES_LINE_PATTERN.search(response)
    if not series_match:
        return None, None
    label_match = _DESCR_LINE_PATTERN.search(response)
    return series_match.group(1), label_match.group(1).strip() if label_match else None


async def gather_indicator_data(question_details: dict) -> str:
    """Build the official-data block for a question's prompt ("" if none).

    A FRED link in the question text wins; otherwise a cheap LLM call maps
    the question to a FRED mirror series. Any failure returns "".
    """
    try:
        searchable = " ".join(
            str(question_details.get(key) or "")
            for key in ("title", "resolution_criteria", "description", "fine_print")
        )
        series_id = extract_fred_series_from_text(searchable)
        label = None
        if series_id is None:
            series_id, label = await _identify_series_with_llm(question_details)
        if not series_id:
            print("Indicator data: none identified for this question")
            return ""
        block = _fetch_and_format_series(series_id, label)
        if block:
            print(f"Indicator data: attached FRED series {series_id}")
        else:
            print(f"Indicator data: series {series_id} fetched no observations")
        return block
    except Exception as exc:  # noqa: BLE001 -- the contract is fail-soft
        print(f"Indicator data unavailable: {exc}")
        return ""