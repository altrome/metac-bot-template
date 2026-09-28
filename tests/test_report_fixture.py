"""Tests for analysis/report.py — the postmortem report.

The fixture is a miniature Summer 2026: the polarity-miss binary (post
44654's story — 2% YES resolving YES for −297.69), an SPR-style numeric
that resolved below its range, one healthy binary, one multiple choice,
and a question that closed without ever being forecast.
"""

import csv

import pytest

from analysis import report


def _numeric_cdf(below=0.001, above=0.001):
    """201-point CDF with the given outside-range masses."""
    span = 1 - below - above
    return [below] + [below + span * (i / 199) for i in range(1, 200)] + [1 - above]


def _scored_post(post_id, title, question, spot_score, baseline=0.0):
    first = question.pop("_first_forecast")
    values = question.pop("_forecast_values")
    question["my_forecasts"] = {
        "history": [{"start_time": first}],
        "latest": {"start_time": first, "forecast_values": values},
        "score_data": {"spot_peer_score": spot_score, "baseline_score": baseline},
    }
    return {"id": post_id, "title": title, "question": question}


def snapshot():
    return {
        "tournament_id": 33022,
        "fetched_at": "2026-09-28T10:00:00Z",
        "post_count": 5,
        "posts": [
            _scored_post(
                44654,
                "Polymarket tracker below 7%?",
                {
                    "type": "binary",
                    "title": "Will the tracker be below 7%?",
                    "status": "resolved",
                    "resolution": "yes",
                    "open_time": "2026-08-01T00:00:00Z",
                    "actual_close_time": "2026-08-31T00:00:00Z",
                    "_first_forecast": "2026-08-02T12:00:00Z",
                    "_forecast_values": [0.98, 0.02],
                },
                -297.69,
                baseline=-464.39,
            ),
            _scored_post(
                52250,
                "SPR index",
                {
                    "type": "numeric",
                    "title": "What will the SPR index be?",
                    "status": "resolved",
                    # The real API sends in-range numeric resolutions as strings.
                    "resolution": "293426.0",
                    "open_time": "2026-08-01T00:00:00Z",
                    "actual_close_time": "2026-08-31T00:00:00Z",
                    "scaling": {"range_min": 300000, "range_max": 450000},
                    "_first_forecast": "2026-08-03T00:00:00Z",
                    "_forecast_values": _numeric_cdf(),
                },
                -230.11,
            ),
            _scored_post(
                40001,
                "A healthy binary",
                {
                    "type": "binary",
                    "title": "Will the obvious thing happen?",
                    "status": "resolved",
                    "resolution": "yes",
                    "open_time": "2026-08-01T00:00:00Z",
                    # Forecast timestamps arrive as Unix epochs, not ISO strings.
                    "_first_forecast": 1785553200.0,
                    "_forecast_values": [0.35, 0.65],
                },
                12.5,
            ),
            _scored_post(
                40002,
                "A multiple choice",
                {
                    "type": "multiple_choice",
                    "title": "Which outcome?",
                    "status": "resolved",
                    "resolution": 1,
                    "options": ["A", "B", "C"],
                    "open_time": "2026-08-01T00:00:00Z",
                    "_first_forecast": "2026-08-04T00:00:00Z",
                    "_forecast_values": [0.2, 0.7, 0.1],
                },
                3.3,
            ),
            # Closed without ever being forecast: coverage lost to cron misses.
            {
                "id": 40003,
                "title": "Missed question",
                "question": {
                    "type": "binary",
                    "title": "A question the cron never delivered",
                    "status": "closed",
                    "open_time": "2026-08-01T00:00:00Z",
                    "my_forecasts": {},
                },
            },
        ],
    }


def test_worst_first_order_and_total():
    scored = report.scored_rows(report.all_rows(snapshot()))
    assert [r["post_id"] for r in scored] == [44654, 52250, 40002, 40001]
    assert sum(r["spot_peer_score"] for r in scored) == pytest.approx(-512.0)


def test_polarity_row_tagged_and_summarized():
    row = next(r for r in report.all_rows(snapshot()) if r["post_id"] == 44654)
    assert row["miss_tag"] == "EXTREME-NO-MISS"
    assert row["forecast"] == "2% YES"
    assert row["resolution"] == "yes"


def test_numeric_outside_range_tagged():
    row = next(r for r in report.all_rows(snapshot()) if r["post_id"] == 52250)
    assert row["miss_tag"] == "RESOLVED-OUTSIDE-RANGE"
    assert row["forecast"] == "0.1% below, 0.1% above range [300,000, 450,000]"
    assert row["resolution"] == "293426.0"


def test_below_lower_bound_string_resolution_tagged():
    # Metaculus encodes out-of-range numeric resolutions as these strings.
    post = _scored_post(
        52251,
        "Resolved below the range",
        {
            "type": "numeric",
            "title": "A question that resolved below_lower_bound",
            "status": "resolved",
            "resolution": "below_lower_bound",
            "open_time": "2026-08-01T00:00:00Z",
            "scaling": {"range_min": 300, "range_max": 450},
            "_first_forecast": "2026-08-02T00:00:00Z",
            "_forecast_values": _numeric_cdf(),
        },
        -5.0,
    )
    row = report.all_rows({"posts": [post]})[0]
    assert row["miss_tag"] == "RESOLVED-OUTSIDE-RANGE"
    assert row["resolution"] == "below_lower_bound"


def test_type_breakdown_sums_damage():
    scored = report.scored_rows(report.all_rows(snapshot()))
    by_type = report.type_breakdown(scored)
    assert by_type["binary"]["count"] == 2
    assert by_type["binary"]["total"] == pytest.approx(-285.19)
    assert by_type["numeric"] == {"count": 1, "total": pytest.approx(-230.11)}
    assert by_type["multiple_choice"] == {"count": 1, "total": pytest.approx(3.3)}


def test_binary_calibration_bins():
    bins = {b["lo"]: b for b in report.binary_calibration(report.all_rows(snapshot())) if b["n"]}
    # The 2% polarity miss resolved YES; so did the healthy 65%.
    assert (bins[0]["n"], bins[0]["yes"]) == (1, 1)
    assert (bins[60]["n"], bins[60]["yes"]) == (1, 1)
    assert set(bins) == {0, 60}


def test_freshness_and_never_forecast():
    fresh = report.freshness_stats(report.all_rows(snapshot()))
    assert fresh["forecast"] == 4
    # Latencies 36h, 48h, 3h, 72h: median 48h, one inside the first day.
    assert fresh["median_hours"] == pytest.approx(48.0)
    assert fresh["within_24h"] == 1
    assert fresh["never"] == 1


def test_render_report_mentions_the_known_damage():
    text = report.render_report(snapshot(), limit=2)
    assert "Total spot peer score (leaderboard number):" in text
    assert "EXTREME-NO-MISS" in text
    assert "RESOLVED-OUTSIDE-RANGE" in text
    assert "1 closed/resolved questions never forecast" in text


def test_csv_export_writes_all_scored_rows(tmp_path):
    path = tmp_path / "scores.csv"
    scored = report.scored_rows(report.all_rows(snapshot()))
    report.export_csv(scored, path)
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["post_id"] == "44654"
    assert float(rows[0]["spot_peer_score"]) == pytest.approx(-297.69)
    assert rows[0]["miss_tag"] == "EXTREME-NO-MISS"