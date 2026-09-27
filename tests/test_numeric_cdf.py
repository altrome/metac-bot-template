"""Regression tests for the numeric/discrete CDF pipeline.

The Summer 2026 season lost ~2,558 points because the old pipeline forced
exactly 0.1% mass outside each open bound (the API validity floor was used
as a target), so every out-of-range resolution cost ~200 points. These tests
pin the fixed behavior: real tail mass on open bounds, exact endpoints on
closed bounds, monotone and API-valid CDFs.
"""

import contextlib
import io

import numpy as np

from numeric_cdf_constrains import enforce_cdf_constraints
from numeric_questions import (
    MIN_OPEN_BOUND_TAIL_MASS,
    extract_tail_probabilities,
    generate_continuous_cdf,
)

# The Summer 2026 Strategic Petroleum Reserve question: range 300k-450k,
# resolved at 293,426 -- just below the range (one of the -230 point hits).
LOWER, UPPER = 300_000.0, 450_000.0
MID_PERCENTILES = {
    10: 330_000,
    20: 345_000,
    40: 370_000,
    60: 385_000,
    80: 400_000,
    90: 415_000,
}


def _build(
    percentiles,
    open_lower,
    open_upper,
    below=None,
    above=None,
    cdf_size=201,
    zero_point=None,
    lower=LOWER,
    upper=UPPER,
):
    with contextlib.redirect_stdout(io.StringIO()):
        return generate_continuous_cdf(
            dict(percentiles),
            "numeric",
            open_upper_bound=open_upper,
            open_lower_bound=open_lower,
            upper_bound=upper,
            lower_bound=lower,
            zero_point=zero_point,
            cdf_size=cdf_size,
            below_bound_probability=below,
            above_bound_probability=above,
        )


def _assert_valid(cdf, open_lower, open_upper):
    cdf = np.asarray(cdf, dtype=float)
    assert np.all(np.diff(cdf) >= -1e-9), "CDF must be non-decreasing"
    assert np.all(np.diff(cdf) <= 0.2 + 1e-9), "no CDF step may exceed 0.2"
    if open_lower:
        assert cdf[0] >= 0.001, "open lower bound needs cdf[0] >= 0.001"
    else:
        assert abs(cdf[0]) < 1e-9, "closed lower bound needs cdf[0] == 0"
    if open_upper:
        assert cdf[-1] <= 0.999, "open upper bound needs cdf[-1] <= 0.999"
    else:
        assert abs(cdf[-1] - 1.0) < 1e-9, "closed upper bound needs cdf[-1] == 1"


def test_open_upper_keeps_elicited_tail_mass():
    cdf = _build(MID_PERCENTILES, open_lower=False, open_upper=True, above=8)
    assert abs(cdf[-1] - 0.92) < 1e-3, "8% above-bound mass must survive"


def test_open_upper_floor_when_tail_small_or_missing():
    small = _build(MID_PERCENTILES, open_lower=False, open_upper=True, above=1)
    missing = _build(MID_PERCENTILES, open_lower=False, open_upper=True, above=None)
    for cdf in (small, missing):
        assert cdf[-1] <= 1 - MIN_OPEN_BOUND_TAIL_MASS + 1e-3


def test_open_lower_keeps_elicited_tail_mass():
    cdf = _build(MID_PERCENTILES, open_lower=True, open_upper=False, below=20)
    assert abs(cdf[0] - 0.20) < 1e-3, "20% below-bound mass must survive"


def test_closed_bounds_exact_endpoints():
    cdf = _build(MID_PERCENTILES, open_lower=False, open_upper=False)
    _assert_valid(cdf, open_lower=False, open_upper=False)


def test_spr_scenario_has_real_below_range_mass():
    # The old pipeline put 0.1% below 300k here and lost ~230 points.
    cdf = _build(MID_PERCENTILES, open_lower=True, open_upper=True, below=15)
    assert abs(cdf[0] - 0.15) < 1e-3


def test_percentile_heights_match_anchors():
    # With floored 5%/5% tails, an elicited P10/P40/P90 must sit at
    # 14%/41%/86% CDF height at its elicited value.
    cdf = np.asarray(_build(MID_PERCENTILES, open_lower=True, open_upper=True))
    x = np.arange(len(cdf))
    for percentile, value in ((10, 330_000), (40, 370_000), (90, 415_000)):
        expected = 0.05 + 0.90 * percentile / 100
        grid_pos = (value - LOWER) / (UPPER - LOWER) * (len(cdf) - 1)
        height = float(np.interp(grid_pos, x, cdf))
        assert abs(height - expected) < 0.02


def test_log_scale_round_trip():
    # Log-scaled question: range 1..1000 with zero_point 0 -> value(x) = 1000**x.
    percentiles = {10: 5.0, 50: 50.0, 90: 300.0}
    cdf = _build(percentiles, open_lower=True, open_upper=True, lower=1.0, upper=1000.0, zero_point=0.0)
    _assert_valid(cdf, open_lower=True, open_upper=True)
    index = round(200 * np.log(50.0) / np.log(1000.0))
    assert abs(cdf[index] - 0.50) < 0.02, "median anchor must land at ~50% height"


def test_discrete_cdf_size():
    cdf = _build(MID_PERCENTILES, open_lower=True, open_upper=True, cdf_size=51)
    assert len(cdf) == 51
    _assert_valid(cdf, open_lower=True, open_upper=True)


def test_validity_matrix():
    cases = [
        (False, False, None, None),
        (True, False, None, 12),
        (False, True, 30, None),
        (True, True, 10, 3),
        (True, True, None, 90),
        (True, True, 80, 80),
    ]
    for open_lower, open_upper, below, above in cases:
        cdf = _build(MID_PERCENTILES, open_lower, open_upper, below=below, above=above)
        assert len(cdf) == 201
        _assert_valid(cdf, open_lower, open_upper)


def test_tail_mass_capped_when_huge():
    cdf = _build(MID_PERCENTILES, open_lower=True, open_upper=True, below=80, above=80)
    total_tail = cdf[0] + (1 - cdf[-1])
    assert abs(total_tail - 0.90) < 1e-2, "combined tails must stay <= 90%"


def test_out_of_range_percentile_responses_still_valid():
    # Model answers entirely above the upper bound: anchors clamp inside,
    # outside mass rides in the tails, CDF stays valid.
    percentiles = {10: 500_000, 50: 550_000, 90: 600_000}
    cdf = _build(percentiles, open_lower=True, open_upper=True, below=5, above=25)
    _assert_valid(cdf, open_lower=True, open_upper=True)
    assert abs(cdf[-1] - 0.75) < 1e-3


def test_enforce_preserves_healthy_endpoints():
    cdf = enforce_cdf_constraints(np.linspace(0.05, 0.92, 201), open_lower=True, open_upper=True)
    assert abs(cdf[0] - 0.05) < 1e-6
    assert abs(cdf[-1] - 0.92) < 1e-6


def test_enforce_applies_floors_to_degenerate_input():
    # Old-style 0/1 input on open bounds still gets the API-valid 0.001/0.999.
    cdf = enforce_cdf_constraints(np.linspace(0, 1, 201), open_lower=True, open_upper=True)
    assert abs(cdf[0] - 0.001) < 1e-6
    assert abs(cdf[-1] - 0.999) < 1e-6
    cdf = enforce_cdf_constraints(np.linspace(0, 1, 201), open_lower=False, open_upper=False)
    assert abs(cdf[0]) < 1e-9
    assert abs(cdf[-1] - 1.0) < 1e-9


def test_median_aggregation_stays_valid():
    # get_numeric_gpt_prediction aggregates runs with a pointwise median.
    cdfs = [
        _build({10: 320_000, 50: 375_000, 90: 430_000}, True, True, below=10, above=8),
        _build({10: 350_000, 50: 390_000, 90: 420_000}, True, True, below=3, above=15),
        _build({10: 310_000, 50: 365_000, 90: 410_000}, True, True, below=25, above=None),
    ]
    median = np.median(np.array(cdfs), axis=0)
    assert np.all(np.diff(median) >= -1e-9)
    assert median[0] >= MIN_OPEN_BOUND_TAIL_MASS - 1e-3
    assert median[-1] <= 1 - MIN_OPEN_BOUND_TAIL_MASS + 1e-3


def test_extract_tail_probabilities():
    text = (
        "Some analysis text with 500 numbers everywhere: 300,000 things.\n"
        "Percentile 10: 320,000\n"
        "Percentile 50: 375,000\n"
        "Probability below 300000: 12\n"
        "Probability above 450,000: 7.5\n"
    )
    below, above = extract_tail_probabilities(text)
    assert below == 12.0
    assert above == 7.5


def test_extract_tail_probabilities_missing_lines():
    text = "Percentile 10: 320,000\nPercentile 50: 375,000\n"
    below, above = extract_tail_probabilities(text)
    assert below is None
    assert above is None


def _mock_llm(responses):
    """Return an async fake for call_gpt5_reasoning_text that also captures prompts."""
    import asyncio

    captured = []

    class _Fake:
        async def reasoning(self, content, **kwargs):
            captured.append(content)
            return responses[len(captured) % len(responses)]

        async def summary(self, **kwargs):
            return "consolidated summary"

    return _Fake(), captured


def test_full_pipeline_carries_tails_with_mocked_llm(monkeypatch):
    import asyncio

    import numeric_questions as nq

    responses = [
        (
            "Time: 3 months\nStatus quo: slow decline\nTrend: slow decline\n"
            "Expectations: none cited\nLow scenario: faster drain\nHigh scenario: refills\n"
            "Percentile 10: 330,000\nPercentile 20: 345,000\nPercentile 40: 370,000\n"
            "Percentile 60: 385,000\nPercentile 80: 400,000\nPercentile 90: 415,000\n"
            "Probability below 300000: 12\nProbability above 450000: 7\n"
        ),
        (
            "Time: 3 months\nStatus quo: slow decline\nTrend: slow decline\n"
            "Expectations: none cited\nLow scenario: faster drain\nHigh scenario: refills\n"
            "Percentile 10: 350,000\nPercentile 20: 360,000\nPercentile 40: 380,000\n"
            "Percentile 60: 395,000\nPercentile 80: 420,000\nPercentile 90: 440,000\n"
            "Probability below 300000: 4\nProbability above 450000: 15\n"
        ),
    ]
    fake, captured = _mock_llm(responses)

    async def fake_research(question_details):
        return "research summary", ["https://example.com"]

    monkeypatch.setattr(nq, "call_gpt5_reasoning_text", fake.reasoning)
    monkeypatch.setattr(nq, "create_rationale_summary", fake.summary)

    question_details = {
        "title": "How many barrels in the SPR?",
        "resolution_criteria": "criteria",
        "description": "background",
        "fine_print": "",
        "type": "numeric",
        "scaling": {"range_min": 300_000, "range_max": 450_000, "zero_point": None},
        "open_upper_bound": True,
        "open_lower_bound": True,
        "unit": "barrels",
    }

    cdf, _comment = asyncio.run(
        nq.get_numeric_gpt_prediction(question_details, 2, fake_research)
    )
    cdf = np.asarray(cdf)
    _assert_valid(cdf, open_lower=True, open_upper=True)
    # The prompt must request tail probabilities on open bounds...
    assert "Probability below 300000: XX" in captured[0]
    assert "Probability above 450000: XX" in captured[0]
    # ...and the elicited tails must reach the final median CDF:
    # below = median(12%, 4%->floor 5%) = 8.5%, above = median(7%, 15%) = 11%.
    assert abs(cdf[0] - 0.085) < 5e-3
    assert abs((1 - cdf[-1]) - 0.11) < 5e-3


def test_full_pipeline_closed_bounds_do_not_request_tails(monkeypatch):
    import asyncio

    import numeric_questions as nq

    responses = [
        (
            "Time: 3 months\nStatus quo: flat\nTrend: flat\nExpectations: none cited\n"
            "Low scenario: dip\nHigh scenario: pop\n"
            "Percentile 10: 330,000\nPercentile 20: 345,000\nPercentile 40: 370,000\n"
            "Percentile 60: 385,000\nPercentile 80: 400,000\nPercentile 90: 415,000\n"
        )
    ]
    fake, captured = _mock_llm(responses)

    async def fake_research(question_details):
        return "research summary", []

    monkeypatch.setattr(nq, "call_gpt5_reasoning_text", fake.reasoning)
    monkeypatch.setattr(nq, "create_rationale_summary", fake.summary)

    question_details = {
        "title": "Bounded question",
        "resolution_criteria": "criteria",
        "description": "background",
        "fine_print": "",
        "type": "numeric",
        "scaling": {"range_min": 300_000, "range_max": 450_000, "zero_point": None},
        "open_upper_bound": False,
        "open_lower_bound": False,
        "unit": "barrels",
    }

    cdf, _comment = asyncio.run(
        nq.get_numeric_gpt_prediction(question_details, 1, fake_research)
    )
    _assert_valid(np.asarray(cdf), open_lower=False, open_upper=False)
    assert "Probability below" not in captured[0]
    assert "Probability above" not in captured[0]