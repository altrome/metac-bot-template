"""Tests for criteria-comprehension prompt rules and the binary [2, 98] clamp.

The catastrophic Summer miss (post 44654: "fails to reach 7.0 points..."
forecast at p=2%, resolved YES, −298) was a polarity-comprehension failure,
not an evidence failure. The prompt templates now force a restatement of the
resolution (negations and date conventions resolved) before any probability,
and binary probabilities are clamped to [2, 98].
"""

from binary_questions import extract_probability_percent
from prompts_gpt5 import (
    BINARY_PROMPT_TEMPLATE,
    MULTIPLE_CHOICE_PROMPT_TEMPLATE,
    NUMERIC_PROMPT_TEMPLATE,
)

# --- binary calibration clamp ---


def test_binary_clamp_is_two_to_ninety_eight():
    assert extract_probability_percent("Probability: 0.5%")[0] == 2.0
    assert extract_probability_percent("Probability: 99.5%")[0] == 98.0


def test_binary_clamp_leaves_normal_values_untouched():
    assert extract_probability_percent("Probability: 37.50%")[0] == 37.5
    assert extract_probability_percent("Probability: 2%")[0] == 2.0
    assert extract_probability_percent("Probability: 98%")[0] == 98.0


# --- criteria restatement + conventions in the templates ---


def test_binary_prompt_requires_resolution_restatement_first():
    assert "1) Resolution:" in BINARY_PROMPT_TEMPLATE
    assert "2) Time:" in BINARY_PROMPT_TEMPLATE, "existing lines must keep their order"
    assert "7) Calibration:" in BINARY_PROMPT_TEMPLATE
    assert BINARY_PROMPT_TEMPLATE.index("1) Resolution:") < BINARY_PROMPT_TEMPLATE.index("Probability: ZZ.ZZ%")


def test_numeric_prompt_requires_resolution_restatement():
    assert "1) Resolution:" in NUMERIC_PROMPT_TEMPLATE
    assert "2) Time:" in NUMERIC_PROMPT_TEMPLATE, "existing lines must keep their order"
    assert "7) High scenario:" in NUMERIC_PROMPT_TEMPLATE


def test_multiple_choice_prompt_requires_resolution_restatement():
    assert "1) Resolution:" in MULTIPLE_CHOICE_PROMPT_TEMPLATE
    assert "4) Surprise:" in MULTIPLE_CHOICE_PROMPT_TEMPLATE


def test_templates_teach_negation_and_date_conventions():
    convention = '"before <date>" excludes that date'
    assert convention in BINARY_PROMPT_TEMPLATE
    assert convention in NUMERIC_PROMPT_TEMPLATE
    assert 'resolve negations ("fails to", "will not")' in BINARY_PROMPT_TEMPLATE
    assert "commit to it (60–90%)" in BINARY_PROMPT_TEMPLATE, "underconfidence fix"


def test_binary_prompt_prefers_dated_source_readings_over_stale_background():
    assert (
        "weigh them over vaguer claims in the Background" in BINARY_PROMPT_TEMPLATE
    ), "the background can be updated after the question window"