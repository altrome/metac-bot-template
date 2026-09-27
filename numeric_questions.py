import datetime
import re

import numpy as np
from scipy.interpolate import PchipInterpolator

from indicator_data import gather_indicator_data
from llm_calls import call_gpt5_reasoning_text, create_rationale_summary
from numeric_cdf_constrains import (
    ascii_plot_cdf,
    cdf_diagnostics,
    enforce_cdf_constraints,
    pdf_sparkline_from_cdf,
)
from prompts_gpt5 import NUMERIC_PROMPT_TEMPLATE

# Minimum probability mass kept outside each open bound, mirroring the Metaculus
# baseline convention. The old pipeline let this collapse to 0.1%, which cost
# ~200 points every time a question resolved outside its range.
MIN_OPEN_BOUND_TAIL_MASS = 0.05
MAX_TOTAL_TAIL_MASS = 0.90


def extract_percentiles_from_response(forecast_text: str) -> dict:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text) -> dict:
        pattern = r"^.*(?:P|p)ercentile.*$"
        number_pattern = r"-\s*(?:[^\d\-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)"
        results = []

        for line in text.split("\n"):
            if re.match(pattern, line):
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [
                    next(num for num in match if num).replace(",", "")
                    for match in numbers
                ]
                numbers = [
                    float(num) if "." in num else int(num)
                    for num in numbers_no_commas
                ]
                if len(numbers) > 1:
                    first_number = numbers[0]
                    last_number = numbers[-1]
                    # Check if the original line had a negative sign before the last number
                    if "-" in line.split(":")[-1]:
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))

        # Convert results to dictionary
        percentile_values = {}
        for first_num, second_num in results:
            key = first_num
            percentile_values[key] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def extract_tail_probabilities(forecast_text: str) -> tuple[float | None, float | None]:
    """
    Return (below_percent, above_percent) from the model's open-bound tail
    probability lines, e.g. "Probability below 300000: 12". Missing lines
    (or missing numbers) give None so the caller can apply the floor.
    """
    below = None
    above = None
    for line in forecast_text.split("\n"):
        match = re.search(r"(?i)^\s*probability\s+below[^:]*:\s*([\d,]+(?:\.\d+)?)", line)
        if match:
            below = float(match.group(1).replace(",", ""))
        match = re.search(r"(?i)^\s*probability\s+above[^:]*:\s*([\d,]+(?:\.\d+)?)", line)
        if match:
            above = float(match.group(1).replace(",", ""))
    return below, above


def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
    cdf_size: int,
    below_bound_probability: float | None = None,
    above_bound_probability: float | None = None,
) -> list[float]:
    """
    Build a Metaculus-compatible CDF from elicited percentiles plus the
    model's probabilities that the outcome falls outside the question's range.

    The percentiles describe the outcome conditional on it landing inside the
    range; on open bounds the tail probabilities carry the outside mass,
    floored at MIN_OPEN_BOUND_TAIL_MASS. The bounds anchor the CDF endpoints
    (closed bounds at 0.0/1.0), the percentiles anchor the interior, and a
    monotone PCHIP interpolation fills the grid.

    Returns: list[float]: A list of cdf_size float values representing the CDF.
    """

    range_min = lower_bound
    range_max = upper_bound
    range_size = range_max - range_min
    buffer = 1 if range_size > 100 else 0.01 * range_size

    # Outside-range mass on open bounds. Values out of range are common enough
    # that this mass is always real, never the 0.1% API validity floor.
    lower_tail_mass = 0.0
    upper_tail_mass = 0.0
    if open_lower_bound:
        elicited = (below_bound_probability or 0.0) / 100.0
        lower_tail_mass = min(max(elicited, MIN_OPEN_BOUND_TAIL_MASS), 0.95)
    if open_upper_bound:
        elicited = (above_bound_probability or 0.0) / 100.0
        upper_tail_mass = min(max(elicited, MIN_OPEN_BOUND_TAIL_MASS), 0.95)
    total_tail_mass = lower_tail_mass + upper_tail_mass
    if total_tail_mass > MAX_TOTAL_TAIL_MASS:
        scale = MAX_TOTAL_TAIL_MASS / total_tail_mass
        lower_tail_mass *= scale
        upper_tail_mass *= scale
    interior_mass = 1.0 - lower_tail_mass - upper_tail_mass

    # Anchor points (value -> CDF height): the bounds carry the tail masses and
    # each elicited percentile carries its share of the interior mass. Values
    # are pulled strictly inside the range; an out-of-range percentile response
    # collapses to the edge and its mass lives in the tails instead.
    anchors: dict[float, float] = {
        range_min: lower_tail_mass,
        range_max: 1.0 - upper_tail_mass,
    }
    for percentile, value in percentile_values.items():
        percentile_fraction = min(max(float(percentile) / 100.0, 0.0), 1.0)
        height = lower_tail_mass + interior_mass * percentile_fraction
        anchored_value = min(max(float(value), range_min + buffer), range_max - buffer)
        anchors[anchored_value] = max(height, anchors.get(anchored_value, 0.0))

    anchor_values = np.array(sorted(anchors.keys()), dtype=float)
    anchor_heights = np.maximum.accumulate(
        np.array([anchors[value] for value in anchor_values])
    )

    # function for log scaled questions
    def generate_cdf_locations(range_min, range_max, zero_point):
        if zero_point is None:
            scale = lambda x: range_min + (range_max - range_min) * x
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scale = lambda x: range_min + (range_max - range_min) * (
                deriv_ratio**x - 1
            ) / (deriv_ratio - 1)
        return np.array([scale(x) for x in np.linspace(0, 1, cdf_size)])

    cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)

    if len(anchor_values) >= 2:
        pchip = PchipInterpolator(anchor_values, anchor_heights)
        continuous_cdf = np.asarray(pchip(cdf_xaxis), dtype=float)
    else:
        # Degenerate input (no usable percentiles): flat interior
        continuous_cdf = np.full(cdf_size, 0.5)
    continuous_cdf = np.clip(continuous_cdf, 0.0, 1.0)

    # Ensure CDF follows metaculus constraints (elicited tail masses survive)
    continuous_cdf = enforce_cdf_constraints(
        continuous_cdf, open_lower_bound, open_upper_bound
    )

    # Console: shape of the PDF (sparkline) and ASCII mini-plot of the CDF
    print("pdf sparkline:", pdf_sparkline_from_cdf(continuous_cdf))
    cdf_diagnostics(continuous_cdf)
    ascii_plot_cdf(continuous_cdf, width=80, height=16, y_ticks=(0.0, 0.25, 0.5, 0.75, 1.0))

    return list(continuous_cdf)


async def get_numeric_gpt_prediction(
    question_details: dict, num_runs: int, run_research_func
) -> tuple[list[float], str]:

    today = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details["open_upper_bound"]
    open_lower_bound = question_details["open_lower_bound"]
    unit_of_measure = question_details["unit"] if question_details["unit"] else "Not stated (please infer this)"
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling["zero_point"]
    if question_type == "discrete":
        outcome_count = question_details["scaling"]["inbound_outcome_count"]
        cdf_size = outcome_count + 1
    else:
        cdf_size = 201
    

    # Create messages about the bounds that are passed in the LLM prompt
    if open_upper_bound:
        upper_bound_message = (
            f"The question's range ends at {upper_bound}, but the outcome may be higher. "
            f"Estimate the percentiles assuming the outcome stays within the range, and "
            f"separately estimate the probability that it is higher than {upper_bound}."
        )
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = (
            f"The question's range starts at {lower_bound}, but the outcome may be lower. "
            f"Estimate the percentiles assuming the outcome stays within the range, and "
            f"separately estimate the probability that it is lower than {lower_bound}."
        )
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    # Tail-probability lines for the final output block (only on open bounds)
    below_probability_line = (
        f"Probability below {lower_bound}: XX\n" if open_lower_bound else ""
    )
    above_probability_line = (
        f"Probability above {upper_bound}: XX\n" if open_upper_bound else ""
    )

    summary_report, source_urls = await run_research_func(question_details)
    official_data_block = await gather_indicator_data(question_details)

    content = NUMERIC_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        official_data=official_data_block,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        below_probability_line=below_probability_line,
        above_probability_line=above_probability_line,
        units=unit_of_measure,
    )

    async def ask_llm_to_get_cdf(content: str) -> tuple[list[float], str]:
        rationale = await call_gpt5_reasoning_text(content, reasoning_effort="medium", verbosity="medium")
        percentile_values = extract_percentiles_from_response(rationale)
        below_probability, above_probability = extract_tail_probabilities(rationale)

        comment = (
            f"Extracted Percentile_values: {percentile_values}\n"
            f"Extracted tail probabilities: below={below_probability}% above={above_probability}%\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )

        cdf = generate_continuous_cdf(
            percentile_values,
            question_type,
            open_upper_bound,
            open_lower_bound,
            upper_bound,
            lower_bound,
            zero_point,
            cdf_size,
            below_bound_probability=below_probability,
            above_bound_probability=above_probability,
        )

        return cdf, comment

    import asyncio

    cdf_and_comment_pairs = await asyncio.gather(
        *[ask_llm_to_get_cdf(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    cdfs: list[list[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    all_cdfs = np.array(cdfs)
    median_cdf: list[float] = np.median(all_cdfs, axis=0).tolist()

    # Create consolidated summary if multiple runs
    consolidated_summary = ""
    if num_runs > 1:
        rationales = [pair[1].split("GPT's Answer: ", 1)[1] if "GPT's Answer: " in pair[1] else pair[1] for pair in cdf_and_comment_pairs]
        consolidated_summary = await create_rationale_summary(
            rationales=rationales,
            question_title=title,
            question_type="numeric",
            final_prediction=f"Median CDF with {len(median_cdf)} points",
            source_urls=source_urls
        )

    # Build final comment with consolidated summary if available
    final_comment_parts = [f"Median CDF: `{str(median_cdf)[:100]}...`"]
    
    if consolidated_summary:
        final_comment_parts.append(f"\n## Consolidated Analysis\n{consolidated_summary}")
    
    final_comment_parts.append("\n" + "\n\n".join(final_comment_sections))
    
    final_comment = "\n\n".join(final_comment_parts)
    return median_cdf, final_comment


def _create_upper_and_lower_bound_messages(
    question: dict
) -> tuple[str, str]:
    if question["open_upper_bound"]:
        upper_bound_message = ""
    else:
        upper_bound_message = (
            f"The outcome can not be higher than {question['scaling']['range_max']}."
        )
    if question["open_lower_bound"]:
        lower_bound_message = ""
    else:
        lower_bound_message = (
            f"The outcome can not be lower than {question['scaling']['range_min']}."
        )
    return upper_bound_message, lower_bound_message