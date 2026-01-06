import re
import datetime
from prompts_gpt5 import BINARY_PROMPT_TEMPLATE, BINARY_META_PROMPT_TEMPLATE
from llm_calls import call_gpt5_reasoning_text, create_rationale_summary

# Non-breaking / narrow spaces that sometimes appear before the '%' sign
NBSPS = ("\u00A0", "\u202F", "\u2009", "\u2007")

# Regex to locate the exact line that contains the final probability.
# Supports both:
#   "Probability: 37.50%" (standard template)
#   "6) Probability: 37.50%" (meta template)
PROB_LINE = re.compile(r"(?mi)^\s*(?:\d+\s*[\)\.]\s*)?Probability\s*:.*$")

# Regex to extract a numeric value on that line; accepts optional '%' and decimals
PROB_VALUE = re.compile(r"Probability\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*%?", re.I)


def is_meta_question(title: str) -> bool:
    """
    Detect if this is a meta-question about community predictions.
    Meta-questions typically ask about community prediction percentages at specific dates.
    """
    # Convert to lowercase for case-insensitive matching
    title_lower = title.lower()
    
    # Patterns that indicate meta-questions
    meta_patterns = [
        r"community prediction.*(?:higher|lower|greater|less|exceed|above|below|at least).*\d+(?:\.\d+)?%",
        r"(?:will|does).*community prediction.*\d+(?:\.\d+)?%",
        r"metaculus.*community prediction.*\d+(?:\.\d+)?%",
        r"cp.*(?:higher|lower|greater|less|exceed|above|below|at least).*\d+(?:\.\d+)?%",
    ]
    
    # Check if any pattern matches
    for pattern in meta_patterns:
        if re.search(pattern, title_lower):
            return True
    
    return False


def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def extract_probability_percent(text: str, clamp_min=1.0, clamp_max=99.0, decimals=2):
    """
    Extract a probability percentage from the 'Probability:' line and return a value in [1, 99].
    Robust to:
      - Optional '%' symbol (e.g., "93", "93%", "93.0 %", "93,0 %")
      - Unicode spacing before '%'
      - Decimal commas (e.g., '93,00')
      - Fractions in [0..1] when '%' is omitted (e.g., '0.93' -> 93%)

    Returns (value_in_percent, status), where status is 'ok' or a failure reason.
    """

    # 1) Normalize any exotic unicode spaces to plain ASCII spaces
    for sp in NBSPS:
        text = text.replace(sp, " ")

    # 2) Restrict parsing to the specific line that contains "Probability:"
    mline = PROB_LINE.search(text)
    if not mline:
        return None, "no-prob-line"
    line = mline.group(0)

    # 3) Convert decimal comma to dot on that line (e.g., "93,0" -> "93.0")
    line_norm = re.sub(r'(\d),(\d)', r'\1.\2', line)

    # 4) Extract the numeric portion; '%' is optional
    m = PROB_VALUE.search(line_norm)
    if not m:
        return None, "no-number"
    val = float(m.group(1))
    had_percent = "%" in line_norm

    # 5) Units: if there's no '%' and the number looks like a fraction, interpret as [0..1] and convert to %
    if (not had_percent) and val <= 1.5:
        val *= 100.0

    # 6) Final clamp and optional rounding
    val = max(clamp_min, min(clamp_max, val))
    if decimals is not None:
        val = round(val, decimals)

    return val, "ok"


async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int, run_research_func
) -> tuple[float, str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    summary_report, source_urls = await run_research_func(question_details)

    # Determine which template to use based on question type
    is_meta = is_meta_question(title)
    if is_meta:
        # Use specialized template for meta-questions about community predictions
        template = BINARY_META_PROMPT_TEMPLATE
        print(f"🎯 DETECTED META-QUESTION: Using BINARY_META_PROMPT_TEMPLATE")
        print(f"   Title: {title}")
    else:
        # Use standard template for regular binary questions
        template = BINARY_PROMPT_TEMPLATE
        print(f"📊 Standard binary question: Using BINARY_PROMPT_TEMPLATE")

    content = template.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )

    async def get_rationale_and_probability(content: str) -> tuple[float, str]:
        last_rationale = None
        last_status = None

        # Small retry loop to handle occasional format drift.
        for _ in range(3):
            rationale = await call_gpt5_reasoning_text(
                content, reasoning_effort="medium", verbosity="medium"
            )
            last_rationale = rationale
            probability, status = extract_probability_percent(rationale)
            last_status = status
            if probability is not None:
                comment = (
                    f"Extracted Probability: {probability}%\n\nGPT's Answer: "
                    f"{rationale}\n\n\n"
                )
                return probability, comment

        raise ValueError(
            f"Could not extract probability from model output (status={last_status}). "
            f"Model output starts with: {str(last_rationale)[:300]}"
        )

    import asyncio
    import numpy as np

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]
    median_probability = float(np.median(probabilities)) / 100

    # Create consolidated summary if multiple runs
    consolidated_summary = ""
    if num_runs > 1:
        rationales = [pair[1].split("GPT's Answer: ", 1)[1] if "GPT's Answer: " in pair[1] else pair[1] for pair in probability_and_comment_pairs]
        consolidated_summary = await create_rationale_summary(
            rationales=rationales,
            question_title=title,
            question_type="binary",
            final_prediction=f"{median_probability:.2%}",
            source_urls=source_urls
        )

    # Build final comment with consolidated summary if available
    final_comment_parts = [f"Median Probability: {median_probability}"]
    
    if consolidated_summary:
        final_comment_parts.append(f"\n## Consolidated Analysis\n{consolidated_summary}")
    
    final_comment_parts.append("\n" + "\n\n".join(final_comment_sections))
    
    final_comment = "\n\n".join(final_comment_parts)
    return median_probability, final_comment