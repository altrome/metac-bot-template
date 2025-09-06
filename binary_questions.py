import re
import datetime
from prompts import BINARY_PROMPT_TEMPLATE, BINARY_META_PROMPT_TEMPLATE
from llm_calls import call_openAI, create_rationale_summary

NBSPS = ("\u00A0", "\u202F", "\u2009", "\u2007")  # NBSP, NNBSP, thin, fig
PROB_LINE = re.compile(r'(?mi)^\s*Probability\s*:.*$')
PROB_VALUE = re.compile(r'Probability\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*%?', re.I)


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


def extract_probability_percent(text: str):
    # 1) Normaliza espacios raros a espacio ASCII
    for sp in NBSPS:
        text = text.replace(sp, " ")
    # 2) Coge SÓLO la línea con "Probability:"
    mline = PROB_LINE.search(text)
    if not mline:
        return None, "no-prob-line"
    line = mline.group(0)
    # 3) Local fix en esa línea: coma decimal -> punto
    line_norm = re.sub(r'(\d),(\d)', r'\1.\2', line)
    # 4) Extrae número (acepta con o sin % final)
    m = PROB_VALUE.search(line_norm)
    if not m:
        return None, "no-number"
    val = float(m.group(1))
    had_percent = "%" in line_norm
    # 5) Unidades: si NO hay % y val<=1.5, asume fracción [0..1] -> pásala a %
    if (not had_percent) and val <= 1.5:
        val *= 100.0
    # 6) Clamp 0..100 y devuelve
    val = max(0.0, min(100.0, val))
    return val, "ok"


async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int, run_research_func
) -> tuple[float, str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    summary_report, source_urls = await run_research_func(title)

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
        rationale = await call_openAI(content)

        probability = extract_probability_percent(rationale)[0]
        comment = (
            f"Extracted Probability: {probability}%\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )
        return probability, comment

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