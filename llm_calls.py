import asyncio
from openai import AsyncOpenAI

CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


def _model_supports_reasoning_config(model: str) -> bool:
    return model.startswith("gpt-5") or model.startswith("o")


def _simple_reasoning_effort_for_model(model: str) -> str:
    """Pick the lowest supported reasoning effort for the given model."""
    # GPT-5.2 family supports "none" as the lowest effort.
    if model.startswith("gpt-5.2"):
        return "none"
    # Older GPT-5 models use "minimal" as the lowest effort.
    if model.startswith("gpt-5"):
        return "minimal"
    # o-series typically supports low/medium/high; use low for speed.
    if model.startswith("o"):
        return "low"
    return "none"


async def call_openAI(prompt: str, model: str = "gpt-5-mini", temperature: float = 0.3) -> str:
    """
    Makes a simple, low-latency text request with concurrent request limiting.

    Intended for lightweight tasks (e.g., query generation, short extraction).
    For heavier reasoning tasks, prefer `call_gpt5_reasoning_text` / `call_gpt5_reasoning`.
    """

    # Remove the base_url parameter to call the OpenAI API directly
    # Also checkout the package 'litellm' for one function that can call any model from any provider
    # Email ben@metaculus.com if you need credit for the Metaculus OpenAI/Anthropic proxy
    client = AsyncOpenAI(
        # base_url="https://llm-proxy.metaculus.com/proxy/openai/v1",
        # default_headers={
        #     "Content-Type": "application/json",
        #     "Authorization": f"Token {METACULUS_TOKEN}",
        # },
        # api_key="Fake API Key since openai requires this not to be NONE. This isn't used",
        max_retries=2,
    )

    # NOTE: Use Responses API for consistency with GPT-5 family.
    async with llm_rate_limiter:
        params = {
            "model": model,
            "input": prompt,
            "text": {"verbosity": "low"},
        }

        if _model_supports_reasoning_config(model):
            effort = _simple_reasoning_effort_for_model(model)
            params["reasoning"] = {"effort": effort}
            # Docs: temperature/top_p/logprobs are only supported when reasoning.effort == "none".
            if effort == "none":
                params["temperature"] = temperature
        else:
            # Non-reasoning models can use temperature normally.
            params["temperature"] = temperature

        response = await client.responses.create(**params)

        answer = getattr(response, "output_text", None) or ""

        # Fallback: extract aggregated text from output items.
        if not answer and getattr(response, "output", None):
            for item in response.output:
                if hasattr(item, "content"):
                    for content_item in item.content:
                        if hasattr(content_item, "text"):
                            answer += content_item.text

        if not answer:
            raise ValueError("No answer returned from OpenAI")
        return answer


async def call_gpt5_reasoning_text(
    prompt: str,
    model: str = "gpt-5.2",
    reasoning_effort: str = "medium",
    verbosity: str = "medium",
    max_output_tokens: int | None = None,
) -> str:
    """Convenience wrapper that returns only the text content."""
    result = await call_gpt5_reasoning(
        prompt,
        model=model,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        max_output_tokens=max_output_tokens,
    )
    return result["content"]


async def call_gpt5_reasoning(
    prompt: str,
    model: str = "gpt-5.2",
    reasoning_effort: str = "medium",
    verbosity: str = "medium",
    max_output_tokens: int = None,
) -> dict:
    """
    Makes a completion request to OpenAI's GPT-5 reasoning model using the Responses API.
    
    GPT-5 models are reasoning models that break down problems step by step with 
    chain of thought reasoning. Use this for complex tasks that benefit from deeper analysis.
    
    Args:
        prompt: The input prompt/question for the model
        model: The GPT-5 model to use. Options: "gpt-5.2", "gpt-5.2-pro", "gpt-5-mini", "gpt-5-nano"
        reasoning_effort: Controls depth of reasoning. Options: "none", "low", "medium", "high", "xhigh"
                         Default is "medium". Use "none" for faster responses, "high" or "xhigh" for harder problems.
        verbosity: Controls output length. Options: "low", "medium", "high". Default is "medium".
        max_output_tokens: Maximum tokens in the response (optional)
    
    Returns:
        A dictionary containing:
        - 'content': The model's response text
        - 'reasoning_tokens': Number of reasoning tokens used (if applicable)
        - 'output_tokens': Number of output tokens used
    
    Note: Parameters like temperature, top_p, and logprobs are only supported when reasoning_effort="none"
    """
    client = AsyncOpenAI(max_retries=2)
    
    # Build request parameters
    params = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
        "text": {"verbosity": verbosity},
    }
    
    if max_output_tokens:
        params["max_output_tokens"] = max_output_tokens
    
    async with llm_rate_limiter:
        response = await client.responses.create(**params)
        
        # Extract the response content (using SDK's convenience property)
        content = response.output_text if hasattr(response, 'output_text') else ""
        
        # If output_text is not available, extract from output array
        if not content and response.output:
            for item in response.output:
                if hasattr(item, 'content'):
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            content += content_item.text
        
        # Extract usage stats
        usage = response.usage if hasattr(response, 'usage') else {}
        reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0) if hasattr(usage, 'output_tokens_details') else 0
        output_tokens = getattr(usage, 'output_tokens', 0)
        
        return {
            "content": content,
            "reasoning_tokens": reasoning_tokens,
            "output_tokens": output_tokens,
        }


async def create_rationale_summary(rationales: list[str], question_title: str, question_type: str, final_prediction: str, source_urls: list[str] = None) -> str:
    """
    Create a consolidated summary of multiple rationales for a forecasting question.
    
    Args:
        rationales: List of individual rationales from multiple runs
        question_title: The forecasting question title
        question_type: Type of question (binary, numeric, multiple_choice)
        final_prediction: The final aggregated prediction
        source_urls: List of source URLs used in research (optional)
    
    Returns:
        A consolidated summary highlighting key insights, contradictions, and justification
    """
    if len(rationales) <= 1:
        return ""  # No summary needed for single rationale
    
    # Import the prompt template
    from prompts_gpt5 import RATIONALE_SUMMARY_PROMPT_TEMPLATE
    
    # Combine all rationales for analysis
    combined_rationales = "\n\n---RATIONALE SEPARATOR---\n\n".join([f"Rationale {i+1}:\n{rationale}" for i, rationale in enumerate(rationales)])
    
    prompt = RATIONALE_SUMMARY_PROMPT_TEMPLATE.format(
        question_title=question_title,
        question_type=question_type,
        final_prediction=final_prediction,
        num_rationales=len(rationales),
        combined_rationales=combined_rationales
    )

    try:
        # summary = await call_openAI(prompt, model="gpt-4.1-mini", temperature=0.2)
        response = await call_gpt5_reasoning(
            prompt,
            model="gpt-5.2",
            max_output_tokens=5000
        )

        summary = response["content"]
        
        # Add source URLs section if available
        if source_urls:
            sources_section = f"\n\n## Sources Used\nThe following sources were used in this analysis:\n"
            for i, url in enumerate(source_urls, 1):
                sources_section += f"{i}. {url}\n"
            summary += sources_section
        
        return summary.strip()
    except Exception as e:
        print(f"Error creating rationale summary: {str(e)}")
        return "Failed to generate consolidated summary."