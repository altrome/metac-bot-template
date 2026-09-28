"""Claude forecast runs for the two-vendor ensemble.

The ensemble alternates reasoning runs between GPT and Claude per question
(see llm_calls.call_forecast_reasoner); this module owns the Claude side.

Env vars: ANTHROPIC_API_KEY (the zero-arg AsyncAnthropic() client reads it,
plus ANTHROPIC_BASE_URL when a custom gateway is in use).
"""

import asyncio

import anthropic

CLAUDE_MODEL = "claude-opus-5"
MAX_TOKENS = 16000
CONCURRENT_REQUESTS_LIMIT = 5
_claude_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


async def call_claude_reasoning_text(prompt: str) -> str:
    """
    One Claude reasoning pass over a forecast prompt; returns the visible text.

    Adaptive thinking is on (the reasoning is billed but never returned — only
    text blocks are), and the server-side refusal fallback is enabled so a
    safety decline re-runs the request instead of failing the ensemble run.
    """
    client = anthropic.AsyncAnthropic()
    async with _claude_rate_limiter:
        response = await client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) or "unknown"
        raise RuntimeError(f"Claude refused the request (category={category})")
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text:
        raise RuntimeError("Claude returned no text content")
    return text