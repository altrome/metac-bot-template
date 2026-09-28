"""Tests for the two-vendor (GPT + Claude) forecast ensemble.

All vendor calls are faked; the live path is covered by the E2E script,
never here.
"""

import asyncio
from types import SimpleNamespace

import claude_calls
import llm_calls
from llm_calls import call_forecast_reasoner, rationale_text_from_comment


def _fake_vendors(monkeypatch, enabled, gpt_answer="GPT answer", claude_answer="Claude answer", claude_raises=None):
    """Patch the ensemble's vendor calls and return the call log."""
    calls = []

    async def fake_gpt(prompt, **kwargs):
        calls.append("gpt")
        return gpt_answer

    async def fake_claude(prompt):
        calls.append("claude")
        if claude_raises is not None:
            raise claude_raises
        return claude_answer

    monkeypatch.setattr(llm_calls, "CLAUDE_ENSEMBLE_ENABLED", enabled)
    monkeypatch.setattr(llm_calls, "call_gpt5_reasoning_text", fake_gpt)
    monkeypatch.setattr(llm_calls, "call_claude_reasoning_text", fake_claude)
    return calls


# --- dispatch ---


def test_even_runs_use_gpt(monkeypatch):
    calls = _fake_vendors(monkeypatch, enabled=True)
    text, vendor = asyncio.run(call_forecast_reasoner("prompt", run_index=0))
    assert (vendor, text) == ("GPT", "GPT answer")
    assert calls == ["gpt"]


def test_odd_runs_use_claude_when_enabled(monkeypatch):
    calls = _fake_vendors(monkeypatch, enabled=True)
    text, vendor = asyncio.run(call_forecast_reasoner("prompt", run_index=1))
    assert (vendor, text) == ("Claude", "Claude answer")
    assert calls == ["claude"]


def test_claude_failure_falls_back_to_gpt(monkeypatch):
    calls = _fake_vendors(monkeypatch, enabled=True, claude_raises=RuntimeError("boom"))
    text, vendor = asyncio.run(call_forecast_reasoner("prompt", run_index=1))
    assert (vendor, text) == ("GPT", "GPT answer")
    assert calls == ["claude", "gpt"], "a broken Claude run must not break the forecast"


def test_claude_never_used_when_disabled(monkeypatch):
    calls = _fake_vendors(monkeypatch, enabled=False)
    text, vendor = asyncio.run(call_forecast_reasoner("prompt", run_index=1))
    assert (vendor, text) == ("GPT", "GPT answer")
    assert calls == ["gpt"]


# --- comment helpers ---


def test_rationale_text_from_comment_strips_either_marker():
    assert rationale_text_from_comment("GPT's Answer: the text") == "the text"
    assert rationale_text_from_comment("Claude's Answer: other text") == "other text"
    assert rationale_text_from_comment("no marker") == "no marker"


# --- claude_calls ---


def _fake_anthropic_module(response, captured=None):
    async def create(**kwargs):
        if captured is not None:
            captured.append(kwargs)
        return response

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    return SimpleNamespace(AsyncAnthropic=lambda: client)


def test_call_claude_returns_visible_text_only(monkeypatch):
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="internal reasoning"),
            SimpleNamespace(type="text", text="Resolution: yes\nProbability: 40.00%"),
        ],
        stop_reason="end_turn",
        stop_details=None,
    )
    monkeypatch.setattr(claude_calls, "anthropic", _fake_anthropic_module(response))
    text = asyncio.run(claude_calls.call_claude_reasoning_text("prompt"))
    assert text == "Resolution: yes\nProbability: 40.00%"


def test_call_claude_sends_expected_request(monkeypatch):
    captured = []
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        stop_details=None,
    )
    monkeypatch.setattr(claude_calls, "anthropic", _fake_anthropic_module(response, captured))
    asyncio.run(claude_calls.call_claude_reasoning_text("prompt"))
    kwargs = captured[0]
    assert kwargs["model"] == claude_calls.CLAUDE_MODEL
    assert kwargs["fallbacks"] == "default"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]


def test_call_claude_raises_on_refusal(monkeypatch):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="")],
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
    )
    monkeypatch.setattr(claude_calls, "anthropic", _fake_anthropic_module(response))
    try:
        asyncio.run(claude_calls.call_claude_reasoning_text("prompt"))
        raise AssertionError("refusal must raise")
    except RuntimeError as exc:
        assert "refused" in str(exc)


def test_call_claude_raises_on_empty_text(monkeypatch):
    response = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="only thought")],
        stop_reason="end_turn",
        stop_details=None,
    )
    monkeypatch.setattr(claude_calls, "anthropic", _fake_anthropic_module(response))
    try:
        asyncio.run(claude_calls.call_claude_reasoning_text("prompt"))
        raise AssertionError("empty text must raise")
    except RuntimeError as exc:
        assert "no text content" in str(exc)