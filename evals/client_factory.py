"""Chat client factory with provider auto-detection.

Free-tier users (the tutorial's audience) set ``GITHUB_TOKEN`` and run on
GitHub Models — exactly what the blog post documents. If you're the post's
author and you've hit GitHub Models' daily request cap mid-iteration, you
can swap in any OpenAI-compatible API by setting one of the alternate
env vars below. The agents and harness are unchanged either way.

Auto-detection priority (first one set wins):
    1. ``OPENAI_API_KEY``   — OpenAI direct       (model: gpt-4.1-mini)
    2. ``GROQ_API_KEY``     — Groq, free tier     (model: llama-3.3-70b-versatile)
    3. ``GITHUB_TOKEN``     — GitHub Models, free (model: openai/gpt-4.1-mini)

Groq's free tier offers ~30 RPM and ~14,400 RPD — substantially more
headroom than GitHub Models when iterating. Different model family
though (Llama, not GPT), so eval scores from a Groq run aren't strictly
comparable to a GitHub Models run; pick one provider for the post's
"official" numbers.

Override the default model on each provider with ``OPENAI_MODEL_ID`` /
``GROQ_MODEL_ID``. Groq's model lineup rotates occasionally; if the
default name is retired, check console.groq.com/docs/models for the
current list.
"""

from __future__ import annotations

import os

from agent_framework.openai import OpenAIChatClient


def get_chat_client() -> OpenAIChatClient:
    """Build a chat client. Provider depends on which env var is present."""
    if openai_key := os.getenv("OPENAI_API_KEY"):
        return OpenAIChatClient(
            model_id=os.getenv("OPENAI_MODEL_ID", "gpt-4.1-mini"),
            api_key=openai_key,
        )
    if groq_key := os.getenv("GROQ_API_KEY"):
        return OpenAIChatClient(
            model_id=os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile"),
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
    if github_token := os.getenv("GITHUB_TOKEN"):
        return OpenAIChatClient(
            model_id="openai/gpt-4.1-mini",
            api_key=github_token,
            base_url="https://models.github.ai/inference",
        )
    raise RuntimeError(
        "No API credentials found. Set one of GITHUB_TOKEN / OPENAI_API_KEY / "
        "GROQ_API_KEY in your .env. See .env.example.",
    )


def active_provider() -> str:
    """Return the active provider name for logging/banners."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GITHUB_TOKEN"):
        return "github"
    return "none"
