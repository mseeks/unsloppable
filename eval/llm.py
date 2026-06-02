"""Shared LLM access for the eval harness — Claude Agent SDK on subscription auth.

Auth resolution (highest priority first): cloud creds → ANTHROPIC_AUTH_TOKEN →
ANTHROPIC_API_KEY → apiKeyHelper → CLAUDE_CODE_OAUTH_TOKEN → existing `claude`
CLI login. So with nothing configured it uses your logged-in Claude plan; for
headless/scheduled runs put CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`)
in a .env. ANTHROPIC_API_KEY, if set, SHADOWS the subscription and bills API
credits — we warn.

All calls go through the Agent SDK (not the plain anthropic SDK) because the
subscription OAuth path is only licensed for Claude Code / the Agent SDK.
"""
from __future__ import annotations

import os
from pathlib import Path

# Load .env if present (python-dotenv ships with claude-agent-sdk's deps).
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# Model defaults. Generation wants cheap+fast; analysis wants strong reasoning.
GEN_MODEL = "claude-haiku-4-5-20251001"
ANALYZE_MODEL = "claude-sonnet-4-6"


def auth_note() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("⚠ ANTHROPIC_API_KEY is set — it shadows subscription OAuth and bills "
                "API credits. `unset ANTHROPIC_API_KEY` to use your Claude plan.")
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return "auth: CLAUDE_CODE_OAUTH_TOKEN (Claude subscription)"
    return "auth: existing `claude` CLI login (Claude subscription)"


async def generate_text(prompt: str, system: str, *, model: str = GEN_MODEL,
                        max_budget_usd: float | None = None) -> tuple[str, float]:
    """One-shot, no-tools text generation. Returns (text, cost_usd).

    setting_sources=[] keeps the run hermetic (no workspace CLAUDE.md/settings),
    which also trims token overhead.
    """
    from claude_agent_sdk import (query, ClaudeAgentOptions, AssistantMessage,
                                  TextBlock, ResultMessage)
    opts = ClaudeAgentOptions(
        model=model,
        system_prompt=system,
        allowed_tools=[],
        setting_sources=[],
        max_turns=1,
        max_budget_usd=max_budget_usd,
    )
    chunks: list[str] = []
    cost = 0.0
    async for msg in query(prompt=prompt, options=opts):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    chunks.append(b.text)
        elif isinstance(msg, ResultMessage):
            cost = getattr(msg, "total_cost_usd", 0.0) or 0.0
    return "".join(chunks).strip(), cost
