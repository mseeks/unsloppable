#!/usr/bin/env python3
"""Example: an agent checks and revises its own writing so it doesn't read as AI.

This is the intended use of unsloppable — a tool an agent calls in its writing
loop. The pattern:

    draft  ->  unsloppable.lint(draft)  ->  if flagged, feed the per-signal advice
    back to the model as revision instructions  ->  re-lint  ->  repeat until clean
    (or budget runs out).

`unsloppable.lint(text, ml=True)` returns a calibrated P(AI) and `r.top_issues()`
gives concrete, per-signal guidance ("sentence lengths too uniform", "marketing
vocabulary", "low Binoculars score") — exactly what a model needs to revise.

Run (uses your Claude subscription via the Agent SDK for the draft/revise calls):
    uv run --extra ml python examples/agent_selfcheck.py "explain how vaccines work"
    uv run --extra ml python examples/agent_selfcheck.py --no-ml --rounds 3 "..."
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import unsloppable  # noqa: E402


def verdict_line(r) -> str:
    p = f" · P(AI)={r.probability}" if r.probability is not None else ""
    return f"score {r.score} ({r.verdict}){p}"


def revision_instructions(r) -> str:
    issues = r.top_issues(6)
    if not issues:
        return ""
    bullets = "\n".join(f"- {s.advice}" for s in issues if s.advice)
    return ("Revise the text below so it reads as natural human writing, NOT as AI. "
            "Keep the meaning and approximate length. Specifically fix these:\n"
            f"{bullets}\n\nText to revise:\n")


async def draft_and_revise(instruction: str, rounds: int, use_ml: bool) -> None:
    from llm import generate_text  # eval/llm.py — Agent SDK on subscription auth

    system = ("You are a writing assistant. Produce only the requested text — no "
              "preamble, heading, markdown, or sign-off. Write the way you naturally would.")
    text, _ = await generate_text(instruction, system)
    for i in range(rounds + 1):
        r = unsloppable.lint(text, ml=use_ml)
        print(f"\n--- round {i} --- {verdict_line(r)}")
        print(text[:280] + ("…" if len(text) > 280 else ""))
        issues = r.top_issues(5)
        if not issues:
            print("  ✓ no AI-leaning signals — done.")
            return
        print("  issues:", ", ".join(s.name for s in issues))
        if i == rounds:
            print("  (reached round limit)")
            return
        instr = revision_instructions(r) + text
        text, _ = await generate_text(instr, system)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instruction", help="what to write, e.g. 'explain how GPS works'")
    ap.add_argument("--rounds", type=int, default=2, help="max revision rounds")
    ap.add_argument("--no-ml", action="store_true",
                    help="heuristic only (fast, no model); default uses the likelihood layer")
    args = ap.parse_args(argv)
    asyncio.run(draft_and_revise(args.instruction, args.rounds, use_ml=not args.no_ml))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
