#!/usr/bin/env python3
"""Phase 3 — advisory agent. Reads the eval results + linter source, proposes fixes.

A Claude Agent SDK agent (read-only: Read/Grep/Glob) opens the evaluation report,
the misclassified examples, and the linter itself, then writes an evidence-grounded,
prioritized improvement plan to eval/report/advice.md. It does NOT edit code — you
review the advice and apply what you agree with, then re-run evaluate.py to confirm
the metric moved. (Auto-tune mode — let the agent edit + gate on held-out AUC — is
the documented next step; this is the human-in-the-loop version.)

Runs on your Claude subscription via the Agent SDK (see llm.py).

Usage:
    uv run python eval/evaluate.py      # make sure reports are fresh first
    uv run python eval/analyze.py
    uv run python eval/analyze.py --model claude-opus-4-7
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from llm import auth_note, ANALYZE_MODEL

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "eval" / "report"
ADVICE_PATH = REPORT_DIR / "advice.md"

SYSTEM = (
    "You are a meticulous NLP engineer improving a heuristic AI-text detector. "
    "You reason strictly from empirical evidence — AUC numbers, score "
    "distributions, and concrete misclassified examples — never from vibes. You "
    "propose specific, minimal, testable changes, cite the evidence for each, and "
    "are candid about false-positive risk. You never sacrifice genuine human "
    "writing just to catch more AI."
)

TASK = """\
You are improving the heuristic linter in this repository. It scores text for \
"AI-slop" tells; a higher score means more AI-like. An empirical harness scores a \
labeled corpus (HC3: real human answers vs ChatGPT answers to the same questions) \
and measures how well the linter's score separates AI from human.

Your current working directory IS the repository root. Read exactly these relative \
paths — do not guess other locations:
1. `unsloppable/` — the linter package: `heuristic.py` (Layer 1 additive score), \
`features.py` (the shared feature registry), `likelihood.py` (Layer 2 perplexity/ \
Binoculars), `combine.py` (the learned combiner). Note every signal and how it feeds the score.
2. `eval/report/report.md` — latest eval: overall AUC, **per-feature AUC**, \
per-domain AUC, accuracy/precision/recall/F1 at thresholds, and score distributions.
3. `eval/report/misclassified.md` — concrete false positives (human flagged as AI) \
and false negatives (AI that slipped through), with text snippets.
4. `eval/report/generation_compare.md` (if present) — the same linter scored across \
sources/eras: HC3 human, HC3 ai (2022 ChatGPT), and fresh current-model ai. CRUCIAL: \
a feature that looks like dead weight on HC3 may fire strongly on a current model, so \
do NOT recommend pruning a feature using HC3 alone — check it against this file first.

Then write a prioritized improvement plan. Ground EVERY recommendation in the data \
you just read: cite the specific AUC numbers and quote misclassified snippets as \
evidence. Cover:

- **Threshold calibration** — compare the score→verdict cutoffs in the linter to the \
actual per-class score distributions. Is the binary cut mis-placed for this corpus? \
Suggest concrete numbers, and note that short-form text scores lower than long-form.
- **Dead-weight features** — which heuristics have per-feature AUC near 0.5 here and \
add little? Should they be down-weighted, dropped, or made conditional on genre/length \
rather than removed (they may matter on other genres)?
- **Strongest signal** — which feature discriminates best, and should scoring lean on \
it more heavily?
- **False positives** — what genuine-human patterns trip the linter? Quote 1-2 \
snippets and propose fixes that cut FPs without gutting recall.
- **False negatives** — what AI text slips through? Quote snippets and propose NEW \
candidate signals: describe the detectable pattern and a rough regex or metric.
- **Scoring / normalization** — any length-sensitivity or weighting issues.

For each recommendation give: the change, the evidence, the expected effect on \
AUC/accuracy, and the false-positive risk. Order by impact-to-effort. Output ONE \
well-structured markdown document (start with a 3-bullet executive summary). Do NOT \
edit any code — this is advisory. End with a short "suggested next experiments" list. \
Respond with ONLY the markdown document — no conversational preamble or closing remarks.
"""


async def run(model: str) -> tuple[str, float]:
    from claude_agent_sdk import (query, ClaudeAgentOptions, AssistantMessage,
                                  TextBlock, ToolUseBlock, ResultMessage)
    opts = ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM,
        allowed_tools=["Read", "Grep", "Glob"],    # read-only allowlist
        disallowed_tools=["Write", "Edit", "NotebookEdit", "Bash"],  # hard block
        setting_sources=[],                        # clean analyst, no workspace CLAUDE.md
        cwd=str(ROOT),
        permission_mode="default",
        max_turns=40,
    )
    final, last_text, cost = None, "", 0.0
    async for msg in query(prompt=TASK, options=opts):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    last_text = b.text
                elif isinstance(b, ToolUseBlock):
                    tgt = (b.input or {}).get("file_path") or (b.input or {}).get("pattern") or ""
                    print(f"  · {b.name} {tgt}", file=sys.stderr)
        elif isinstance(msg, ResultMessage):
            cost = getattr(msg, "total_cost_usd", 0.0) or 0.0
            final = getattr(msg, "result", None)
    return (final or last_text or "").strip(), cost


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=ANALYZE_MODEL)
    args = ap.parse_args(argv)

    report = REPORT_DIR / "report.md"
    if not report.exists():
        print("no eval/report/report.md — run `uv run python eval/evaluate.py` first",
              file=sys.stderr)
        return 1

    print(auth_note(), file=sys.stderr)
    print(f"advisory agent ({args.model}) reading reports + linter …", file=sys.stderr)
    advice, cost = asyncio.run(run(args.model))
    if not advice:
        print("agent returned no advice", file=sys.stderr)
        return 1

    ADVICE_PATH.write_text(advice)
    print(f"\nwrote advice → {ADVICE_PATH.relative_to(ROOT)}  (${cost:.3f})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
