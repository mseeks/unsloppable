"""unsloppable — a layered, programmatically-callable linter that flags prose
reading as AI-written, so an agent can check and revise its own output.

Layers:
  1. heuristic  — fast, pure-stdlib stylometric tells (always available)
  2. likelihood — perplexity / Binoculars-style signals from a small local model
                  (optional; needs the `ml` extra)
  combiner      — a calibrated logistic regression over both layers, returning a
                  probability + per-signal explanation (optional; needs `ml`).

Quick start:

    import unsloppable
    r = unsloppable.lint("Your text here…")        # heuristic-only by default
    print(r.score, r.verdict)
    for s in r.top_issues():                      # actionable, for revision
        print(s.name, "→", s.advice)

    # full layered verdict with a calibrated probability:
    r = unsloppable.lint("…", ml=True)
    print(r.probability)
"""
from __future__ import annotations

from .heuristic import lint_text, score_text, verdict
from .likelihood import LikelihoodScorer, default_scorer  # torch is lazy; safe to import
from .text import context
from .types import Result, Signal

__all__ = ["lint", "lint_text", "score_text", "verdict", "Result", "Signal", "context",
           "LikelihoodScorer", "default_scorer"]


def lint(text: str, *, ml: bool = False, model=None, scorer=None) -> Result:
    """Lint `text` and return a Result.

    ml=False (default): Layer-1 heuristic only — fast, no dependencies.
    ml=True: add the likelihood layer + learned combiner for a calibrated
             probability and a stronger verdict (requires the `ml` extra and a
             trained combiner). Falls back to heuristic-only with a note if the
             ML pieces aren't available (missing `torch`, missing combiner.json…).

    scorer: an optional pre-warmed LikelihoodScorer to reuse across calls; defaults
            to a process-wide singleton (models load once).
    """
    r = score_text(text)
    if not ml:
        return r
    try:
        # NB: torch is imported lazily inside the scorer and combiner.json is read
        # inside augment(), so the guard must wrap the CALL, not just the import.
        from .combine import augment
        return augment(r, text, model=model, scorer=scorer)
    except Exception as e:
        r.notes.append(f"ml=True requested but unavailable ({type(e).__name__}: "
                       f"{str(e)[:80]}); returning heuristic-only result")
        return r
