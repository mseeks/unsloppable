"""The feature registry — the single source of truth for *what we measure*.

Every feature is defined exactly once here as a normalized value extractor plus
metadata (direction + human-readable advice). The same registry feeds:
  * Layer 1's transparent additive score (heuristic.py),
  * the eval harness's per-feature AUC (no more duplicated per-1k math), and
  * the learned combiner's feature vector (combine.py),
so a feature can never silently mean three different things in three places.

Pure stdlib. `direction` says which way is AI ("high=AI" or "low=AI"); `advice`
is the actionable note surfaced to a revising agent when the feature leans AI.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from .text import Context, FUNCTION_WORDS

# --- lexical resources (ported from the original linter) ----------------------

AI_VOCAB = [
    "leverage", "leveraging", "leveraged",
    "robust", "scalable", "innovative", "cutting-edge", "state-of-the-art",
    "seamless", "seamlessly", "synergy", "synergistic",
    "passionate", "thrilled", "honored", "delighted",
    "spearhead", "spearheaded", "spearheading",
    "pivotal", "transformative", "transformational", "paradigm",
    "unlock", "unleash", "empower", "empowering",
    "deep dive", "delve", "delving", "navigate the landscape",
    "ever-evolving", "fast-paced", "dynamic environment",
    "moreover", "furthermore", "additionally", "in essence",
    "in today's", "in the realm of", "at the intersection of",
    "look forward to", "would welcome the opportunity",
    "excited to discuss", "i'd love to discuss",
    "tapestry", "underscore", "underscores", "underscoring",
    "testament to", "speaks volumes",
]

NOT_JUST_RE = re.compile(
    r"\bnot\s+(just|only|merely|simply)\b[^.!?]{1,80}\bbut\b", re.IGNORECASE)
TRICOLON_RE = re.compile(r"\b\w+,\s+\w+,\s+and\s+\w+\b", re.IGNORECASE)
HEDGE_RE = re.compile(
    r"\bit\s+is\s+(?:generally|always|usually|commonly|strongly|highly|often|"
    r"worth\s+noting\s+that|well[-\s]known\s+that|"
    r"important|recommended|advisable|crucial|essential|"
    r"safe|possible|common|normal|natural|typical)\b", re.IGNORECASE)
BRIDGE_RE = re.compile(
    r"\b(?:in\s+other\s+words|to\s+put\s+it\s+(?:simply|another\s+way)|"
    r"put\s+simply|more\s+specifically|that\s+is\s+to\s+say)\b", re.IGNORECASE)

EM_DASH = "—"
SMART_QUOTES = ["“", "”", "‘", "’"]
_VOCAB_RES = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in AI_VOCAB}


# --- raw extractors -----------------------------------------------------------

def _vocab_hits(ctx: Context) -> list[tuple[str, int]]:
    low = ctx.text.lower()
    out = []
    for term, rx in _VOCAB_RES.items():
        c = len(rx.findall(low))
        if c:
            out.append((term, c))
    return out


def _mattr(tokens: list[str], window: int = 40) -> float:
    """Moving-average type-token ratio — a length-robust lexical-diversity metric.

    Plain TTR falls mechanically as text lengthens, so it partly measures length;
    MATTR averages TTR over a sliding window and is stable across sample sizes —
    important here because corpus samples range ~50–400 words. For text shorter
    than the window we fall back to plain TTR.
    """
    n = len(tokens)
    if n == 0:
        return 1.0
    if n <= window:
        return len(set(tokens)) / n
    ratios = []
    for i in range(0, n - window + 1):
        ratios.append(len(set(tokens[i:i + window])) / window)
    return sum(ratios) / len(ratios)


@dataclass(frozen=True)
class Feature:
    name: str
    direction: str            # "high=AI" or "low=AI"
    extract: Callable[[Context], float]
    advice: str               # shown to a revising agent when this leans AI


# Ordered registry. `extract` returns the normalized value used everywhere.
REGISTRY: list[Feature] = [
    Feature("em_per_1k", "high=AI",
            lambda c: c.text.count(EM_DASH) * c.per_1k,
            "Heavy em-dash use reads as AI; replace some with commas, periods, or parentheses."),
    Feature("smart_quotes_per_1k", "high=AI",
            lambda c: sum(c.text.count(q) for q in SMART_QUOTES) * c.per_1k,
            "Typographic “smart” quotes suggest paste-from-formatted-source; use plain quotes."),
    Feature("vocab_per_1k", "high=AI",
            lambda c: sum(n for _, n in _vocab_hits(c)) * c.per_1k,
            "Marketing/abstract vocabulary (leverage, robust, delve, tapestry…) reads as AI; prefer concrete words."),
    Feature("vocab_distinct", "high=AI",
            lambda c: float(len(_vocab_hits(c))),
            "Multiple distinct AI-buzzwords; cut the clichés."),
    Feature("not_just_per_1k", "high=AI",
            lambda c: len(NOT_JUST_RE.findall(c.text)) * c.per_1k,
            "“Not just X but Y” is an overused AI construction; rephrase directly."),
    Feature("tricolon_per_1k", "high=AI",
            lambda c: len(TRICOLON_RE.findall(c.text)) * c.per_1k,
            "Frequent “X, Y, and Z” triads read as AI; vary list structure or drop some."),
    Feature("hedge_per_1k", "high=AI",
            lambda c: len(HEDGE_RE.findall(c.text)) * c.per_1k,
            "“It is important/worth noting that…” hedging is a strong AI tell; state it plainly."),
    Feature("bridge_per_1k", "high=AI",
            lambda c: len(BRIDGE_RE.findall(c.text)) * c.per_1k,
            "Reflexive restating (“in other words”, “put simply”) reads as AI; say it once."),
    Feature("sentence_cv", "low=AI",
            lambda c: c.cv if c.cv is not None else math.nan,
            "Sentence lengths are too uniform; mix short and long sentences for human rhythm."),
    Feature("type_token_ratio", "low=AI",
            lambda c: _mattr(c.tokens),
            "Low lexical diversity (repeated words); vary word choice."),
    Feature("function_word_ratio", "high=AI",
            lambda c: (sum(1 for t in c.tokens if t in FUNCTION_WORDS) / len(c.tokens))
            if c.tokens else 0.0,
            "Unusually high glue-word density; tighten phrasing."),
    Feature("mean_word_length", "high=AI",
            lambda c: (sum(len(t) for t in c.tokens) / len(c.tokens)) if c.tokens else 0.0,
            "Inflated average word length; prefer plainer, shorter words."),
    Feature("mean_sentence_length", "high=AI",
            lambda c: (sum(c.sent_lengths) / len(c.sent_lengths)) if c.sent_lengths else 0.0,
            "Long average sentence length; break up some sentences."),
    Feature("comma_per_1k", "high=AI",
            lambda c: c.text.count(",") * c.per_1k,
            "Dense comma use (sub-clauses) reads as AI; simplify some sentences."),
]

FEATURE_NAMES = [f.name for f in REGISTRY]
DIRECTION = {f.name: f.direction for f in REGISTRY}
ADVICE = {f.name: f.advice for f in REGISTRY}


def extract_features(text: str, ctx: Context | None = None) -> dict[str, float]:
    """Normalized feature values for one text — the shared vector."""
    c = ctx if ctx is not None else Context(text=text)
    return {f.name: float(f.extract(c)) for f in REGISTRY}


def vocab_hits(ctx: Context) -> list[tuple[str, int]]:
    """Exposed for the heuristic's human-readable note ('AI vocabulary: …')."""
    return _vocab_hits(ctx)
