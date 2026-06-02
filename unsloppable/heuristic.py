"""Layer 1 — the transparent, dependency-free heuristic scorer.

A faithful, debugged version of the original additive linter: same empirically
tuned constants (now in config.Weights), the same tells, but driven by the shared
feature registry + the fixed sentence splitter, and with the two correctness bugs
removed (abbreviation over-splitting; the cv=1.0 "looks human" fallback for short
text is now an explicit low-confidence state).

This is the fast path: pure stdlib, runs anywhere, fully explainable. The learned
combiner (combine.py) layers likelihood signals on top for the reliable verdict.
"""
from __future__ import annotations

from .config import DEFAULT, Weights
from .features import (BRIDGE_RE, EM_DASH, HEDGE_RE, NOT_JUST_RE, SMART_QUOTES,
                       TRICOLON_RE, extract_features, vocab_hits)
from .text import Context, context
from .types import Result, Signal


def _verdict(score: float, w: Weights) -> str:
    if score < w.tier_human:
        return "human"
    if score < w.tier_light:
        return "mostly human, light tells"
    if score < w.tier_lean:
        return "AI-leaning"
    return "reads as AI"


def score_text(text: str, ctx: Context | None = None, w: Weights = DEFAULT) -> Result:
    c = ctx if ctx is not None else context(text)
    wc = c.word_count
    feats = extract_features(text, c)

    if wc == 0:
        return Result(score=0.0, verdict="human", low_confidence=True, word_count=0,
                      features=feats, notes=["empty text"])

    per_1k = c.per_1k
    cv = c.cv
    signals: list[Signal] = []
    notes: list[str] = []
    score = 0.0

    def add(name: str, value: float, pts: float, advice: str | None,
            note: str | None = None):
        nonlocal score
        score += pts
        signals.append(Signal(name=name, value=round(value, 3),
                              leaning="ai" if pts > 0 else "neutral",
                              contribution=round(pts, 2), advice=advice if pts > 0 else None))
        if note:
            notes.append(note)

    # --- em-dash density (suppressed when prose is structurally varied) -------
    em = c.text.count(EM_DASH)
    em_1k = em * per_1k
    if em_1k > w.em_floor:
        pts = min((em_1k - w.em_floor) * w.em_slope, w.em_cap)
        if cv is not None and cv >= w.em_cv_suppress_above:
            pts *= w.em_cv_suppress_factor
        add("em_per_1k", em_1k, pts,
            "Heavy em-dash use reads as AI; vary punctuation.",
            f"em-dash density high: {em} in {wc} words ({em_1k:.1f}/1k)")

    # --- smart quotes ---------------------------------------------------------
    sq = sum(c.text.count(q) for q in SMART_QUOTES)
    if sq:
        add("smart_quotes_per_1k", float(sq), min(sq * w.smart_slope, w.smart_cap),
            "Typographic “smart” quotes; use plain quotes.",
            f"smart quotes: {sq}")

    # --- AI vocabulary (require >=2 distinct terms) --------------------------
    hits = vocab_hits(c)
    vtotal = sum(n for _, n in hits)
    if vtotal and len(hits) >= w.vocab_min_distinct:
        sample = ", ".join(f"{t}×{n}" for t, n in sorted(hits, key=lambda x: -x[1])[:5])
        add("vocab_per_1k", float(vtotal), min(vtotal * w.vocab_slope, w.vocab_cap),
            "Marketing/abstract vocabulary; prefer concrete words.",
            f"AI vocabulary: {sample}")

    # --- "not just X but Y" ---------------------------------------------------
    nj = len(NOT_JUST_RE.findall(c.text))
    if nj:
        add("not_just_per_1k", float(nj), min(nj * w.not_just_slope, w.not_just_cap),
            "“Not just X but Y” construction; rephrase.",
            f"'not just/only X but Y': {nj}")

    # --- tricolon density -----------------------------------------------------
    tri = len(TRICOLON_RE.findall(c.text))
    tri_1k = tri * per_1k
    if tri_1k > w.tri_floor and wc >= w.tri_min_words:
        add("tricolon_per_1k", tri_1k, min((tri_1k - w.tri_floor) * w.tri_slope, w.tri_cap),
            "Frequent “X, Y, and Z” triads; vary list structure.",
            f"tricolon density: {tri} ({tri_1k:.1f}/1k)")

    # --- "it is [adj]…" hedging ----------------------------------------------
    hedge = len(HEDGE_RE.findall(c.text))
    if hedge:
        cap = w.hedge_cap_short if wc < w.hedge_short_words else w.hedge_cap_long
        add("hedge_per_1k", float(hedge), min(hedge * w.hedge_slope, cap),
            "“It is important/worth noting that…” hedging; state it plainly.",
            f"hedge phrasing ('it is [adj]…'): {hedge}")

    # --- definitional bridges -------------------------------------------------
    br = len(BRIDGE_RE.findall(c.text))
    br_1k = br * per_1k
    if br_1k > w.bridge_floor:
        add("bridge_per_1k", br_1k, min((br_1k - w.bridge_floor) * w.bridge_slope, w.bridge_cap),
            "Reflexive restating (“in other words”…); say it once.",
            f"definitional bridges: {br}")

    # --- sentence-length-variance ramp (the dominant signal) ------------------
    if cv is not None:
        if cv < w.cv_t1:
            pts = w.cv_p1
        elif cv < w.cv_t2:
            pts = w.cv_p2
        elif cv < w.cv_t3:
            pts = w.cv_p3
        elif cv < w.cv_t4 and wc >= w.cv_t4_min_words:
            pts = w.cv_p4
        else:
            pts = 0.0
        if pts:
            add("sentence_cv", cv, pts,
                "Sentence lengths are too uniform; mix short and long sentences.",
                f"low sentence-length variance (cv={cv:.2f})")

    verdict = _verdict(score, w)
    if c.low_confidence:
        notes.append("low confidence: too few sentences to trust structural signals")
    return Result(score=round(score, 1), verdict=verdict,
                  low_confidence=c.low_confidence, word_count=wc,
                  features=feats, signals=signals, layers={"heuristic": round(score, 1)},
                  notes=notes)


def lint_text(text: str) -> dict:
    """Dict-returning shim for the eval harness and back-compat.

    Returns the Layer-1 score plus the full normalized feature vector, so the
    harness has one place to read both the composite score and per-feature values.
    """
    r = score_text(text)
    out = {"score": r.score, "verdict": r.verdict, "word_count": r.word_count,
           "low_confidence": r.low_confidence, "notes": r.notes}
    out.update(r.features)            # all normalized feature values
    return out


def verdict(score: float, w: Weights = DEFAULT) -> str:
    return _verdict(score, w)
