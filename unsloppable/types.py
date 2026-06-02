"""Public result types — what a calling agent receives.

The whole point of the tool is *actionable* output: not just a number, but which
signals leaned AI and what to do about each, so a revising agent can fix the
specific problems. `Result.top_issues()` is the surface built for that loop.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


def _fix(v):
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {k: _fix(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_fix(x) for x in v]
    return v


def _json_safe(d: dict) -> dict:
    """Recursively replace NaN/inf with None so output is valid JSON (the spec has
    no NaN token; bare NaN crashes non-Python parsers — and the CLI is the agent
    self-check surface)."""
    return {k: _fix(v) for k, v in d.items()}


@dataclass
class Signal:
    name: str
    value: float
    leaning: str          # "ai" | "neutral"
    contribution: float   # points toward AI (Layer 1) / coef·z (combiner)
    advice: str | None = None

    def to_dict(self) -> dict:
        return _json_safe(asdict(self))


@dataclass
class Result:
    score: float                       # Layer-1 additive score (interpretable, 0..~15)
    verdict: str                       # human-readable label
    probability: float | None = None   # calibrated P(AI) from the combiner, if available
    low_confidence: bool = False       # too short to trust the structural signals
    word_count: int = 0
    features: dict[str, float] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    layers: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def top_issues(self, k: int = 5) -> list[Signal]:
        """The strongest AI-leaning signals, with advice — for agent revision."""
        ai = [s for s in self.signals if s.leaning == "ai"]
        ai.sort(key=lambda s: -abs(s.contribution))
        return ai[:k]

    def to_dict(self) -> dict:
        d = _json_safe(asdict(self))
        d["top_issues"] = [s.to_dict() for s in self.top_issues()]
        return d
