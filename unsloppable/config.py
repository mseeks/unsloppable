"""Layer-1 scoring weights, externalized from the scoring logic.

These were magic numbers inline in `lint_text`. Pulling them into a dataclass with
JSON load/save means tuning is a config diff, not a code edit, and a future
auto-tuner can write weights back without touching logic. The defaults reproduce
the original linter's calibration (the empirically-tuned constants), now driven by
the shared feature registry and the fixed sentence splitter.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Weights:
    # em-dash density (suppressed in structurally varied prose)
    em_floor: float = 4.0
    em_slope: float = 1.0
    em_cap: float = 6.0
    em_cv_suppress_above: float = 0.7
    em_cv_suppress_factor: float = 0.25
    # smart quotes (per raw count)
    smart_slope: float = 0.2
    smart_cap: float = 2.0
    # AI vocabulary (require >=2 distinct terms)
    vocab_slope: float = 1.1
    vocab_cap: float = 8.0
    vocab_min_distinct: int = 2
    # "not just X but Y"
    not_just_slope: float = 1.0
    not_just_cap: float = 3.0
    # tricolon density
    tri_floor: float = 3.5
    tri_slope: float = 0.4
    tri_cap: float = 3.0
    tri_min_words: int = 80
    # "it is [adj]…" hedging
    hedge_slope: float = 0.8
    hedge_cap_short: float = 1.6
    hedge_cap_long: float = 2.4
    hedge_short_words: int = 300
    # definitional bridges ("in other words"…)
    bridge_floor: float = 2.0
    bridge_slope: float = 0.4
    bridge_cap: float = 1.5
    # sentence-length-variance ramp (the dominant signal)
    cv_t1: float = 0.35
    cv_p1: float = 4.0
    cv_t2: float = 0.45
    cv_p2: float = 2.5
    cv_t3: float = 0.55
    cv_p3: float = 1.5
    cv_t4: float = 0.65
    cv_p4: float = 0.75
    cv_t4_min_words: int = 80
    # verdict tiers (score -> label)
    tier_human: float = 1.5
    tier_light: float = 3.0
    tier_lean: float = 5.0

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Weights":
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


DEFAULT = Weights()
