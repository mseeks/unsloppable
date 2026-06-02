"""The learned combiner — a calibrated logistic regression over both layers.

Layer 1 (stylometric) is era-locked; Layer 2 (likelihood) is model-relative and
generalizes better. The combiner learns how to weight them into one calibrated
P(AI), and—because it's linear over interpretable features—every verdict comes
with a sorted list of which signals pushed it toward AI and by how much, so a
revising agent gets actionable feedback, not a black-box number.

Training (eval/train_combiner.py) needs sklearn (the `ml` extra). Inference here
is pure-Python (a dot product) plus the likelihood scorer, so loading the combiner
doesn't pull sklearn. The model is a small JSON of coefficients + standardization.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .features import ADVICE, DIRECTION as REG_DIRECTION, FEATURE_NAMES
from .likelihood import DIRECTION as LL_DIRECTION
from .likelihood import FEATURES as LL_FEATURES
from .types import Result, Signal

MODEL_PATH = Path(__file__).resolve().parent / "combiner.json"
MIN_RELIABLE_WORDS = 20   # below this, even the likelihood layer is unreliable

# The combiner's feature space. We deliberately drop two redundancies the review
# surfaced: `heuristic_score` (a deterministic function of the stylometric features
# already in the vector — it double-counts and muddies the explanation) and
# `ll_perplexity` (corr -1.0 with ll_mean_logprob — collinear, and it flipped the
# coefficient sign so the per-signal advice could contradict the value).
COMBINER_FEATURES = list(FEATURE_NAMES) + [f for f in LL_FEATURES if f != "ll_perplexity"]

# Direction per feature, for deriving an honest "leaning" from the value itself
# (not from the LR coefficient sign, which collinearity can flip).
COMBINER_DIRECTION = {**REG_DIRECTION, **LL_DIRECTION}

_LL_ADVICE = {
    "ll_perplexity": "Text is unusually predictable to a language model (low perplexity); "
                     "add specific, surprising detail and vary phrasing.",
    "ll_mean_logprob": "Token choices are highly probable to a language model; be less generic.",
    "ll_surprisal_stdev": "Uniform token-level surprisal (low burstiness); human writing is uneven — "
                          "mix the expected with the unexpected.",
    "ll_binoculars": "Low Binoculars score — reads as machine-generated to a detector; "
                     "revise toward more idiosyncratic, specific phrasing.",
}
ADVICE_ALL = {**ADVICE, **_LL_ADVICE, "heuristic_score": "Multiple stylometric AI tells (see Layer 1)."}


def _sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def feature_vector(text: str, result: Result | None = None, ll: dict | None = None,
                   scorer=None) -> dict[str, float]:
    """Assemble the full combiner feature dict for one text.

    `result` (a Layer-1 Result) supplies stylometric values + heuristic_score;
    `ll` supplies cached likelihood features, else `scorer` computes them.
    """
    from .heuristic import score_text
    r = result if result is not None else score_text(text)
    vec: dict[str, float] = dict(r.features)
    vec["heuristic_score"] = float(r.score)
    if ll is None and scorer is not None:
        ll = scorer.score(text)
    if ll:
        vec.update({k: ll.get(k, math.nan) for k in LL_FEATURES})
    return vec


def load_model(path: Path | str = MODEL_PATH) -> dict:
    return json.loads(Path(path).read_text())


def predict(vec: dict[str, float], model: dict) -> tuple[float, list[tuple[str, float, float]]]:
    """Return (probability, [(feature, z, contribution)...]) for a feature dict.

    Standardized values are clipped to ±`clip` (default 6) so an out-of-distribution
    input (e.g. very short, buzzword-dense text with an extreme per-1k rate) can't
    make one feature's linear extrapolation dominate the verdict — the failure mode
    that otherwise flips the prediction. In-distribution this is a no-op (|z|≪6).
    """
    feats = model["features"]
    mean, std, coef = model["mean"], model["std"], model["coef"]
    impute = model.get("impute", {})
    clip = model.get("clip", 6.0)
    logit = model["intercept"]
    contribs = []
    for i, name in enumerate(feats):
        v = vec.get(name, math.nan)
        if v is None or math.isnan(v):
            v = impute.get(name, mean[i])
        z = (v - mean[i]) / std[i] if std[i] else 0.0
        z = max(-clip, min(clip, z))
        c = coef[i] * z
        logit += c
        contribs.append((name, z, c))
    return _sigmoid(logit), contribs


def _verdict_from_prob(p: float) -> str:
    if p < 0.2:
        return "human"
    if p < 0.5:
        return "mostly human, light tells"
    if p < 0.8:
        return "AI-leaning"
    return "reads as AI"


def augment(result: Result, text: str, model=None, scorer=None) -> Result:
    """Add Layer-2 + the calibrated probability to a Layer-1 Result.

    model: path or loaded dict (default: the bundled combiner.json).
    scorer: a LikelihoodScorer (default: the process singleton).
    """
    mdl = model if isinstance(model, dict) else load_model(model or MODEL_PATH)
    if scorer is None:
        from .likelihood import default_scorer
        scorer = default_scorer()
    ll = scorer.score(text)
    result.layers.update({k: round(v, 3) for k, v in ll.items() if not math.isnan(v)})

    vec = feature_vector(text, result=result, ll=ll)
    prob, contribs = predict(vec, mdl)
    result.probability = round(prob, 4)
    # The likelihood layer works on short multi-sentence text, so the ml verdict
    # keys on word count (too short for ANY layer), not the structural <4-sentence
    # low_confidence flag.
    if result.word_count < MIN_RELIABLE_WORDS:
        result.verdict = "uncertain (text too short to score reliably)"
    else:
        result.verdict = _verdict_from_prob(prob)

    # Leaning comes from the VALUE's direction (never contradicts the data); the LR
    # contribution magnitude only ranks how much the model leaned on it.
    sigs = []
    for name, z, c in contribs:
        direction = COMBINER_DIRECTION.get(name, "high=AI")
        on_ai_side = (z > 0.4 and direction == "high=AI") or (z < -0.4 and direction == "low=AI")
        leaning = "ai" if (on_ai_side and abs(c) > 0.05) else "neutral"
        sigs.append(Signal(name=name, value=round(vec.get(name, float("nan")), 3),
                           leaning=leaning, contribution=round(c, 3),
                           advice=ADVICE_ALL.get(name) if leaning == "ai" else None))
    result.signals = sigs
    result.notes.append(f"combiner P(AI)={prob:.2f} over {len(contribs)} features "
                        f"(model: {mdl.get('meta', {}).get('name', 'combiner')})")
    return result
