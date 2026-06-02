"""Core unit tests — pure stdlib, no network, no models. Run offline:

    uv run --extra test python -m pytest tests/ -q
"""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None

from unsloppable import lint, score_text                       # noqa: E402
from unsloppable.combine import COMBINER_FEATURES, predict      # noqa: E402
from unsloppable.features import DIRECTION, FEATURE_NAMES, extract_features  # noqa: E402
from unsloppable.text import context, split_sentences           # noqa: E402


# --- sentence splitter (the bug-prone heart of the structural signal) --------

def test_splitter_handles_abbreviations_and_decimals():
    s = split_sentences("The U.S. economy grew 3.5 percent that year, Dr. Smith said. Wow.")
    assert s == ["The U.S. economy grew 3.5 percent that year, Dr. Smith said.", "Wow."]


def test_splitter_recovers_missing_space_after_period():
    s = split_sentences("Programs were cut.Victory came next.Furthermore, growth resumed.")
    assert len(s) == 3


def test_splitter_keeps_initials_together():
    s = split_sentences("George W. Bush spoke today. It rained.")
    assert s == ["George W. Bush spoke today.", "It rained."]


def test_splitter_single_sentence():
    assert len(split_sentences("Just one sentence here with no breaks")) == 1


# --- registry / features ------------------------------------------------------

def test_registry_directions_cover_all_features():
    assert set(DIRECTION) == set(FEATURE_NAMES)
    assert all(d in ("high=AI", "low=AI") for d in DIRECTION.values())


def test_extract_features_returns_all_names():
    f = extract_features("This is a normal sentence. Here is another one, slightly longer.")
    assert set(f) == set(FEATURE_NAMES)
    assert 0.0 <= f["type_token_ratio"] <= 1.0
    assert 0.0 <= f["function_word_ratio"] <= 1.0


def test_cv_undefined_is_nan_not_zero():
    # one sentence -> cv undefined -> NaN (not 0.0, which would read as max-AI)
    f = extract_features("A single run-on sentence with no internal breaks at all here")
    assert math.isnan(f["sentence_cv"])


def test_context_low_confidence_for_short_text():
    assert context("One. Two.").low_confidence is True
    assert context("One. Two. Three. Four. Five sentences here. Six now.").low_confidence is False


# --- heuristic scorer ---------------------------------------------------------

def test_heuristic_separates_obvious_cases():
    ai = ("In today's fast-paced world, it is important to leverage robust, scalable "
          "solutions. Moreover, we must delve into the ever-evolving landscape. Furthermore, "
          "this is a testament to innovation. In essence, we unlock seamless synergy.")
    human = ("the bus was late again so i just walked. took like 25 min. my feet hurt but "
             "whatever, the weather was nice and i grabbed a coffee which honestly helped a lot.")
    assert score_text(ai).score > score_text(human).score


def test_empty_text_is_safe():
    r = score_text("")
    assert r.score == 0.0 and r.low_confidence and r.word_count == 0


def test_result_top_issues_have_advice():
    r = lint("Leverage robust scalable synergy. Moreover, delve into the paradigm. "
             "Furthermore, unlock transformative value. It is important to note this.")
    issues = r.top_issues()
    assert issues and all(s.advice for s in issues)


# --- combiner inference (pure python; no training needed) --------------------

def test_combiner_predict_is_deterministic_and_bounded():
    # a synthetic 2-feature model
    model = {"features": ["a", "b"], "mean": [0.0, 0.0], "std": [1.0, 1.0],
             "coef": [2.0, -1.0], "intercept": 0.0, "impute": {}}
    p, contribs = predict({"a": 1.0, "b": 0.0}, model)
    assert 0.0 <= p <= 1.0
    assert len(contribs) == 2
    # higher 'a' (positive coef) raises probability
    p2, _ = predict({"a": 3.0, "b": 0.0}, model)
    assert p2 > p


def test_combiner_imputes_missing_feature():
    model = {"features": ["a", "b"], "mean": [0.0, 5.0], "std": [1.0, 1.0],
             "coef": [1.0, 1.0], "intercept": 0.0, "impute": {"b": 5.0}}
    p, _ = predict({"a": 0.0}, model)  # b missing -> imputed to 5.0 -> z=0
    assert abs(p - 0.5) < 1e-6


def test_combiner_feature_space_nonempty():
    assert len(COMBINER_FEATURES) > len(FEATURE_NAMES)  # includes heuristic_score + likelihood


# --- ml layer (skipped if torch absent) --------------------------------------

def test_to_dict_is_valid_json_even_with_nan_features():
    # short text -> sentence_cv is NaN; to_dict() must sanitize to null so strict
    # parsers (the agent self-check surface) don't choke on a bare NaN token.
    import json
    r = lint("A single run-on sentence with no internal breaks at all here")
    json.dumps(r.to_dict(), allow_nan=False)  # raises if any NaN/inf leaks through


def test_heuristic_signal_names_are_registry_names():
    # heuristic and ml paths must use the same signal vocabulary so agent code
    # keying on s.name doesn't break when ml is toggled.
    r = lint("Leverage robust scalable synergy. Moreover, delve into the paradigm. "
             "Furthermore, unlock transformative value. It is important to note this.")
    assert all(s.name in FEATURE_NAMES for s in r.signals)


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_likelihood_scorer_shape_and_empty_safe():
    from unsloppable.likelihood import FEATURES, LikelihoodScorer
    s = LikelihoodScorer()
    feats = s.score("This is a short test sentence for the scorer to read.")
    assert set(feats) == set(FEATURES)
    # empty / blank must not crash the forward pass — returns all-NaN
    batch = s.score_batch(["", "   ", "A real sentence here for scoring purposes."])
    assert len(batch) == 3 and all(set(b) == set(FEATURES) for b in batch)


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_ml_path_runs_and_short_text_is_uncertain():
    r = lint("It is crucial to leverage synergy now.", ml=True)  # 7 words
    assert "uncertain" in r.verdict  # too short to score reliably
    long = lint("The quarterly report shows steady growth across all regions. Revenue rose "
                "twelve percent while costs held flat. Management credits the new logistics "
                "system, though some analysts remain cautious about next year.", ml=True)
    assert long.probability is not None and 0.0 <= long.probability <= 1.0
