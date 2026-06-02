"""Shared text utilities: tokenization, sentence splitting, and a precomputed
Context object so every feature extractor draws from the same parse instead of
re-splitting/re-tokenizing the text.

Pure stdlib. This is Layer 1's foundation and the harness's single source of truth
for "how the text was parsed", so a splitter fix changes the linter and the eval
in lockstep.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from functools import cached_property

# --- sentence splitting -------------------------------------------------------

# Abbreviations that take a trailing period mid-sentence. The old splitter
# (`re.split(r"(?<=[.!?])\s+")`) over-split on these and on initials, corrupting
# the sentence-length-variance signal — and because HC3 humans used abbreviations
# ~6x more than 2022 ChatGPT, that bug LEAKED a label-correlated artifact into the
# headline AUC. We split conservatively instead (see split_sentences).
_ABBREV = frozenset("""
mr mrs ms dr prof sr jr st vs etc al ca cf eg ie eg. ie. dept est gen gov rep sen
inc ltd co corp no vol fig pp ed eds rev approx min max ph sc ba ma md phd
jan feb mar apr jun jul aug sep sept oct nov dec
mon tue wed thu fri sat sun
a.m p.m u.s u.k u.n e.g i.e a.d b.c
""".split())

_CLOSERS = "\"”’')]"  # closing quotes/brackets that belong to the sentence being ended

_PUNCT_RUN = re.compile(r"[.!?]+")


def split_sentences(text: str) -> list[str]:
    """Conservative sentence splitter.

    Splits on runs of .!? at a sentence boundary, NOT when the boundary is:
      * inside a decimal ("3.5") or acronym-without-space ("U.S.A") — the char
        after the period is a digit/lowercase, never an uppercase sentence start;
      * after a known abbreviation token ("Dr.", "etc.", "U.S.");
      * after a single-letter initial ("George W. Bush");
      * before a lowercase word (real sentences start uppercase) — one rule that
        catches most mid-sentence abbreviations regardless of the abbrev list.

    A boundary is taken either before whitespace+Uppercase ("end. Next") OR
    directly before an uppercase letter with NO space ("end.Next") — the latter
    handles a common scrape artifact where the period-space was lost, which would
    otherwise collapse a multi-sentence passage into one and destroy its
    sentence-length-variance signal.
    """
    pieces: list[str] = []
    start = 0
    for m in _PUNCT_RUN.finditer(text):
        i, j = m.start(), m.end()
        # absorb trailing closing quotes/brackets into the sentence being ended,
        # so '… left." Then …' splits correctly.
        e = j
        while e < len(text) and text[e] in _CLOSERS:
            e += 1
        if e < len(text) and not text[e].isspace():
            # no-space boundary only if next char is an uppercase letter
            if not (text[e].isalpha() and text[e].isupper()):
                continue  # "3.5", "foo.bar", "U.S.a" → not a boundary
            nxt = text[e]
        else:
            k = e
            while k < len(text) and text[k].isspace():
                k += 1
            nxt = text[k] if k < len(text) else ""
        p = i
        while p > 0 and (text[p - 1].isalnum() or text[p - 1] in ".'’"):
            p -= 1
        prev_tok = text[p:i].strip(".'’").lower()
        if nxt and nxt.islower():
            continue  # mid-sentence abbreviation: "...U.S. economy grew..."
        if prev_tok in _ABBREV:
            continue
        if len(prev_tok) == 1 and prev_tok.isalpha():
            continue  # initial: "W."
        piece = text[start:e].strip()
        if piece:
            pieces.append(piece)
        start = e
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


# --- tokenization -------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # words, keeping contractions

# A standard closed-class function-word list (articles, prepositions, pronouns,
# conjunctions, auxiliaries, particles). Used for the function-word-ratio feature
# — a classic, authorship/register-robust stylometric signal.
FUNCTION_WORDS = frozenset("""
the a an this that these those some any no every each either neither
i me my mine we us our ours you your yours he him his she her hers it its
they them their theirs who whom whose which what
is am are was were be been being have has had do does did
will would shall should can could may might must ought need dare
and or but nor for yet so as if then than because while although though
unless until whether since whereas
of to in on at by from with about against between into through during before after
above below under over up down out off near upon among around across behind beyond
not no none never very too also just only even still yet again ever almost
i'm you're he's she's it's we're they're i've you've we've they've i'd you'd
""".split())


@dataclass
class Context:
    """Everything a feature might need, parsed once."""
    text: str

    @cached_property
    def words(self) -> list[str]:
        return self.text.split()

    @cached_property
    def word_count(self) -> int:
        return len(self.words)

    @cached_property
    def tokens(self) -> list[str]:
        """Lowercased alphabetic tokens (contractions kept)."""
        return [t.lower() for t in _WORD.findall(self.text)]

    @cached_property
    def sentences(self) -> list[str]:
        return split_sentences(self.text)

    @cached_property
    def sent_lengths(self) -> list[int]:
        return [len(s.split()) for s in self.sentences if s.split()]

    @cached_property
    def sentence_count(self) -> int:
        return len(self.sent_lengths)

    @cached_property
    def cv(self) -> float | None:
        """Coefficient of variation of sentence length. None if undefined
        (need >=2 sentences). `low_confidence` flags <4 sentences separately —
        we no longer fake cv=1.0 ('looks human') for short text."""
        lengths = self.sent_lengths
        if len(lengths) < 2:
            return None
        mean = statistics.mean(lengths)
        if not mean:
            return None
        return statistics.pstdev(lengths) / mean

    @cached_property
    def low_confidence(self) -> bool:
        return self.sentence_count < 4

    @property
    def per_1k(self) -> float:
        return 1000.0 / self.word_count if self.word_count else 0.0


def context(text: str) -> Context:
    return Context(text=text)
