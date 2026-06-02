# unsloppable

A **layered, programmatically-callable linter that flags prose reading as
AI-written** — built so an agent can check and revise its own output before
shipping it. It pairs fast stylometric heuristics with model-relative likelihood
signals (perplexity / Binoculars), combines them into a calibrated probability,
and returns **per-signal, actionable advice** ("sentence lengths too uniform",
"marketing vocabulary", "low Binoculars score") — not just a number.

It is a **writing-quality tool, not a forensic detector.** A false positive costs
nothing (it's advice you can ignore); it must never be used to accuse a person of
using AI. See [Honesty notes](#honesty-notes).

```python
import unsloppable

r = unsloppable.lint(draft)            # fast: stdlib heuristic layer only
print(r.score, r.verdict)
for s in r.top_issues():             # what to fix, for a revising agent
    print(s.name, "→", s.advice)

r = unsloppable.lint(draft, ml=True)   # add the likelihood layer + calibrated P(AI)
print(r.probability)                 # e.g. 0.93
```

```sh
pip install unsloppable                  # zero-dependency CLI (Layer 1)
unsloppable draft.md                      # CI/pre-commit gate (exit 1 if flagged)
echo "your text" | unsloppable --json -   # machine-readable, for an agent
unsloppable --json --ml draft.md          # add the likelihood layer + P(AI)  (pip install 'unsloppable[ml]')
```

## Why layered

Surface stylometric tells (em-dashes, "delve", tricolons, sentence uniformity) are
**era-locked**: they were tuned on 2022 ChatGPT and largely stop working on current
models. We measured it on a modern multi-model corpus (RAID):

| signal | HC3 (2022) | RAID (2023–24 models) |
|---|---|---|
| stylometric heuristic (Layer 1) | 0.84 AUC | **0.61 AUC** (barely works) |
| perplexity / burstiness (Layer 2) | 0.99 | **0.84 / 0.87** |
| Binoculars cross-perplexity (Layer 2) | 0.93 | **0.82** |

Likelihood signals are *model-relative*, not lexical, so they generalize across
model eras far better. The two layers are complementary, and the learned combiner
weights them.

## Architecture

```
unsloppable/                  the linter package (importable + `python -m unsloppable`)
  text.py        shared parse Context + a correct sentence splitter
  features.py    the FEATURE REGISTRY — one definition per signal (value/direction/advice),
                 consumed by the linter, the harness, and the combiner
  config.py      Layer-1 weights, externalized to a JSON-loadable dataclass
  heuristic.py   Layer 1 — transparent additive stylometric score (pure stdlib)
  likelihood.py  Layer 2 — perplexity / burstiness / Binoculars from two small local
                 models on MPS/CUDA/CPU  (needs the `ml` extra)
  combine.py     the learned combiner — calibrated LR over both layers; pure-Python
                 inference + sorted per-signal contributions as the explanation
  cli.py         argparse CLI: stdin, --json, --ml, --threshold, fail-the-build exit code
eval/
  fetch_hc3.py        labeled corpus from HC3 (2022 human vs ChatGPT)
  fetch_raid.py       labeled modern multi-model corpus from RAID (MIT, download-on-demand)
  generate.py         fresh current-model AI via the Claude Agent SDK (the 2026 era anchor)
  score_likelihood.py cache Layer-2 features per row (expensive; resumable)
  evaluate.py         AUC + bootstrap CI + within-question paired AUC + per-source/era AUC
  compare_sources.py  the drift view: separation per source/model/era
  train_combiner.py   fit + evaluate the combiner; writes unsloppable/combiner.json
  analyze.py          advisory agent (read-only): proposes evidence-grounded fixes
tests/                offline stdlib tests (splitter, registry, heuristic, combiner)
corpus/               generated/downloaded (git-ignored): hc3 · raid · generated · scores
```

## Setup

[uv](https://docs.astral.sh/uv/). The core linter + harness are **pure stdlib**.
The likelihood layer and the combiner live behind the `ml` extra.

```sh
uv sync                 # core CLI — pure stdlib, zero dependencies
uv sync --extra ml      # + torch / transformers / scikit-learn / numpy (Layer 2 + training)
uv sync --extra gen     # + claude-agent-sdk (fresh-corpus generation)
```

## The loop

The eval **is** the test: measure how well a score separates known-AI from
known-human, change the model, re-measure on a held-out, grouped split.

```sh
# 1. build corpora (provenance-tagged JSONL; harness is source-agnostic)
uv run python eval/fetch_hc3.py
uv run python eval/fetch_raid.py
uv run --extra gen python eval/generate.py --genres qa --n 80   # current-model anchor (subscription)

# 2. cache Layer-2 likelihood features, then train + evaluate the combiner
uv run --extra ml python eval/score_likelihood.py --corpus corpus/hc3.jsonl corpus/raid.jsonl corpus/generated.jsonl --balance 900
uv run --extra ml python eval/train_combiner.py  --corpus corpus/hc3.jsonl corpus/raid.jsonl corpus/generated.jsonl

# 3. read the numbers (held-out, bootstrap CI, per-era drift)
uv run python eval/evaluate.py --corpus corpus/hc3.jsonl corpus/raid.jsonl --scores corpus/scores.jsonl --split test
uv run python eval/compare_sources.py
```

## Results

_Held-out, **grouped-by-question** split (no topic leakage). Measured on the 3,406
rows with cached Layer-2 features — full HC3 (1,476) + a balanced 900/900 RAID sample
+ 130 current-Claude — held-out test n=1,017; combiner and heuristic on the identical
rows. Reproduce with `eval/train_combiner.py`; bootstrap 95% CIs._

**The combiner is the tool.** Held-out ROC AUC:

| | overall | HC3 (2022) | RAID (2023–24 modern) |
|---|---|---|---|
| **combiner (both layers)** | **0.931** `[0.914, 0.946]` | **0.985** | **0.883** |
| heuristic alone (Layer 1) | 0.73 | 0.84 | **0.61** |

Layer 1 collapses on modern text (0.61); the combiner recovers it to ~0.88 by leaning
on the likelihood layer. Per-signal AUC (measured only on rows where the feature is
genuinely present — never on imputed rows):

| signal | layer | overall AUC | HC3 | modern RAID |
|---|---|---|---|---|
| surprisal burstiness | 2 | 0.91 | 0.99 | 0.87 |
| perplexity / mean-logprob | 2 | 0.89 | 0.99 | 0.84 |
| Binoculars cross-perplexity | 2 | 0.87 | 0.93 | 0.82 |
| sentence-length variance (cv) | 1 | 0.70 | — | 0.64 |
| type-token ratio, vocab, tricolon… | 1 | ≤0.65 | — | ≈0.5 |

The combiner leans hardest on the likelihood features; stylometry adds complementary
signal. Because the likelihood features are highly collinear, the LR's per-feature
coefficient *signs* aren't individually interpretable (the ensemble is) — read each
feature's direction from its standalone AUC, not its coefficient. Notably, **current
Claude output trips the *stylometric* layer hardest** (AUC 0.89 on the generated set)
while base models like mpt/mistral are caught mainly by likelihood — different model
families fall to different layers, and the combiner covers both. Full per-model/per-era
breakdown: `eval/report/generation_compare.md`. (Heuristic-alone on the *full* 8.7k-row
corpus is ~0.69; the 0.73 above is on the balanced Layer-2 subset.)

## Honesty notes

- **Style tool, not a forensic detector.** Detection of an unknown author is a
  losing treadmill (light paraphrasing collapses surface signals; false positives
  fall hardest on non-native English writers — Liang et al. 2023). This tool is for
  the *opposite* setting: an author (your agent) checking *its own* output, where a
  false positive just prompts an optional revision. Don't use it to accuse anyone.
- **The harness keeps it honest.** The headline AUC fell from a previously-reported
  0.90 to ~0.83 on HC3 once we fixed a metric-inflating sentence-splitter bug and
  removed topic leakage. We kept the correct splitter and the lower number.
- **Era drift is real and measured.** The stylometric layer collapses on modern
  models (0.65 on RAID); the likelihood layer is what makes the tool work on current
  output. Provenance (model/era) is tracked so drift stays visible.
- **Scorer-model caveat.** Likelihood is measured under a small local model; it's a
  relative signal, not a calibrated-against-the-world probability. The combiner's
  P(AI) is calibrated to the training mix and is base-rate dependent.
- **Corpora aren't redistributed.** `fetch_*.py` download on demand; `corpus/` is
  git-ignored. RAID is MIT; HC3 per its dataset terms.

## Roadmap

- **Auto-tune** the Layer-1 weights / combiner, gated on held-out AUC + an
  adversarial-robustness metric (not on a single corpus's number).
- **More eras / models** as they ship (the JSONL schema is provenance-ready).
- **Lighter Layer-2 backend** (Ollama logprobs / a distilled scorer) so the agent
  path needs no torch.

[hc3]: https://huggingface.co/datasets/Hello-SimpleAI/HC3
[raid]: https://github.com/liamdugan/raid
