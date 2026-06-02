# unsloppable — findings & next steps

_Status: 2026-06-02. The plan below has now largely been **implemented** — see the
"Implemented" box. The analysis that motivated it is kept as the rationale._

> ## Implemented (2026-06-02)
> The findings here drove a rebuild into a **layered, calibrated, agent-callable
> detector**. What shipped:
> - **Package + registry + debugged harness** — `unsloppable/` with one feature source
>   of truth; sentence-splitter + cv bugs fixed (HC3 AUC honestly fell 0.90→0.83);
>   bootstrap CIs, within-question paired AUC, group-by-question split, train-only
>   threshold selection.
> - **Modernized corpus** — HC3 (2022) + **RAID** (7.2k rows, 10 models × 6 genres,
>   MIT) + 130 current-Claude samples, all provenance-tagged. This exposed the
>   measured era-drift: the stylometric heuristic drops to **0.65 AUC on modern text**.
> - **Layer 2 (likelihood)** — perplexity / burstiness / Binoculars from two
>   Qwen2.5-0.5B models on MPS; **~0.90 AUC on modern RAID** where the heuristic fails.
> - **Learned combiner** — calibrated LR over both layers (±6 clipped, interpretable),
>   held-out **AUC 0.931** [0.914, 0.946] (HC3 0.985 / modern-RAID 0.883) vs the
>   heuristic's 0.73; ships per-signal revision advice. Hardened against an
>   adversarial review (empty-input/fallback/JSON/collinearity bugs all fixed).
> - **Agent interface** — `unsloppable.lint(text, ml=True)` → `Result.top_issues()`, a
>   CI-gate CLI (`python -m unsloppable`), and `examples/agent_selfcheck.py` (the
>   draft→lint→revise loop). 15 offline tests.
>
> Still open (see §4–§6): lighter Layer-2 backend (no torch), auto-tune gated on a
> held-out + adversarial metric, more model eras, precision-at-low-base-rate.

_Original analysis (2026-06-01). Every number below was measured on the committed
harness (HC3, seed 13) and cross-checked by independent reviews._

## TL;DR

1. **The linter works, but for a narrower reason than it looks.** The 0.896 held-out AUC is
   real, but it's essentially a **two-feature model** (`sentence_cv` does ~78% of the work,
   `tricolon` adds a small-but-real increment) with **four dead features** (`em_dash`,
   `smart_quotes`, `ai_vocab`, `not_just` all sit at AUC ≈ 0.50 on HC3). The headline isn't a
   six-signal detector — it's a sentence-length-uniformity detector with garnishes.
2. **The real artifact is the eval harness, and the honest framing is "slop/style linter," not
   "forensic detector."** The harness is senior-grade work *because* it falsifies its own
   linter. As a forensic AI detector this is a treadmill it loses (paraphrase breaks it; HC3 is
   a dead 2022 distribution; false positives concentrate on a register, not on authorship).
   That self-falsification is the portfolio signal — lead with it.
3. **The order of operations is causal, not preferential.** Cheap honesty/correctness fixes
   first (they make every later number trustworthy *and* are the differentiator), then the one
   measured capability win (new stylometric features), then modernize the corpus, and only then
   the ML model — shipped as an **interpretable** logistic regression, not a black box. The
   transformer and the "auto-tune the weights" item are measured low-leverage and are
   explicitly deprioritized.

---

## 1. What the harness actually shows (verified)

Run `python3 eval/fetch_hc3.py && python3 eval/evaluate.py --split test` to reproduce the
baseline. Key measured facts:

| Fact | Number | How measured |
|---|---|---|
| Held-out test AUC | **0.896** (n=442); full corpus 0.891; 5-seed mean 0.889 ± 0.010 | `evaluate.py --split test` |
| Bootstrap 95% CI on test AUC | **[0.866, 0.923]** (width 0.057) | 2000-rep percentile bootstrap |
| `sentence_cv` per-feature AUC | **0.874** (the workhorse) | per-feature AUC table |
| `tricolon` per-feature AUC | 0.65 | per-feature AUC table |
| `em_dash`, `smart_quotes`, `ai_vocab`, `not_just` | **≈ 0.49–0.50 (dead)** | per-feature AUC table |
| Composite − `sentence_cv` AUC | n=442: [-0.0004, +0.046] (n.s.); **n=1476: [+0.013, +0.038] (significant)** | paired bootstrap |
| Em-dashes in HC3 **AI** vs **human** | **0 vs ~41** → AUC 0.486, *wrong sign* | direct count in corpus |
| Sentence-splitter bug leak | human abbreviation rate ~23–24% vs AI ~3–4%; fixing the splitter moves AUC **0.896 → 0.889** | abbreviation-aware splitter A/B |
| `cv=1.0` short-text fallback | fires on **281/1476 (19%)**, skewed 248 human / 33 AI; silently scores uniform short AI as "human" | fallback counter |
| Adversarial fragility | mechanical sentence merge/split → recall on caught AI **100% → ~76%** (zero semantic change) | `eval/adversarial` prototype |
| Genre confound direction | within-question **paired** AUC **0.908 > pooled 0.891** → confound *deflates*, doesn't inflate; AI beat its own paired human on 647/738 | paired AUC |
| Threshold-selection leak | `best_threshold()` chosen on the eval split → +0.010 F1 optimism (up to +0.031) | train-select / test-apply A/B |
| Split topic leak | 304/738 questions straddle the split; **69% of test shares a question with train** — harmless now, a memorization trap for any fitted model | group overlap count |

### The four things that matter most

- **It's a `sentence_cv` detector.** Four of six features are noise *on this corpus*; the
  composite's edge over `cv` alone is real but small. Say this plainly — it's more impressive
  than pretending six hand-tuned heuristics each pull weight.
- **It's tuned to a dead distribution.** HC3 is Jan-2023 ChatGPT. The single most-memed
  2024–26 tell — the em-dash — appears **zero** times in HC3-AI and is *negatively* weighted by
  a learned model on this data. Every weight in `lint_text` is fit to a model that no longer
  exists. This is the dominant risk to the project's relevance, not a footnote.
- **The "held-out" split isn't.** `evaluate.py` defaults to `--split test`, and `analyze.py`
  feeds the test-derived `report.md`/`misclassified.md` to the advisory agent; six `advice.md`
  edits are baked into the linter from test-set failures. The number is mildly optimistic and
  the discipline is muddy (README does claim train-split confirmation — so it's not fabricated,
  just leaky). Trivial to fix.
- **There's a label-leaking bug in the one feature that works.** The regex sentence splitter
  over-splits on abbreviations/decimals (`"U.S."`, `"3.5"`), and because HC3 humans use
  abbreviations ~6× more than 2022 ChatGPT, the bug borrows discrimination from a corpus
  artifact. Fixing it *lowers* AUC by ~0.007 — and choosing correctness over the number, in
  writing, is exactly the signal this portfolio wants.

---

## 2. The reframe (positioning — do this, it's nearly free)

Rewrite the README headline and the linter docstring:

> **An empirical eval harness that measures — and falsifies — how well cheap stylometric
> heuristics separate AI from human text, plus a writing-quality ("slop") linter that flags
> cliché density, marketing vocab, and sentence-length monotony _regardless of authorship_.**

- Demote the linter from "detector" to "style/quality tool." A false positive on a style tool
  costs nothing (it's advice: "vary your sentence lengths"); a false positive on a *detector*
  wrongly accuses a person — and the literature (Liang et al. 2023; OpenAI withdrawing its own
  classifier) shows that harm is real and concentrated on non-native English writers.
- State that the 0.896 is ~one-feature and register-sensitive and **must not be used as a
  forensic verdict on individuals.**
- Write a short `NOTES.md`: _"I built a detector, my own harness falsified it, here's exactly
  how and what I learned."_ For a senior AI-eng job search, **that synthesized narrative is the
  deliverable** — more than any single AUC.

---

## 3. The three tracks — what each is really worth

### A. Refactor (improve the current form) — _high value, but as enablement_
- **Feature registry.** `lint_text` and `evaluate.featurize` re-derive the same per-1k features
  in two codepaths; a learned model would be a third. Collapse to one declarative registry
  (`extract / normalize / score / direction` per feature) consumed by linter, harness, and
  model. Golden-test for score-for-score equivalence.
- **Externalize weights** (the ~16 inline magic constants) into a JSON config so tuning ≠
  editing logic and a learned model can write weights back.
- **Real CLI**: `argparse`, stdin (`-`), `--json`, `--threshold`, and **exit 1 when any file
  exceeds threshold** so it can actually gate CI/pre-commit (today it returns exit 0 even for
  text that "reads as AI"). Make the linter installable (`console_script`); keep `eval/` as
  scripts.
- **First unit tests + a tiny committed fixture corpus** (NOT git-ignored) so CI runs offline,
  plus a pinned HC3 manifest/checksum so the headline is reproducible.

### B. Corpus (grow the base) — _the scientific unlock_
- **RAID (no-adversarial), first.** MIT-licensed, `test_none.csv` ≈ 77 MB, 11 modern models ×
  11 genres, human rows included. Download-on-demand (keep `corpus/` git-ignored). This is the
  fastest way to measure whether the four "dead" features wake up on 2024-era text. Expect
  overall AUC to **drop** on the harder modern mix — _that drop is the finding._
- **A clean, non-encyclopedic pre-2022 human pool** (e.g. GPT-2-output webtext, pre-2019).
  Every current false positive is Wikipedia register; a varied human pool is the only way to
  separate AI-ness from encyclopedic-register detection. If pooled AUC collapses toward 0.5,
  that's the honest result the whole project exists to surface.
- **Provenance fields now** (`model`, `era`, `decoding`, `license`) before sources multiply,
  backfilled onto HC3 — turns the corpus into a drift instrument. Group-by-question splitting
  must land **before** any fitted model.
- **Self-generation via Ollama** (free, over Tailscale) is valuable but *secondary and
  circular* — use it as a drift baseline, not primary training data. A Temporal-scheduled drift
  monitor → ClickStack is a great infra flex but is **premature** until the corpus story works
  by hand once.

### C. ML model (the next phase) — _measured, so we know what's real_
The crux experiment was run (from-scratch + sklearn, on the harness's own split):
- **Auto-tuning the existing 6 weights ties the heuristic** (LR 0.895 / GBM 0.895 vs 0.896).
  The README's flagship "auto-tune for higher AUC" item is **not a capability play** — sell it
  as maintainability/honesty (weights become data, gated on held-out AUC), never as "it scores
  higher," because it doesn't.
- **The real lever is new features.** 6 hand-features + **type-token ratio + function-word
  ratio** → LR **0.957** / GBM **0.963** (+~0.07, ~12× seed noise). **TTR alone = 0.847**,
  nearly the whole current heuristic. These are textbook stylometry signals the linter ignores.
- **Ship an interpretable LR**, not a transformer: a fine-tuned DeBERTa would reach ~0.97–0.99
  but buys ~1–2 points over cheap features while destroying the interpretability that is this
  project's differentiator. The LR's coefficient table _is_ the explanation, and it recovers
  conditional signal a hand-tuner never would (e.g. the negative em-dash coefficient that
  survives L1 — concrete drift evidence).
- **Zero-shot likelihood (Fast-DetectGPT / Binoculars) via local Ollama** is the honest "real
  SOTA" baseline, but defer it: a small, mismatched local scoring model against 2022-ChatGPT
  text is exactly the regime where paper numbers don't transfer. Build it after the modern
  corpus, framed as _"implemented SOTA, here's where it breaks."_

---

## 4. Sequenced plan (the recommendation)

Order is dependency-driven: honesty → enablement → data → model. The one place the two
syntheses disagreed was **when to add the stylometry features** — before corpus modernization
(so RAID is a clean external check) or after (so you don't ship a register artifact). Adjudication:
**measure them early (it's free and it's the most interesting result), but gate _shipping_ them
into the scored linter on the modern + varied-human validation.** The experiment is cheap; the
productization should wait for proof it isn't just learning 2022 encyclopedic register.

1. **(S, methodology) Close the test-split leak.** Default `evaluate.py` to `--split train`;
   advisory + tuning see train-only reports; touch test once at the end. Note that advice
   recs #2–#7 were chosen with test knowledge.
2. **(S, methodology) Rigor primitives.** Bootstrap 95% CI next to every AUC; select F1
   threshold on train / apply to test; add within-question **paired AUC** (0.908) as a headline
   metric. Without a CI, every "improvement" under +0.03 is noise-chasing.
3. **(S, methodology) Fix the splitter + the `cv` fallback,** and **publish the honest AUC drop**
   (~0.896 → 0.889) with the abbreviation-leak explanation. Make `<4`-sentence text return
   `low_confidence`, not a silent "human."
4. **(M, refactor) Feature registry + JSON weight config,** golden-tested for equivalence. The
   seam every later step needs.
5. **(M, product) Real CLI** (exit codes, `--json`, stdin, `--threshold`) **+ first tests +
   offline fixture corpus + pinned HC3 checksum.** Can run parallel to the science track.
6. **(S, strategy) Reframe README + write `NOTES.md`** citing the verified falsifications
   (one-feature reality, em-dash 0/41 sign-flip, ~22-pt paraphrase recall drop).
7. **(M, corpus) Ingest RAID no-adversarial + add provenance fields + group-by-question split.**
   Report per-source/per-era AUC slices, never a single blended headline.
8. **(M, corpus) Add the clean non-encyclopedic human pool;** re-measure pooled vs paired AUC
   and the false-positive composition — the genre-confound test.
9. **(S, ml) Measure TTR + function-word ratio now (per-domain AUC + CI);** ship them into the
   scored linter only after #7–#8 confirm they aren't just register artifacts.
10. **(M, ml) Ship the interpretable LR** over the registry's features (train-only
    standardization persisted, gated on group-split held-out + CI, per-feature contribution as
    the explanation). Optionally add **one** Ollama zero-shot-likelihood baseline as "real SOTA,
    and where it breaks." Keep sklearn/numpy behind an optional extra so the core eval stays
    stdlib-only.

**Rough cadence:** weeks 1–2 = #1–#6 (mostly S, the integrity + product spine); week 3 =
#7–#8 (the modern-data reckoning); week 4 = #9–#10 (the interpretable model).

---

## 5. Explicitly NOT doing (and why)

- **A fine-tuned transformer (now).** ~10 lines of stdlib stylometry already reach ~0.96 on
  HC3; a no-GPU XL effort buys ~1–2 points and forfeits interpretability — the differentiator.
- **The auto-tune-the-linter agent as a capability play.** Measured to tie the heuristic. Keep
  only as a maintainability story, and if kept, gate it on drift/adversarial metrics — *not* on
  HC3 AUC, which would optimize a register detector against an extinct model.
- **RAID's adversarial split.** Designed to break detectors; a heuristic linter will collapse to
  ~0.5. Scope adversarial robustness *out* explicitly rather than pretend regexes handle it.
- **The Temporal/ClickStack drift monitor (yet).** Don't automate a pipeline you haven't
  validated by hand once.

---

## 6. Open experiments worth running (each is a clean result)

- Recompute `em_per_1k` / `vocab` AUC on RAID's GPT-4/Claude-era rows — do the dead features
  cross ~0.6? (the central drift question)
- Does the +0.07 stylometry lift land in the easy `medicine` slice or the hard `wiki_csai`
  slice? (real discrimination vs amplified encyclopedic confound)
- Does pooled AUC survive a varied human pool, or collapse toward 0.5? (AI-ness vs register)
- What's the AUC on the genre the tool is *actually* for — LinkedIn/marketing/cover-letter
  prose — where em-dashes and AI-vocab plausibly fire on current models? (the dead features may
  be alive on the target genre; the single most relevant un-run experiment)
- Precision at a realistic low base rate (5–10% AI), not the 50/50 corpus — the number that
  decides whether a CI slop-gate is usable.
