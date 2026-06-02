#!/usr/bin/env python3
"""Evaluate the linter against a labeled corpus — the empirical 'test'.

Measures how well a score separates AI from human text:
  * ROC AUC (threshold-independent) with a bootstrap 95% CI
  * within-question PAIRED AUC — the genre-controlled separation (each AI answer
    vs its OWN human answer), so we measure AI-ness, not topic
  * accuracy / precision / recall / F1, with the operating threshold selected on
    TRAIN and applied to the eval split (no threshold-on-test leakage)
  * per-FEATURE AUC (which heuristics discriminate; the rest is the registry)
  * per-DOMAIN / per-SOURCE / per-ERA AUC (where it's strong vs blind, and drift)
  * worst misclassifications, dumped to report/misclassified.md

Honest-split discipline: the advisory/tuning loop must read TRAIN reports, so the
default report split is `train`; touch `test` once at the end for the headline.
The split is grouped by question so a question's human+AI answers never straddle
train/test (a memorization trap for any fitted model).

Usage:
    python3 evaluate.py                          # hc3, TRAIN split (for tuning)
    python3 evaluate.py --split test             # held-out headline number
    python3 evaluate.py --corpus corpus/hc3.jsonl corpus/raid.jsonl --split test
    python3 evaluate.py --scores corpus/scores.jsonl   # merge cached Layer-2 features
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from unsloppable.features import DIRECTION as REG_DIRECTION  # noqa: E402
from unsloppable.features import FEATURE_NAMES  # noqa: E402
from unsloppable.heuristic import score_text  # noqa: E402
from unsloppable.likelihood import DIRECTION as LL_DIRECTION  # noqa: E402 (no torch at import)

DIRECTIONS = {**REG_DIRECTION, **LL_DIRECTION}

REPORT_DIR = ROOT / "eval" / "report"
DEFAULT_CORPUS = ROOT / "corpus" / "hc3.jsonl"


# ---------------------------------------------------------------- metrics ----

def auc(scores: list[float], labels: list[int]) -> float:
    """ROC AUC = P(score_positive > score_negative), ties=0.5 (Mann-Whitney U,
    average ranks for ties). label 1 = AI."""
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(paired)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = sum(r for r, (_, lab) in zip(ranks, paired) if lab == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_auc_ci(scores: list[float], labels: list[int], reps: int = 2000,
                     seed: int = 13) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for AUC — so a future 'improvement' inside the
    noise band isn't mistaken for a real gain."""
    rng = random.Random(seed)
    n = len(scores)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = range(n)
    boots = []
    for _ in range(reps):
        sample = [rng.choice(idx) for _ in range(n)]
        s = [scores[i] for i in sample]
        lab = [labels[i] for i in sample]
        a = auc(s, lab)
        if not math.isnan(a):
            boots.append(a)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return (round(lo, 4), round(hi, 4))


def paired_auc(rows: list[dict], scores: list[float], labels: list[int]) -> dict:
    """Within-question AUC: for each question with both an AI and a human answer,
    does the AI answer score higher? Controls for topic/genre entirely."""
    by_q: dict[str, dict[str, list[float]]] = {}
    for r, s, lab in zip(rows, scores, labels):
        q = r.get("question") or r["id"]
        d = by_q.setdefault(q, {"ai": [], "human": []})
        d["ai" if lab == 1 else "human"].append(s)
    wins = ties = losses = pairs = 0
    deltas = []
    for d in by_q.values():
        if not d["ai"] or not d["human"]:
            continue
        a = statistics.mean(d["ai"])
        h = statistics.mean(d["human"])
        pairs += 1
        deltas.append(a - h)
        if a > h:
            wins += 1
        elif a == h:
            ties += 1
        else:
            losses += 1
    if pairs == 0:
        return {}
    return {"paired_auc": round((wins + 0.5 * ties) / pairs, 4), "pairs": pairs,
            "wins": wins, "ties": ties, "losses": losses,
            "mean_delta": round(statistics.mean(deltas), 2)}


def confusion_at(scores: list[float], labels: list[int], thr: float) -> dict:
    tp = fp = fn = tn = 0
    for s, lab in zip(scores, labels):
        pred = 1 if s >= thr else 0
        if pred and lab:
            tp += 1
        elif pred and not lab:
            fp += 1
        elif not pred and lab:
            fn += 1
        else:
            tn += 1
    acc = (tp + tn) / len(labels) if labels else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"threshold": round(thr, 2), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


def best_threshold(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Threshold maximizing F1. Call this on TRAIN, then apply to the eval split."""
    best_thr, best_f1 = 0.0, -1.0
    for t in sorted(set(scores)):
        m = confusion_at(scores, labels, t)
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], t
    return best_thr, best_f1


def dist(values: list[float]) -> dict:
    if not values:
        return {}
    sv = sorted(values)
    return {"mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "p25": round(sv[len(sv) // 4], 2), "p75": round(sv[(3 * len(sv)) // 4], 2),
            "min": round(min(values), 2), "max": round(max(values), 2)}


# ----------------------------------------------------------------- corpus ----

def load_corpus(paths: list[Path], scores_path: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"  warn: corpus not found: {p}", file=sys.stderr)
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if scores_path and scores_path.exists():           # merge cached Layer-2 features by id
        cache = {}
        with scores_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    cache[d["id"]] = d.get("features", d)
        merged = 0
        for r in rows:
            extra = cache.get(r["id"])
            if extra:
                r.setdefault("_ll", {}).update({k: v for k, v in extra.items() if k != "id"})
                merged += 1
        print(f"  merged Layer-2 scores for {merged}/{len(rows)} rows", file=sys.stderr)
    return rows


def split(rows: list[dict], test_frac: float, seed: int,
          group_key: str = "question") -> tuple[list, list]:
    """Grouped split: whole question-groups go to train or test together, so a
    question's human+AI answers never straddle the split. Falls back to per-row
    when a row has no group key."""
    rng = random.Random(seed)
    groups: dict[str, list] = {}
    for r in rows:
        g = r.get(group_key) or f"__solo__{r['id']}"
        groups.setdefault(g, []).append(r)
    gids = list(groups)
    rng.shuffle(gids)
    cut = int(len(gids) * test_frac)
    test_g = set(gids[:cut])
    train, test = [], []
    for g, items in groups.items():
        (test if g in test_g else train).extend(items)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def featurize(row: dict) -> dict:
    """Normalized feature vector for a row — stylometric (always) plus any cached
    Layer-2 likelihood features merged onto the row under '_ll'."""
    r = score_text(row["text"])
    feats = dict(r.features)
    feats["score"] = float(r.score)
    if row.get("_ll"):
        feats.update(row["_ll"])
    return {"feats": feats, "_raw": r, "_row": row}


def _label(r: dict) -> int:
    return 1 if r["label"] == "ai" else 0


def slice_auc(rows, scores, labels, key) -> dict:
    out = {}
    for v in sorted({r.get(key, "?") for r in rows}):
        idx = [i for i, r in enumerate(rows) if r.get(key, "?") == v]
        if len({labels[i] for i in idx}) == 2:
            out[v] = round(auc([scores[i] for i in idx], [labels[i] for i in idx]), 4)
    return out


def _impute(feats: list[dict], names: list[str]) -> None:
    """Replace NaN feature values (e.g. cv undefined for <2-sentence text) with the
    feature's median over present values — neutral, not a false signal."""
    for name in names:
        present = [f["feats"][name] for f in feats
                   if name in f["feats"] and not math.isnan(f["feats"][name])]
        med = statistics.median(present) if present else 0.0
        for f in feats:
            v = f["feats"].get(name)
            if v is None or math.isnan(v):
                f["feats"][name] = med


def _present(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def evaluate(rows: list[dict], train_rows: list[dict], threshold: float | None) -> dict:
    feats = [featurize(r) for r in rows]
    labels = [_label(r) for r in rows]
    all_names = sorted({k for f in feats for k in f["feats"]})
    # Capture each feature's genuinely-present (value, label) pairs BEFORE imputation,
    # so per-feature AUC is computed only on real values — NOT on constant-imputed rows,
    # which would mechanically crush the AUC toward 0.5 (e.g. Layer-2 features on rows
    # without cached scores). This is the honesty fix for the diluted Layer-2 numbers.
    present = {name: [(f["feats"][name], labels[i]) for i, f in enumerate(feats)
                      if name in f["feats"] and _present(f["feats"][name])]
               for name in all_names}
    _impute(feats, all_names)
    scores = [f["feats"]["score"] for f in feats]

    overall = auc(scores, labels)
    ci = bootstrap_auc_ci(scores, labels)
    paired = paired_auc(rows, scores, labels)

    names = list(FEATURE_NAMES)
    for f in feats:
        for k in f["feats"]:
            if k not in names and k != "score":
                names.append(k)
    feature_auc = {}
    for name in names:
        pairs = present.get(name, [])
        direction = DIRECTIONS.get(name, "high=AI")
        cov = len(pairs)
        if cov < 10 or len({lab for _, lab in pairs}) < 2:
            feature_auc[name] = {"auc": None, "direction": direction, "coverage": cov}
            continue
        a = auc([v for v, _ in pairs], [lab for _, lab in pairs])
        if direction == "low=AI" and not math.isnan(a):
            a = 1 - a
        feature_auc[name] = {"auc": round(a, 4) if not math.isnan(a) else None,
                             "direction": direction, "coverage": cov}

    domain_auc = slice_auc(rows, scores, labels, "domain")
    source_auc = slice_auc(rows, scores, labels, "source")
    era_auc = slice_auc(rows, scores, labels, "era")

    # operating threshold: select on TRAIN, apply to eval split (no leakage).
    if threshold is None and train_rows:
        tr_scores = [score_text(r["text"]).score for r in train_rows]
        tr_labels = [_label(r) for r in train_rows]
        threshold, _ = best_threshold(tr_scores, tr_labels)
    op = confusion_at(scores, labels, threshold if threshold is not None else 2.5)
    leaky_thr, _ = best_threshold(scores, labels)        # diagnostic upper bound only
    leaky = confusion_at(scores, labels, leaky_thr)

    ai_s = [s for s, lab in zip(scores, labels) if lab == 1]
    hu_s = [s for s, lab in zip(scores, labels) if lab == 0]

    fps, fns = [], []
    for f, lab, s in zip(feats, labels, scores):
        row = f["_row"]
        rec = {"id": row["id"], "domain": row.get("domain"), "source": row.get("source"),
               "score": s, "sentence_cv": round(f["feats"].get("sentence_cv", 0), 2),
               "ttr": round(f["feats"].get("type_token_ratio", 0), 2),
               "snippet": row["text"][:280]}
        if lab == 0 and s >= op["threshold"]:
            fps.append(rec)
        elif lab == 1 and s < op["threshold"]:
            fns.append(rec)
    fps.sort(key=lambda r: -r["score"])
    fns.sort(key=lambda r: r["score"])

    return {"n": len(rows), "n_ai": sum(labels), "n_human": len(rows) - sum(labels),
            "overall_auc": round(overall, 4), "auc_ci95": list(ci), "paired": paired,
            "score_dist": {"ai": dist(ai_s), "human": dist(hu_s)},
            "operating_threshold": op, "leaky_best_f1": leaky,
            "feature_auc": feature_auc, "domain_auc": domain_auc,
            "source_auc": source_auc, "era_auc": era_auc,
            "false_positives": fps, "false_negatives": fns}


# ------------------------------------------------------------------ report ----

def write_reports(m: dict, meta: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(
        {"meta": meta, **{k: v for k, v in m.items()
                          if k not in ("false_positives", "false_negatives")}}, indent=2))

    fa = m["feature_auc"]
    feat_rows = "\n".join(
        f"| `{n}` | {v['auc'] if v['auc'] is not None else 'n/a'} | {v['direction']} | {v.get('coverage', '?')} |"
        for n, v in sorted(fa.items(), key=lambda kv: -(kv[1]['auc'] or 0)))
    dom_rows = "\n".join(f"| {d} | {a} |" for d, a in m["domain_auc"].items())
    src_rows = "\n".join(f"| {d} | {a} |" for d, a in m["source_auc"].items()) or "| — | — |"
    op, leaky = m["operating_threshold"], m["leaky_best_f1"]
    ci = m["auc_ci95"]
    p = m["paired"]
    paired_line = (f"**Paired (within-question) AUC: {p['paired_auc']}** "
                   f"(AI beat its own human on {p['wins']}/{p['pairs']}, mean Δ {p['mean_delta']})"
                   if p else "_paired AUC: n/a (no within-question pairs in this corpus)_")

    report = f"""# unsloppable — evaluation report

_generated {meta['generated']} · corpus {meta['corpus']} · split `{meta['split']}` (n={m['n']}, ai={m['n_ai']}, human={m['n_human']})_

## Headline

**Overall ROC AUC: {m['overall_auc']}**  (95% CI [{ci[0]}, {ci[1]}]) — probability the
linter scores a random AI sample above a random human one. 0.5 = coin flip.

{paired_line}

| metric | operating cut (score≥{op['threshold']}, chosen on train) | best-F1 on this split (diagnostic only) |
|---|---|---|
| accuracy  | {op['accuracy']} | {leaky['accuracy']} |
| precision | {op['precision']} | {leaky['precision']} |
| recall    | {op['recall']} | {leaky['recall']} |
| F1        | {op['f1']} | {leaky['f1']} |
| TP/FP/FN/TN | {op['tp']}/{op['fp']}/{op['fn']}/{op['tn']} | {leaky['tp']}/{leaky['fp']}/{leaky['fn']}/{leaky['tn']} |

## Score distribution

| class | mean | median | p25 | p75 | min | max |
|---|---|---|---|---|---|---|
| ai    | {m['score_dist']['ai'].get('mean')} | {m['score_dist']['ai'].get('median')} | {m['score_dist']['ai'].get('p25')} | {m['score_dist']['ai'].get('p75')} | {m['score_dist']['ai'].get('min')} | {m['score_dist']['ai'].get('max')} |
| human | {m['score_dist']['human'].get('mean')} | {m['score_dist']['human'].get('median')} | {m['score_dist']['human'].get('p25')} | {m['score_dist']['human'].get('p75')} | {m['score_dist']['human'].get('min')} | {m['score_dist']['human'].get('max')} |

## Per-feature discrimination (individual AUC)

_AUC computed only on rows where the feature is genuinely present (`coverage`) —
never on imputed rows, which would crush the number toward 0.5._

| feature | AUC | direction | coverage (n rows) |
|---|---|---|---|
{feat_rows}

## Per-domain AUC
| domain | AUC |
|---|---|
{dom_rows}

## Per-source AUC
| source | AUC |
|---|---|
{src_rows}

## Misclassifications

{len(m['false_positives'])} false positives (human flagged AI) and
{len(m['false_negatives'])} false negatives (AI missed) at the operating cut.
See **misclassified.md**.
"""
    (REPORT_DIR / "report.md").write_text(report)

    def _block(title, recs, note):
        out = [f"## {title} ({len(recs)})\n", note, ""]
        for r in recs[:15]:
            out.append(f"- **{r['id']}** · {r.get('domain')} · {r.get('source')} · "
                       f"score {r['score']} · cv {r['sentence_cv']} · ttr {r['ttr']}\n"
                       f"  > {r['snippet']}…\n")
        return "\n".join(out)

    (REPORT_DIR / "misclassified.md").write_text(
        f"# Misclassified (operating cut, score≥{op['threshold']})\n\n"
        + _block("False positives — human flagged as AI", m['false_positives'],
                 "_Tells firing on genuine human writing. Candidates for softening._")
        + "\n\n"
        + _block("False negatives — AI that slipped through", m['false_negatives'],
                 "_AI prose with none of the current tells. Candidates for new signals._"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", default=[str(DEFAULT_CORPUS)])
    ap.add_argument("--scores", default=None, help="JSONL of cached Layer-2 features by id")
    ap.add_argument("--split", choices=["train", "test", "all"], default="train",
                    help="default train (the tuning view); use test for the headline")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--threshold", type=float, default=None,
                    help="fix the operating cut (default: best-F1 chosen on train)")
    args = ap.parse_args(argv)

    rows = load_corpus([Path(p) for p in args.corpus],
                       Path(args.scores) if args.scores else None)
    if not rows:
        print("no corpus rows — run fetch_hc3.py first", file=sys.stderr)
        return 1

    train_rows, test_rows = split(rows, args.test_frac, args.seed)
    if args.split == "all":
        eval_rows, name = rows, "all"
    elif args.split == "test":
        eval_rows, name = test_rows, f"test {int(args.test_frac*100)}%"
    else:
        eval_rows, name = train_rows, f"train {100-int(args.test_frac*100)}%"

    m = evaluate(eval_rows, train_rows, args.threshold)
    meta = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "corpus": ", ".join(Path(p).name for p in args.corpus),
            "split": name, "seed": args.seed}
    write_reports(m, meta)

    ci = m["auc_ci95"]
    print(f"\n=== unsloppable eval ({name}, n={m['n']}, ai={m['n_ai']}, human={m['n_human']}) ===")
    print(f"overall AUC: {m['overall_auc']}  CI95 [{ci[0]}, {ci[1]}]")
    if m["paired"]:
        print(f"paired AUC : {m['paired']['paired_auc']} "
              f"({m['paired']['wins']}/{m['paired']['pairs']} wins, Δ{m['paired']['mean_delta']})")
    op = m["operating_threshold"]
    print(f"@score>={op['threshold']} (train-chosen): acc {op['accuracy']} "
          f"prec {op['precision']} rec {op['recall']} f1 {op['f1']}")
    print("feature AUC:", {k: v["auc"] for k, v in
                           sorted(m["feature_auc"].items(), key=lambda x: -(x[1]["auc"] or 0))})
    print("domain AUC :", m["domain_auc"])
    if m["source_auc"]:
        print("source AUC :", m["source_auc"])
    if m["era_auc"]:
        print("era AUC    :", m["era_auc"])
    print(f"\nreports → {REPORT_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
