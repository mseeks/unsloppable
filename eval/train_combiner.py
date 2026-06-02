#!/usr/bin/env python3
"""Train the learned combiner (logistic regression over both layers).

Fits a calibrated LR over the stylometric registry + the heuristic score + the
cached Layer-2 likelihood features, on the harness's grouped train split, and
evaluates held-out AUC vs the Layer-1 heuristic baseline. Saves a small JSON model
(coefficients + standardization) that unsloppable.combine loads at inference.

Trains only on rows that have cached likelihood features (run score_likelihood.py
first), so the likelihood inputs are real, not imputed.

Needs the `ml` extra. Usage:
    uv run --extra ml python eval/score_likelihood.py --corpus corpus/hc3.jsonl corpus/raid.jsonl corpus/generated.jsonl
    uv run --extra ml python eval/train_combiner.py  --corpus corpus/hc3.jsonl corpus/raid.jsonl corpus/generated.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from evaluate import auc, bootstrap_auc_ci, load_corpus, split  # noqa: E402
from unsloppable.combine import COMBINER_FEATURES, MODEL_PATH  # noqa: E402
from unsloppable.heuristic import score_text  # noqa: E402


def vectorize(rows):
    X, y, srcs, eras = [], [], [], []
    for r in rows:
        res = score_text(r["text"])
        vec = dict(res.features)
        vec["heuristic_score"] = float(res.score)
        vec.update(r.get("_ll", {}))
        X.append([vec.get(n, math.nan) for n in COMBINER_FEATURES])
        y.append(1 if r["label"] == "ai" else 0)
        srcs.append(r.get("source", "?"))
        eras.append(r.get("era", "?"))
    return X, y, srcs, eras


def main(argv: list[str]) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--scores", default=str(ROOT / "corpus" / "scores.jsonl"))
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out", default=str(MODEL_PATH))
    args = ap.parse_args(argv)

    rows = load_corpus([Path(p) for p in args.corpus], Path(args.scores))
    rows = [r for r in rows if r.get("_ll")]          # only rows with real Layer-2 features
    if not rows:
        print("no rows with cached likelihood features — run score_likelihood.py first", file=sys.stderr)
        return 1
    train_rows, test_rows = split(rows, args.test_frac, args.seed)
    print(f"train {len(train_rows)}  test {len(test_rows)}  (rows with Layer-2: {len(rows)})")

    Xtr, ytr, _, _ = vectorize(train_rows)
    Xte, yte, ste, ete = vectorize(test_rows)
    Xtr, Xte = np.array(Xtr, float), np.array(Xte, float)
    ytr, yte = np.array(ytr), np.array(yte)

    # impute (train medians) + standardize (train stats)
    impute = {}
    for j, name in enumerate(COMBINER_FEATURES):
        col = Xtr[:, j]
        med = float(np.nanmedian(col)) if np.isfinite(col).any() else 0.0
        impute[name] = med
        for M in (Xtr, Xte):
            mask = ~np.isfinite(M[:, j])
            M[mask, j] = med
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0
    CLIP = 6.0   # bound standardized values so OOD inputs can't dominate via extrapolation
    Ztr = np.clip((Xtr - mean) / std, -CLIP, CLIP)
    Zte = np.clip((Xte - mean) / std, -CLIP, CLIP)

    clf = LogisticRegression(max_iter=2000, C=args.C)
    clf.fit(Ztr, ytr)
    ptr = clf.predict_proba(Ztr)[:, 1]
    pte = clf.predict_proba(Zte)[:, 1]

    # heuristic baseline AUC on the same test rows
    heur = [score_text(r["text"]).score for r in test_rows]
    auc_comb = auc(list(pte), list(yte))
    auc_heur = auc(heur, list(yte))
    ci = bootstrap_auc_ci(list(pte), list(yte))

    def by(group, vals):
        out = {}
        for g in sorted(set(group)):
            idx = [i for i, x in enumerate(group) if x == g]
            if len({yte[i] for i in idx}) == 2:
                out[g] = round(auc([vals[i] for i in idx], [int(yte[i]) for i in idx]), 4)
        return out

    print(f"\n=== combiner vs heuristic (held-out test, n={len(yte)}) ===")
    print(f"  combiner  AUC: {auc_comb:.4f}  CI95 [{ci[0]}, {ci[1]}]   train {auc(list(ptr), list(ytr)):.4f}")
    print(f"  heuristic AUC: {auc_heur:.4f}   (Layer-1 alone)")
    print(f"  by source  (combiner): {by(ste, list(pte))}")
    print(f"  by source  (heuristic):{by(ste, heur)}")
    print(f"  by era     (combiner): {by(ete, list(pte))}")
    coefs = sorted(zip(COMBINER_FEATURES, clf.coef_[0]), key=lambda x: -abs(x[1]))
    print("  top coefficients (standardized):")
    for name, c in coefs[:12]:
        print(f"    {name:22} {c:+.3f}")

    model = {"features": COMBINER_FEATURES, "mean": list(map(float, mean)),
             "std": list(map(float, std)), "coef": list(map(float, clf.coef_[0])),
             "intercept": float(clf.intercept_[0]), "impute": impute, "clip": CLIP,
             "meta": {"name": "lr-stylo+likelihood", "C": args.C, "seed": args.seed,
                      "n_train": len(ytr), "test_auc": round(auc_comb, 4),
                      "test_auc_heuristic": round(auc_heur, 4),
                      "corpus": [Path(p).name for p in args.corpus]}}
    Path(args.out).write_text(json.dumps(model, indent=2))
    print(f"\nsaved combiner -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
