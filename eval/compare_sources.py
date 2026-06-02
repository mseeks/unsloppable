#!/usr/bin/env python3
"""Compare the linter's separation across sources / models / eras — the drift view.

Groups every sample by provenance (source · era · label, and per-model for AI) and
reports each AI group's AUC vs the pooled human baseline, for the Layer-1 heuristic
score AND (if cached) the Layer-2 likelihood features. This is how we SEE era drift:
the stylometric heuristic that scores ~0.83 on 2022 HC3 collapses on modern models,
while likelihood features hold up — quantified here, not asserted.

Writes eval/report/generation_compare.md.

Usage:
    uv run python eval/compare_sources.py
    uv run python eval/compare_sources.py --corpus corpus/hc3.jsonl corpus/raid.jsonl corpus/generated.jsonl --scores corpus/scores.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from evaluate import REPORT_DIR, auc, dist, featurize, load_corpus  # noqa: E402
from unsloppable.likelihood import FEATURES as LL_FEATURES  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+",
                    default=[str(ROOT / "corpus" / "hc3.jsonl"),
                             str(ROOT / "corpus" / "raid.jsonl"),
                             str(ROOT / "corpus" / "generated.jsonl")])
    ap.add_argument("--scores", default=str(ROOT / "corpus" / "scores.jsonl"))
    args = ap.parse_args(argv)

    rows = load_corpus([Path(p) for p in args.corpus],
                       Path(args.scores) if Path(args.scores).exists() else None)
    if not rows:
        print("no corpus rows", file=sys.stderr)
        return 1

    feats = [(r, featurize(r)["feats"]) for r in rows]
    has_ll = any("ll_binoculars" in f for _, f in feats)
    human_scores = [f["score"] for r, f in feats if r["label"] == "human"]
    human_ll = {k: [f[k] for r, f in feats if r["label"] == "human" and k in f]
                for k in LL_FEATURES}

    def group_key(r):
        era = r.get("era", "?")
        if r["label"] == "human":
            return f'{r.get("source","?")} · {era} · human'
        return f'{r.get("source","?")} · {era} · {r.get("model","ai")}'

    groups: dict[str, list] = {}
    for r, f in feats:
        groups.setdefault(group_key(r), []).append((r, f))

    cols = "| group | n | median score | heuristic AUC vs human"
    if has_ll:
        cols += " | perplexity AUC | binoculars AUC | burstiness AUC"
    cols += " |"
    sep = "|" + "|".join("---" for _ in cols.split("|")[1:-1]) + "|"
    lines = ["# Source / era comparison — drift view\n",
             "AUC = each AI group vs the pooled human baseline (0.5 = coin flip). "
             "Higher heuristic score / lower perplexity & binoculars = more AI-like.\n",
             cols, sep]
    print(f"{'group':34} {'n':>5} {'med':>6} {'heurAUC':>8}", end="")
    if has_ll:
        print(f" {'pplAUC':>7} {'binoAUC':>8} {'burstAUC':>8}", end="")
    print()

    def grp_auc(vals_ai, vals_hu, low_is_ai):
        if not vals_ai or not vals_hu:
            return None
        a = auc(vals_ai + vals_hu, [1] * len(vals_ai) + [0] * len(vals_hu))
        return round(1 - a if low_is_ai else a, 3)

    for key in sorted(groups):
        grp = groups[key]
        scs = [f["score"] for _, f in grp]
        d = dist(scs)
        is_ai = not key.endswith("· human")
        row_md = f"| {key} | {len(grp)} | {d['median']} | "
        if is_ai:
            a = grp_auc(scs, human_scores, low_is_ai=False)
            row_md += f"{a}"
            line = f"{key:34} {len(grp):>5} {d['median']:>6} {str(a):>8}"
        else:
            row_md += "—"
            line = f"{key:34} {len(grp):>5} {d['median']:>6} {'—':>8}"
        if has_ll:
            for fn, low in [("ll_perplexity", True), ("ll_binoculars", True),
                            ("ll_surprisal_stdev", True)]:
                if is_ai:
                    va = [f[fn] for _, f in grp if fn in f]
                    a = grp_auc(va, human_ll[fn], low_is_ai=low)
                    row_md += f" | {a}"
                    line += f" {str(a):>8}"
                else:
                    row_md += " | —"
                    line += f" {'—':>8}"
        row_md += " |"
        lines.append(row_md)
        print(line)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "generation_compare.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT_DIR.relative_to(ROOT)}/generation_compare.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
