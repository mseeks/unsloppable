#!/usr/bin/env python3
"""Precompute Layer-2 likelihood features for a corpus and cache them by id.

The likelihood layer is expensive (two transformer forward passes per text), so we
score once into corpus/scores.jsonl and let evaluate.py merge it via --scores.
Needs the `ml` extra. Runs on MPS/CUDA/CPU.

Usage:
    uv run --extra ml python eval/score_likelihood.py --corpus corpus/hc3.jsonl corpus/raid.jsonl
    uv run --extra ml python eval/score_likelihood.py --corpus corpus/raid.jsonl --balance 1000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from unsloppable.likelihood import LikelihoodScorer  # noqa: E402


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        if not p.exists():
            print(f"  warn: {p} not found", file=sys.stderr)
            continue
        with p.open() as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def balance_sample(rows: list[dict], n_per_source_label: int, seed: int) -> list[dict]:
    """Cap rows per (source, label) so the expensive pass stays bounded + balanced."""
    rng = random.Random(seed)
    buckets: dict[tuple, list] = {}
    for r in rows:
        buckets.setdefault((r.get("source", "?"), r["label"]), []).append(r)
    out = []
    for items in buckets.values():
        rng.shuffle(items)
        out.extend(items[:n_per_source_label])
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "corpus" / "scores.jsonl"))
    ap.add_argument("--balance", type=int, default=0,
                    help="cap rows per (source,label); 0 = score everything")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--observer", default=None)
    ap.add_argument("--performer", default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args(argv)

    rows = load([Path(p) for p in args.corpus])
    if args.balance:
        rows = balance_sample(rows, args.balance, args.seed)
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    # skip ids already scored (resumable)
    out_path = Path(args.out)
    done: set[str] = set()
    if args.append and out_path.exists():
        with out_path.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    rows = [r for r in rows if r["id"] not in done]
    print(f"scoring {len(rows)} rows ({len(done)} already cached) …", file=sys.stderr)

    kw = {"max_tokens": args.max_tokens}
    if args.observer:
        kw["observer"] = args.observer
    if args.performer:
        kw["performer"] = args.performer
    scorer = LikelihoodScorer(**kw)

    mode = "a" if (args.append and out_path.exists()) else "w"
    t0 = time.time()
    n = 0
    with out_path.open(mode) as f:
        B = 256  # write in waves
        for i in range(0, len(rows), B):
            wave = rows[i:i + B]
            feats = scorer.score_batch([r["text"] for r in wave], batch_size=args.batch_size)
            for r, fe in zip(wave, feats):
                f.write(json.dumps({"id": r["id"], "features": fe}) + "\n")
            f.flush()
            n += len(wave)
            rate = n / (time.time() - t0)
            print(f"\r  {n}/{len(rows)}  ({rate:.1f}/s, {len(rows)-n} left, "
                  f"~{(len(rows)-n)/rate:.0f}s)", end="", file=sys.stderr)
    print(f"\ndone: {n} scored in {time.time()-t0:.0f}s -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
