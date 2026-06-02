#!/usr/bin/env python3
"""Ingest RAID (no-adversarial, labeled) into the harness JSONL schema.

RAID (Dugan et al. 2024, MIT) is a large modern detection benchmark: human text
plus generations from many 2023–24 models across genres. HC3 is a dead 2022
distribution; RAID is the modern, multi-model, genre-matched reality check — and
the human side is varied prose (reddit/news/reviews/books), which breaks HC3's
encyclopedic-register confound.

We pull the LABELED `train_none.csv` (the public test set hides labels), keep
English prose domains, sample a model+domain-balanced ~1:1 subset, and tag full
provenance (model/era/decoding/license) so per-era/per-source drift stays analyzable.

stdlib only. Downloads ~258 MB once to a gitignored cache.

Usage:
    python3 fetch_raid.py                 # ~2400 balanced rows -> corpus/raid.jsonl
    python3 fetch_raid.py --per-cell 60 --min-words 50 --max-words 600
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path

URL = "https://dataset.raid-bench.xyz/train_none.csv"
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
CACHE = CORPUS / "_raid_train_none.csv"

# English prose domains (drop code, poetry, recipes, czech_news, german_news).
KEEP_DOMAINS = {"abstracts", "books", "news", "reddit", "reviews", "wiki"}
# Modern models worth detecting; gpt2 (2019) dropped to avoid a third era.
DROP_MODELS = {"gpt2"}


def download(force: bool = False) -> Path:
    CORPUS.mkdir(parents=True, exist_ok=True)
    if CACHE.exists() and not force and CACHE.stat().st_size > 1_000_000:
        print(f"  using cached {CACHE.name} ({CACHE.stat().st_size/1e6:.0f} MB)", file=sys.stderr)
        return CACHE
    print(f"  downloading RAID train_none.csv (~258 MB) …", file=sys.stderr)
    req = urllib.request.Request(URL, headers={"User-Agent": "unsloppable-eval/0.2"})
    with urllib.request.urlopen(req, timeout=600) as resp, CACHE.open("wb") as out:
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            print(f"\r    {done/1e6:.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    return CACHE


def build(per_cell: int, min_words: int, max_words: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    csv.field_size_limit(1 << 24)
    # bucket rows by (label-cell): human by domain; ai by (domain, model)
    human: dict[str, list] = {}
    ai: dict[tuple, list] = {}
    with CACHE.open(newline="") as f:
        for row in csv.DictReader(f):
            dom = row.get("domain", "")
            if dom not in KEEP_DOMAINS:
                continue
            model = row.get("model", "")
            if model in DROP_MODELS:
                continue
            text = " ".join((row.get("generation") or "").split()).strip()
            wc = len(text.split())
            if wc < min_words or wc > max_words:
                continue
            rec = {"text": text, "domain": dom, "model": model,
                   "decoding": row.get("decoding") or None,
                   "title": (row.get("title") or "")[:300],
                   "source_id": row.get("source_id") or row.get("id")}
            if model == "human":
                human.setdefault(dom, []).append(rec)
            else:
                ai.setdefault((dom, model), []).append(rec)

    # sample balanced cells
    picked_ai = []
    for items in ai.values():
        rng.shuffle(items)
        picked_ai.extend(items[:per_cell])
    # match human count to ai count, spread across domains
    n_target = len(picked_ai)
    doms = list(human)
    picked_human = []
    per_dom = max(1, n_target // max(1, len(doms)))
    for dom in doms:
        items = human[dom]
        rng.shuffle(items)
        picked_human.extend(items[:per_dom])
    rng.shuffle(picked_human)
    picked_human = picked_human[:n_target]

    samples = []
    for lab, recs in (("human", picked_human), ("ai", picked_ai)):
        for i, r in enumerate(recs):
            samples.append({
                "id": f"raid-{r['domain']}-{r['model']}-{i}",
                "text": r["text"], "label": lab, "domain": r["domain"],
                "source": "RAID", "model": r["model"], "era": "2023-24",
                "decoding": r["decoding"], "license": "MIT",
                # group key: same source document -> human + its machine continuations
                "question": f"raid:{r['domain']}:{r['source_id']}",
            })
    rng.shuffle(samples)
    return samples


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cell", type=int, default=60,
                    help="max AI samples per (domain, model) cell")
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--max-words", type=int, default=600)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--out", default=str(CORPUS / "raid.jsonl"))
    args = ap.parse_args(argv)

    download(force=args.force_download)
    samples = build(args.per_cell, args.min_words, args.max_words, args.seed)
    out = Path(args.out)
    with out.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_h = sum(1 for s in samples if s["label"] == "human")
    n_a = len(samples) - n_h
    from collections import Counter
    by_model = Counter(s["model"] for s in samples if s["label"] == "ai")
    by_dom = Counter(s["domain"] for s in samples)
    print(f"\nwrote {len(samples)} -> {out}  (human {n_h} / ai {n_a})", file=sys.stderr)
    print(f"  models: {dict(by_model)}", file=sys.stderr)
    print(f"  domains: {dict(by_dom)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
