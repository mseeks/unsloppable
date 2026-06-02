#!/usr/bin/env python3
"""Ingest the HC3 corpus (Human-ChatGPT Comparison Corpus) into a flat JSONL.

HC3 gives, per question, a list of real human answers and a list of ChatGPT
answers. We flatten that into labeled samples the linter can score:

    {"id", "text", "label" ("human"|"ai"), "domain", "source": "HC3:<rev>", "question"}

stdlib only — no `datasets`, no API key. Source:
https://huggingface.co/datasets/Hello-SimpleAI/HC3

Usage:
    python3 fetch_hc3.py                          # default domains, balanced
    python3 fetch_hc3.py --domains open_qa medicine --limit 400
    python3 fetch_hc3.py --out corpus/hc3.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main"
# Domains with real content (reddit_eli5 / all are ~1KB stubs on the hub).
ALL_DOMAINS = ["open_qa", "wiki_csai", "medicine", "finance"]
DEFAULT_DOMAINS = ["open_qa", "wiki_csai", "medicine"]

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def _download(domain: str) -> list[dict]:
    url = f"{BASE}/{domain}.jsonl"
    req = urllib.request.Request(url, headers={"User-Agent": "unsloppable-eval/0.1"})
    print(f"  downloading {domain} …", file=sys.stderr)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _clean(text: str) -> str:
    # Light touch: collapse runaway whitespace, strip. Keep the prose intact —
    # we want to detect AI tells in real output, not in a sanitized version.
    return " ".join(text.split()).strip()


def build(
    domains: list[str],
    per_question_human: int,
    per_question_ai: int,
    min_words: int,
    limit: int | None,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []
    for domain in domains:
        records = _download(domain)
        rng.shuffle(records)
        kept = 0
        for idx, rec in enumerate(records):
            if limit is not None and kept >= limit:
                break
            humans = [_clean(a) for a in rec.get("human_answers", [])]
            ais = [_clean(a) for a in rec.get("chatgpt_answers", [])]
            humans = [a for a in humans if len(a.split()) >= min_words]
            ais = [a for a in ais if len(a.split()) >= min_words]
            if not humans or not ais:
                continue  # keep it paired so the split stays class-balanced
            q = _clean(rec.get("question", ""))[:300]
            for j, a in enumerate(humans[:per_question_human]):
                samples.append({
                    "id": f"hc3-{domain}-{idx}-human-{j}",
                    "text": a, "label": "human", "domain": domain,
                    "source": "HC3", "model": "human", "era": "2022",
                    "license": "HC3 (Guo et al. 2023)", "question": q,
                })
            for j, a in enumerate(ais[:per_question_ai]):
                samples.append({
                    "id": f"hc3-{domain}-{idx}-ai-{j}",
                    "text": a, "label": "ai", "domain": domain,
                    "source": "HC3", "model": "chatgpt-2022", "era": "2022",
                    "license": "HC3 (Guo et al. 2023)", "question": q,
                })
            kept += 1
    return samples


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS,
                    choices=ALL_DOMAINS, help="HC3 domains to ingest")
    ap.add_argument("--limit", type=int, default=300,
                    help="max questions per domain (None = all)")
    ap.add_argument("--per-question-human", type=int, default=1)
    ap.add_argument("--per-question-ai", type=int, default=1)
    ap.add_argument("--min-words", type=int, default=50,
                    help="drop answers shorter than this (too little signal)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(CORPUS_DIR / "hc3.jsonl"))
    args = ap.parse_args(argv)

    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    samples = build(args.domains, args.per_question_human, args.per_question_ai,
                    args.min_words, limit, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_human = sum(1 for s in samples if s["label"] == "human")
    n_ai = sum(1 for s in samples if s["label"] == "ai")
    print(f"\nwrote {len(samples)} samples → {out}", file=sys.stderr)
    print(f"  human: {n_human}   ai: {n_ai}   domains: {', '.join(args.domains)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
