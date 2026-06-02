#!/usr/bin/env python3
"""Phase 2 — generate fresh AI text on random topics, labeled for the corpus.

HC3's AI side is 2022 ChatGPT. This generates text from a *current* model so we
can measure whether the linter still fires on today's output. Defaults to the
Q&A genre so samples are directly comparable to HC3's human answers (same genre
= the AUC measures AI-ness, not genre); other genres stress-test for blind spots.

Runs on your Claude subscription via the Agent SDK (see llm.py). Each sample is
one model call, so mind the budget: --n and --max-cost bound the spend.

Usage:
    uv run python eval/generate.py                          # 16 Q&A samples
    uv run python eval/generate.py --genres qa blog --n 24
    uv run python eval/generate.py --n 40 --max-cost 2.0 --append
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from llm import generate_text, auth_note, GEN_MODEL

ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = ROOT / "corpus" / "generated.jsonl"

# Neutral system prompt: we want the model's DEFAULT style, not coached prose —
# the whole point is to see what tells current models emit unprompted.
SYSTEM = ("You are a writing assistant. Produce the requested text and nothing "
          "else: no preamble, no title or heading, no markdown, no sign-off. "
          "Write it the way you naturally would.")

# Genre → instruction template. {topic} is filled per sample.
GENRES = {
    "qa": "Write a clear, informative ~150-word answer to a general-knowledge question about {topic}.",
    "blog": "Write the opening 2-3 paragraphs (~150 words) of a blog post about {topic}.",
    "linkedin": "Write a ~120-word LinkedIn post about {topic}.",
    "product": "Write a ~100-word marketing product description for {topic}.",
    "explainer": "Explain {topic} to a curious adult in about 150 words.",
}

# Reproducible "random topics" — a diverse bank sampled with a seed.
TOPICS = [
    "how vaccines train the immune system", "the causes of the 2008 financial crisis",
    "why the sky is blue", "the history of the printing press", "how black holes form",
    "the basics of compound interest", "photosynthesis", "the fall of the Roman Empire",
    "how GPS determines your location", "the water cycle", "what causes inflation",
    "the rules of chess", "how antibiotics work", "the invention of the transistor",
    "ocean currents and climate", "how a refrigerator works", "the French Revolution",
    "what DNA does", "the theory of plate tectonics", "how credit scores are calculated",
    "the Apollo moon landings", "why leaves change color in autumn", "how encryption works",
    "the origins of jazz", "what happens during an earthquake", "how bees make honey",
    "the Industrial Revolution", "how the stock market works", "the human sleep cycle",
    "the discovery of penicillin", "how solar panels generate electricity",
    "the causes of World War I", "what makes a good cup of coffee", "how memory works",
    "the life cycle of a star", "the basics of machine learning", "how vaccines are developed",
    "the Silk Road trade routes", "why we dream", "how a combustion engine works",
    "the greenhouse effect", "the rise of social media", "how muscles grow",
    "the printing of money by central banks", "the domestication of dogs",
]


async def _gen_one(sem: asyncio.Semaphore, genre: str, topic: str, idx: int,
                   model: str) -> dict | None:
    instruction = GENRES[genre].format(topic=topic)
    async with sem:
        try:
            text, cost = await generate_text(instruction, SYSTEM, model=model)
        except Exception as e:
            print(f"  ! {genre}/{idx} failed: {type(e).__name__}: {str(e)[:120]}",
                  file=sys.stderr)
            return None
    if len(text.split()) < 30:
        print(f"  ! {genre}/{idx} too short ({len(text.split())}w), skipped",
              file=sys.stderr)
        return None
    print(f"  ✓ {genre}/{idx}  {len(text.split())}w  ${cost:.4f}  «{topic[:40]}»",
          file=sys.stderr)
    return {"id": f"gen-{genre}-{idx}", "text": text, "label": "ai",
            "domain": genre, "source": f"gen:{model}", "model": model, "era": "2026",
            "license": "self-generated", "question": topic, "_cost": cost}


async def run(genres: list[str], n: int, model: str, concurrency: int,
              max_cost: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    # Build a topic/genre work list of length n, cycling topics as needed.
    jobs = []
    for i in range(n):
        genre = genres[i % len(genres)]
        topic = rng.choice(TOPICS)
        jobs.append((genre, topic, i))

    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    total_cost = 0.0
    # Process in concurrency-sized waves so we can honor the budget cap.
    pending = [_gen_one(sem, g, t, i, model) for (g, t, i) in jobs]
    for coro in asyncio.as_completed(pending):
        rec = await coro
        if rec is None:
            continue
        total_cost += rec.pop("_cost", 0.0)
        results.append(rec)
        if total_cost >= max_cost:
            print(f"  budget cap ${max_cost} reached (spent ${total_cost:.3f}); "
                  f"stopping early with {len(results)} samples", file=sys.stderr)
            break
    print(f"\ntotal cost: ${total_cost:.3f} for {len(results)} samples", file=sys.stderr)
    return results


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genres", nargs="+", default=["qa"], choices=list(GENRES))
    ap.add_argument("--n", type=int, default=16, help="number of samples to generate")
    ap.add_argument("--model", default=GEN_MODEL)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-cost", type=float, default=1.5, help="USD budget cap")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--append", action="store_true",
                    help="append to the output file instead of overwriting")
    args = ap.parse_args(argv)

    print(auth_note(), file=sys.stderr)
    print(f"generating {args.n} samples · genres {args.genres} · model {args.model}",
          file=sys.stderr)

    results = asyncio.run(run(args.genres, args.n, args.model, args.concurrency,
                              args.max_cost, args.seed))
    if not results:
        print("no samples generated", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with out.open(mode) as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(results)} samples → {out} ({'appended' if args.append else 'overwrote'})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
