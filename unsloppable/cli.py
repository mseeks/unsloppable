"""Command-line interface — usable as a CI/pre-commit slop gate and by agents.

    unsloppable file.md [more.md ...]      # score files
    echo "text" | unsloppable -            # read stdin
    unsloppable --json draft.md            # machine-readable (for an agent)
    unsloppable --ml draft.md              # likelihood layer + P(AI)  (needs the ml extra)
    echo "text" | unsloppable --json --ml -   # agent self-check shape

Exit code is 1 if any input is flagged (score≥threshold, or P(AI)≥--prob with
--ml), 0 otherwise — so it fails a build/commit on sloppy text. --quiet suppresses
per-file output (exit code only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import lint
from .config import DEFAULT


def _emit_human(name: str, r, show_advice: bool) -> None:
    print(f"\n=== {name} ===")
    line = f"score: {r.score}  ({r.verdict})"
    if r.probability is not None:
        line += f"  ·  P(AI)={r.probability}"
    if r.low_confidence:
        line += "  ·  [low confidence: short text]"
    print(line)
    cv = r.features.get("sentence_cv")
    cv_s = "n/a" if cv is None or (isinstance(cv, float) and cv != cv) else f"{cv:.2f}"
    print(f"words: {r.word_count}  ·  cv: {cv_s}  "
          f"·  ttr: {round(r.features.get('type_token_ratio', 0), 2)}")
    issues = r.top_issues()
    if issues and show_advice:
        print("issues to revise:")
        for s in issues:
            print(f"  - {s.name} (+{s.contribution}) → {s.advice}")
    elif not issues:
        print("  no AI-leaning signals")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="unsloppable", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="files to lint; use '-' for stdin")
    ap.add_argument("--json", action="store_true", help="emit JSON (one object per input)")
    ap.add_argument("--ml", action="store_true",
                    help="add the likelihood layer + calibrated P(AI) (needs the ml extra)")
    ap.add_argument("--threshold", type=float, default=DEFAULT.tier_light,
                    help="heuristic score at/above which an input is flagged (exit 1)")
    ap.add_argument("--prob", type=float, default=0.5,
                    help="with --ml, P(AI) at/above which an input is flagged")
    ap.add_argument("--quiet", action="store_true", help="no per-file output, just exit code")
    args = ap.parse_args(argv)

    if not args.paths:
        ap.print_help()
        return 2

    inputs: list[tuple[str, str]] = []
    for p in args.paths:
        if p == "-":
            inputs.append(("<stdin>", sys.stdin.read()))
        else:
            path = Path(p)
            if not path.exists():
                print(f"{p}: not found", file=sys.stderr)
                continue
            try:
                inputs.append((str(path), path.read_text(encoding="utf-8", errors="replace")))
            except OSError as e:
                print(f"{p}: {e}", file=sys.stderr)
                continue

    flagged = False
    results = []
    for name, text in inputs:
        r = lint(text, ml=args.ml)
        # Don't fail the gate on text too short to score reliably.
        if r.word_count < 20:
            hit = False
        elif args.ml and r.probability is not None:
            hit = r.probability >= args.prob
        else:
            hit = r.score >= args.threshold
        flagged = flagged or hit
        if args.json:
            results.append({"name": name, "flagged": hit, **r.to_dict()})
        elif not args.quiet:
            _emit_human(name, r, show_advice=True)

    if args.json:
        print(json.dumps(results, indent=2, allow_nan=False))
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
