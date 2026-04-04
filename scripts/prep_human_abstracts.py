#!/usr/bin/env python
from __future__ import annotations

"""
Prepare human abstracts for prompting: load YYYY.txt files, sample N per year,
split into first/second halves, and write (i) a JSONL of records and (ii) a
prompts .txt containing only the first halves.
"""

import argparse
import json
from pathlib import Path

from lhf_lex.data.human_abstracts import (
    load_human_abstracts_dir,
    sample_per_year,
    to_jsonl_rows,
)


def main() -> None:
    """CLI entry: sample, split, and write outputs.

    Reads abstracts from a root folder of YYYY.txt files, samples a fixed number
    per year, converts to JSONL rows (with first/second halves), and writes:
      • --out-jsonl: one JSON object per abstract record
      • --out-prompts: line-delimited first halves for generation prompts

    Exits with a message if no abstracts are found for the requested years.
    """
    ap = argparse.ArgumentParser(
        description="Prepare human abstracts: sample per year, split into halves, write JSONL and prompts."
    )
    ap.add_argument("root", help="Folder containing YYYY.txt files (one abstract per line)")
    ap.add_argument("--years", nargs="*", type=int, default=None, help="Years to include (default: auto-detect)")
    ap.add_argument("--sample-per-year", type=int, default=5, help="N per year to sample (default: 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-jsonl", default="data/prompts/human_halves.jsonl")
    ap.add_argument("--out-prompts", default="data/prompts/human_first_halves.txt")
    args = ap.parse_args()

    recs = load_human_abstracts_dir(args.root, args.years)
    if not recs:
        raise SystemExit(f'No abstracts found under {args.root} for years {args.years or "(auto)"}')

    sampled = sample_per_year(recs, n_per_year=args.sample_per_year, seed=args.seed)
    rows = to_jsonl_rows(sampled)

    outj = Path(args.out_jsonl)
    outj.parent.mkdir(parents=True, exist_ok=True)
    with outj.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote JSONL: {outj} ({len(rows)} rows)")

    outp = Path(args.out_prompts)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write((r.get("first_half") or "").strip() + "\n")
    print(f"Wrote prompts: {outp} ({len(rows)} prompts)")


if __name__ == "__main__":
    main()
