#!/usr/bin/env python
from __future__ import annotations

"""
Compute normalised frequency (OPM) for a given vocabulary from a ufb-lex:1.0
JSONL file. Extracts texts from the chosen/rejected/both sides, computes
occurrences-per-million per vocab item, and writes a TSV {word, opm} to --out.
"""

import argparse
from pathlib import Path

import pandas as pd

from lhf_lex.io.ufb_lex import stream_texts
from lhf_lex.metrics.freq import normalised_frequency


def read_vocab(path: str) -> list[str]:
    """Load a lowercase vocabulary from a text file.

    Empty lines are skipped; lines starting with '#' are treated as comments.

    Args:
        path: UTF-8 text file with one item per line.

    Returns:
        List of vocab items (lowercased).
    """
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip().lower() for ln in f if ln.strip() and not ln.startswith("#")]


def main() -> None:
    """Parse CLI args, compute OPM over the vocab, and write a TSV."""
    ap = argparse.ArgumentParser(description="Compute placeholder normalised frequency (OPM).")
    ap.add_argument("jsonl", type=str, help="Path to ufb-lex:1.0 JSONL file.")
    ap.add_argument("--vocab", type=str, required=True, help="Path to vocab txt (one item per line).")
    ap.add_argument("--side", type=str, default="both", choices=["chosen", "rejected", "both"])
    ap.add_argument("--out", type=str, required=True, help="Output TSV path.")
    args = ap.parse_args()

    vocab = read_vocab(args.vocab)
    texts = stream_texts(args.jsonl, side=args.side)
    opm = normalised_frequency(texts, vocab)

    rows = [{"word": w, "opm": opm[w]} for w in sorted(opm.keys())]
    df = pd.DataFrame(rows)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, sep="\t", index=False)
    print(f"wrote {len(df)} rows to {outp}")


if __name__ == "__main__":
    main()
