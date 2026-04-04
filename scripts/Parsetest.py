import argparse

def main():
    ap = argparse.ArgumentParser(description="Toy example")
    ap.add_argument("input_path", help="Positional arg: required path")        # e.g., data.jsonl
    ap.add_argument("--limit", type=int, default=100, help="Optional flag")    # e.g., --limit 50
    args = ap.parse_args()

    print("input_path =", args.input_path)
    print("limit =", args.limit)

if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import sys
import pandas as pd

from lhf_lex.io.ufb_lex import stream_texts
from lhf_lex.metrics.freq import normalised_frequency


def read_vocab(path):
    """Load a lowercase vocabulary from a text file.
    Skips empty lines and lines starting with '#'.
    """
    vocab = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            stripped = ln.strip()
            if not stripped or stripped.startswith("#"):
                continue
            vocab.append(stripped.lower())
    return vocab


def main():
    # ---- argparse: define CLI ----
    parser = argparse.ArgumentParser(
        prog="opm_freq",
        description="Compute occurrences-per-million (OPM) for a vocabulary over a ufb-lex:1.0 JSONL corpus."
    )
    parser.add_argument(
        "jsonl",
        type=str,
        help="Path to ufb-lex:1.0 JSONL file."
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to UTF-8 vocab file (one item per line; '#' lines are comments)."
    )
    parser.add_argument(
        "--side",
        type=str,
        default="both",
        choices=["chosen", "rejected", "both"],
        help="Which side(s) of the preference data to use (default: both)."
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output TSV path."
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="word",
        choices=["word", "opm"],
        help="Sort output by 'word' or 'opm' (default: word)."
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="If set, sort in descending order (useful with --sort-by opm)."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="If set, only write the top K rows after sorting."
    )

    args = parser.parse_args()

    # ---- basic file checks (fail fast with clear messages) ----
    jsonl_path = Path(args.jsonl)
    vocab_path = Path(args.vocab)
    out_path = Path(args.out)

    if not jsonl_path.is_file():
        print(f"ERROR: JSONL file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(2)
    if not vocab_path.is_file():
        print(f"ERROR: vocab file not found: {vocab_path}", file=sys.stderr)
        sys.exit(2)

    # ---- load vocab ----
    try:
        vocab = read_vocab(str(vocab_path))
    except UnicodeDecodeError as e:
        print(f"ERROR: Failed to read vocab as UTF-8: {vocab_path}\n{e}", file=sys.stderr)
        sys.exit(2)

    if not vocab:
        print("ERROR: vocab is empty after filtering blanks/comments.", file=sys.stderr)
        sys.exit(2)

    # ---- stream texts + compute OPM ----
    try:
        texts = stream_texts(str(jsonl_path), side=args.side)  # should be an iterator/generator
        opm = normalised_frequency(texts, vocab)               # expected to return dict: {word: float}
    except Exception as e:
        print(f"ERROR: failed during frequency computation: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- rows -> DataFrame ----
    # Ensure every vocab item has an entry; if missing, default to 0.0
    rows = [{"word": w, "opm": float(opm.get(w, 0.0))} for w in vocab]

    df = pd.DataFrame(rows)

    # ---- sort + (optional) top-k ----
    df = df.sort_values(by=args.sort_by, ascending=not args.descending, kind="mergesort")
    # mergesort is stable; if sorting by opm ties, original vocab order is preserved

    if args.top is not None and args.top > 0:
        df = df.head(args.top)

    # ---- write TSV ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()