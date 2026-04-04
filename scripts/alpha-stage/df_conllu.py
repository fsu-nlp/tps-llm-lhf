#!/usr/bin/env python3
"""
Compute document frequencies (DF) from CoNLL-U, per model × submodel, and
write Top-25 and Bottom-25 items to TSV files.

Definition (default):
  - Document = span between lines starting with "# newdoc id ="
  - DF(w) = (# docs containing w ≥1×) / (total # docs in the group)
  - Token key = lemma or lemma+UPOS (configurable)

Grouping:
  - Root is expected like: <POS_ROOT>/<model_name>/*.conllu[.gz]
  - Submodel inferred from filename: contains "_base_" -> "base",
    "_instruct_" -> "instruct", contains "human" -> "human", else "unknown".

Outputs:
  - <OUT_ROOT>/<model_name>/<submodel>_df_top_bottom.tsv
    Columns: rank_type, rank, key, docs_with, total_docs, df

Usage:
  python scripts-ig/df_conllu.py \
    --pos-root out-ig/pre/pos-tr-full \
    --out-root out-ig/pre/analyses/df \
    --key lemma_pos \
    --min-docs 1 \
    --topk 25 \
    --exclude-upos PUNCT SYM

Notes:
  - Empty documents (no tokens) are ignored for DF denominators.
  - Multi-word tokens (ID with '-') and empty nodes (ID with '.') are skipped.
"""

from __future__ import annotations
import argparse
import gzip
from pathlib import Path
from collections import Counter, defaultdict
import re
import sys

NEWDOC_RE = re.compile(r"^#\s*newdoc id\s*=")

def open_maybe_gz(p: Path):
    return gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, "r", encoding="utf-8")

def infer_submodel(filename: str) -> str:
    f = filename.lower()
    if "_base_" in f:
        return "base"
    if "_instruct_" in f:
        return "instruct"
    if "human" in f:
        return "human"
    return "unknown"

def parse_conllu_docs(path: Path, key_mode: str, exclude_upos: set[str]) -> list[set[str]]:
    """
    Return a list of per-document sets of keys (lemma or lemma+UPOS).
    Empty docs (no tokens) are returned as empty sets (and filtered by caller).
    """
    docs: list[set[str]] = []
    cur: set[str] = set()

    with open_maybe_gz(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                if NEWDOC_RE.match(line):
                    # Start of a new doc => push previous
                    docs.append(cur)
                    cur = set()
                continue
            # token row
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tid = cols[0]
            if "-" in tid or "." in tid:  # skip multi-word/empty nodes
                continue
            lemma = cols[2].strip()
            upos = cols[3].strip()
            if upos in exclude_upos:
                continue
            if not lemma or lemma == "_":
                continue
            lemma = lemma.lower()
            if key_mode == "lemma":
                key = lemma
            elif key_mode == "lemma_pos":
                key = f"{lemma}_{upos}"
            else:
                raise ValueError(f"Unsupported key mode: {key_mode}")
            cur.add(key)

    # Append last doc
    docs.append(cur)
    # First element can be empty if file didn't start with newdoc; keep as is and let caller filter empties
    return docs

def collect_groups(pos_root: Path, key_mode: str, exclude_upos: set[str]) -> dict[tuple[str,str], list[set[str]]]:
    """
    Returns dict[(model, submodel)] -> list of doc-sets.
    """
    groups: dict[tuple[str,str], list[set[str]]] = defaultdict(list)
    for model_dir in sorted(p for p in pos_root.iterdir() if p.is_dir()):
        model = model_dir.name
        conllus = sorted(list(model_dir.glob("*.conllu")) + list(model_dir.glob("*.conllu.gz")))
        if not conllus:
            continue
        for f in conllus:
            submodel = infer_submodel(f.name)
            docs = parse_conllu_docs(f, key_mode=key_mode, exclude_upos=exclude_upos)
            # Filter truly empty docs (no tokens seen)
            docs = [d for d in docs if len(d) > 0]
            if not docs:
                continue
            groups[(model, submodel)].extend(docs)
    return groups

def df_stats(doc_sets: list[set[str]]) -> tuple[Counter, int]:
    """
    Returns (doc_count_per_key, total_docs)
    """
    total_docs = len(doc_sets)
    counts = Counter()
    for s in doc_sets:
        counts.update(s)
    return counts, total_docs

def select_top_bottom(counts: Counter, total_docs: int, topk: int, min_docs: int) -> tuple[list[tuple], list[tuple]]:
    """
    Return sorted top and bottom lists with entries: (key, docs_with, df)
    - Top: highest DF (desc), tiebreak by docs_with desc then key asc
    - Bottom: among keys with docs_with >= min_docs (>=1 by default), lowest DF (asc), tiebreak by docs_with asc then key asc
    """
    rows = [(k, c, c / total_docs) for k, c in counts.items() if c >= min_docs]
    if not rows:
        return [], []
    top = sorted(rows, key=lambda x: (-x[2], -x[1], x[0]))[:topk]
    bottom = sorted(rows, key=lambda x: (x[2], x[1], x[0]))[:topk]
    return top, bottom

def write_tsv(out_path: Path, top, bottom, total_docs: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        w.write("rank_type\trank\tkey\tdocs_with\ttotal_docs\tdf\n")
        for i, (k, c, df) in enumerate(top, 1):
            w.write(f"top\t{i}\t{k}\t{c}\t{total_docs}\t{df:.6f}\n")
        for i, (k, c, df) in enumerate(bottom, 1):
            w.write(f"bottom\t{i}\t{k}\t{c}\t{total_docs}\t{df:.6f}\n")

def main():
    ap = argparse.ArgumentParser(description="Compute document frequencies (DF) from CoNLL-U by model×submodel.")
    ap.add_argument("--pos-root", required=True, type=Path, help="Root with parsed CoNLL-U subfolders per model.")
    ap.add_argument("--out-root", required=True, type=Path, help="Where to write per-group TSV outputs.")
    ap.add_argument("--key", choices=["lemma", "lemma_pos"], default="lemma_pos", help="Key granularity.")
    ap.add_argument("--topk", type=int, default=25, help="How many top and bottom entries to write.")
    ap.add_argument("--min-docs", type=int, default=1, help="Minimum #docs for an item to be considered (bottom/top).")
    ap.add_argument("--exclude-upos", nargs="*", default=["PUNCT", "SYM"], help="UPOS tags to exclude from DF.")
    args = ap.parse_args()

    exclude_upos = set(args.exclude_upos)

    groups = collect_groups(args.pos_root, key_mode=args.key, exclude_upos=exclude_upos)
    if not groups:
        print(f"[df] No groups found under {args.pos_root}", file=sys.stderr)
        sys.exit(1)

    wrote = 0
    for (model, submodel), doc_sets in sorted(groups.items()):
        counts, total_docs = df_stats(doc_sets)
        if total_docs == 0 or not counts:
            continue
        top, bottom = select_top_bottom(counts, total_docs, topk=args.topk, min_docs=args.min_docs)
        out_dir = args.out_root / model
        out_file = out_dir / f"{submodel}_df_top_bottom.tsv"
        write_tsv(out_file, top, bottom, total_docs)
        wrote += 1
        print(f"[df] {model}/{submodel}: docs={total_docs}, vocab={len(counts)} → {out_file}")

    if wrote == 0:
        print("[df] Nothing written (no eligible docs/items).", file=sys.stderr)

if __name__ == "__main__":
    main()
