#!/usr/bin/env python3
"""
DF shift (instruct−base) on paired docs using the first K eligible tokens.

Additions:
  - --windowk K  (preferred) sets K, the window size (default: 50).
  - --N is retained for backwards-compatibility; if both given, --windowk wins.

Other behaviour unchanged:
  - Pair docs by idx from '# newdoc id = ... idx<NUM> ...'
  - Exclude pairs where either side has < K eligible tokens (after UPOS exclusion).
  - Key granularity via --key {lemma,lemma_pos}.
  - Outputs per model: <OUT_ROOT>/<model>/dfdiff_top_bottom.tsv

Example:
  python scripts-ig/df_diff_base-instruct.py \
    --pos-root out-ig/pre/pos-tr-full \
    --out-root out-ig/pre/analyses/diff \
    --key lemma_pos \
    --topk 25 \
    --exclude-upos PUNCT SYM \
    --windowk 50
"""

from __future__ import annotations
import argparse
import gzip
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

NEWDOC_RE = re.compile(r"^#\s*newdoc id\s*=")
IDX_RE = re.compile(r"idx(\d+)")
BASE_HINT = "_base_"
INSTR_HINT = "_instruct_"

def open_maybe_gz(p: Path):
    return gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, "r", encoding="utf-8")

def extract_idx(header_line: str) -> int | None:
    m = IDX_RE.search(header_line)
    return int(m.group(1)) if m else None

def conllu_firstK_keys(path: Path, key_mode: str, exclude_upos: Set[str], K: int) -> Dict[int, Tuple[Set[str], int]]:
    """
    Return dict: idx -> (set(keys within first K eligible tokens), total_eligible_in_doc)
    Eligible = token after UPOS exclusion; skip multiword/empty nodes.
    """
    out: Dict[int, Tuple[Set[str], int]] = {}
    cur_idx: int | None = None
    seen_keys: Set[str] = set()
    taken: int = 0
    total_elig: int = 0

    with open_maybe_gz(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                if NEWDOC_RE.match(line):
                    if cur_idx is not None:
                        out[cur_idx] = (seen_keys, total_elig)
                    cur_idx = extract_idx(line)
                    seen_keys = set()
                    taken = 0
                    total_elig = 0
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tid = cols[0]
            if "-" in tid or "." in tid:
                continue
            lemma = cols[2].strip()
            upos = cols[3].strip()
            if upos in exclude_upos:
                continue
            if not lemma or lemma == "_":
                continue
            total_elig += 1
            if taken >= K:
                continue
            lemma = lemma.lower()
            if key_mode == "lemma":
                key = lemma
            elif key_mode == "lemma_pos":
                key = f"{lemma}_{upos}"
            else:
                raise ValueError(f"Unsupported --key={key_mode}")
            seen_keys.add(key)
            taken += 1

    if cur_idx is not None:
        out[cur_idx] = (seen_keys, total_elig)
    return out

def find_pair_files(model_dir: Path) -> Tuple[Path | None, Path | None]:
    base = None
    instr = None
    for p in sorted(list(model_dir.glob("*.conllu")) + list(model_dir.glob("*.conllu.gz"))):
        name = p.name.lower()
        if BASE_HINT in name:
            base = p
        elif INSTR_HINT in name:
            instr = p
    return base, instr

def write_tsv(path: Path, top_rows, bottom_rows, total_pairs: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        w.write("rank_type\trank\tkey\tbase_docs\tinstr_docs\ttotal_pairs\tdf_base\tdf_instr\tdelta\n")
        for i, (k, cb, ci, db, di, dd) in enumerate(top_rows, 1):
            w.write(f"up\t{i}\t{k}\t{cb}\t{ci}\t{total_pairs}\t{db:.6f}\t{di:.6f}\t{dd:.6f}\n")
        for i, (k, cb, ci, db, di, dd) in enumerate(bottom_rows, 1):
            w.write(f"down\t{i}\t{k}\t{cb}\t{ci}\t{total_pairs}\t{db:.6f}\t{di:.6f}\t{dd:.6f}\n")

def main():
    ap = argparse.ArgumentParser(description="DF shift (instruct−base) on paired docs using the first K eligible tokens.")
    ap.add_argument("--pos-root", required=True, type=Path, help="Root with subfolders per model containing .conllu[.gz].")
    ap.add_argument("--out-root", required=True, type=Path, help="Output root for per-model TSVs.")
    ap.add_argument("--key", choices=["lemma", "lemma_pos"], default="lemma_pos", help="Key granularity.")
    ap.add_argument("--exclude-upos", nargs="*", default=["PUNCT", "SYM"], help="UPOS tags to exclude.")
    # Window size flags: preferred --windowk; legacy --N
    ap.add_argument("--windowk", type=int, default=None, help="Window size K: number of eligible tokens from start of each doc.")
    ap.add_argument("--N", type=int, default=None, help="(Deprecated) Window size; use --windowk instead.")
    ap.add_argument("--topk", type=int, default=25, help="How many largest positive and negative deltas to write.")
    ap.add_argument("--min_docs", type=int, default=1, help="Minimum docs per side to include in ranking.")
    args = ap.parse_args()

    # Resolve K (prefer --windowk, fallback to --N, default 50)
    if args.windowk is not None:
        K = int(args.windowk)
    elif args.N is not None:
        K = int(args.N)
    else:
        K = 50

    if K <= 0:
        print("[diff] K must be > 0", file=sys.stderr)
        sys.exit(2)

    exclude_upos = set(args.exclude_upos)

    model_dirs = [p for p in args.pos_root.iterdir() if p.is_dir()]
    if not model_dirs:
        print(f"[diff] No model directories under {args.pos_root}", file=sys.stderr)
        sys.exit(1)

    wrote_any = False
    for model_dir in sorted(model_dirs):
        model = model_dir.name
        base_f, instr_f = find_pair_files(model_dir)
        if not base_f or not instr_f:
            print(f"[diff] {model}: missing base or instruct file → skip", file=sys.stderr)
            continue

        base_map = conllu_firstK_keys(base_f, args.key, exclude_upos, K)
        instr_map = conllu_firstK_keys(instr_f, args.key, exclude_upos, K)

        # Pair by idx; retain only pairs with both sides having ≥ K eligible tokens
        common_idxs = set(base_map.keys()) & set(instr_map.keys())
        pairs = []
        for idx in common_idxs:
            base_keys, base_elig = base_map[idx]
            instr_keys, instr_elig = instr_map[idx]
            if base_elig >= K and instr_elig >= K:
                pairs.append((base_keys, instr_keys))
        total_pairs = len(pairs)
        if total_pairs == 0:
            print(f"[diff] {model}: no retained pairs (after K={K}) → skip", file=sys.stderr)
            continue

        # Document counts per key on each side
        c_base = Counter()
        c_instr = Counter()
        vocab = set()
        for bk, ik in pairs:
            c_base.update(bk)
            c_instr.update(ik)
            vocab.update(bk)
            vocab.update(ik)

        rows = []
        for k in vocab:
            cb = c_base.get(k, 0)
            ci = c_instr.get(k, 0)
            if max(cb, ci) < args.min_docs:
                continue
            db = cb / total_pairs
            di = ci / total_pairs
            dd = di - db
            rows.append((k, cb, ci, db, di, dd))

        if not rows:
            print(f"[diff] {model}: nothing passes min_docs={args.min_docs}", file=sys.stderr)
            continue

        top_up = sorted(rows, key=lambda r: (-r[5], -r[2], r[0]))[:args.topk]
        top_down = sorted(rows, key=lambda r: (r[5], r[1], r[0]))[:args.topk]

        out_file = args.out_root / model / "dfdiff_top_bottom.tsv"
        write_tsv(out_file, top_up, top_down, total_pairs)
        wrote_any = True
        print(f"[diff] {model}: K={K}, pairs={total_pairs}, vocab={len(vocab)} → {out_file}")

    if not wrote_any:
        print("[diff] No outputs written.", file=sys.stderr)

if __name__ == "__main__":
    main()
