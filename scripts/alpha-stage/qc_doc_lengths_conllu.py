#!/usr/bin/env python3
"""
QC: Count documents (abstract continuations) above/below a token-length threshold
in CoNLL-U files, using '# newdoc id' boundaries and excluding UPOS ∈ {PUNCT, SYM, X}.

Now also reports, per model directory, how many (Human, Base, Instruct) *triplets*
meet the threshold, aligned by prompt index:
- Primary alignment key: the integer extracted from '...-idx<NUM>-...' in '# newdoc id'.
- Fallback: 0-based document order if no idx is found.

Outputs:
- Per-file CSV (unchanged schema).
- Per-model CSV (augmented with triplet columns).

Usage:
  python scripts/qc_doc_lengths_conllu.py \
    --pos-root out/pre/pos \
    --out-csv results/doclen_qc_per_file.csv \
    --out-summary results/doclen_qc_by_model.csv \
    --threshold 60
"""

from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

EXCLUDE_UPOS = {"PUNCT", "SYM", "X"}
NEWDOC_RE = re.compile(r"^#\s*newdoc id\s*=\s*(.+)\s*$")
IDX_IN_ID_RE = re.compile(r"(?:^|[^A-Za-z0-9])idx(\d+)(?:[^0-9]|$)")

def find_conllu_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.conllu"))

def infer_model_name(pos_root: Path, file_path: Path) -> str:
    try:
        rel = file_path.relative_to(pos_root)
    except ValueError:
        return "UNKNOWN"
    parts = rel.parts
    return parts[0] if parts else "UNKNOWN"

def _flush_doc(acc_len: int,
               order_idx: int,
               last_newdoc_id: Optional[str]) -> Optional[Tuple[int, int, Optional[int], Optional[str]]]:
    """
    Prepare a record for the current document and reset accumulators.

    Returns a tuple: (order_idx, length, idx_key_or_None, newdoc_id_or_None)
    """
    if order_idx < 0:
        return None
    idx_key: Optional[int] = None
    if last_newdoc_id:
        m = IDX_IN_ID_RE.search(last_newdoc_id)
        if m:
            try:
                idx_key = int(m.group(1))
            except Exception:
                idx_key = None
    return (order_idx, acc_len, idx_key, last_newdoc_id)

def parse_doc_lengths_with_keys(path: Path) -> List[Tuple[int, int, Optional[int], Optional[str]]]:
    """
    Parse a CoNLL-U file into a list of docs with:
      (order_idx, length_excl_[PUNCT|SYM|X], idx_key_or_None, newdoc_id_or_None)

    - Groups tokens by '# newdoc id' boundaries.
    - Skips multiword tokens (ID contains '-') and empty nodes (ID contains '.').
    """
    docs: List[Tuple[int, int, Optional[int], Optional[str]]] = []
    order_idx = -1
    acc_len = 0
    last_newdoc_id: Optional[str] = None
    saw_any_newdoc = False

    def commit():
        nonlocal order_idx, acc_len, last_newdoc_id
        rec = _flush_doc(acc_len, order_idx, last_newdoc_id)
        if rec is not None:
            docs.append(rec)
        acc_len = 0
        last_newdoc_id = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("#"):
                m = NEWDOC_RE.match(line)
                if m:
                    # starting a new doc: commit previous, then prepare a fresh one
                    if order_idx >= 0:
                        commit()
                    saw_any_newdoc = True
                    order_idx += 1
                    last_newdoc_id = m.group(1)
                continue
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue
            upos = cols[3]
            if upos not in EXCLUDE_UPOS:
                acc_len += 1

    # flush final doc
    if order_idx >= 0:
        commit()
    elif not saw_any_newdoc:
        # No newdoc markers; treat whole file as one doc with order_idx=0
        order_idx = 0
        rec = _flush_doc(acc_len, order_idx, last_newdoc_id)
        if rec is not None:
            docs.append(rec)

    return docs

def count_doc_lengths(path: Path, threshold: int) -> Tuple[int, int, int, int, float]:
    """
    Return (n_docs, n_gt, n_lt, n_eq, mean_len) for one file.
    """
    docs = parse_doc_lengths_with_keys(path)
    n_docs = len(docs)
    n_gt = n_lt = n_eq = 0
    total_len = 0
    for _, k, _, _ in docs:
        if k > threshold:
            n_gt += 1
        elif k < threshold:
            n_lt += 1
        else:
            n_eq += 1
        total_len += k
    mean_len = (total_len / n_docs) if n_docs > 0 else 0.0
    return n_docs, n_gt, n_lt, n_eq, mean_len

def pct(part: int, whole: int) -> float:
    return (100.0 * part / whole) if whole > 0 else 0.0

def select_model_files(model_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (base_file, instruct_file) chosen from model_dir by filename heuristics.
    """
    base = instruct = None
    for fp in sorted(model_dir.glob("*.conllu")):
        name = fp.name.lower()
        if "instruct" in name:
            instruct = fp
        elif "base" in name:
            base = fp
    return base, instruct

def build_key_to_len(docs: List[Tuple[int, int, Optional[int], Optional[str]]],
                     prefer_idx: bool) -> Dict[int, int]:
    """
    Build a mapping from alignment key -> length.
    - If prefer_idx and idx_key exists, use it; otherwise use order_idx.
    - If the chosen key duplicates, keep the first and ignore later duplicates.
    """
    m: Dict[int, int] = {}
    for order_idx, k, idx_key, _ in docs:
        key = idx_key if (prefer_idx and idx_key is not None) else order_idx
        if key not in m:
            m[key] = k
    return m

def decide_alignment_key(h_docs, b_docs, i_docs) -> bool:
    """
    Decide whether to align by idx (True) or by order (False).
    Heuristic: use idx if all three sides have >= 95% of docs with an idx_key present.
    """
    def idx_rate(docs) -> float:
        have = sum(1 for _, _, idx_key, _ in docs if idx_key is not None)
        total = len(docs)
        return (have / total) if total else 0.0

    r_h = idx_rate(h_docs)
    r_b = idx_rate(b_docs)
    r_i = idx_rate(i_docs)
    return (r_h >= 0.95) and (r_b >= 0.95) and (r_i >= 0.95)

def compute_triplets_for_model(model_name: str,
                               human_file: Path,
                               base_file: Path,
                               instruct_file: Path,
                               threshold: int,
                               quiet: bool) -> Tuple[int, int, float]:
    """
    Compute triplet counts for one model:
    - total triplet keys (intersection of keys present in H, B, I)
    - number meeting >= threshold across all three
    - percentage
    """
    h_docs = parse_doc_lengths_with_keys(human_file)
    b_docs = parse_doc_lengths_with_keys(base_file)
    i_docs = parse_doc_lengths_with_keys(instruct_file)

    prefer_idx = decide_alignment_key(h_docs, b_docs, i_docs)
    key_mode = "idx" if prefer_idx else "order"

    h_map = build_key_to_len(h_docs, prefer_idx)
    b_map = build_key_to_len(b_docs, prefer_idx)
    i_map = build_key_to_len(i_docs, prefer_idx)

    common_keys = set(h_map).intersection(b_map).intersection(i_map)
    total = len(common_keys)
    n_ge = sum(
        1 for k in common_keys
        if (h_map[k] >= threshold and b_map[k] >= threshold and i_map[k] >= threshold)
    )
    p_ge = pct(n_ge, total)

    if not quiet:
        print(f"[tri] {model_name} :: keys={total}, ≥{threshold}={n_ge} ({p_ge:.2f}%) [{key_mode}-aligned]")

    return total, n_ge, p_ge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-root", type=Path, required=True,
                    help="Root directory with model subfolders containing .conllu files (e.g., out/pre/pos)")
    ap.add_argument("--out-csv", type=Path, required=True,
                    help="Per-file CSV output")
    ap.add_argument("--out-summary", type=Path, required=True,
                    help="Per-model summary CSV output")
    ap.add_argument("--threshold", type=int, default=60,
                    help="Token-length threshold (exclusive of PUNCT/SYM/X); default=60")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-file progress")
    args = ap.parse_args()

    files = find_conllu_files(args.pos_root)
    if not files:
        print(f"[qc] No .conllu files found under: {args.pos_root}", file=sys.stderr)
        sys.exit(2)

    # Locate human file
    human_dir = args.pos_root / "human"
    human_files = sorted(human_dir.glob("*.conllu")) if human_dir.exists() else []
    human_file: Optional[Path] = human_files[0] if human_files else None
    if not human_file:
        print("[qc] WARNING: No human file found at pos-root/human/*.conllu; triplet stats will be skipped.", file=sys.stderr)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    # Aggregates per model
    per_model: Dict[str, Dict[str, float]] = {}

    # Per-file CSV
    with args.out_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "model",
            "file",
            "n_docs",
            f"n_ge_{args.threshold}",
            f"pct_ge_{args.threshold}",
            f"n_lt_{args.threshold}",
            f"pct_lt_{args.threshold}",
            f"n_eq_{args.threshold}",  # kept for audit
            "mean_len_excl_PUNCT_SYM_X"
        ])

        for fp in files:
            model = infer_model_name(args.pos_root, fp)
            n_docs, n_gt, n_lt, n_eq, mean_len = count_doc_lengths(fp, args.threshold)

            # Fold ties into >= bucket for reporting
            n_ge = n_gt + n_eq
            p_ge = pct(n_ge, n_docs)
            p_lt = pct(n_lt, n_docs)

            writer.writerow([
                model,
                str(fp.relative_to(args.pos_root)),
                n_docs,
                n_ge, f"{p_ge:.2f}",
                n_lt, f"{p_lt:.2f}",
                n_eq,
                f"{mean_len:.3f}",
            ])

            if not args.quiet:
                print(
                    f"[qc] {model} :: {fp.name} → docs={n_docs}, "
                    f"≥{args.threshold}={n_ge} ({p_ge:.2f}%), "
                    f"<{args.threshold}={n_lt} ({p_lt:.2f}%), "
                    f"mean={mean_len:.2f}"
                )

            agg = per_model.setdefault(
                model, {"n_docs": 0, "n_ge": 0, "n_lt": 0, "n_eq": 0, "sum_len": 0.0,
                        "trip_total": 0, "trip_ge": 0}
            )
            agg["n_docs"] += n_docs
            agg["n_ge"] += n_ge
            agg["n_lt"] += n_lt
            agg["n_eq"] += n_eq
            agg["sum_len"] += mean_len * n_docs

    # Triplet stats per model (requires human + base + instruct)
    # We iterate subfolders directly under pos-root excluding 'human'
    if human_file:
        for model_dir in sorted(p for p in args.pos_root.iterdir() if p.is_dir() and p.name != "human"):
            model = model_dir.name
            base_fp, instr_fp = select_model_files(model_dir)
            if not base_fp or not instr_fp:
                print(f"[tri] {model} :: skipped (base/instruct file not found)", file=sys.stderr)
                continue
            trip_total, trip_ge, _ = compute_triplets_for_model(
                model, human_file, base_fp, instr_fp, args.threshold, args.quiet
            )
            agg = per_model.setdefault(
                model, {"n_docs": 0, "n_ge": 0, "n_lt": 0, "n_eq": 0, "sum_len": 0.0,
                        "trip_total": 0, "trip_ge": 0}
            )
            agg["trip_total"] = trip_total
            agg["trip_ge"] = trip_ge

    # Per-model summary CSV (augmented with triplet columns)
    with args.out_summary.open("w", newline="", encoding="utf-8") as f_sum:
        writer = csv.writer(f_sum)
        writer.writerow([
            "model",
            "n_docs",
            f"n_ge_{args.threshold}",
            f"pct_ge_{args.threshold}",
            f"n_lt_{args.threshold}",
            f"pct_lt_{args.threshold}",
            f"n_eq_{args.threshold}",
            "mean_len_excl_PUNCT_SYM_X",
            "triplet_total",
            f"triplet_ge_{args.threshold}",
            f"triplet_pct_ge_{args.threshold}",
        ])
        for model, agg in sorted(per_model.items()):
            n_docs = int(agg["n_docs"])
            n_ge = int(agg["n_ge"])
            n_lt = int(agg["n_lt"])
            n_eq = int(agg["n_eq"])
            p_ge = pct(n_ge, n_docs)
            p_lt = pct(n_lt, n_docs)
            mean_len = (agg["sum_len"] / n_docs) if n_docs > 0 else 0.0

            trip_total = int(agg.get("trip_total", 0))
            trip_ge = int(agg.get("trip_ge", 0))
            trip_p_ge = pct(trip_ge, trip_total)

            writer.writerow([
                model,
                n_docs,
                n_ge, f"{p_ge:.2f}",
                n_lt, f"{p_lt:.2f}",
                n_eq,
                f"{mean_len:.3f}",
                trip_total,
                trip_ge, f"{trip_p_ge:.2f}",
            ])

if __name__ == "__main__":
    main()
