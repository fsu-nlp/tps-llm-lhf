#!/usr/bin/env python3
"""
Annotate TPS word-level CSV(s) with Wiktionary etymology lookup.

Inputs:
- Wiktionary-derived TSV: en_etymology.tsv (columns: lemma, pos, key, ety_class, donor_langs, etym_text)
- TPS CSV file(s) such as: tps_word_*.csv (must have column: key)

Outputs:
- Annotated CSV(s) written to: /home/tom/delve7-35langs/etymology/tps-annotated/

Etymology status:
- FOUND: key exists in en_etymology.tsv
- NOT_IN_WIKTIONARY: key not found in TSV lookup
Note: even if FOUND, ety_class may be UNKNOWN (that's different from NOT_IN_WIKTIONARY).

Usage examples:
  python3 annotate_tps_with_etymology.py \
    --tps /home/tom/ablation/.../tps_word_olmo2-1124-7b.csv \
    --etym /home/tom/delve7-35langs/etymology/wiktionary/en_etymology.tsv

  python3 annotate_tps_with_etymology.py \
    --tps-dir /home/tom/ablation/lhf-lex-ablation/out-ig/pre/analyses/tps-all \
    --pattern tps_word_*.csv \
    --etym /home/tom/delve7-35langs/etymology/wiktionary/en_etymology.tsv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Tuple


OUT_DIR_DEFAULT = "/home/tom/delve7-35langs/etymology/tps-annotated"


def load_etym_lookup(etym_tsv_path: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Return dict:
      key -> (ety_class, donor_langs, etym_text)
    """
    lookup: Dict[str, Tuple[str, str, str]] = {}
    with open(etym_tsv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        required = {"key", "ety_class", "donor_langs", "etym_text"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"Etym TSV missing required columns: {sorted(missing)}")

        for row in r:
            k = (row.get("key") or "").strip()
            if not k:
                continue
            lookup[k] = (
                (row.get("ety_class") or "").strip(),
                (row.get("donor_langs") or "").strip(),
                (row.get("etym_text") or "").strip(),
            )
    return lookup


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def annotate_one_csv(tps_csv_path: str, lookup: Dict[str, Tuple[str, str, str]], out_dir: str) -> str:
    """
    Annotate a single TPS CSV and write annotated CSV to out_dir.
    Returns output path.
    """
    base = os.path.basename(tps_csv_path)
    stem, ext = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{stem}.etym{ext}")

    with open(tps_csv_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError(f"No header found in TPS CSV: {tps_csv_path}")
        if "key" not in reader.fieldnames:
            raise ValueError(f"TPS CSV missing required 'key' column: {tps_csv_path}")

        # Add new columns at the end
        new_cols = ["ety_status", "ety_class", "donor_langs", "etym_text"]
        fieldnames = list(reader.fieldnames) + [c for c in new_cols if c not in reader.fieldnames]

        ensure_dir(out_dir)
        with open(out_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            found = 0
            missing = 0

            for row in reader:
                k = (row.get("key") or "").strip()
                if k in lookup:
                    ety_class, donor_langs, etym_text = lookup[k]
                    row["ety_status"] = "FOUND"
                    row["ety_class"] = ety_class
                    row["donor_langs"] = donor_langs
                    row["etym_text"] = etym_text
                    found += 1
                else:
                    row["ety_status"] = "NOT_IN_WIKTIONARY"
                    row["ety_class"] = ""
                    row["donor_langs"] = ""
                    row["etym_text"] = ""
                    missing += 1

                writer.writerow(row)

    print(f"[annotated] {tps_csv_path} -> {out_path}  (FOUND={found}, NOT_IN_WIKTIONARY={missing})", file=sys.stderr)
    return out_path


def find_files(tps_dir: str, pattern: str) -> List[str]:
    """
    Recursively find files matching pattern under tps_dir.
    """
    matches: List[str] = []
    for root, _, _ in os.walk(tps_dir):
        matches.extend(glob.glob(os.path.join(root, pattern)))
    matches = sorted(set(matches))
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etym", required=True, help="Path to en_etymology.tsv")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tps", help="Path to a single TPS CSV file (e.g. tps_word_*.csv)")
    g.add_argument("--tps-dir", help="Directory to search for TPS CSV files")
    ap.add_argument("--pattern", default="tps_word_*.csv", help="Glob pattern used with --tps-dir (default: tps_word_*.csv)")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT, help=f"Output directory (default: {OUT_DIR_DEFAULT})")
    args = ap.parse_args()

    print(f"[load] etymology TSV: {args.etym}", file=sys.stderr)
    lookup = load_etym_lookup(args.etym)
    print(f"[load] lookup entries: {len(lookup):,}", file=sys.stderr)

    if args.tps:
        annotate_one_csv(args.tps, lookup, args.out_dir)
    else:
        files = find_files(args.tps_dir, args.pattern)
        if not files:
            raise SystemExit(f"No files found under {args.tps_dir} matching pattern {args.pattern}")
        print(f"[scan] found {len(files)} file(s)", file=sys.stderr)
        for fp in files:
            annotate_one_csv(fp, lookup, args.out_dir)


if __name__ == "__main__":
    main()

