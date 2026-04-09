#!/usr/bin/env python3
"""
Compute etymology "market share" distributions within each POS category
from an annotated TPS CSV (output of annotate_tps_with_etymology.py).

We treat c_B and c_I as the mass measures (window counts) and compute, per POS:
  share_B(ety_class | pos) and share_I(ety_class | pos)
and the delta: share_I - share_B.

Outputs TSVs to an output directory:
  - pos_etym_dist*.tsv   (pos, model, ety_class, mass, share, denom_mass)
  - pos_etym_delta*.tsv  (pos, ety_class, share_B, share_I, delta_I_minus_B)
  - coverage.tsv         (pos, model, total_mass, found_mass, found_share, found_unknown_mass, ...)
  - pos_etym_binary_content.tsv
      High-level split for ADJ/ADV/NOUN/VERB only:
        compares GERMANIC vs ROMANCE_ALL, excluding MIXED/OTHER/UNKNOWN and NOT_IN_WIKTIONARY.
      The two shares add up to 1 (within each POS and model), unless no eligible mass exists.

Key feature:
- --finegrained (default: False): if False, collapse etymology into big buckets:
    GERMANIC vs LATIN-FRENCH-ROMANCE (ROMANCE_ALL) + OTHER + UNKNOWN + NOT_IN_WIKTIONARY
  If True, keep original labels (GERMANIC, ROMANCE, LATIN-FRENCH, OTHER, UNKNOWN, NOT_IN_WIKTIONARY).

Notes:
- By default we compute shares among FOUND items only (so missing coverage doesn't distort).
  We also report coverage in coverage.tsv.
- You can include NOT_IN_WIKTIONARY mass in the denominator with --denom include_all
- You can collapse FOUND UNKNOWN -> OTHER with --collapse-unknown-to-other (optional)
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Tuple


CONTENT_POS = {"ADJ", "ADV", "NOUN", "VERB"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def norm_ety_class(ety_status: str, ety_class: str) -> str:
    """
    Normalise / sanitise etym labels.
    """
    s = (ety_status or "").strip()
    c = (ety_class or "").strip()
    if s != "FOUND":
        return "NOT_IN_WIKTIONARY"
    if not c:
        return "UNKNOWN"
    return c


def collapse_class(cls: str, finegrained: bool) -> str:
    """
    Collapse etymology classes unless finegrained=True.
    Default collapsed scheme:
      - ROMANCE and LATIN-FRENCH -> ROMANCE_ALL
      - GERMANIC stays GERMANIC
      - MIXED stays MIXED (if present)
      - OTHER stays OTHER
      - UNKNOWN stays UNKNOWN
      - NOT_IN_WIKTIONARY stays NOT_IN_WIKTIONARY
    """
    if finegrained:
        return cls
    if cls in {"ROMANCE", "LATIN-FRENCH"}:
        return "ROMANCE_ALL"
    return cls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Annotated TPS CSV (e.g. tps_word_*.etym.csv)")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g. /home/tom/delve7-35langs/etymology/analysis)")
    ap.add_argument("--pos-col", default="upos", help="Column name for POS (default: upos)")
    ap.add_argument(
        "--denom",
        choices=["found_only", "include_all"],
        default="found_only",
        help="Denominator for shares: found_only (default) or include_all (includes NOT_IN_WIKTIONARY).",
    )
    ap.add_argument(
        "--collapse-unknown-to-other",
        action="store_true",
        help="If set, treat ety_class=UNKNOWN (when FOUND) as OTHER.",
    )
    ap.add_argument(
        "--finegrained",
        action="store_true",
        help="If set, keep fine-grained ety_class labels (ROMANCE vs LATIN-FRENCH). Default collapses to ROMANCE_ALL.",
    )
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Mass accumulators:
    # mass[(pos, model, ety_class)] = sum counts
    mass: Dict[Tuple[str, str, str], float] = defaultdict(float)

    # Totals by POS+model for denominator
    total_found: Dict[Tuple[str, str], float] = defaultdict(float)  # FOUND only
    total_all: Dict[Tuple[str, str], float] = defaultdict(float)    # FOUND + NOT_IN_WIKTIONARY

    # Coverage diagnostics
    cov_total: Dict[Tuple[str, str], float] = defaultdict(float)
    cov_found: Dict[Tuple[str, str], float] = defaultdict(float)
    cov_found_unknown: Dict[Tuple[str, str], float] = defaultdict(float)
    cov_not_in_wikt: Dict[Tuple[str, str], float] = defaultdict(float)

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {args.pos_col, "c_B", "c_I", "ety_status", "ety_class"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in input: {sorted(missing)}")

        for row in reader:
            pos = (row.get(args.pos_col) or "").strip()
            if not pos:
                continue

            ety_status = (row.get("ety_status") or "").strip()
            ety_class_raw = (row.get("ety_class") or "").strip()

            # normalise classes
            cls = norm_ety_class(ety_status, ety_class_raw)

            # optional: collapse FOUND UNKNOWN -> OTHER (before romance collapsing)
            if (
                args.collapse_unknown_to_other
                and ety_status == "FOUND"
                and (ety_class_raw.strip() == "UNKNOWN" or not ety_class_raw.strip())
            ):
                cls = "OTHER"

            # collapse ROMANCE/LATIN-FRENCH unless finegrained
            cls = collapse_class(cls, finegrained=args.finegrained)

            cB = as_float(row.get("c_B") or "0")
            cI = as_float(row.get("c_I") or "0")

            # Update totals and coverage (per model)
            for model, c in [("B", cB), ("I", cI)]:
                if c == 0.0:
                    continue

                cov_total[(pos, model)] += c

                if ety_status == "FOUND":
                    cov_found[(pos, model)] += c
                    if (ety_class_raw.strip() == "UNKNOWN") or (not ety_class_raw.strip()):
                        cov_found_unknown[(pos, model)] += c
                    total_found[(pos, model)] += c
                else:
                    cov_not_in_wikt[(pos, model)] += c

                total_all[(pos, model)] += c

                # Update mass by class
                mass[(pos, model, cls)] += c

    # Filenames depend on finegrained flag
    suffix = "_fine" if args.finegrained else ""
    dist_path = os.path.join(args.outdir, f"pos_etym_dist{suffix}.tsv")
    delta_path = os.path.join(args.outdir, f"pos_etym_delta{suffix}.tsv")
    cov_path = os.path.join(args.outdir, "coverage.tsv")
    binary_path = os.path.join(args.outdir, "pos_etym_binary_content.tsv")

    # Write pos_etym_dist*.tsv
    with open(dist_path, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["pos", "model", "ety_class", "mass", "share", "denom_mass"])

        denom_map = total_found if args.denom == "found_only" else total_all
        keys = sorted(mass.keys(), key=lambda t: (t[0], t[1], t[2]))

        for pos, model, cls in keys:
            denom = denom_map.get((pos, model), 0.0)
            if denom <= 0:
                continue
            m = mass[(pos, model, cls)]

            # If denom is found_only, shares for NOT_IN_WIKTIONARY are undefined; skip them.
            if args.denom == "found_only" and cls == "NOT_IN_WIKTIONARY":
                continue

            share = m / denom
            w.writerow([pos, model, cls, f"{m:.6f}", f"{share:.10f}", f"{denom:.6f}"])

    # Build share lookup (for delta table)
    share_lookup: Dict[Tuple[str, str, str], float] = {}
    denom_map = total_found if args.denom == "found_only" else total_all

    for (pos, model, cls), m in mass.items():
        denom = denom_map.get((pos, model), 0.0)
        if denom <= 0:
            continue
        if args.denom == "found_only" and cls == "NOT_IN_WIKTIONARY":
            continue
        share_lookup[(pos, model, cls)] = m / denom

    # Universe of (pos, cls)
    pos_cls = sorted({(p, c) for (p, _m, c) in share_lookup.keys()})

    # Write pos_etym_delta*.tsv
    with open(delta_path, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow(["pos", "ety_class", "share_B", "share_I", "delta_I_minus_B"])
        for pos, cls in pos_cls:
            sb = share_lookup.get((pos, "B", cls), 0.0)
            si = share_lookup.get((pos, "I", cls), 0.0)
            w.writerow([pos, cls, f"{sb:.10f}", f"{si:.10f}", f"{(si - sb):.10f}"])

    # Write coverage.tsv (always)
    with open(cov_path, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow([
            "pos", "model",
            "total_mass",
            "found_mass", "found_share",
            "found_unknown_mass", "found_unknown_share_of_total", "found_unknown_share_of_found",
            "not_in_wiktionary_mass", "not_in_wiktionary_share"
        ])

        for (pos, model), total in sorted(cov_total.items(), key=lambda x: (x[0][0], x[0][1])):
            if total <= 0:
                continue
            found = cov_found.get((pos, model), 0.0)
            unk = cov_found_unknown.get((pos, model), 0.0)
            niw = cov_not_in_wikt.get((pos, model), 0.0)

            w.writerow([
                pos, model,
                f"{total:.6f}",
                f"{found:.6f}", f"{(found/total):.10f}",
                f"{unk:.6f}", f"{(unk/total):.10f}", f"{(unk/found):.10f}" if found > 0 else "0.0",
                f"{niw:.6f}", f"{(niw/total):.10f}",
            ])

    # Write pos_etym_binary_content.tsv
    #
    # Only for pos in {ADJ, ADV, NOUN, VERB}
    # Only include cls in {GERMANIC, ROMANCE_ALL} (ROMANCE_ALL only exists if not finegrained;
    # if finegrained=True we still compute binary by mapping ROMANCE and LATIN-FRENCH into ROMANCE_ALL here).
    #
    # Denominator excludes MIXED/OTHER/UNKNOWN/NOT_IN_WIKTIONARY.
    #
    # Output columns:
    #   pos, model, germanic_mass, romance_mass, denom_mass, share_germanic, share_romance
    #
    with open(binary_path, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        w.writerow([
            "pos", "model",
            "germanic_mass", "romance_mass", "denom_mass",
            "share_germanic", "share_romance"
        ])

        for pos in sorted(CONTENT_POS):
            for model in ["B", "I"]:
                # Get masses (note: if finegrained=True, romance may be split into ROMANCE/LATIN-FRENCH;
                # we unify here explicitly.)
                germanic = mass.get((pos, model, "GERMANIC"), 0.0)

                romance = 0.0
                if args.finegrained:
                    romance += mass.get((pos, model, "ROMANCE"), 0.0)
                    romance += mass.get((pos, model, "LATIN-FRENCH"), 0.0)
                else:
                    romance += mass.get((pos, model, "ROMANCE_ALL"), 0.0)

                denom = germanic + romance
                if denom <= 0:
                    # No eligible mass (e.g. no coverage or all excluded)
                    w.writerow([pos, model, f"{germanic:.6f}", f"{romance:.6f}", "0.000000", "0.0", "0.0"])
                    continue

                w.writerow([
                    pos, model,
                    f"{germanic:.6f}", f"{romance:.6f}", f"{denom:.6f}",
                    f"{(germanic/denom):.10f}", f"{(romance/denom):.10f}"
                ])

    print(f"[ok] wrote:\n  {dist_path}\n  {delta_path}\n  {cov_path}\n  {binary_path}")


if __name__ == "__main__":
    main()
