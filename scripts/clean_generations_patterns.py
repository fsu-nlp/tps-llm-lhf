#!/usr/bin/env python3
"""
Pattern-based pre-cleaning for model/human outputs (no external models).

Transforms (applied in this order):
  1) Drop everything from the first 'Introduction'   -> 'intro_truncated'
  2) Collapse any consecutive newlines to a space    -> 'newlines_collapsed'
  3) Collapse runs of spaces/tabs to a single space  -> 'spaces_collapsed'
  4) Trim leading/trailing whitespace                -> 'strip_applied'

NOTE: Angle-bracketed content like <0.5) or <p>…</p> is NOT touched.

Inputs:
  - One or more --in-out-dir subtrees (e.g., out/pre, out/pre-sample). Recurse *.jsonl.
  - Optional --in-human-halves JSONL files (repeatable). For those, clean 'second_half' if present, else 'text'.

Outputs:
  - Mirrored tree under --out-root, file names suffixed with '.preclean.jsonl'
  - Rollups at {out_root}/cleaning_diffs.jsonl, cleaning_summary.json, cleaning_summary.md

All paths are resolved relative to --repo-root (default: '.').
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List, Optional

# ---------- cleaning primitives ----------

INTRO_SPLIT_RE  = re.compile(r"\bIntroduction\b")  # capitalised, exact token
NEWLINES_RE     = re.compile(r"\n+")
SPACES_RE       = re.compile(r"[ \t]+")

def _sub_with_capture(pattern: re.Pattern, replacement: str, s: str, removed: List[str]) -> str:
    """re.sub but append matched substrings to 'removed' before replacing."""
    def _repl(m: re.Match) -> str:
        removed.append(m.group(0))
        return replacement
    return pattern.sub(_repl, s)

def clean_text(text: str) -> Tuple[str, Dict[str, Any], List[str]]:
    """
    Apply transforms and return:
      cleaned_text, flags/meta, removed_substrings (full substrings removed/replaced).
    """
    meta = {
        "intro_truncated": False,
        "newlines_collapsed": False,
        "spaces_collapsed": False,
        "strip_applied": False,
    }
    removed_substrings: List[str] = []

    if not isinstance(text, str):
        return text, meta, removed_substrings

    s0 = text
    # Normalise non-breaking spaces first (not counted separately)
    s = s0.replace("\u00A0", " ")

    # 1) delete from first 'Introduction' onwards (capture the truncated tail)
    m = INTRO_SPLIT_RE.search(s)
    if m:
        meta["intro_truncated"] = True
        removed_substrings.append(s[m.start():])  # everything from 'Introduction' to end
        s = s[: m.start()]

    # 2) collapse consecutive newlines to single space (capture runs)
    if "\n" in s:
        before = s
        s = _sub_with_capture(NEWLINES_RE, " ", s, removed_substrings)
        if s != before:
            meta["newlines_collapsed"] = True

    # 3) collapse spaces/tabs to single space (capture runs)
    if SPACES_RE.search(s):
        before = s
        s = _sub_with_capture(SPACES_RE, " ", s, removed_substrings)
        if s != before:
            meta["spaces_collapsed"] = True

    # 4) strip leading/trailing whitespace (capture the stripped segments)
    leading_len = len(s) - len(s.lstrip())
    trailing_len = len(s) - len(s.rstrip())
    if leading_len or trailing_len:
        meta["strip_applied"] = True
        if leading_len:
            removed_substrings.append(s[:leading_len])
        if trailing_len:
            removed_substrings.append(s[len(s) - trailing_len:])
        s = s.strip()

    return s, meta, removed_substrings


# ---------- IO helpers ----------

def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                yield i, {}
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path} line {i+1}: {e}") from e
            yield i, obj


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------- processing ----------

def decide_fields(obj: Dict[str, Any], is_human_halves: bool) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (primary_field, secondary_field) to clean.
    - For model outputs: clean "text" if present.
    - For human halves: prefer "second_half", else "text".
    Only one field is cleaned to keep diffs simple.
    """
    if is_human_halves:
        if "second_half" in obj and isinstance(obj["second_half"], str):
            return "second_half", None
        if "text" in obj and isinstance(obj["text"], str):
            return "text", None
        return None, None
    else:
        if "text" in obj and isinstance(obj["text"], str):
            return "text", None
        return None, None

def pick_record_id(obj: Dict[str, Any]) -> Any:
    """Best-effort record identifier for diffs."""
    for k in ("record_id", "idx", "id", "prompt_id", "prompt_sha", "decode_hash"):
        if k in obj:
            return obj.get(k)
    return None

def process_file(
    src_path: Path,
    dest_path: Path,
    diffs_sink: Path,
    is_human_halves_file: bool,
    summary_acc: Dict[str, Any],
    repo_root: Path,
) -> None:
    """
    Read src jsonl, clean appropriate field, write dest, append to diffs,
    and update summary accumulator.
    """
    rows_out = []
    diffs_out = []

    file_stats = {
        "items": 0,
        "touched": 0,
        "chars_removed": 0,
        "intro_truncated": 0,
        "newlines_collapsed": 0,
        "spaces_collapsed": 0,
        "strip_applied": 0,
    }

    rel_src = src_path.relative_to(repo_root)
    rel_dest = dest_path.relative_to(repo_root)

    for idx, obj in iter_jsonl(src_path):
        file_stats["items"] += 1
        if not isinstance(obj, dict):
            rows_out.append(obj)  # pass through
            continue

        field, _ = decide_fields(obj, is_human_halves_file)

        if field is None or field not in obj or not isinstance(obj[field], str):
            rows_out.append(obj)
            continue

        original = obj[field]
        cleaned, flags, removed_substrings = clean_text(original)

        changed = (cleaned != original)
        if changed:
            file_stats["touched"] += 1
            removed_chars = max(0, len(original) - len(cleaned))
            file_stats["chars_removed"] += removed_chars
            for k in ("intro_truncated", "newlines_collapsed", "spaces_collapsed", "strip_applied"):
                if flags[k]:
                    file_stats[k] += 1

            diffs_out.append({
                "src_file": str(rel_src),
                "dest_file": str(rel_dest),
                "mode": "patterns",
                "item_index": idx,
                "record_id": pick_record_id(obj),
                "source_field": field,
                "changed": True,
                "orig_len": len(original),
                "clean_len": len(cleaned),
                "chars_removed": removed_chars,
                "removed_substrings": removed_substrings,   # FULL substrings removed/replaced
                "original_text": original,                  # FULL original
                "cleaned_text": cleaned,                    # FULL cleaned
                "flags": flags
            })

            obj[field] = cleaned

        rows_out.append(obj)

    # write cleaned file
    ensure_parent(dest_path)
    write_jsonl(dest_path, rows_out)

    # append diffs
    if diffs_out:
        ensure_parent(diffs_sink)
        with diffs_sink.open("a", encoding="utf-8") as f:
            for d in diffs_out:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # update summary accumulator
    summary_acc["files"].append({
        "src_file": str(rel_src),
        "dest_file": str(rel_dest),
        **file_stats,
    })
    summary_acc["totals"]["files"] += 1
    summary_acc["totals"]["items"] += file_stats["items"]
    summary_acc["totals"]["touched"] += file_stats["touched"]
    summary_acc["totals"]["chars_removed"] += file_stats["chars_removed"]
    for k in ("intro_truncated", "newlines_collapsed", "spaces_collapsed", "strip_applied"):
        summary_acc["totals"][k] += file_stats[k]


def build_summary_md(summary: Dict[str, Any]) -> str:
    t = summary["totals"]
    touched_rate = (t["touched"] / t["items"] * 100) if t["items"] else 0.0
    avg_removed_touched = (t["chars_removed"] / t["touched"]) if t["touched"] else 0.0

    lines = []
    lines.append("# Pattern Cleaning Summary\n")
    lines.append(f"- Files processed: **{t['files']}**")
    lines.append(f"- Records (lines): **{t['items']}**")
    lines.append(f"- Records touched: **{t['touched']}** ({touched_rate:.2f}%)")
    lines.append(f"- Total characters removed: **{t['chars_removed']}**")
    lines.append(f"- Avg chars removed among touched: **{avg_removed_touched:.2f}**\n")
    lines.append("## Pattern hit counts (by record)")
    lines.append("")
    lines.append(f"- Truncated at 'Introduction': **{t['intro_truncated']}**")
    lines.append(f"- Newlines collapsed: **{t['newlines_collapsed']}**")
    lines.append(f"- Spaces collapsed: **{t['spaces_collapsed']}**")
    lines.append(f"- Strip applied: **{t['strip_applied']}**")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| src_file | items | touched | chars_removed | intro | newlines | spaces | strip |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for f in summary["files"]:
        lines.append(
            f"| {f['src_file']} | {f['items']} | {f['touched']} | {f['chars_removed']} | "
            f"{f['intro_truncated']} | {f['newlines_collapsed']} | "
            f"{f['spaces_collapsed']} | {f['strip_applied']} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Pattern-based cleaner for LHF Lex Ablation outputs.")
    ap.add_argument("--repo-root", default=".", help="Repository root (base for relative paths).")
    ap.add_argument("--in-out-dir", action="append", required=False, default=[],
                    help="Input subtree to recurse (repeatable). Example: out/pre")
    ap.add_argument("--in-human-halves", action="append", required=False, default=[],
                    help="Human halves JSONL file(s) (repeatable).")
    ap.add_argument("--out-root", required=True, help="Root for mirrored cleaned outputs (e.g., out/pre-precleaned)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = (repo_root / args.out_root).resolve()

    # Accumulator for summary
    summary = {
        "totals": {
            "files": 0,
            "items": 0,
            "touched": 0,
            "chars_removed": 0,
            "intro_truncated": 0,
            "newlines_collapsed": 0,
            "spaces_collapsed": 0,
            "strip_applied": 0,
        },
        "files": [],
        "run": {
            "repo_root": str(Path(args.repo_root)),
            "in_out_dirs": args.in_out_dir,
            "in_human_halves": args.in_human_halves,
            "out_root": args.out_root,
        },
    }

    # Global diffs & summaries paths (under out_root)
    diffs_path = out_root / "cleaning_diffs.jsonl"
    summary_json_path = out_root / "cleaning_summary.json"
    summary_md_path = out_root / "cleaning_summary.md"

    # Wipe previous diffs if present to avoid mixing runs
    if diffs_path.exists():
        diffs_path.unlink()

    # 1) Process input trees (model outputs)
    for subtree in args.in_out_dir:
        in_root = (repo_root / subtree).resolve()
        if not in_root.exists():
            print(f"[WARN] --in-out-dir not found: {subtree}", file=sys.stderr)
            continue
        for src_path in in_root.rglob("*.jsonl"):
            rel_to_in = src_path.relative_to(in_root)
            dest_path = out_root / rel_to_in
            dest_path = dest_path.with_name(dest_path.stem + ".preclean.jsonl")

            process_file(
                src_path=src_path,
                dest_path=dest_path,
                diffs_sink=diffs_path,
                is_human_halves_file=False,
                summary_acc=summary,
                repo_root=repo_root
            )

    # 2) Process human halves (individual files)
    for hh in args.in_human_halves:
        src_path = (repo_root / hh).resolve()
        if not src_path.exists():
            print(f"[WARN] --in-human-halves not found: {hh}", file=sys.stderr)
            continue
        rel_to_repo = src_path.relative_to(repo_root)
        dest_path = out_root / rel_to_repo
        dest_path = dest_path.with_name(dest_path.stem + ".preclean.jsonl")

        process_file(
            src_path=src_path,
            dest_path=dest_path,
            diffs_sink=diffs_path,
            is_human_halves_file=True,
            summary_acc=summary,
            repo_root=repo_root
        )

    # 3) Write summaries
    ensure_parent(summary_json_path)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with summary_md_path.open("w", encoding="utf-8") as f:
        f.write(build_summary_md(summary))

    print(f"[OK] Files processed: {summary['totals']['files']}")
    print(f"[OK] Diffs: {diffs_path.relative_to(repo_root)}")
    print(f"[OK] Summary JSON: {summary_json_path.relative_to(repo_root)}")
    print(f"[OK] Summary MD: {summary_md_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
