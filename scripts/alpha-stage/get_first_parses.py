#!/usr/bin/env python3
"""
Mirror a directory tree of .conllu files, writing only the first N lines of each file.

Example:
  python scripts-ig/get_first_parses.py \
    --src /home/tom/ablation/lhf-lex-ablation/out-ig/pre/pos-sm \
    --dst /home/tom/ablation/lhf-lex-ablation/out-ig/pre/pos-sm-sample \
    --lines 600

Notes:
- Only files with suffix ".conllu" are processed.
- Directory structure beneath --src is mirrored under --dst.
- If a file has fewer than N lines, the whole file is copied.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import os
import sys
import tempfile
import shutil

def head_write(src: Path, dst: Path, n_lines: int) -> None:
    """Write the first n_lines from src to dst (UTF-8). Uses atomic replace."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Count-as-you-go; if file shorter than N, we just copy all lines read.
    # Write to a temp file first for atomicity.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(dst.parent)) as tmp:
        tmp_path = Path(tmp.name)
        try:
            written = 0
            with src.open("r", encoding="utf-8", errors="strict") as fin:
                for line in fin:
                    if written < n_lines:
                        tmp.write(line)
                        written += 1
                    else:
                        break
        except Exception:
            # Clean up temp on error, then re-raise.
            try:
                tmp.close()
            finally:
                tmp_path.unlink(missing_ok=True)
            raise

    # Atomic replace into place
    os.replace(tmp_path, dst)

def process_tree(src_root: Path, dst_root: Path, n_lines: int, overwrite: bool) -> int:
    """Process all .conllu under src_root, mirror to dst_root; return count of files written."""
    count = 0
    for src in src_root.rglob("*.conllu"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel

        if dst.exists() and not overwrite:
            print(f"[skip] {dst} (exists, use --overwrite to replace)", file=sys.stderr)
            continue

        head_write(src, dst, n_lines)
        print(f"[ok]   {dst}")
        count += 1
    return count

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mirror .conllu files, writing first N lines.")
    p.add_argument("--src", required=True, type=Path, help="Source root (e.g., out-ig/pre/pos-sm)")
    p.add_argument("--dst", required=True, type=Path, help="Destination root (e.g., out-ig/pre/pos-sm-sample)")
    p.add_argument("--lines", "-n", required=True, type=int, help="Number of lines to keep from each file (N)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    if not args.src.exists() or not args.src.is_dir():
        print(f"Source directory does not exist or is not a directory: {args.src}", file=sys.stderr)
        sys.exit(1)

    args.dst.mkdir(parents=True, exist_ok=True)

    try:
        total = process_tree(args.src, args.dst, args.lines, args.overwrite)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"\nDone. Wrote {total} file(s) to {args.dst}")

if __name__ == "__main__":
    main()
