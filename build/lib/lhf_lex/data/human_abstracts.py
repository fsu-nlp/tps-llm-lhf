from __future__ import annotations

"""
Utilities for handling human-written abstracts stored as `YYYY.txt` (one abstract
per line): load per year, sample per year, split into halves, and produce
JSONL-ready rows for prompting/evaluation.
"""

import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Tuple

from lhf_lex.text.sentences import split_sentences


@dataclass
class AbstractRecord:
    """Container for a single abstract and metadata.

    Attributes:
        year: Four-digit year inferred from the filename.
        idx: Zero-based line index within the year file.
        text: Full abstract text.
        sentences: Sentence segmentation of `text`.
    """
    year: int
    idx: int
    text: str
    sentences: List[str]


def _read_year_file(path: str, year: int) -> Iterable[AbstractRecord]:
    """Yield records from a `YYYY.txt` file (one abstract per non-empty line).

    Splits each line into sentences with `split_sentences`; empty or
    sentence-less lines are skipped.

    Args:
        path: Path to the year file.
        year: Four-digit year label to attach to each record.

    Yields:
        `AbstractRecord` instances.
    """
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            t = line.strip()
            if not t:
                continue
            sents = split_sentences(t)
            if not sents:
                continue
            yield AbstractRecord(year=year, idx=i, text=t, sentences=sents)


def load_human_abstracts_dir(root_dir: str, years: Optional[List[int]] = None) -> List[AbstractRecord]:
    """Load abstracts from a directory of `YYYY.txt` files.

    If `years` is None, auto-detect all 4-digit `*.txt` files; otherwise, load
    only the specified years that exist.

    Args:
        root_dir: Directory containing `YYYY.txt` files.
        years: Optional list of years to include.

    Returns:
        List of `AbstractRecord` objects.
    """
    records: List[AbstractRecord] = []
    if years is None:
        for path in glob.glob(os.path.join(root_dir, "*.txt")):
            base = os.path.basename(path)
            name, _ = os.path.splitext(base)
            if name.isdigit() and len(name) == 4:
                y = int(name)
                records.extend(list(_read_year_file(path, y)))
    else:
        for y in years:
            path = os.path.join(root_dir, f"{y}.txt")
            if os.path.exists(path):
                records.extend(list(_read_year_file(path, y)))
    return records


def sample_per_year(records: List[AbstractRecord], n_per_year: int = 5, seed: int = 42) -> List[AbstractRecord]:
    """Stratified sample: up to `n_per_year` abstracts per year (with seed).

    If a year has ≤ `n_per_year` records, all are kept.

    Args:
        records: Input abstracts.
        n_per_year: Maximum number per year.
        seed: RNG seed for reproducibility.

    Returns:
        Sampled list of `AbstractRecord`.
    """
    by_year: Dict[int, List[AbstractRecord]] = {}
    for r in records:
        by_year.setdefault(r.year, []).append(r)
    rng = random.Random(seed)
    out: List[AbstractRecord] = []
    for y, lst in sorted(by_year.items()):
        if len(lst) <= n_per_year:
            out.extend(lst)
        else:
            out.extend(rng.sample(lst, n_per_year))
    return out


def split_into_halves(sentences: List[str]) -> Tuple[str, str]:
    """Split a sentence list into near-equal halves (left-biased on odd lengths).

    Args:
        sentences: Tokenised sentences of an abstract.

    Returns:
        Tuple `(first_half, second_half)` as strings (may be empty).
    """
    if not sentences:
        return "", ""
    k = (len(sentences) + 1) // 2
    first = " ".join(sentences[:k]).strip()
    second = " ".join(sentences[k:]).strip()
    return first, second


def to_jsonl_rows(records: List[AbstractRecord]) -> List[Dict]:
    """Convert records to JSONL-ready dicts including first/second halves.

    Args:
        records: Input `AbstractRecord` list.

    Returns:
        List of dicts with keys:
        {year, idx, text, sentences, first_half, second_half}.
    """
    rows: List[Dict] = []
    for r in records:
        first, second = split_into_halves(r.sentences)
        rows.append(
            {
                "year": r.year,
                "idx": r.idx,
                "text": r.text,
                "sentences": r.sentences,
                "first_half": first,
                "second_half": second,
            }
        )
    return rows
