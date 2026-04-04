from __future__ import annotations

"""
Naïve sentence splitter: split on punctuation boundaries (., !, ?) followed by
whitespace/newlines and before an uppercase/number/paren/bracket or end-of-text.
Collapses internal whitespace and returns trimmed sentence strings.
"""

import re
from typing import List

# Boundary: lookbehind for [.!?], then space/newline, then lookahead to [A-Z0-9([] or EOS.
_SENT_BOUNDARY_RE = re.compile(r'(?<=[.!?])(?:\s+|\s*\n\s*)(?=[A-Z0-9\(\[]|$)')


def split_sentences(text: str) -> List[str]:
    """Split `text` into sentences using a light-weight regex heuristic.

    Normalises runs of whitespace, then splits at `_SENT_BOUNDARY_RE`.

    Args:
        text: Raw input string.

    Returns:
        List of non-empty, trimmed sentences.
    """
    text = text.strip()
    if not text:
        return []
    text = re.sub(r'\s+', ' ', text)
    parts = re.split(_SENT_BOUNDARY_RE, text)
    return [p.strip() for p in parts if p and p.strip()]
