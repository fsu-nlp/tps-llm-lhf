from __future__ import annotations

"""
Lightweight normalisation utilities: strip simple Markdown/formatting artefacts
and produce a lowercase token stream of `[a-z']+` sequences.
"""

import re
from typing import List

# Match alphabetic tokens with optional apostrophes (e.g., don't, it's).
_TOKEN_RE = re.compile(r"[a-z']+", re.IGNORECASE)


def strip_formatting(text: str) -> str:
    """Remove basic Markdown-style formatting and list markers.
    
    Args:
        text: Input string possibly containing simple Markdown.

    Returns:
        Plaintext with common formatting artefacts removed (spacing preserved loosely).
    """
    t = text
    t = re.sub(r"\[.*?\]\(.*?\)", " ", t)          # links
    t = re.sub(r"\*\*|__|`", " ", t)               # emphasis/code markers
    t = re.sub(r"^\s*[-*]\s+", " ", t, flags=re.MULTILINE)  # bullets
    return t


def tokenize(text: str) -> List[str]:
    """Tokenise text after formatting strip, lowercasing to `[a-z']+` tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase tokens (letters + apostrophes).
    """
    t = strip_formatting(text).lower()
    return _TOKEN_RE.findall(t)
