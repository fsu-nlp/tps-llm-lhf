# src/lhf_lex/triangulate/features.py
from __future__ import annotations

"""
Feature extractors for triangulation analyses.

Modes:
  - surface: lowercase word tokens (>1 char) via `lhf_lex.text.normalise.tokenize`
  - lemma_pos: spaCy lemmas with coarse POS tags, formatted as "lemma/POS"
  - markers: intersection of tokens with a provided marker lexicon
"""

from typing import Iterable, List, Set, Optional
import os
import re

from lhf_lex.text.normalise import tokenize as _tok

try:
    import spacy  # optional
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

__NLP_CACHE = {}


def _get_spacy(model: str = "en_core_web_sm", *, require: bool = False):
    """Load and cache a spaCy pipeline (NER/parser disabled).

    Args:
        model: spaCy model name.
        require: If True, raise a RuntimeError when spaCy/model are unavailable.

    Returns:
        A loaded spaCy Language object or None (when unavailable and require=False).
    """
    if not _SPACY_AVAILABLE:
        if require:
            raise RuntimeError(
                "spaCy is not installed but feature-mode 'lemma_pos' was requested. "
                "Install with: pip install 'spacy>=3.7,<4' && python -m spacy download en_core_web_sm"
            )
        return None
    nlp = __NLP_CACHE.get(model)
    if nlp is None:
        try:
            nlp = spacy.load(model, disable=["ner", "parser"])
        except OSError:
            if require:
                raise RuntimeError(
                    f"spaCy model '{model}' not found. Install with: python -m spacy download {model}"
                )
            return None
        __NLP_CACHE[model] = nlp
    return nlp


def surface_features(text: str, vocab: Optional[Set[str]] = None) -> Set[str]:
    """Extract surface-form token features (len>1, lowercased).

    Args:
        text: Input text.
        vocab: Optional whitelist to intersect with.

    Returns:
        Set of token strings.
    """
    toks = _tok(text)
    feats = set(t for t in toks if len(t) > 1)
    if vocab:
        feats &= vocab
    return feats


def lemma_pos_features(
    text: str,
    vocab: Optional[Set[str]] = None,
    spacy_model: str = "en_core_web_sm",
    *,
    require_spacy: bool = True,
) -> Set[str]:
    """Extract lemma/POS features using spaCy (fallback to surface if unavailable).

    Each feature is "lemma/POS" (lemma lowercased; spaCy coarse POS).

    Args:
        text: Input text.
        vocab: Optional whitelist to intersect with.
        spacy_model: spaCy model to load.
        require_spacy: If False, fall back to `surface_features` when spaCy is missing.

    Returns:
        Set of "lemma/POS" strings (or surface tokens on fallback).
    """
    nlp = _get_spacy(spacy_model, require=require_spacy)
    if nlp is None:  # only if require_spacy=False
        return surface_features(text, vocab=vocab)
    doc = nlp(text)
    feats = {f"{t.lemma_.lower()}/{t.pos_}" for t in doc if t.is_alpha and len(t) > 1}
    if vocab:
        feats &= vocab
    return feats


def markers_features(text: str, markers: Iterable[str]) -> Set[str]:
    """Return tokens present in a provided marker lexicon.

    Args:
        text: Input text.
        markers: Iterable of marker strings (case-insensitive).

    Returns:
        Set of matching token strings.
    """
    v = {m.lower() for m in markers}
    toks = _tok(text)
    return set(t for t in toks if t in v)


def build_feature_set(
    text: str,
    mode: str = "surface",
    vocab: Optional[Iterable[str]] = None,
    markers: Optional[Iterable[str]] = None,
    spacy_model: str = "en_core_web_sm",
) -> Set[str]:
    """Dispatch to a feature extractor according to `mode`.

    Args:
        text: Input text.
        mode: One of {"surface", "lemma_pos", "markers"} (aliases: "lemma-pos", "lemmapos").
        vocab: Optional iterable to restrict features (lowercased).
        markers: Required when mode="markers" (marker lexicon).
        spacy_model: spaCy model for lemma_pos mode.

    Returns:
        A set of feature strings.

    Raises:
        ValueError: Unknown mode or missing markers when required.
    """
    mode = (mode or "surface").lower()
    vset = set(w.lower() for w in (vocab or [])) or None
    if mode == "surface":
        return surface_features(text, vocab=vset)
    elif mode in ("lemma_pos", "lemma-pos", "lemmapos"):
        return lemma_pos_features(text, vocab=vset, spacy_model=spacy_model, require_spacy=True)
    elif mode == "markers":
        if markers is None:
            raise ValueError("markers mode requires a markers list")
        return markers_features(text, markers)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
