from __future__ import annotations
from collections import Counter
from typing import Dict, Iterable

from lhf_lex.text.normalise import tokenize

def normalised_frequency(texts: Iterable[str], vocab: Iterable[str]) -> Dict[str, float]:
    vset = {w.lower() for w in vocab}
    counts = Counter()
    total = 0
    for t in texts:
        toks = tokenize(t)
        total += len(toks)
        for tok in toks:
            if tok in vset:
                counts[tok] += 1
    if total == 0:
        return {w: 0.0 for w in vset}
    scale = 1_000_000 / total
    return {w: counts.get(w, 0) * scale for w in sorted(vset)}
