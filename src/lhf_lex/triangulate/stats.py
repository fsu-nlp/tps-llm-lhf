from __future__ import annotations

"""
Statistical utilities for triangulation.

- `aggregate_counts`: convert per-prompt feature sets (H/B/I) into
  document-frequency counts per feature.
- `paired_permutation`: within-prompt label permutation test for TPS
  (null: exchangeability of H, B, I), with BH–FDR correction.
"""

from typing import Dict, List, Tuple
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .prevalence import Counts, compute_tps


@dataclass
class PromptFeatures:
    """Feature sets for one prompt: Human (H), Base (B), Instruction (I)."""
    H: set
    B: set
    I: set


def aggregate_counts(blocks: List[PromptFeatures]) -> Dict[str, Counts]:
    """Aggregate per-feature document frequencies across prompts.

    For each feature w, count presence in H/B/I over all prompts and attach N
    (number of prompts), assuming equal N across conditions.

    Args:
        blocks: List of per-prompt feature triples.

    Returns:
        Mapping feature -> Counts(H, B, I, N).
    """
    vocab = set().union(*(pf.H | pf.B | pf.I for pf in blocks)) if blocks else set()
    N = len(blocks)
    counts: Dict[str, Counts] = {}
    for w in vocab:
        H = sum(1 for pf in blocks if w in pf.H)
        B = sum(1 for pf in blocks if w in pf.B)
        I = sum(1 for pf in blocks if w in pf.I)
        counts[w] = Counts(H=H, B=B, I=I, N=N)
    return counts


def paired_permutation(blocks: List[PromptFeatures], n_perm: int = 200, seed: int = 42) -> pd.DataFrame:
    """Paired (within-prompt) permutation test for TPS with BH–FDR.

    Procedure:
      1) Compute observed TPS per feature via `compute_tps(aggregate_counts(...))`.
      2) For each permutation, shuffle the labels (H, B, I) *within each prompt*,
         recompute TPS, and tally how often TPS_perm >= TPS_obs for each feature.
      3) Empirical p-values: (count_ge + 1) / (n_perm + 1).
      4) Apply Benjamini–Hochberg to obtain q-values.

    Args:
        blocks: Per-prompt feature triples.
        n_perm: Number of label permutations.
        seed: RNG seed.

    Returns:
        DataFrame with observed TPS and components plus `p_perm` and `q_bh`,
        sorted by TPS (desc).
    """
    rng = random.Random(seed)
    # observed
    obs_df = compute_tps(aggregate_counts(blocks))
    obs_tps = dict(zip(obs_df["word"], obs_df["TPS"]))

    words = list(obs_tps.keys())
    ge_counts = {w: 0 for w in words}

    for _ in range(n_perm):
        perm_blocks: List[PromptFeatures] = []
        for pf in blocks:
            triples = [pf.H, pf.B, pf.I]
            rng.shuffle(triples)  # permute labels within the prompt
            perm_blocks.append(PromptFeatures(H=triples[0], B=triples[1], I=triples[2]))
        perm_df = compute_tps(aggregate_counts(perm_blocks))
        perm_tps = dict(zip(perm_df["word"], perm_df["TPS"]))
        for w in words:
            if perm_tps.get(w, float("-inf")) >= obs_tps[w]:
                ge_counts[w] += 1

    pvals = {w: (ge_counts[w] + 1) / (n_perm + 1) for w in words}
    out = obs_df.copy()
    out["p_perm"] = out["word"].map(pvals)

    # BH-FDR
    out = out.sort_values("p_perm", ascending=True).reset_index(drop=True)
    m = len(out)
    qvals = []
    prev_q = 1.0
    for i, p in enumerate(out["p_perm"], start=1):
        q = min(prev_q, p * m / i)
        qvals.append(q)
        prev_q = q
    out["q_bh"] = qvals
    # sort back by TPS
    out = out.sort_values("TPS", ascending=False).reset_index(drop=True)
    return out
