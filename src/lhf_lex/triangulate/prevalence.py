from __future__ import annotations

"""
Token Prevalence Score (TPS) utilities.

Given document-frequency counts for features under human (H), base (B), and
instruction-tuned (I) continuations, compute smoothed logits (Jeffreys prior)
and derive TPS:
    TPS = min(Δ_IH, Δ_IB) − max(0, Δ_BH),
where Δ_XY is the difference in smoothed logits between conditions X and Y.
"""

from dataclasses import dataclass
from typing import Dict
import math

import pandas as pd


@dataclass
class Counts:
    """Per-feature document-frequency counts across conditions.

    Attributes:
        H: Document frequency under human continuations.
        B: Document frequency under base-model continuations.
        I: Document frequency under instruction-tuned continuations.
        N: Number of prompts/docs per condition (assumed equal across H/B/I).
    """
    H: int
    B: int
    I: int
    N: int  # number of prompts/docs per condition (assumed equal across H/B/I)


def smoothed_logit(df: int, N: int, alpha: float = 0.5, beta: float = 0.5) -> float:
    """Smoothed log-odds of prevalence with a Beta(α,β) prior.

    Uses Jeffreys prior by default (α=β=0.5):
        p̂ = (df + α) / (N + α + β)
        logit = log(p̂ / (1 − p̂))

    Args:
        df: Document frequency (0..N).
        N: Number of documents.
        alpha: Prior pseudo-count for successes.
        beta: Prior pseudo-count for failures.

    Returns:
        Smoothed logit (float).
    """
    p = (df + alpha) / (N + alpha + beta)
    return float(math.log(p / (1.0 - p)))


def compute_tps(counts: Dict[str, Counts]) -> pd.DataFrame:
    """Compute TPS and components for each feature.

    For each word/feature w:
        lH,lB,lI := smoothed logits for H,B,I
        Δ_IH := lI − lH
        Δ_IB := lI − lB
        Δ_BH := lB − lH
        TPS  := min(Δ_IH, Δ_IB) − max(0, Δ_BH)

    Args:
        counts: Mapping feature -> Counts(H,B,I,N).

    Returns:
        DataFrame sorted by TPS (desc) with columns:
        {word, df_H, df_B, df_I, logit_H, logit_B, logit_I,
         Delta_IH, Delta_IB, Delta_BH, TPS}.
    """
    rows = []
    # counts maps word -> Counts(H,B,I,N_per_condition)
    for w, c in counts.items():
        H, B, I, N = c.H, c.B, c.I, c.N
        lH = smoothed_logit(H, N)
        lB = smoothed_logit(B, N)
        lI = smoothed_logit(I, N)
        dIH = lI - lH
        dIB = lI - lB
        dBH = lB - lH
        TPS = min(dIH, dIB) - max(0.0, dBH)
        rows.append(
            {
                "word": w,
                "df_H": H,
                "df_B": B,
                "df_I": I,
                "logit_H": lH,
                "logit_B": lB,
                "logit_I": lI,
                "Delta_IH": dIH,
                "Delta_IB": dIB,
                "Delta_BH": dBH,
                "TPS": TPS,
            }
        )
    df = pd.DataFrame(rows).sort_values("TPS", ascending=False).reset_index(drop=True)
    return df
