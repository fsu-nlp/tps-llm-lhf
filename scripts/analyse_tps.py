#!/usr/bin/env python3
"""
TPS toolkit (discover vs evaluate) on spaCy/CoNLL-U parses.

Modes
-----
1) --mode discover
   Input:  --pos-root containing subfolders: human/, <modelA>/, <modelB>/...
            Each model folder must have exactly one 'base' and one 'instruct' .conllu.
   Output: Per-model wTPS (per-lexeme), optional cross-model aggregate; QC and summaries.

2) --mode eval --level {sequences,documents,dataset}
   Input:  --wtable (CSV with columns including 'key' and a τ column),
           --eval-conllu (one parsed dataset: H or B or I or any other corpus),
           --key (must match the keying used to build wTPS).
   Output: sTPS per sentence / doc-level TPS / dataset-level mTPS.
           Canonical unit score is L^p magnitude over τ(w) with p=--lp (default 2).
           NOTE: By default (unless you pass an explicit flag), τ(w) is rectified at 0
                 (τ⁺=max(0,τ)) during eval to focus on preference-stage *increases*.

Conventions
-----------
τ_stage(w) = ell_I(w) - ell_B(w) by default, but we prefer a dedicated 'TPS' column.
wTPS_M(w) as defined by triangulation (I exceeds both H and B, penalising B>H).
Sequence/document/corpus scores use the same L^p functional and are token-exact.

Author: ablation_study
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------
# Stopwords (English)
# -----------------------
STOPWORDS_EN = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","s","t","can","will","just","don","should","now",
    "ll","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn"
}

NEWDOC_RE = re.compile(r"^#\s*newdoc id\s*=\s*(.+)\s*$", re.IGNORECASE)
SENTID_RE = re.compile(r"^#\s*sent_id\s*=\s*(.+)\s*$", re.IGNORECASE)
INT_ID_RE = re.compile(r"^\d+$", re.ASCII)

@dataclass
class Token:
    form: str
    lemma: str
    upos: str

@dataclass
class Sentence:
    sent_id: str
    tokens: List[Token]

@dataclass
class Doc:
    doc_id: str
    sentences: List[Sentence]

# ---------- I/O: CoNLL-U ----------

def read_conllu_docs(path: Path) -> List[Doc]:
    """Read CoNLL-U into Docs with sentence boundaries preserved."""
    docs: List[Doc] = []
    cur_doc_id: Optional[str] = None
    cur_sent_id: Optional[str] = None
    cur_sents: List[Sentence] = []
    cur_tokens: List[Token] = []
    sent_counter = 0

    def flush_sentence():
        nonlocal cur_tokens, cur_sent_id, sent_counter, cur_sents
        if cur_tokens:
            sid = cur_sent_id or f"sent-{sent_counter}"
            cur_sents.append(Sentence(sent_id=sid, tokens=cur_tokens))
            sent_counter += 1
            cur_tokens = []
            cur_sent_id = None

    def flush_doc():
        nonlocal cur_sents, cur_doc_id, docs, sent_counter
        if cur_doc_id is not None:
            flush_sentence()
            docs.append(Doc(doc_id=cur_doc_id, sentences=cur_sents))
        cur_sents = []
        sent_counter = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                flush_sentence()
                continue
            if line.startswith("#"):
                mdoc = NEWDOC_RE.match(line)
                if mdoc:
                    flush_doc()
                    cur_doc_id = mdoc.group(1).strip()
                    continue
                ms = SENTID_RE.match(line)
                if ms:
                    cur_sent_id = ms.group(1).strip()
                continue
            parts = line.split("\t")
            if len(parts) < 10:
                continue
            tok_id = parts[0]
            if not INT_ID_RE.match(tok_id):
                continue
            form = parts[1]
            lemma = parts[2]
            upos = parts[3]
            cur_tokens.append(Token(form=form, lemma=lemma, upos=upos))

    if cur_doc_id is None:
        flush_sentence()
        if cur_sents:
            docs.append(Doc(doc_id=path.stem, sentences=cur_sents))
    else:
        flush_doc()
    return docs

# ---------- Keying & helpers ----------

def make_key(tok: Token, key_type: str, lowercase: bool) -> Tuple[str, str, str, str]:
    lemma = tok.lemma or ""
    form  = tok.form  or ""
    upos  = tok.upos  or ""
    if lowercase:
        lemma = lemma.lower()
        form = form.lower()
    if key_type == "lemma_pos":
        key = f"{lemma}_{upos}"
    elif key_type == "lemma":
        key = lemma
    elif key_type == "form_pos":
        key = f"{form}_{upos}"
    elif key_type == "form":
        key = form
    else:
        raise ValueError(f"Unsupported key type: {key_type}")
    return key, lemma, form, upos

def flatten_tokens(doc: Doc) -> List[Token]:
    out: List[Token] = []
    for s in doc.sentences:
        out.extend(s.tokens)
    return out

def seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)

def quantiles_stratified(M: int, jitter: bool, rng: random.Random) -> List[float]:
    if M <= 0:
        return []
    base = [(m + 0.5) / M for m in range(M)]
    if not jitter:
        return base
    wiggle = 0.5 / M
    return [min(1.0, max(0.0, q + rng.uniform(-wiggle, wiggle))) for q in base]

def prevalence_with_jeffreys(count: float, n: int) -> float:
    return (count + 0.5) / (n + 1.0)

# ---------- Triangulation (discover) ----------

def compute_starts_for_triplet(L_H: int, L_B: int, L_I: int, K: int,
                               policy: str, trim_common: bool,
                               M: int, jitter: bool, base_seed: int, r_index: int
                               ) -> Optional[Tuple[List[int], List[int], List[int], int]]:
    spans = [L_H - K, L_B - K, L_I - K]
    if trim_common:
        S = min(spans)
        if S < 0:
            return None
        span_H = span_B = span_I = S
    else:
        if min(spans) < 0:
            return None
        span_H, span_B, span_I = spans

    rng = seeded_rng(base_seed + r_index)

    def starts_from_span(span: int) -> List[int]:
        if span < 0:
            return []
        if policy == "first":
            return [0] * max(1, M)
        if policy == "last":
            return [span] * max(1, M)
        qs = quantiles_stratified(M, jitter=jitter, rng=rng)
        if span == 0:
            return [0] * max(1, len(qs))
        return [min(span, max(0, int(q * span))) for q in qs]

    sH = starts_from_span(span_H)
    sB = starts_from_span(span_B)
    sI = starts_from_span(span_I)
    M_eff = min(len(sH), len(sB), len(sI))
    sH, sB, sI = sH[:M_eff], sB[:M_eff], sI[:M_eff]
    if M_eff == 0:
        return None
    return sH, sB, sI, M_eff

def triangulate_wtps(human_docs: List[Doc], base_docs: List[Doc], instr_docs: List[Doc],
                     K: int, key_type: str, exc_upos_calc: set, lowercase: bool,
                     seed0: int, policy: str, trim_common: bool,
                     M: int, jitter: bool, rel_floor: float) -> Tuple[List[Dict], Dict]:
    n_triplets = min(len(human_docs), len(base_docs), len(instr_docs))
    n_eligible = 0
    dropped_short = 0

    H_counts: Dict[str, float] = defaultdict(float)
    B_counts: Dict[str, float] = defaultdict(float)
    I_counts: Dict[str, float] = defaultdict(float)

    key2upos: Dict[str, str] = {}
    key2lemma: Dict[str, str] = {}
    key2form: Dict[str, str] = {}

    for r in range(n_triplets):
        h_tokens = flatten_tokens(human_docs[r])
        b_tokens = flatten_tokens(base_docs[r])
        i_tokens = flatten_tokens(instr_docs[r])
        L_H, L_B, L_I = len(h_tokens), len(b_tokens), len(i_tokens)

        starts = compute_starts_for_triplet(L_H, L_B, L_I, K, policy, trim_common,
                                            M, jitter, seed0, r)
        if starts is None:
            dropped_short += 1
            continue
        sH, sB, sI, M_eff = starts

        presH: Dict[str, int] = defaultdict(int)
        presB: Dict[str, int] = defaultdict(int)
        presI: Dict[str, int] = defaultdict(int)

        for m in range(M_eff):
            sh, sb, si = sH[m], sB[m], sI[m]
            wh = h_tokens[sh:sh+K]
            wb = b_tokens[sb:sb+K]
            wi = i_tokens[si:si+K]

            setH, setB, setI = set(), set(), set()
            for tok in wh:
                if tok.upos in exc_upos_calc: continue
                k, lem, frm, up = make_key(tok, key_type, lowercase)
                setH.add(k); key2upos.setdefault(k, up); key2lemma.setdefault(k, lem); key2form.setdefault(k, frm)
            for tok in wb:
                if tok.upos in exc_upos_calc: continue
                k, lem, frm, up = make_key(tok, key_type, lowercase)
                setB.add(k); key2upos.setdefault(k, up); key2lemma.setdefault(k, lem); key2form.setdefault(k, frm)
            for tok in wi:
                if tok.upos in exc_upos_calc: continue
                k, lem, frm, up = make_key(tok, key_type, lowercase)
                setI.add(k); key2upos.setdefault(k, up); key2lemma.setdefault(k, lem); key2form.setdefault(k, frm)

            for k in setH: presH[k] += 1
            for k in setB: presB[k] += 1
            for k in setI: presI[k] += 1

        for k, c in presH.items(): H_counts[k] += c / M_eff
        for k, c in presB.items(): B_counts[k] += c / M_eff
        for k, c in presI.items(): I_counts[k] += c / M_eff
        n_eligible += 1

    rows: List[Dict] = []
    if n_eligible == 0:
        return rows, {"n_triplets": n_triplets, "n_eligible": 0,
                      "dropped_short": dropped_short,
                      "note": "No eligible triplets for K."}

    all_keys = set(H_counts) | set(B_counts) | set(I_counts)
    for k in all_keys:
        cH = H_counts.get(k, 0.0); cB = B_counts.get(k, 0.0); cI = I_counts.get(k, 0.0)
        lH = prevalence_with_jeffreys(cH, n_eligible)
        lB = prevalence_with_jeffreys(cB, n_eligible)
        lI = prevalence_with_jeffreys(cI, n_eligible)
        dIH = lI - lH; dIB = lI - lB; dBH = lB - lH
        TPS = min(dIH, dIB) - max(0.0, dBH)
        rows.append({
            "key": k, "lemma": key2lemma.get(k,""), "form": key2form.get(k,""),
            "upos": key2upos.get(k,""), "n_docs": n_eligible,
            "c_H": round(cH,6), "c_B": round(cB,6), "c_I": round(cI,6),
            "ell_H": lH, "ell_B": lB, "ell_I": lI,
            "Delta_IH": dIH, "Delta_IB": dIB, "Delta_BH": dBH,
            "TPS": TPS,
            "pp_IH": 100.0*dIH, "pp_IB": 100.0*dIB, "pp_BH": 100.0*dBH,
            "relpct_IH": 100.0*dIH/max(lH, 1e-9),
            "relpct_IB": 100.0*dIB/max(lB, 1e-9),
            "relpct_BH": 100.0*dBH/max(lH, 1e-9),
            "IH_decomp_base": dBH, "IH_decomp_lhf": dIB,
            "IH_share_lhf": (dIB/dIH) if abs(dIH) > 1e-12 else float("nan"),
        })
    rows.sort(key=lambda r: r["TPS"], reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank_TPS"] = rank

    Vpos = [r for r in rows if r["TPS"] > 0]
    def mean_or_nan(vals: List[float]) -> float:
        return float(sum(vals)/len(vals)) if vals else float("nan")
    qc = {
        "n_triplets": n_triplets, "n_eligible": n_eligible, "dropped_short": dropped_short,
        "unique_keys": len(all_keys), "Vpos_tokens_TPS_gt_0": len(Vpos),
        "pLAS_B_over_Vpos": mean_or_nan([r["ell_B"]-r["ell_H"] for r in Vpos]),
        "pLAS_I_over_Vpos": mean_or_nan([r["ell_I"]-r["ell_H"] for r in Vpos]),
        "IH_share_lhf_over_Vpos": mean_or_nan([r["IH_share_lhf"] for r in Vpos if not math.isnan(r["IH_share_lhf"])]),
    }
    return rows, qc

def write_wtps_csv(path: Path, rows: List[Dict], topk: int,
                   exc_upos_outp: set, exc_stpw_outp: bool):
    out_rows = rows
    if exc_upos_outp:
        out_rows = [r for r in out_rows if r.get("upos","") not in exc_upos_outp]
    if exc_stpw_outp:
        def is_stop(r: Dict) -> bool:
            lem = (r.get("lemma") or "").lower()
            frm = (r.get("form") or "").lower()
            token = lem if lem else frm
            return token in STOPWORDS_EN if token else False
        out_rows = [r for r in out_rows if not is_stop(r)]
    if topk and topk > 0:
        out_rows = out_rows[:topk]

    fieldnames = [
        "rank_TPS","key","lemma","form","upos","n_docs","c_H","c_B","c_I",
        "ell_H","ell_B","ell_I","Delta_IH","Delta_IB","Delta_BH","TPS",
        "pp_IH","pp_IB","pp_BH","relpct_IH","relpct_IB","relpct_BH",
        "IH_decomp_base","IH_decomp_lhf","IH_share_lhf",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

# ---------- Aggregation over models (discover cross-model) ----------

def aggregate_over_models(per_model_rows: Dict[str, List[Dict]], mode: str = "macro") -> List[Dict]:
    """
    Aggregate per-key wTPS across models.

    mode:
      - "macro": average ell_* across models (equal weight).
      - "micro": pool counts and n_docs, then recompute ell_* via Jeffreys at pooled level.
    """
    from collections import defaultdict
    keys_models = defaultdict(list)  # key -> list[(model, row)]
    for m, rows in per_model_rows.items():
        for r in rows:
            keys_models[r["key"]].append((m, r))

    agg_rows: List[Dict] = []
    rel_floor = 1e-3
    for key, lst in keys_models.items():
        if mode == "macro":
            ell_H = sum(r["ell_H"] for _, r in lst) / len(lst)
            ell_B = sum(r["ell_B"] for _, r in lst) / len(lst)
            ell_I = sum(r["ell_I"] for _, r in lst) / len(lst)
            n_docs = int(round(sum(r["n_docs"] for _, r in lst) / len(lst)))
            cH = cB = cI = float("nan")
        elif mode == "micro":
            cH = sum(r["c_H"] for _, r in lst)
            cB = sum(r["c_B"] for _, r in lst)
            cI = sum(r["c_I"] for _, r in lst)
            n_docs = sum(r["n_docs"] for _, r in lst)
            ell_H = prevalence_with_jeffreys(cH, n_docs)
            ell_B = prevalence_with_jeffreys(cB, n_docs)
            ell_I = prevalence_with_jeffreys(cI, n_docs)
        else:
            raise ValueError("mode must be 'macro' or 'micro'")

        dIH = ell_I - ell_H
        dIB = ell_I - ell_B
        dBH = ell_B - ell_H
        TPS = min(dIH, dIB) - max(0.0, dBH)

        _, r0 = lst[0]
        agg_rows.append({
            "key": key,
            "lemma": r0["lemma"], "form": r0["form"], "upos": r0["upos"],
            "n_docs": n_docs,
            "c_H": cH, "c_B": cB, "c_I": cI,
            "ell_H": ell_H, "ell_B": ell_B, "ell_I": ell_I,
            "Delta_IH": dIH, "Delta_IB": dIB, "Delta_BH": dBH, "TPS": TPS,
            "pp_IH": 100.0 * dIH, "pp_IB": 100.0 * dIB, "pp_BH": 100.0 * dBH,
            "relpct_IH": 100.0 * dIH / max(ell_H, rel_floor),
            "relpct_IB": 100.0 * dIB / max(ell_B, rel_floor),
            "relpct_BH": 100.0 * dBH / max(ell_H, rel_floor),
            "IH_decomp_base": dBH,
            "IH_decomp_lhf": dIB,
            "IH_share_lhf": (dIB / dIH) if dIH > 1e-12 else float("nan"),
        })

    agg_rows.sort(key=lambda r: r["TPS"], reverse=True)
    for rank, r in enumerate(agg_rows, start=1):
        r["rank_TPS"] = rank
    return agg_rows

# ---------- τ map & evaluation ----------

def tau_from_wtable(wtable_csv: Path, tau_col: str = "TPS") -> Dict[str, float]:
    """
    Load per-lexeme weights τ(w). Preferred column is `tau_col` (default: 'TPS').
    If `tau_col` is missing, fall back to τ(w)=ell_I-ell_B.
    """
    tau: Dict[str, float] = {}
    with wtable_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["key"]
            if tau_col in row and row[tau_col] not in (None, "", "nan"):
                tau_val = float(row[tau_col])
            else:
                # fallback: Δ_IB = ell_I - ell_B
                ell_I = float(row.get("ell_I", "0.0"))
                ell_B = float(row.get("ell_B", "0.0"))
                tau_val = ell_I - ell_B
            tau[key] = tau_val
    return tau

def stps_per_sentence(
    docs: List[Doc],
    tau: Dict[str, float],
    key_type: str,
    lowercase: bool,
    exc_upos_calc: set,
    source_tag: str,
    p: float,
    rectify_tau: bool
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Compute sentence-level:
      - sTPS_p      : L^p magnitude (canonical for TPS; uses |τ|^p over possibly rectified τ)
      - sTPS_signed : mean of raw τ(w) (diagnostic; *not* rectified)
      - sTPS_pos    : max(0, sTPS_signed) (diagnostic)
    Also return totals needed for exact doc/dataset aggregation.
    """
    if p <= 0:
        raise ValueError("--lp must be > 0")

    rows: List[Dict] = []
    ds_sum_p = 0.0
    ds_sum_signed = 0.0
    ds_len_used = 0
    n_sentences = 0

    for d in docs:
        for s_idx, sent in enumerate(d.sentences):
            contribs: List[float] = []
            contribs_signed: List[float] = []
            for tok in sent.tokens:
                if tok.upos in exc_upos_calc:
                    continue
                key, _, _, _ = make_key(tok, key_type, lowercase)
                v = tau.get(key, 0.0)
                contribs_signed.append(v)  # raw for signed diagnostics
                if rectify_tau and v < 0.0:
                    v = 0.0
                contribs.append(v)

            n_used = len(contribs)
            n_len = len(sent.tokens)
            if n_used == 0:
                sTPS_p = 0.0
                sTPS_signed = 0.0
                sTPS_pos = 0.0
                sum_p = 0.0
                sum_signed = 0.0
            else:
                # L^p magnitude on (possibly rectified) tokens
                sum_p = float(sum((abs(x) ** p) for x in contribs))
                sTPS_p = float((sum_p / n_used) ** (1.0 / p))
                # signed diagnostics on raw τ
                sum_signed = float(sum(contribs_signed))
                sTPS_signed = float(sum_signed / n_used)
                sTPS_pos = float(max(0.0, sTPS_signed))

            ds_sum_p += sum_p
            ds_sum_signed += sum_signed
            ds_len_used += n_used
            n_sentences += 1

            rows.append({
                "source": source_tag,
                "doc_id": d.doc_id,
                "sent_index": s_idx,
                "sent_id": sent.sent_id,
                "len_tokens": n_len,
                "len_used": n_used,
                "sum_p": sum_p,
                "sum_signed": sum_signed,
                "sTPS_p": sTPS_p,
                "sTPS_signed": sTPS_signed,
                "sTPS_pos": sTPS_pos
            })

    ds = {
        "ds_sum_p": ds_sum_p,
        "ds_sum_signed": ds_sum_signed,
        "ds_len_used": float(ds_len_used),
        "n_sentences": float(n_sentences),
        # convenience: unweighted means (not canonical for L^p aggregation)
        "mean_sTPS_p_unweighted": float(sum(r["sTPS_p"] for r in rows) / n_sentences) if n_sentences else float("nan"),
        "mean_sTPS_signed_unweighted": float(sum(r["sTPS_signed"] for r in rows) / n_sentences) if n_sentences else float("nan"),
    }
    return rows, ds

def dtps_per_document(
    seq_rows: List[Dict],
    p: float
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Aggregate sentence rows to document-level exact L^p and signed means.

    For each document d:
      dTPS_p      = ( sum_{s∈d} sum_p_s / sum_{s∈d} len_used_s )^(1/p)
      dTPS_signed =   sum_{s∈d} sum_signed_s / sum_{s∈d} len_used_s
    Dataset-level (canonical):
      mTPS_p      = ( sum_{all s} sum_p_s / sum_{all s} len_used_s )^(1/p)
      mTPS_signed =   sum_{all s} sum_signed_s / sum_{all s} len_used_s
    """
    by_doc_sum_p: Dict[str, float] = defaultdict(float)
    by_doc_sum_signed: Dict[str, float] = defaultdict(float)
    by_doc_len_used: Dict[str, int] = defaultdict(int)
    doc_source: Dict[str, str] = {}

    # dataset totals (canonical)
    ds_sum_p = 0.0
    ds_sum_signed = 0.0
    ds_len_used = 0

    for r in seq_rows:
        doc = r["doc_id"]
        doc_source[doc] = r["source"]
        nu = int(r["len_used"])
        sp = float(r["sum_p"])
        ss = float(r["sum_signed"])

        by_doc_sum_p[doc] += sp
        by_doc_sum_signed[doc] += ss
        by_doc_len_used[doc] += nu

        ds_sum_p += sp
        ds_sum_signed += ss
        ds_len_used += nu

    doc_rows: List[Dict] = []
    for doc in sorted(by_doc_len_used.keys()):
        L = by_doc_len_used[doc]
        if L <= 0:
            dTPS_p = 0.0
            dTPS_signed = 0.0
        else:
            dTPS_p = float((by_doc_sum_p[doc] / L) ** (1.0 / p))
            dTPS_signed = float(by_doc_sum_signed[doc] / L)
        doc_rows.append({
            "doc_id": doc,
            "source": doc_source.get(doc, ""),
            "len_used": L,
            "sum_p": by_doc_sum_p[doc],
            "sum_signed": by_doc_sum_signed[doc],
            "dTPS_p": dTPS_p,
            "dTPS_signed": dTPS_signed
        })

    if ds_len_used <= 0:
        mTPS_p = 0.0
        mTPS_signed = 0.0
    else:
        mTPS_p = float((ds_sum_p / ds_len_used) ** (1.0 / p))
        mTPS_signed = float(ds_sum_signed / ds_len_used)

    summary = {
        "mTPS_p": mTPS_p,
        "mTPS_signed": mTPS_signed,
        "ds_sum_p": ds_sum_p,
        "ds_sum_signed": ds_sum_signed,
        "ds_len_used": ds_len_used,
        "n_docs": len(by_doc_len_used)
    }
    return doc_rows, summary

# ---------- CLI subroutines ----------

def discover_main(args):
    pos_root: Path = args.pos_root
    if pos_root is None:
        raise ValueError("--pos-root is required for --mode discover")

    human_dir = pos_root / "human"
    human_file = discover_human_file(human_dir, args.human_file)
    human_docs = read_conllu_docs(human_file)

    # model dirs (exclude 'human')
    model_dirs = [p for p in pos_root.iterdir() if p.is_dir() and p.name != "human"]
    if args.models:
        want = set(args.models)
        model_dirs = [p for p in model_dirs if p.name in want]
        missing = want - {p.name for p in model_dirs}
        if missing:
            raise FileNotFoundError(f"Requested model folders not found: {sorted(missing)}")

    exc_calc = set(args.exc_upos_calc)
    exc_outp = set(args.exc_upos_outp)
    lowercase = not args.no_lowercase

    variant = (f"policy={args.window_policy}_trim={'yes' if args.trim_common else 'no'}"
               f"_M={args.num_windows}_K={args.windowk}_jitter={'no' if args.no_jitter else 'yes'}")
    out_root: Path = args.out_root / variant
    out_root.mkdir(parents=True, exist_ok=True)

    per_model_rows: Dict[str, List[Dict]] = {}

    for mdir in sorted(model_dirs, key=lambda p: p.name.lower()):
        model_name = mdir.name
        try:
            base_file, instr_file = discover_model_files(mdir)
        except FileNotFoundError as e:
            print(f"[skip] {model_name}: {e}")
            continue

        base_docs = read_conllu_docs(base_file)
        instr_docs = read_conllu_docs(instr_file)

        rows, qc = triangulate_wtps(
            human_docs, base_docs, instr_docs,
            K=args.windowk, key_type=args.key, exc_upos_calc=exc_calc, lowercase=lowercase,
            seed0=args.seed, policy=args.window_policy, trim_common=args.trim_common,
            M=max(1, args.num_windows), jitter=not args.no_jitter, rel_floor=args.rel_floor
        )
        per_model_rows[model_name] = rows

        model_out = out_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        csv_word = model_out / f"tps_word_{model_name}.csv"
        write_wtps_csv(csv_word, rows, topk=args.topk, exc_upos_outp=exc_outp,
                       exc_stpw_outp=args.exc_stpw_outp)

        summary = {
            "mode": "discover", "model": model_name,
            "human_file": str(human_file), "base_file": str(base_file), "instruct_file": str(instr_file),
            "params": {
                "key": args.key, "windowk": args.windowk, "topk": args.topk,
                "exc_upos_calc": sorted(exc_calc), "exc_upos_outp": sorted(exc_outp),
                "exc_stpw_outp": args.exc_stpw_outp, "seed": args.seed,
                "lowercase_keys": lowercase, "window_policy": args.window_policy,
                "trim_common": args.trim_common, "num_windows": args.num_windows,
                "jitter": not args.no_jitter
            },
            "qc": qc, "rows_written": (args.topk if args.topk>0 else len(rows)),
        }
        with (model_out / f"summary_{model_name}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[ok] discover: {model_name} → {csv_word}")

    # ---- cross-model aggregation (optional) ----
    if per_model_rows:
        agg_dir = out_root / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)

        modes = [args.aggregate] if args.aggregate in {"macro", "micro"} else ["macro", "micro"]
        for mode in modes:
            agg_rows = aggregate_over_models(per_model_rows, mode=mode)
            agg_csv = agg_dir / f"tps_word_ALLMODELS_{mode}.csv"
            write_wtps_csv(agg_csv, agg_rows, topk=args.topk,
                           exc_upos_outp=exc_outp,
                           exc_stpw_outp=args.exc_stpw_outp)

            # quick summary
            Vpos = [r for r in agg_rows if r["TPS"] > 0]
            def mean_or_nan(vals: List[float]) -> float:
                return float(sum(vals)/len(vals)) if vals else float("nan")
            pLAS_B = mean_or_nan([r["ell_B"] - r["ell_H"] for r in Vpos])
            pLAS_I = mean_or_nan([r["ell_I"] - r["ell_H"] for r in Vpos])
            share_lhf = mean_or_nan([r["IH_share_lhf"] for r in Vpos if not math.isnan(r["IH_share_lhf"])])

            with (agg_dir / f"summary_ALLMODELS_{mode}.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "mode": mode,
                    "variant_folder": str(out_root),
                    "n_models": len(per_model_rows),
                    "models": sorted(per_model_rows.keys()),
                    "Vpos_tokens_TPS_gt_0": len(Vpos),
                    "pLAS_B_over_Vpos": pLAS_B,
                    "pLAS_I_over_Vpos": pLAS_I,
                    "IH_share_lhf_over_Vpos": share_lhf,
                    "params": {
                        "key": args.key,
                        "windowk": args.windowk,
                        "topk": args.topk,
                        "exc_upos_calc": sorted(args.exc_upos_calc),
                        "exc_upos_outp": sorted(args.exc_upos_outp),
                        "exc_stpw_outp": args.exc_stpw_outp,
                        "seed": args.seed,
                        "lowercase_keys": not args.no_lowercase,
                        "window_policy": args.window_policy,
                        "trim_common": args.trim_common,
                        "num_windows": args.num_windows,
                        "jitter": not args.no_jitter,
                    },
                }, f, ensure_ascii=False, indent=2)
            print(f"[ok] aggregate ({mode}): {agg_csv}")

def eval_main(args):
    if args.wtable is None or args.eval_conllu is None:
        raise ValueError("--wtable and --eval-conllu are required for --mode eval")

    tau = tau_from_wtable(args.wtable, tau_col=args.tau_col)
    docs = read_conllu_docs(args.eval_conllu)

    exc_calc = set(args.exc_upos_calc)
    lowercase = not args.no_lowercase
    source_tag = args.source_tag or "SRC"
    p = float(args.lp)

    # Resolve rectification default:
    # - If the user *didn't* pass either flag, default to True and warn.
    # - If they passed one, honour it and stay quiet.
    rectify_tau = args.tau_rectify
    if rectify_tau is None:
        rectify_tau = True
        print("[warn] TPS eval: defaulting to τ rectification (τ⁺=max(0,τ)). "
              "Pass --no-tau-rectify to use raw τ magnitudes.", file=sys.stderr)

    out_dir: Path = args.out_root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # sequences
    seq_rows, ds_sent = stps_per_sentence(
        docs, tau, key_type=args.key, lowercase=lowercase,
        exc_upos_calc=exc_calc, source_tag=source_tag, p=p, rectify_tau=rectify_tau
    )
    stem = args.eval_conllu.stem

    if args.level == "sequences":
        seq_csv = out_dir / f"tps_seq_{stem}.csv"
        with seq_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "source","doc_id","sent_index","sent_id",
                "len_tokens","len_used","sum_p","sum_signed",
                "sTPS_p","sTPS_signed","sTPS_pos"
            ])
            w.writeheader()
            for r in seq_rows:
                w.writerow(r)
        with (out_dir / f"summary_seq_{stem}.json").open("w", encoding="utf-8") as f:
            json.dump({
                "mode":"eval","level":"sequences","source":source_tag,
                "lp": p,
                "tau_rectified": bool(rectify_tau),
                "mTPS_p_canonical": float((ds_sent["ds_sum_p"]/ds_sent["ds_len_used"])**(1.0/p)) if ds_sent["ds_len_used"]>0 else 0.0,
                "mTPS_signed_canonical": float(ds_sent["ds_sum_signed"]/ds_sent["ds_len_used"]) if ds_sent["ds_len_used"]>0 else 0.0,
                "mean_sTPS_p_unweighted": ds_sent["mean_sTPS_p_unweighted"],
                "mean_sTPS_signed_unweighted": ds_sent["mean_sTPS_signed_unweighted"],
                "n_sentences": int(ds_sent["n_sentences"]),
                "wtable": str(args.wtable),
                "tau_col": args.tau_col,
                "eval_conllu": str(args.eval_conllu)
            }, f, indent=2)
        print(f"[ok] eval/sequences: {seq_csv}")
        return

    # documents
    doc_rows, ds_doc = dtps_per_document(seq_rows, p=p)
    if args.level == "documents":
        doc_csv = out_dir / f"tps_doc_{stem}.csv"
        with doc_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "source","doc_id","len_used","sum_p","sum_signed","dTPS_p","dTPS_signed"
            ])
            w.writeheader()
            for r in doc_rows:
                w.writerow(r)
        with (out_dir / f"summary_doc_{stem}.json").open("w", encoding="utf-8") as f:
            json.dump({
                "mode":"eval","level":"documents","source":source_tag,
                "lp": p,
                "tau_rectified": bool(rectify_tau),
                # canonical dataset (token-weighted)
                "mTPS_p": ds_doc["mTPS_p"],
                "mTPS_signed": ds_doc["mTPS_signed"],
                "docs": ds_doc["n_docs"],
                "wtable": str(args.wtable),
                "tau_col": args.tau_col,
                "eval_conllu": str(args.eval_conllu)
            }, f, indent=2)
        print(f"[ok] eval/documents: {doc_csv}")
        return

    # dataset
    if args.level == "dataset":
        mTPS_p = float((ds_sent["ds_sum_p"]/ds_sent["ds_len_used"])**(1.0/p)) if ds_sent["ds_len_used"]>0 else 0.0
        mTPS_signed = float(ds_sent["ds_sum_signed"]/ds_sent["ds_len_used"]) if ds_sent["ds_len_used"]>0 else 0.0
        with (out_dir / f"summary_dataset_{stem}.json").open("w", encoding="utf-8") as f:
            json.dump({
                "mode":"eval","level":"dataset","source":source_tag,
                "lp": p,
                "tau_rectified": bool(rectify_tau),
                "mTPS_p": mTPS_p,
                "mTPS_signed": mTPS_signed,
                "n_sentences": int(ds_sent["n_sentences"]),
                "n_docs": len({r['doc_id'] for r in seq_rows}),
                "wtable": str(args.wtable),
                "tau_col": args.tau_col,
                "eval_conllu": str(args.eval_conllu)
            }, f, indent=2)
        print(f"[ok] eval/dataset: summary_dataset_{stem}.json")
        return

    raise ValueError("--level must be one of: sequences, documents, dataset")

# ---------- Helpers to locate files (discover) ----------

def discover_model_files(model_dir: Path) -> Tuple[Path, Path]:
    conllus = [p for p in model_dir.glob("*.conllu")]
    base  = [p for p in conllus if re.search(r"base", p.name, re.IGNORECASE)]
    instr = [p for p in conllus if re.search(r"instruct", p.name, re.IGNORECASE)]
    if len(base) != 1 or len(instr) != 1:
        raise FileNotFoundError(
            f"Expected exactly one base and one instruct .conllu in {model_dir}, "
            f"got base={len(base)} instruct={len(instr)}"
        )
    return base[0], instr[0]

def discover_human_file(human_dir: Path, override: Optional[Path] = None) -> Path:
    if override is not None:
        return override
    conllus = [p for p in human_dir.glob("*.conllu")]
    if len(conllus) != 1:
        raise FileNotFoundError(f"Expected exactly one human .conllu in {human_dir}, got {len(conllus)}")
    return conllus[0]

# ---------- Argument parsing ----------

def build_argparser():
    ap = argparse.ArgumentParser(description="TPS (discover vs evaluate) on spaCy/CoNLL-U parses.")
    ap.add_argument("--mode", choices=["discover","eval"], required=True,
                    help="discover: compute wTPS from H/B/I. eval: score sequences/docs/dataset using τ(w).")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="Output root directory.")
    ap.add_argument("--key", default="lemma_pos", choices=["lemma_pos","lemma","form_pos","form"],
                    help="Keying for lexical items.")
    ap.add_argument("--no-lowercase", action="store_true",
                    help="Do NOT lowercase forms/lemmas in keys.")
    ap.add_argument("--exc-upos-calc", nargs="*", default=[],
                    help="UPOS tags to exclude from calculation/scoring (e.g., PUNCT SYM).")
    ap.add_argument("--topk", type=int, default=0,
                    help="Top-k rows to output for wTPS (0 = all).")
    ap.add_argument("--exc-upos-outp", nargs="*", default=[],
                    help="(discover) UPOS tags to exclude from written wTPS CSV.")
    ap.add_argument("--exc-stpw-outp", action="store_true",
                    help="(discover) Exclude English stopwords from written wTPS CSV.")

    # discover-specific
    ap.add_argument("--pos-root", type=Path,
                    help="(discover) Root with subfolders 'human' and <model>/ .conllu files.")
    ap.add_argument("--human-file", type=Path, default=None,
                    help="(discover) Override human .conllu.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="(discover) Optional list of model folder names.")
    ap.add_argument("--windowk", type=int, default=60,
                    help="(discover) Fixed window length K (tokens).")
    ap.add_argument("--window-policy", choices=["quantile","first","last"], default="quantile",
                    help="(discover) Window placement policy.")
    ap.add_argument("--trim-common", action="store_true",
                    help="(discover) Trim starts to common support across H/B/I per triplet.")
    ap.add_argument("--num-windows", type=int, default=1,
                    help="(discover) Number of stratified quantile slices per triplet (M).")
    ap.add_argument("--no-jitter", action="store_true",
                    help="(discover) Disable quantile jitter.")
    ap.add_argument("--seed", type=int, default=13,
                    help="(discover) Base seed.")
    ap.add_argument("--rel-floor", type=float, default=0.005,
                    help="(discover) Floor for relative %% denominators (display only).")
    ap.add_argument("--aggregate", choices=["macro","micro","both"], default="macro",
                    help="(discover) Aggregate over models for wTPS (optional).")

    # eval-specific
    ap.add_argument("--level", choices=["sequences","documents","dataset"],
                    help="(eval) Output level.")
    ap.add_argument("--wtable", type=Path,
                    help="(eval) CSV from discover containing wTPS columns (e.g., TPS, ell_I, ell_B).")
    ap.add_argument("--eval-conllu", type=Path,
                    help="(eval) One parsed dataset to score (H, B, I, or any corpus).")
    ap.add_argument("--source-tag", type=str, default=None,
                    help="(eval) Label to tag the scored source (e.g., I, B, H).")
    ap.add_argument("--lp", type=float, default=2.0,
                    help="(eval) p for L^p unit score; default 2.0 (RMS). Must be > 0.")
    ap.add_argument("--tau-col", type=str, default="TPS",
                    help="(eval) Column in --wtable to use as τ(w). Default 'TPS'. "
                         "If missing, falls back to ell_I-ell_B.")

    # rectification flags (mutually exclusive); default is resolved at runtime with warning
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--tau-rectify", dest="tau_rectify", action="store_true",
                     help="(eval) Clip τ(w) at 0 (use τ⁺). Recommended for TPS.")
    grp.add_argument("--no-tau-rectify", dest="tau_rectify", action="store_false",
                     help="(eval) Use raw τ(w). Can inflate Base if many negative τ(w).")
    ap.set_defaults(tau_rectify=None)

    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()
    if args.mode == "discover":
        discover_main(args)
    else:
        if args.level is None:
            raise ValueError("--level is required for --mode eval")
        eval_main(args)

if __name__ == "__main__":
    main()
