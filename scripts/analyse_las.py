#!/usr/bin/env python3
"""
LAS toolkit (discover vs evaluate) on spaCy/CoNLL-U parses.

Enhancements in this revision:
- EVAL/SCORING supports either:
  (A) L^p mean per unit (sequence/document/dataset):
      uLAS_p(U; M) = ( (1/|T(U)|) * sum_{t in T(U)} | delta(w(t)) |^p )^(1/p)
      where delta(w) = ell_M(w) - ell_H(w) and T(U) are tokens scored
      (after --exc-upos-calc filtering). Choose with --lp <float>, e.g., 1, 1.5, 2.
  (B) Signed mean with no L^p normalisation (cancellation allowed):
      uLAS_signed(U; M) = (1/|T(U)|) * sum_{t in T(U)} delta(w(t))
      Select by passing --lp NONE
- CSV field names remain sLAS/dLAS. JSON summaries include 'norm_p' (number or null)
  and 'scoring_mode' in {"lp","signed"}.

Author: ablation_study
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
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
    return [t for s in doc.sentences for t in s.tokens]

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

# ---------- Windowing for H–M pairs ----------

def compute_starts_for_pair(L_H: int, L_M: int, K: int,
                            policy: str, trim_common: bool,
                            M: int, jitter: bool, base_seed: int, r_index: int
                            ) -> Optional[Tuple[List[int], List[int], int]]:
    spans = [L_H - K, L_M - K]
    if trim_common:
        S = min(spans)
        if S < 0:
            return None
        span_H = span_M = S
    else:
        if min(spans) < 0:
            return None
        span_H, span_M = spans

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
    sM = starts_from_span(span_M)
    M_eff = min(len(sH), len(sM))
    sH, sM = sH[:M_eff], sM[:M_eff]
    if M_eff == 0:
        return None
    return sH, sM, M_eff

# ---------- Discovery: per-model wLAS ----------

def discover_wlas(human_docs: List[Doc], model_docs: List[Doc],
                  K: int, key_type: str, exc_upos_calc: set, lowercase: bool,
                  seed0: int, policy: str, trim_common: bool,
                  M: int, jitter: bool) -> Tuple[List[Dict], Dict]:
    n_pairs = min(len(human_docs), len(model_docs))
    n_eligible = 0
    dropped_short = 0

    H_counts: Dict[str, float] = defaultdict(float)
    M_counts: Dict[str, float] = defaultdict(float)

    key2upos: Dict[str, str] = {}
    key2lemma: Dict[str, str] = {}
    key2form: Dict[str, str] = {}

    for r in range(n_pairs):
        h_tokens = flatten_tokens(human_docs[r])
        m_tokens = flatten_tokens(model_docs[r])
        L_H, L_M = len(h_tokens), len(m_tokens)

        starts = compute_starts_for_pair(L_H, L_M, K, policy, trim_common,
                                         M, jitter, seed0, r)
        if starts is None:
            dropped_short += 1
            continue
        sH, sM, M_eff = starts

        presH: Dict[str, int] = defaultdict(int)
        presM: Dict[str, int] = defaultdict(int)

        for m in range(M_eff):
            sh, sm = sH[m], sM[m]
            wh = h_tokens[sh:sh+K]
            wm = m_tokens[sm:sm+K]

            setH, setM = set(), set()
            for tok in wh:
                if tok.upos in exc_upos_calc:
                    continue
                k, lem, frm, up = make_key(tok, key_type, lowercase)
                setH.add(k); key2upos.setdefault(k, up); key2lemma.setdefault(k, lem); key2form.setdefault(k, frm)
            for tok in wm:
                if tok.upos in exc_upos_calc:
                    continue
                k, lem, frm, up = make_key(tok, key_type, lowercase)
                setM.add(k); key2upos.setdefault(k, up); key2lemma.setdefault(k, lem); key2form.setdefault(k, frm)

            for k in setH: presH[k] += 1
            for k in setM: presM[k] += 1

        for k, c in presH.items(): H_counts[k] += c / M_eff
        for k, c in presM.items(): M_counts[k] += c / M_eff
        n_eligible += 1

    rows: List[Dict] = []
    if n_eligible == 0:
        return rows, {"n_pairs": n_pairs, "n_eligible": 0,
                      "dropped_short": dropped_short,
                      "note": "No eligible pairs for K."}

    all_keys = set(H_counts) | set(M_counts)
    for k in all_keys:
        cH = H_counts.get(k, 0.0); cM = M_counts.get(k, 0.0)
        lH = prevalence_with_jeffreys(cH, n_eligible)
        lM = prevalence_with_jeffreys(cM, n_eligible)
        dMH = lM - lH  # LAS
        rows.append({
            "key": k, "lemma": key2lemma.get(k,""), "form": key2form.get(k,""),
            "upos": key2upos.get(k,""), "n_pairs": n_eligible,
            "c_H": round(cH,6), "c_M": round(cM,6),
            "ell_H": lH, "ell_M": lM,
            "Delta_MH": dMH, "LAS": dMH,
            "pp_MH": 100.0*dMH,
            "relpct_MH": 100.0*dMH/max(lH, 1e-9),
        })
    rows.sort(key=lambda r: r["LAS"], reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank_LAS"] = rank

    Vpos = [r for r in rows if r["LAS"] > 0]
    qc = {
        "n_pairs": n_pairs, "n_eligible": n_eligible, "dropped_short": dropped_short,
        "unique_keys": len(all_keys), "Vpos_tokens_LAS_gt_0": len(Vpos),
    }
    return rows, qc

def write_wlas_csv(path: Path, rows: List[Dict], topk: int,
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
        "rank_LAS","key","lemma","form","upos","n_pairs","c_H","c_M",
        "ell_H","ell_M","Delta_MH","LAS","pp_MH","relpct_MH",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

# ---------- Cross-model aggregation (wLAS) ----------

def aggregate_over_models(per_model_rows: Dict[str, List[Dict]], mode: str = "macro") -> List[Dict]:
    """
    Aggregate per-key wLAS across models (subvariants included).
    mode: 'macro' = mean of ell_*, 'micro' = pool counts+n_pairs then Jeffreys.
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
            ell_M = sum(r["ell_M"] for _, r in lst) / len(lst)
            n_pairs = int(round(sum(r["n_pairs"] for _, r in lst) / len(lst)))
            cH = cM = float("nan")
        elif mode == "micro":
            cH = sum(r["c_H"] for _, r in lst)
            cM = sum(r["c_M"] for _, r in lst)
            n_pairs = sum(r["n_pairs"] for _, r in lst)
            ell_H = prevalence_with_jeffreys(cH, n_pairs)
            ell_M = prevalence_with_jeffreys(cM, n_pairs)
        else:
            raise ValueError("mode must be 'macro' or 'micro'")

        dMH = ell_M - ell_H
        _, r0 = lst[0]
        agg_rows.append({
            "key": key,
            "lemma": r0["lemma"], "form": r0["form"], "upos": r0["upos"],
            "n_pairs": n_pairs, "c_H": cH, "c_M": cM,
            "ell_H": ell_H, "ell_M": ell_M,
            "Delta_MH": dMH, "LAS": dMH,
            "pp_MH": 100.0*dMH,
            "relpct_MH": 100.0*dMH/max(ell_H, rel_floor),
        })

    agg_rows.sort(key=lambda r: r["LAS"], reverse=True)
    for rank, r in enumerate(agg_rows, start=1):
        r["rank_LAS"] = rank
    return agg_rows

# ---------- V* selection and pLAS ----------

def select_vstar(wrows: List[Dict], mode: str, min_hprev: float) -> List[Dict]:
    if mode == "all":
        base = wrows
    elif mode == "pos":
        base = [r for r in wrows if r["LAS"] > 0]
    else:
        raise ValueError("--vstar must be 'all' or 'pos'")
    if min_hprev > 0.0:
        base = [r for r in base if r["ell_H"] >= min_hprev]
    return base

def corpus_plas(vstar_rows: List[Dict]) -> Dict[str, float]:
    if not vstar_rows:
        return {"pLAS_signed": float("nan"), "pLAS_pos": float("nan"), "Vstar_size": 0}
    deltas = [r["LAS"] for r in vstar_rows]
    p_signed = float(sum(deltas) / len(deltas))
    p_pos = float(sum(max(0.0, d) for d in deltas) / len(deltas))
    return {"pLAS_signed": p_signed, "pLAS_pos": p_pos, "Vstar_size": len(vstar_rows)}

# ---------- Eval: sLAS / dLAS / mLAS (L^p or signed) ----------

def delta_from_wtable(wtable_csv: Path) -> Dict[str, float]:
    dmap: Dict[str, float] = {}
    with wtable_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["key"]
            if "LAS" in row and row["LAS"] != "":
                dmap[key] = float(row["LAS"])
            else:
                lM = float(row["ell_M"]); lH = float(row["ell_H"])
                dmap[key] = lM - lH
    return dmap

def lp_mean_abs(values: List[float], p: float) -> float:
    """Return (mean |x|^p)^(1/p). Safe for empty input (returns 0.0)."""
    n = len(values)
    if n == 0:
        return 0.0
    if p <= 0:
        raise ValueError("p must be > 0 for L^p mean.")
    acc = 0.0
    for v in values:
        acc += abs(v) ** p
    return (acc / n) ** (1.0 / p)

def signed_mean(values: List[float]) -> float:
    """Return arithmetic mean of signed values. Empty input -> 0.0."""
    n = len(values)
    if n == 0:
        return 0.0
    return sum(values) / n

def slas_per_sentence(docs: List[Doc], delta: Dict[str, float],
                      key_type: str, lowercase: bool, exc_upos_calc: set,
                      source_tag: str, lp: Optional[float]) -> Tuple[List[Dict], float]:
    """
    Compute per-sequence score:
      - If lp is None: signed mean (cancellation allowed).
      - Else: L^p mean of absolute values.
    Returns (rows, mean_over_sequences).
    """
    rows: List[Dict] = []
    s_vals: List[float] = []
    for d in docs:
        for s_idx, sent in enumerate(d.sentences):
            contribs: List[float] = []
            used = 0
            for tok in sent.tokens:
                if tok.upos in exc_upos_calc:
                    continue
                key, _, _, _ = make_key(tok, key_type, lowercase)
                used += 1
                contribs.append(delta.get(key, 0.0))
            n_len = len(sent.tokens)
            if used > 0:
                s_val = signed_mean(contribs) if lp is None else lp_mean_abs(contribs, lp)
            else:
                s_val = 0.0
            s_vals.append(s_val)
            rows.append({
                "source": source_tag, "doc_id": d.doc_id,
                "sent_index": s_idx, "sent_id": sent.sent_id,
                "len_tokens": n_len, "len_used": used,
                # sLAS holds the chosen metric (signed or L^p)
                "sLAS": s_val,
                # Kept for backwards compatibility; equals sLAS even if negative in signed mode.
                "sLAS_pos": s_val
            })
    mean_seq = float(sum(s_vals)/len(s_vals)) if s_vals else float("nan")
    return rows, mean_seq

def dlas_per_document(seq_rows: List[Dict]) -> Tuple[List[Dict], float]:
    """
    Aggregate per document by arithmetic mean of sequence-level sLAS (already chosen metric).
    Returns (doc_rows, mean_over_docs).
    """
    by_doc: Dict[str, List[float]] = defaultdict(list)
    doc_src: Dict[str, str] = {}
    for r in seq_rows:
        doc = r["doc_id"]
        by_doc[doc].append(float(r["sLAS"]))
        doc_src[doc] = r["source"]
    doc_rows: List[Dict] = []
    all_vals: List[float] = []
    for doc, vals in by_doc.items():
        m = float(sum(vals)/len(vals)) if vals else 0.0
        doc_rows.append({"doc_id": doc, "source": doc_src.get(doc,"SRC"),
                         "dLAS": m, "dLAS_pos": m})
        all_vals.append(m)
    mean_doc = float(sum(all_vals)/len(all_vals)) if all_vals else float("nan")
    return doc_rows, mean_doc

# ---------- Helpers to locate files (discover) ----------

def discover_human_file(human_dir: Path, override: Optional[Path] = None) -> Path:
    if override is not None:
        return override
    conllus = [p for p in human_dir.glob("*.conllu")]
    if len(conllus) != 1:
        raise FileNotFoundError(f"Expected exactly one human .conllu in {human_dir}, got {len(conllus)}")
    return conllus[0]

def parse_variant_label(stem: str, variant_specs: List[str]) -> str:
    """
    Map a filename stem to a subvariant label.
    If --variant label:regex given, return first label whose regex matches stem (case-insensitive).
    Else: 'instruct' if 'instruct' in stem; 'base' if 'base' in stem; otherwise the stem itself.
    """
    if variant_specs:
        for spec in variant_specs:
            if ":" not in spec:
                raise ValueError(f"--variant expects label:regex, got '{spec}'")
            label, rgx = spec.split(":", 1)
            if re.search(rgx, stem, flags=re.IGNORECASE):
                return label
    low = stem.lower()
    if "instruct" in low:
        return "instruct"
    if "base" in low:
        return "base"
    return stem

def discover_model_files(model_dir: Path, file_glob: str, variant_specs: List[str]) -> List[Tuple[str, Path]]:
    """
    Return a list of (subvariant_label, path) for all matching .conllu files in model_dir.
    Ensures labels are unique within a directory by appending -2, -3, ... if needed.
    """
    files = sorted(model_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No model .conllu in {model_dir} matching '{file_glob}'")
    used: Dict[str, int] = {}
    out: List[Tuple[str, Path]] = []
    for p in files:
        stem = p.stem
        label = parse_variant_label(stem, variant_specs)
        cnt = used.get(label, 0) + 1
        used[label] = cnt
        if cnt > 1:
            label = f"{label}-{cnt}"
        out.append((label, p))
    return out

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

    variant_folder = (f"policy={args.window_policy}_trim={'yes' if args.trim_common else 'no'}"
                      f"_M={args.num_windows}_K={args.windowk}_jitter={'no' if args.no_jitter else 'yes'}")
    out_root: Path = args.out_root / variant_folder
    out_root.mkdir(parents=True, exist_ok=True)

    per_model_rows: Dict[str, List[Dict]] = {}

    for mdir in sorted(model_dirs, key=lambda p: p.name.lower()):
        model_name = mdir.name
        try:
            subfiles = discover_model_files(mdir, args.file_glob, args.variant)
        except FileNotFoundError as e:
            print(f"[skip] {model_name}: {e}")
            continue

        for sublabel, model_file in subfiles:
            model_docs = read_conllu_docs(model_file)

            rows, qc = discover_wlas(
                human_docs, model_docs,
                K=args.windowk, key_type=args.key, exc_upos_calc=exc_calc, lowercase=lowercase,
                seed0=args.seed, policy=args.window_policy, trim_common=args.trim_common,
                M=max(1, args.num_windows), jitter=not args.no_jitter
            )

            # register distinct name for aggregation
            agg_key = f"{model_name}:{sublabel}"
            per_model_rows[agg_key] = rows

            # output dirs/files
            model_out = (out_root / model_name / sublabel)
            model_out.mkdir(parents=True, exist_ok=True)

            csv_word = model_out / f"las_word_{model_name}_{sublabel}.csv"
            write_wlas_csv(csv_word, rows, topk=args.topk, exc_upos_outp=exc_outp,
                           exc_stpw_outp=args.exc_stpw_outp)

            # pLAS over V*
            vstar_rows = select_vstar(rows, mode=args.vstar, min_hprev=args.min_hprev)
            p_summary = corpus_plas(vstar_rows)

            summary = {
                "mode": "discover",
                "model_parent": model_name,
                "model_variant": sublabel,
                "model_agg_key": agg_key,
                "human_file": str(human_file),
                "model_file": str(model_file),
                "params": {
                    "key": args.key, "windowk": args.windowk, "topk": args.topk,
                    "exc_upos_calc": sorted(exc_calc), "exc_upos_outp": sorted(exc_outp),
                    "exc_stpw_outp": args.exc_stpw_outp, "seed": args.seed,
                    "lowercase_keys": lowercase, "window_policy": args.window_policy,
                    "trim_common": args.trim_common, "num_windows": args.num_windows,
                    "jitter": not args.no_jitter, "vstar": args.vstar, "min_hprev": args.min_hprev,
                    "file_glob": args.file_glob, "variant_specs": args.variant,
                },
                "qc": qc,
                "pLAS": p_summary,
                "rows_written": (args.topk if args.topk>0 else len(rows)),
            }
            with (model_out / f"summary_{model_name}_{sublabel}.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[ok] discover: {model_name}/{sublabel} → {csv_word}")

    # ---- cross-model aggregation (optional) ----
    if per_model_rows:
        agg_dir = out_root / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)

        modes = [args.aggregate] if args.aggregate in {"macro","micro"} else ["macro","micro"]
        for mode in modes:
            agg_rows = aggregate_over_models(per_model_rows, mode=mode)
            agg_csv = agg_dir / f"las_word_ALLMODELS_{mode}.csv"
            write_wlas_csv(agg_csv, agg_rows, topk=args.topk,
                           exc_upos_outp=exc_outp,
                           exc_stpw_outp=args.exc_stpw_outp)

            # pLAS for aggregate table
            vstar_rows = select_vstar(agg_rows, mode=args.vstar, min_hprev=args.min_hprev)
            p_summary = corpus_plas(vstar_rows)

            with (agg_dir / f"summary_ALLMODELS_{mode}.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "mode": mode,
                    "variant_folder": str(out_root),
                    "n_models": len(per_model_rows),
                    "models": sorted(per_model_rows.keys()),
                    "pLAS": p_summary,
                    "params": {
                        "key": args.key, "windowk": args.windowk, "topk": args.topk,
                        "exc_upos_calc": sorted(args.exc_upos_calc),
                        "exc_upos_outp": sorted(args.exc_upos_outp),
                        "exc_stpw_outp": args.exc_stpw_outp,
                        "seed": args.seed,
                        "lowercase_keys": not args.no_lowercase,
                        "window_policy": args.window_policy,
                        "trim_common": args.trim_common,
                        "num_windows": args.num_windows,
                        "jitter": not args.no_jitter,
                        "vstar": args.vstar, "min_hprev": args.min_hprev,
                        "file_glob": args.file_glob, "variant_specs": args.variant,
                    },
                }, f, ensure_ascii=False, indent=2)
            print(f"[ok] aggregate ({mode}): {agg_csv}")

def eval_main(args):
    if args.wtable is None or args.eval_conllu is None:
        raise ValueError("--wtable and --eval-conllu are required for --mode eval")

    # Parse --lp: accept numeric or NONE for signed mode
    lp = parse_lp_arg(args.lp)  # Optional[float]
    if lp is not None and lp <= 0:
        raise ValueError("--lp must be > 0 (or NONE).")

    delta = delta_from_wtable(args.wtable)
    docs = read_conllu_docs(args.eval_conllu)

    exc_calc = set(args.exc_upos_calc)
    lowercase = not args.no_lowercase
    source_tag = args.source_tag or "SRC"

    out_dir: Path = args.out_root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # sequences (sLAS is signed mean if lp=None; otherwise L^p mean)
    seq_rows, m_seq = slas_per_sentence(
        docs, delta, key_type=args.key, lowercase=lowercase,
        exc_upos_calc=exc_calc, source_tag=source_tag, lp=lp
    )

    stem = args.eval_conllu.stem
    scoring_mode = "signed" if lp is None else "lp"

    if args.level == "sequences":
        seq_csv = out_dir / f"las_seq_{stem}.csv"
        with seq_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source","doc_id","sent_index","sent_id","len_tokens","len_used","sLAS","sLAS_pos"])
            w.writeheader()
            for r in seq_rows: w.writerow(r)
        with (out_dir / f"summary_seq_{stem}.json").open("w", encoding="utf-8") as f:
            json.dump({"mode":"eval","level":"sequences","source":source_tag,
                       "scoring_mode": scoring_mode,
                       "norm_p": lp,
                       # Legacy aliases keep same number; interpret per 'scoring_mode'
                       "mLAS_abs": m_seq,
                       "mLAS_signed": m_seq,
                       "mLAS_pos": m_seq,
                       "rows": len(seq_rows), "wtable": str(args.wtable),
                       "eval_conllu": str(args.eval_conllu)}, f, indent=2)
        print(f"[ok] eval/sequences: {seq_csv}")
        return

    # documents (mean over sLAS)
    doc_rows, mean_doc = dlas_per_document(seq_rows)
    if args.level == "documents":
        doc_csv = out_dir / f"las_doc_{stem}.csv"
        with doc_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source","doc_id","dLAS","dLAS_pos"])
            w.writeheader()
            for r in doc_rows: w.writerow(r)
        with (out_dir / f"summary_doc_{stem}.json").open("w", encoding="utf-8") as f:
            json.dump({"mode":"eval","level":"documents","source":source_tag,
                       "scoring_mode": scoring_mode,
                       "norm_p": lp,
                       "mLAS_from_sentences": m_seq,
                       "mean_doc_dLAS": mean_doc,
                       "docs": len(doc_rows),
                       "wtable": str(args.wtable),
                       "eval_conllu": str(args.eval_conllu)}, f, indent=2)
        print(f"[ok] eval/documents: {doc_csv}")
        return

    # dataset (mean over sequence-level sLAS)
    if args.level == "dataset":
        with (out_dir / f"summary_dataset_{stem}.json").open("w", encoding="utf-8") as f:
            json.dump({"mode":"eval","level":"dataset","source":source_tag,
                       "scoring_mode": scoring_mode,
                       "norm_p": lp,
                       "mLAS": m_seq,
                       "n_sentences": len(seq_rows), "n_docs": len({r['doc_id'] for r in seq_rows}),
                       "wtable": str(args.wtable),
                       "eval_conllu": str(args.eval_conllu)}, f, indent=2)
        print(f"[ok] eval/dataset: summary_dataset_{stem}.json")
        return

    raise ValueError("--level must be one of: sequences, documents, dataset")

# ---------- Argument parsing ----------

def parse_lp_arg(lp_arg: Optional[str]) -> Optional[float]:
    """
    Parse --lp value. Returns:
      - None if lp_arg indicates signed mode ('none', 'signed', case-insensitive)
      - float(p) otherwise
    """
    if lp_arg is None:
        return 2.0  # default if not provided by argparse (shouldn't happen with default)
    s = str(lp_arg).strip().lower()
    if s in {"none", "signed"}:
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Invalid --lp value: {lp_arg}. Use a float (e.g., 1, 1.5, 2) or NONE.")

def build_argparser():
    ap = argparse.ArgumentParser(
        description="LAS (discover vs eval) on spaCy/CoNLL-U parses. Eval supports L^p mean (--lp <float>) or signed mean (--lp NONE)."
    )
    ap.add_argument("--mode", choices=["discover","eval"], required=True,
                    help="discover: compute wLAS from H/M. eval: score sequences/docs/dataset using precomputed δ(w).")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="Output root directory.")
    ap.add_argument("--key", default="lemma_pos", choices=["lemma_pos","lemma","form_pos","form"],
                    help="Keying for lexical items.")
    ap.add_argument("--no-lowercase", action="store_true",
                    help="Do NOT lowercase forms/lemmas in keys.")
    ap.add_argument("--exc-upos-calc", nargs="*", default=[],
                    help="UPOS tags to exclude from calculation/scoring (e.g., PUNCT SYM).")
    ap.add_argument("--topk", type=int, default=0,
                    help="Top-k rows to output for wLAS (0 = all).")
    ap.add_argument("--exc-upos-outp", nargs="*", default=[],
                    help="(discover) UPOS tags to exclude from written wLAS CSV.")
    ap.add_argument("--exc-stpw-outp", action="store_true",
                    help="(discover) Exclude English stopwords from written wLAS CSV.")

    # discover-specific
    ap.add_argument("--pos-root", type=Path,
                    help="(discover) Root with subfolders 'human' and <model>/ .conllu files.")
    ap.add_argument("--human-file", type=Path, default=None,
                    help="(discover) Override human .conllu.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="(discover) Optional list of model folder names.")

    # NEW: multi-file support within model dirs
    ap.add_argument("--file-glob", type=str, default="*.conllu",
                    help="(discover) Glob for model .conllu files inside each model dir (default: *.conllu).")
    ap.add_argument("--variant", action="append", default=[],
                    help="(discover) Map filename stems to subvariant labels via label:regex (case-insensitive). "
                         "Example: --variant base:_base_ --variant instruct:_instruct_. "
                         "If none given, auto-detect 'base'/'instruct', else use stem.")

    ap.add_argument("--windowk", type=int, default=60,
                    help="(discover) Fixed window length K (tokens).")
    ap.add_argument("--window-policy", choices=["quantile","first","last"], default="quantile",
                    help="(discover) Window placement policy.")
    ap.add_argument("--trim-common", action="store_true",
                    help="(discover) Trim starts to common support across H/M per pair.")
    ap.add_argument("--num-windows", type=int, default=1,
                    help="(discover) Number of stratified quantile slices per pair (M).")
    ap.add_argument("--no-jitter", action="store_true",
                    help="(discover) Disable quantile jitter.")
    ap.add_argument("--seed", type=int, default=42,
                    help="(discover) Base seed.")
    ap.add_argument("--aggregate", choices=["macro","micro","both"], default="macro",
                    help="(discover) Aggregate over models for wLAS (optional).")
    ap.add_argument("--vstar", choices=["all","pos"], default="all",
                    help="(discover) V* for pLAS: all keys or LAS>0 only.")
    ap.add_argument("--min-hprev", type=float, default=0.0,
                    help="(discover) Optional floor on ℓ_H for V* selection.")

    # eval-specific
    ap.add_argument("--level", choices=["sequences","documents","dataset"],
                    help="(eval) Output level.")
    ap.add_argument("--wtable", type=Path,
                    help="(eval) CSV from discover containing wLAS columns (LAS or ell_M/ell_H).")
    ap.add_argument("--eval-conllu", type=Path,
                    help="(eval) One parsed dataset to score (typically that model’s outputs).")
    ap.add_argument("--source-tag", type=str, default=None,
                    help="(eval) Label to tag the scored source (e.g., M, H).")
    ap.add_argument("--lp", type=str, default="2.0",
                    help="(eval) L^p exponent p over token contributions (use e.g. 1, 1.5, 2), "
                         "or 'NONE' for signed mean without L^p normalisation.")
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
