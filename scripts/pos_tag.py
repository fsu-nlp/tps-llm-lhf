#!/usr/bin/env python3
"""
POS-tag GPT-cleaned JSONL files with spaCy and write CoNLL-U.

Example:
  # POS-only (fast): disables parser and NER
  python scripts/pos_tag.py \
    --model en_core_web_sm \
    --in-out-dir out/pre/cleaned \
    --out-root out/pre/pos \
    --n-process 6 \
    --posonly

  # Transformer, "lite" CPU: keep transformer+tagger; disable parser/NER/lemma/morph/attr_ruler
  python scripts/pos_tag.py \
    --model en_core_web_trf \
    --in-out-dir out/pre/cleaned \
    --out-root out/pre/pos_trf_lite \
    --n-process 1 \
    --batch-size 1024 \
    --lite

  # Full parse (default): POS + lemmatisation + morph + dependencies + NER
  python scripts/pos_tag.py \
    --model en_core_web_trf \
    --in-out-dir out/pre/cleaned \
    --out-root out/pre/pos_trf_full \
    --n-process 1 \
    --batch-size 256

Inputs (read-only, JSONL per line):
  <IN_OUT_DIR>/<model>/**.jsonl
  Each line: {..., "cleaned_text": "<text to tag>", "idx": <int>, "model_name": "...", ...}

Outputs (mirrors input structure under OUT_ROOT):
  <OUT_ROOT>/<model>/**.conllu

CoNLL-U fields: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC.
- If the parser is unavailable, HEAD/DEPREL are "_".
- If NER is unavailable, MISC has no NER tags.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tag JSONL files with spaCy and write CoNLL-U.")
    p.add_argument("--model", type=str, default="en_core_web_sm",
                   help="spaCy model to load (e.g., en_core_web_sm or en_core_web_trf).")
    p.add_argument("--in-out-dir", dest="in_root", type=Path, default=Path("out/pre/cleaned"),
                   help="Input root containing model subfolders with JSONL files.")
    p.add_argument("--out-root", type=Path, default=Path("out/pre/pos"),
                   help="Output root; folder structure mirrors --in-out-dir.")
    p.add_argument("--n-process", type=int, default=1,
                   help="Number of worker processes for spaCy nlp.pipe.")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Batch size for spaCy nlp.pipe.")
    p.add_argument("--field", type=str, default="cleaned_text",
                   help="JSON key containing the text to tag.")
    p.add_argument("--file-glob", type=str, default="*.jsonl",
                   help="Glob for input files (recursed under model subfolders).")

    # Pipeline slimming options
    p.add_argument("--posonly", action="store_true",
                   help="Disable parser and NER (POS/lemma/morph only if available).")
    p.add_argument("--lite", action="store_true",
                   help="Disable CPU-heavy pipes: parser, ner, lemmatizer, morphologizer, attribute_ruler.")
    p.add_argument("--disable", type=str, default="",
                   help="Comma-separated list of spaCy pipes to disable (e.g., 'parser,ner,lemmatizer').")

    p.add_argument("--no-gpu", action="store_true",
                   help="Explicitly disable GPU usage (debugging).")
    return p.parse_args()


def check_gpu_availability() -> None:
    print("[INFO] Checking GPU availability...", file=sys.stderr)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"[INFO] PyTorch CUDA available: {cuda_available}", file=sys.stderr)
            print(f"[INFO] CUDA device count: {device_count}", file=sys.stderr)
            print(f"[INFO] Primary CUDA device: {device_name}", file=sys.stderr)
            print(f"[INFO] PyTorch version: {torch.__version__}", file=sys.stderr)
            print(f"[INFO] CUDA version (PyTorch): {torch.version.cuda}", file=sys.stderr)
        else:
            print("[WARN] PyTorch CUDA not available!", file=sys.stderr)
            print(f"[INFO] PyTorch version: {torch.__version__}", file=sys.stderr)
    except ImportError:
        print("[WARN] PyTorch not installed!", file=sys.stderr)

    try:
        import spacy
        print(f"[INFO] spaCy version: {spacy.__version__}", file=sys.stderr)
        try:
            import cupy
            print(f"[INFO] CuPy installed (version: {cupy.__version__})", file=sys.stderr)
        except ImportError:
            print("[WARN] CuPy not installed - spaCy GPU support requires spacy[cuda].", file=sys.stderr)
    except ImportError:
        print("[ERROR] spaCy not installed!", file=sys.stderr)


def iter_model_files(in_root: Path, pattern: str) -> Iterable[Path]:
    in_root = in_root.resolve()
    for p in in_root.rglob(pattern):
        if p.is_file():
            try:
                rel = p.relative_to(in_root)
            except ValueError:
                continue
            if len(rel.parts) >= 2:
                yield p


def ensure_out_path(out_root: Path, in_root: Path, in_file: Path, new_suffix: str = ".conllu") -> Path:
    rel = in_file.resolve().relative_to(in_root.resolve())
    out_file = (out_root / rel).with_suffix(new_suffix)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    return out_file


def _compute_disable_list(name: str, posonly: bool, lite: bool, disable_csv: str) -> List[str]:
    disable: List[str] = []
    if posonly:
        disable.extend(["parser", "ner"])
    if lite:
        # Aggressive CPU trim; keep tagger for POS
        disable.extend(["parser", "ner", "lemmatizer", "morphologizer", "attribute_ruler"])
    if disable_csv.strip():
        extra = [x.strip() for x in disable_csv.split(",") if x.strip()]
        disable.extend(extra)
    # De-dup while preserving order
    seen = set()
    deduped = []
    for d in disable:
        if d not in seen:
            deduped.append(d)
            seen.add(d)
    # Warn if tagger disabled (POS would be blank)
    if "tagger" in deduped:
        print("[WARN] 'tagger' is disabled; UPOS/XPOS will be '_'.", file=sys.stderr)
    return deduped


def _activate_cuda_and_warmup_if_transformer(nlp) -> None:
    """Set CUDA device for transformer and do a light warmup (no logging)."""
    if "transformer" not in nlp.pipe_names:
        return
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    # Set default device and warm up once
    try:
        torch.set_default_device("cuda")
    except Exception:
        pass
    try:
        long_text = "Warmup " + " ".join(["token"] * 2000)
        with torch.inference_mode():
            _ = nlp(long_text)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def load_spacy_model(name: str, posonly: bool, lite: bool, disable_csv: str, use_gpu: bool = True):
    try:
        import spacy
    except ImportError:
        print("spaCy is not installed. Install with: pip install spacy", file=sys.stderr)
        raise

    disable = _compute_disable_list(name, posonly, lite, disable_csv)

    if use_gpu and "trf" in name:
        print("[INFO] Attempting to require GPU for transformer model...", file=sys.stderr)
        try:
            is_gpu = spacy.require_gpu()
            print("[INFO] GPU activated in spaCy." if is_gpu else "[WARN] spaCy did not activate GPU.", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Failed to require GPU: {e}", file=sys.stderr)
            print("[INFO] Continuing with CPU processing...", file=sys.stderr)

    try:
        nlp = spacy.load(name, disable=disable)
        print(f"[INFO] Loaded '{name}' with disable={disable}", file=sys.stderr)
    except OSError:
        print(f"spaCy model '{name}' not found.\nTry: python -m spacy download {name}", file=sys.stderr)
        raise

    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    if "transformer" in nlp.pipe_names and use_gpu:
        _activate_cuda_and_warmup_if_transformer(nlp)

    print(f"[INFO] Final pipes: {nlp.pipe_names}", file=sys.stderr)
    return nlp


def to_conllu(doc, doc_id: str) -> str:
    has_dep = doc.has_annotation("DEP")
    lines: List[str] = [f"# newdoc id = {doc_id}"]
    sent_idx = 0
    for sent in doc.sents:
        sent_idx += 1
        sent_text = sent.text.strip().replace("\t", " ").replace("\n", " ")
        lines.append(f"# sent_id = {doc_id}-{sent_idx}")
        lines.append(f"# text = {sent_text}")
        for i, token in enumerate(sent, start=1):
            form = token.text
            lemma = token.lemma_ if token.lemma_ else "_"
            upos = token.pos_ if token.pos_ else "_"
            xpos = token.tag_ if token.tag_ else "_"
            feats = str(token.morph) if str(token.morph) else "_"
            if has_dep:
                if token.dep_ == "ROOT":
                    head, deprel = 0, "root"
                else:
                    head = (token.head.i - sent.start) + 1 if sent.start <= token.head.i < sent.end else 0
                    deprel = token.dep_ if token.dep_ else "_"
            else:
                head, deprel = "_", "_"
            deps = "_"
            misc_parts = []
            if token.whitespace_ == "":
                misc_parts.append("SpaceAfter=No")
            if token.ent_iob_ != "O":
                ner_tag = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else token.ent_iob_
                misc_parts.append(f"NER={ner_tag}")
            misc = "|".join(misc_parts) if misc_parts else "_"
            lines.append(f"{i}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{misc}")
        lines.append("")
    if lines[-1] != "":
        lines.append("")
    return "\n".join(lines) + "\n"


def read_jsonl_texts(path: Path, text_field: str) -> Tuple[List[str], List[Dict]]:
    texts: List[str] = []
    metas: List[Dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (obj.get(text_field) or "").strip()
            meta = {
                "idx": obj.get("idx"),
                "prompt_id": obj.get("prompt_id"),
                "model_name": obj.get("model_name") or obj.get("model_id") or "",
                "source_file": str(path),
            }
            texts.append(text)
            metas.append(meta)
    return texts, metas


def open_atomic_writer(path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    f = tmp.open("w", encoding="utf-8", newline="\n")
    return f, tmp


def process_file(nlp, in_root: Path, in_path: Path, out_root: Path, batch_size: int, n_process: int, text_field: str) -> None:
    out_path = ensure_out_path(out_root, in_root, in_path, new_suffix=".conllu")
    texts, metas = read_jsonl_texts(in_path, text_field=text_field)
    out_f, tmp_path = open_atomic_writer(out_path)

    if not texts:
        out_f.close()
        tmp_path.replace(out_path)
        return

    stream = ((t, metas[i]) for i, t in enumerate(texts))

    batch_count = 0
    start_time = time.time()

    for doc_obj, ctx in nlp.pipe(stream, as_tuples=True, batch_size=batch_size, n_process=n_process):
        batch_count += 1

        base_stem = Path(ctx.get("source_file", "doc")).stem
        idx = ctx.get("idx")
        prompt_id = ctx.get("prompt_id")
        model_name = ctx.get("model_name", "")
        if idx is not None:
            doc_id = f"{base_stem}-idx{idx}"
        elif prompt_id is not None:
            doc_id = f"{base_stem}-pid{prompt_id}"
        else:
            doc_id = base_stem
        if model_name:
            doc_id = f"{doc_id}-{str(model_name).replace('/', '_')}"
        out_f.write(to_conllu(doc_obj, doc_id=doc_id))

    print(f"[INFO] Processed {batch_count} batches in {time.time() - start_time:.1f}s", file=sys.stderr)

    out_f.close()
    tmp_path.replace(out_path)


def main():
    args = parse_args()
    in_root = args.in_root
    out_root = args.out_root
    use_gpu = not args.no_gpu

    check_gpu_availability()

    nlp = load_spacy_model(
        args.model,
        posonly=args.posonly,
        lite=args.lite,
        disable_csv=args.disable,
        use_gpu=use_gpu,
    )
    print(f"[INFO] Using spaCy model: {args.model} | pipes: {nlp.pipe_names}", file=sys.stderr)
    print(f"[INFO] Reading from: {in_root}", file=sys.stderr)
    print(f"[INFO] Writing to:   {out_root}", file=sys.stderr)
    print(f"[INFO] n_process={args.n_process} batch_size={args.batch_size} posonly={args.posonly} lite={args.lite}", file=sys.stderr)

    if "transformer" in nlp.pipe_names and args.n_process > 1:
        print("[INFO] Transformer present; prefer --n-process 1 and larger --batch-size.", file=sys.stderr)

    files = sorted(iter_model_files(in_root, args.file_glob))
    if not files:
        print("[WARN] No input files found under model subfolders.", file=sys.stderr)
        return

    for i, in_path in enumerate(files, start=1):
        rel = in_path.resolve().relative_to(in_root.resolve())
        print(f"[{i}/{len(files)}] {rel}", file=sys.stderr)

        try:
            process_file(
                nlp=nlp,
                in_root=in_root,
                in_path=in_path,
                out_root=out_root,
                batch_size=args.batch_size,
                n_process=args.n_process,
                text_field=args.field,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {rel}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()