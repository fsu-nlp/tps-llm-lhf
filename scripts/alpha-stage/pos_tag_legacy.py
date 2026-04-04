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
- If NER is available, MISC includes NER=B-XXX/I-XXX. If not, it's omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tag JSONL files with spaCy and write CoNLL-U.")
    p.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to load (e.g., en_core_web_sm or en_core_web_trf).",
    )
    p.add_argument(
        "--in-out-dir",
        dest="in_root",
        type=Path,
        default=Path("out/pre/cleaned"),
        help="Input root containing model subfolders with JSONL files.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("out/pre/pos"),
        help="Output root; folder structure mirrors --in-out-dir.",
    )
    p.add_argument(
        "--n-process",
        type=int,
        default=1,
        help="Number of worker processes for spaCy nlp.pipe.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for spaCy nlp.pipe.",
    )
    p.add_argument(
        "--field",
        type=str,
        default="cleaned_text",
        help="JSON key containing the text to tag.",
    )
    p.add_argument(
        "--file-glob",
        type=str,
        default="*.jsonl",
        help="Glob for input files (recursed under model subfolders).",
    )
    p.add_argument(
        "--posonly",
        action="store_true",
        help="If set, disable parser and NER (POS/lemma/morph only). Default (unset) runs full pipeline.",
    )
    return p.parse_args()


def iter_model_files(in_root: Path, pattern: str) -> Iterable[Path]:
    """Yield JSONL files under any subfolder of in_root (skip top-level files like cleaning summaries)."""
    in_root = in_root.resolve()
    for p in in_root.rglob(pattern):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(in_root)
        except ValueError:
            continue
        # Only process files that are at least one level down from in_root (i.e., within a model folder)
        if len(rel.parts) >= 2:
            yield p


def ensure_out_path(out_root: Path, in_root: Path, in_file: Path, new_suffix: str = ".conllu") -> Path:
    rel = in_file.resolve().relative_to(in_root.resolve())
    out_file = (out_root / rel).with_suffix(new_suffix)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    return out_file


def _activate_cuda_and_warmup_if_transformer(nlp) -> None:
    """For curated-transformers in spaCy 3.7+: set torch default device, warm up, then move shim model to CUDA."""
    if "transformer" not in nlp.pipe_names:
        return
    try:
        import torch
    except Exception:
        print("[WARN] torch not available; transformer will run on CPU.", file=sys.stderr)
        return

    default_set = False
    if torch.cuda.is_available():
        try:
            torch.set_default_device("cuda")
            default_set = True
        except Exception as e:
            print(f"[WARN] Could not set torch default device to CUDA: {e}", file=sys.stderr)
    else:
        print("[WARN] CUDA not available; transformer will run on CPU.", file=sys.stderr)

    try:
        long_text = "Warmup " + " ".join(["token"] * 2000)
        with torch.inference_mode():
            _ = nlp(long_text)
    except Exception as e:
        print(f"[WARN] Warm-up forward failed (continuing): {e}", file=sys.stderr)

    try:
        trf = nlp.get_pipe("transformer")
        model = getattr(trf, "model", None)
        shims = getattr(model, "shims", None)
        if not shims or len(shims) == 0:
            print(f"[INFO] Transformer shim not exposed; default_set={default_set}.", file=sys.stderr)
            return

        torch_mod = getattr(shims[0], "model", None) or getattr(shims[0], "_model", None)
        if torch_mod is None:
            print("[INFO] Transformer shim has no torch model attribute; leaving as-is.", file=sys.stderr)
            return

        if torch.cuda.is_available():
            torch_mod.to(torch.device("cuda:0"))
            try:
                p = next(torch_mod.parameters())
                print(f"[INFO] Transformer param device after move: {p.device}", file=sys.stderr)
            except Exception:
                print("[INFO] Transformer moved to CUDA, but could not inspect parameters.", file=sys.stderr)
        else:
            print("[INFO] CUDA not available post warm-up; staying on CPU.", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Could not finalise CUDA placement: {e}", file=sys.stderr)


def load_spacy_model(name: str, posonly: bool):
    """
    Load spaCy model.

    Modes:
    - posonly=True: disable parser and NER (fast POS/lemma/morph). Adds sentenciser if no parser/senter.
    - posonly=False (default): full pipeline as provided by the model (e.g., parser + NER for _trf).

    For transformer pipelines, CUDA warm-up is attempted.
    """
    try:
        import spacy
    except ImportError:
        print("spaCy is not installed. Install with: pip install spacy", file=sys.stderr)
        raise

    disable: List[str] = []
    if posonly:
        disable = ["parser", "ner"]

    try:
        nlp = spacy.load(name, disable=disable)
        print(f"[INFO] Loaded '{name}' with disable={disable}", file=sys.stderr)
    except OSError:
        print(
            f"spaCy model '{name}' not found.\n"
            f"Try: python -m spacy download {name}",
            file=sys.stderr,
        )
        raise

    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    if "transformer" in nlp.pipe_names:
        _activate_cuda_and_warmup_if_transformer(nlp)
        try:
            import torch  # noqa: F401
            trf = nlp.get_pipe("transformer")
            model = getattr(trf, "model", None)
            shims = getattr(model, "shims", None)
            if shims and len(shims) > 0 and hasattr(shims[0], "model"):
                mod = shims[0].model
                p = next(mod.parameters())
                print(f"[INFO] Final transformer param device: {p.device}", file=sys.stderr)
            else:
                print("[INFO] Final transformer device check: shim unavailable (likely fine).", file=sys.stderr)
        except Exception as e:
            print(f"[INFO] Final transformer device check failed: {e}", file=sys.stderr)

    print(f"[INFO] Final pipes: {nlp.pipe_names}", file=sys.stderr)
    return nlp


def to_conllu(doc, doc_id: str) -> str:
    """
    Render a spaCy Doc to a CoNLL-U string with minimal metadata, guaranteeing:
      - one blank line after every sentence; and
      - if the doc has zero sentences, still emit one blank line after the header,
        so consecutive '# newdoc' lines are separated by a blank line.
    """
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

            # Dependency columns
            if has_dep:
                if token.dep_ == "ROOT":
                    head = 0
                    deprel = "root"
                else:
                    if sent.start <= token.head.i < sent.end:
                        head = (token.head.i - sent.start) + 1
                    else:
                        head = 0
                    deprel = token.dep_ if token.dep_ else "_"
            else:
                head = "_"
                deprel = "_"

            deps = "_"

            # MISC: whitespace + optional NER tag
            misc_parts = []
            if token.whitespace_ == "":
                misc_parts.append("SpaceAfter=No")
            if token.ent_iob_ != "O":
                ner_tag = f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else token.ent_iob_
                misc_parts.append(f"NER={ner_tag}")

            misc = "|".join(misc_parts) if misc_parts else "_"

            lines.append(f"{i}\t{form}\t{lemma}\t{upos}\t{xpos}\t{feats}\t{head}\t{deprel}\t{deps}\t{misc}")

        # mandatory blank line after each sentence
        lines.append("")

    # If the doc had zero sentences, the last element is the header;
    # append an empty string so the file ends with a sentence-terminating blank line.
    if lines[-1] != "":
        lines.append("")

    # Join and add one trailing newline => ensures exactly one empty line at end-of-doc.
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
            # Keep empties to preserve alignment with upstream indices; handled in renderer.
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

    # Always create the output file for traceability (even if empty)
    out_f, tmp_path = open_atomic_writer(out_path)

    if not texts:
        out_f.close()
        tmp_path.replace(out_path)
        return

    # Prepare (text, context) tuples for nlp.pipe
    stream = ((t, metas[i]) for i, t in enumerate(texts))

    for doc_obj, ctx in nlp.pipe(stream, as_tuples=True, batch_size=batch_size, n_process=n_process):
        # Build a stable doc_id from metadata
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

    out_f.close()
    tmp_path.replace(out_path)


def main():
    args = parse_args()

    in_root = args.in_root
    out_root = args.out_root

    nlp = load_spacy_model(args.model, posonly=args.posonly)
    print(f"[INFO] Using spaCy model: {args.model} | pipes: {nlp.pipe_names}", file=sys.stderr)
    print(f"[INFO] Reading from: {in_root}", file=sys.stderr)
    print(f"[INFO] Writing to:   {out_root}", file=sys.stderr)
    print(f"[INFO] n_process={args.n_process} batch_size={args.batch_size} posonly={args.posonly}", file=sys.stderr)

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
