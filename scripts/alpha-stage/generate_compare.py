#!/usr/bin/env python
from __future__ import annotations

"""
Generate texts from Base, SFT, and DPO variants with a shared decoding
configuration, then compute occurrences-per-million (OPM) for a vocabulary.
Writes a combined generations JSONL and a TSV with {variant, word, opm}.
"""

import argparse
import json
from pathlib import Path

from lhf_lex.infer.generate import generate_texts, DecodingConfig
from lhf_lex.metrics.freq import normalised_frequency


def read_prompts_txt(path: str, limit: int | None = None) -> list[str]:
    """Read prompts from a UTF-8 .txt (one per line).

    Args:
        path: Path to a text file containing one prompt per line.
        limit: Optional maximum number of prompts to return.

    Returns:
        List of non-empty prompt strings (stripped).
    """
    out = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    return out[:limit] if limit else out


def read_vocab(path: str) -> list[str]:
    """Read a lowercase vocabulary (one item per line).

    Lines starting with '#' are ignored.

    Args:
        path: Path to a UTF-8 text file with one token per line.

    Returns:
        List of vocab items (lowercased).
    """
    return [
        ln.strip().lower()
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def main() -> None:
    """CLI: generate Base/SFT/DPO outputs and compute OPM.

    Parses arguments, runs generation with a DecodingConfig, and writes:
      • --out-jsonl: all generations (one record per prompt×variant)
      • --out-opm: OPM table with columns {variant, word, opm}
    """
    ap = argparse.ArgumentParser(description="Generate from Base vs SFT vs DPO and compute placeholder OPM.")
    ap.add_argument("prompts", help="Prompts .txt (one per line)")
    ap.add_argument("--model", required=True, help="Base HF model id (e.g., allenai/OLMo-2-0425-1B)")
    ap.add_argument("--sft", required=True, help="Path to SFT LoRA adapter dir")
    ap.add_argument("--dpo", required=True, help="Path to DPO LoRA adapter dir")
    ap.add_argument("--vocab", required=True, help="Vocabulary file for OPM")
    ap.add_argument("--out-jsonl", required=True, help="Combined generations JSONL")
    ap.add_argument("--out-opm", required=True, help="OPM TSV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    prompts = read_prompts_txt(args.prompts, args.limit)
    dec = DecodingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        do_sample=(not args.greedy),
        seed=args.seed,
    )

    all_rows = []
    for variant, adapter in [("base", None), ("sft", args.sft), ("dpo", args.dpo)]:
        recs = generate_texts(
            prompts,
            model_name=args.model,
            adapter_path=adapter,
            decoding=dec,
            bf16=args.bf16,
        )
        for r in recs:
            r["variant"] = variant
        all_rows.extend(recs)

    out_j = Path(args.out_jsonl)
    out_j.parent.mkdir(parents=True, exist_ok=True)
    with out_j.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # OPM per variant
    vocab = read_vocab(args.vocab)
    import pandas as pd  # local import to keep CLI startup light

    rows = []
    for variant in ["base", "sft", "dpo"]:
        texts = [r["text"] for r in all_rows if r["variant"] == variant]
        opm = normalised_frequency(texts, vocab)
        for w, val in opm.items():
            rows.append({"variant": variant, "word": w, "opm": val})
    df = pd.DataFrame(rows).sort_values(["word", "variant"])
    out_tsv = Path(args.out_opm)
    df.to_csv(out_tsv, sep="\t", index=False)

    print(f"Wrote {len(all_rows)} generations to {out_j}")
    print(f"Wrote OPM table to {out_tsv}")


if __name__ == "__main__":
    main()
