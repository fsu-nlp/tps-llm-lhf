#!/usr/bin/env python
from __future__ import annotations

"""
Generate continuations with continuation-only output, streaming to a single JSONL.

- Always resumes if the output JSONL already exists (skips completed idx).
- Writes in fixed chunks of 1000 prompts.
- After each chunk: flush() + os.fsync() to reduce data-loss risk.
"""

import argparse, json, os, hashlib
from pathlib import Path
from typing import List, Dict, Any, Set

from lhf_lex.infer.generate import DecodingConfig, generate_records
from lhf_lex.infer.prompts import PROMPT_SCHEMAS

# Hard-coded chunk size as requested
CHUNK_SIZE = 1000


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_prompts(path: str, limit: int | None) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s:
                lines.append(s)
            if limit and len(lines) >= limit:
                break
    return lines


def ensure_trailing_newline(path: Path) -> None:
    """If file exists and does not end with newline, append one."""
    if not path.exists():
        return
    try:
        if path.stat().st_size == 0:
            return
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with open(path, "a", encoding="utf-8") as out:
                out.write("\n")
    except Exception:
        # Best-effort; ignore if file is too small or on other IO issues
        pass


def main():
    ap = argparse.ArgumentParser(description="Continuation-only text generation (streamed JSONL with resume).")
    ap.add_argument("input", help="Path to prompts .txt (one per line)")
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter path")
    ap.add_argument("--out", required=True, help="Output JSONL path")

    # Decoding
    ap.add_argument("--backend", choices=["hf-generate", "hf-pipeline"], default="hf-generate")
    ap.add_argument("--bf16", action="store_true", help="Prefer bfloat16 on supported hardware")
    ap.add_argument("--greedy", action="store_true", help="Compatibility alias for --no-sample (forces greedy)")
    ap.add_argument("--model-type", choices=["base", "instruct"], default="base",
                    help="base: raw first half (no tags); instruct: add <|system|>/<|user|>/<|assistant|> wrapper.")
    ap.add_argument("--do-sample", action="store_true", help="Enable sampling (default True)")
    ap.add_argument("--no-sample", action="store_true", help="Disable sampling (greedy)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=6)

    # Prompting
    ap.add_argument("--prompt-schema", choices=list(PROMPT_SCHEMAS.keys()), default=None,
                    help="Use 'chat_v1' to embed system/user/assistant tags; 'none' to pass raw text.")

    # Run control
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", choices=["base", "sft", "dpo"], default=None)
    ap.add_argument("--dose", type=float, default=None)
    ap.add_argument("--run-id", default=None)

    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(args.input, args.limit)

    # Resume support: collect completed idx from existing global JSONL (if any)
    completed_ids: Set[int] = set()
    if outp.exists():
        with open(outp, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    j = json.loads(ln)
                    if "idx" in j:
                        completed_ids.add(int(j["idx"]))
                except Exception:
                    # ignore malformed/blank lines (e.g., truncated last line)
                    pass

    # Decide which prompts to run (skip already present indices)
    to_run = [(i, s) for i, s in enumerate(prompts) if i not in completed_ids]
    if not to_run:
        print(f"All prompts already present in {outp}; nothing to do.")
        return

    # Sampling flags
    do_sample_flag = True
    if args.no_sample or args.greedy:
        do_sample_flag = False
    if args.do_sample:
        do_sample_flag = True

    dec = DecodingConfig(
        do_sample=do_sample_flag,
        min_new_tokens=None,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        include_prompt_in_output=False,
    )

    # Resolve schema: explicit --prompt-schema wins; else from --model-type
    if args.prompt_schema is not None:
        resolved_schema = args.prompt_schema
    else:
        resolved_schema = 'none' if args.model_type == 'base' else 'chat_v1'

    # Prepare output for appends (and ensure a clean newline boundary)
    ensure_trailing_newline(outp)

    total = len(to_run)
    processed = 0

    # Stream in fixed chunks of 1000
    for start in range(0, total, CHUNK_SIZE):
        batch_pairs = to_run[start:start + CHUNK_SIZE]
        raw_batch = [s for _, s in batch_pairs]

        # Generate for this chunk
        records = generate_records(
            model_id=args.model,
            adapter_path=args.adapter,
            raw_prompts=raw_batch,
            schema=resolved_schema,
            decoding=dec,
            seed=args.seed,
            backend=args.backend,
            stage=args.stage,
            dose=args.dose,
            bf16=args.bf16,
        )

        # Append chunk and sync
        with open(outp, "a", encoding="utf-8") as out:
            for (i, _), rec in zip(batch_pairs, records):
                rec["idx"] = i
                rec["model_type"] = args.model_type
                out.write(json.dumps(rec, ensure_ascii=False))
                out.write("\n")
            out.flush()
            os.fsync(out.fileno())

        processed += len(records)
        print(f"[{processed}/{total}] appended to {outp}")

    print(f"Done. Wrote {processed} records to {outp}.")


if __name__ == "__main__":
    main()
