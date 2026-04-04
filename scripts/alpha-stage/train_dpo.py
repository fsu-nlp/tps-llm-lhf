#!/usr/bin/env python
from __future__ import annotations

"""
Train a LoRA DPO adapter on a ufb-lex:1.0 preference JSONL using TRL-style
Direct Preference Optimisation. Optionally continue from an SFT LoRA adapter.
Writes the trained adapter to --out.
"""

import argparse

from lhf_lex.train.dpo import train_dpo


def main() -> None:
    """CLI: configure and launch DPO training.

    Parses paths and hyperparameters, then calls `train_dpo` with optional
    SFT initialisation (`--adapter`). Outputs a LoRA adapter to `--out`.
    """
    ap = argparse.ArgumentParser(description="LoRA DPO training scaffold (TRL 0.21.0).")
    ap.add_argument("data", help="Path to ufb-lex:1.0 preference JSONL")
    ap.add_argument("--model", required=True, help="Base HF model id/path (e.g., allenai/OLMo-2-0425-1B)")
    ap.add_argument("--adapter", default=None, help="Path to SFT LoRA adapter dir to continue from")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    train_dpo(
        data_path=args.data,
        output_dir=args.out,
        model_name=args.model,
        beta=args.beta,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bf16=args.bf16,
        adapter_path=args.adapter,
    )


if __name__ == "__main__":
    main()
