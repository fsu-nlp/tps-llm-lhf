#!/usr/bin/env python
from __future__ import annotations

"""
Train a LoRA SFT adapter on a supervised fine-tuning dataset and save it to
`--out`. Supports bf16 and a comma-separated `--target-modules` list.
"""

import argparse

from lhf_lex.train.sft import train_sft


def main() -> None:
    """CLI: configure and launch SFT training.

    Parses paths and hyperparameters, then calls `train_sft`. The resulting
    LoRA adapter is written to `--out`.
    """
    ap = argparse.ArgumentParser(description="LoRA SFT training scaffold")
    ap.add_argument("data", help="Path to SFT dataset (format as expected by train_sft)")
    ap.add_argument("--model", required=True, help="HF id (e.g., ai2-llm/OLMo-1B) or local path")
    ap.add_argument("--out", required=True, help="Output directory for the LoRA adapter")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names to target with LoRA",
    )
    args = ap.parse_args()

    train_sft(
        args.data,
        args.out,
        args.model,
        args.max_steps,
        args.per_device_train_batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.bf16,
        args.target_modules,
    )


if __name__ == "__main__":
    main()
