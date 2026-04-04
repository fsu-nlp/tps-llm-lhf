#!/usr/bin/env python
from __future__ import annotations

"""
Quick sanity check for Base vs SFT vs DPO adapters: load models, generate a
single continuation for a fixed prompt, and report occurrences-per-million
(OPM) for a given vocabulary alongside the raw texts.
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from lhf_lex.metrics.freq import normalised_frequency


def gen(model, tok, prompt: str, max_new_tokens: int = 64) -> str:
    """Generate a continuation for `prompt` with `model`.

    Args:
        model: Causal LM (optionally with a PEFT adapter loaded).
        tok: Matching tokenizer.
        prompt: Input string to complete.
        max_new_tokens: Maximum number of new tokens to sample.

    Returns:
        The decoded continuation (without special tokens).
    """
    import torch
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)


def _load_base(model_id: str):
    """Load a base CausalLM using the new `dtype` kwarg, with a fallback for older HF versions."""
    kwargs = {"device_map": "auto"}
    try:
        # Newer transformers prefer `dtype`
        return AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", **kwargs)
    except TypeError:
        # Older transformers still use `torch_dtype`
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", **kwargs)


def main() -> None:
    """CLI: load Base/SFT/DPO, generate one prompt, print texts and OPM.

    Loads the base model and two LoRA adapters (SFT, DPO), runs one generation
    per variant for `--prompt`, computes OPM for `--vocab`, and prints results.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--dpo", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--prompt", default="Explain Learning from Human Feedback in one sentence.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)

    base = _load_base(args.base)
    sft = PeftModel.from_pretrained(_load_base(args.base), args.sft)
    dpo = PeftModel.from_pretrained(_load_base(args.base), args.dpo)

    texts = {}
    for name, model in [("base", base), ("sft", sft), ("dpo", dpo)]:
        texts[name] = gen(model, tok, args.prompt, max_new_tokens=64)

    vocab = [ln.strip() for ln in open(args.vocab, "r", encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    from lhf_lex.text.normalise import tokenize  # noqa: F401 (kept for parity with project imports)

    for name, text in texts.items():
        opm = normalised_frequency([text], vocab)
        print(f"== {name.upper()} ==")
        print(text)
        print("OPM:", {k: round(v, 2) for k, v in opm.items()})
        print()


if __name__ == "__main__":
    main()
