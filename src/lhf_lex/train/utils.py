from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Dict
import os
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class ModelSpec:
    """
    Minimal spec for loading a causal LM with (or without) LoRA adapters.
    """
    model_name: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # If None, we fall back to common proj names; override for non-standard blocks.
    target_modules: Optional[list[str]] = None
    # Set True on bf16-capable GPUs (A100/H100/L4 etc.); ignored otherwise.
    bf16: bool = True
    # Where to place weights; "auto" is fine for single/multi-GPU.
    device_map: Union[str, Dict[str, int], None] = "auto"


def load_tokenizer(model_name: str):
    """
    Prefer the fast tokenizer; if conversion deps (e.g. tiktoken/protobuf) are missing,
    fall back to the slow tokenizer automatically. Ensure PAD token exists (pad→eos).
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    except Exception as e:
        warnings.warn(
            f"Falling back to slow tokenizer for {model_name} ({e}). "
            "Consider `pip install tiktoken protobuf sentencepiece`."
        )
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _default_targets_if_needed(model, targets: Optional[list[str]]) -> list[str] | None:
    """
    If the caller did not specify target modules, return a conservative default that
    works for many GPT-style blocks (OLMo/LLaMA-like): q/k/v/o projections.
    For exotic blocks, pass an explicit list in ModelSpec.target_modules or let PEFT infer.
    """
    if targets is not None:
        return targets
    # Safe default for common attention MLP stacks:
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _select_dtype(bf16: bool) -> torch.dtype:
    """
    Prefer bf16 when requested and CUDA is available; otherwise fp16 on CUDA, fp32 on CPU.
    """
    if torch.cuda.is_available():
        return torch.bfloat16 if bf16 else torch.float16
    return torch.float32


def load_lora_model(
    spec: ModelSpec,
    adapter_path: str | None = None,
):
    """
    Load base model + tokenizer, then either:
      (a) attach fresh LoRA adapters (new SFT), or
      (b) load an existing LoRA adapter from `adapter_path` (continued training, e.g. DPO).

    Returns: (tokenizer, model)
    """
    tok = load_tokenizer(spec.model_name)

    dtype = _select_dtype(spec.bf16)

    # Try new Transformers API (`dtype=`). Fall back to deprecated `torch_dtype=` if needed.
    try:
        base = AutoModelForCausalLM.from_pretrained(
            spec.model_name,
            dtype=dtype,                 # new API
            device_map=spec.device_map,
            trust_remote_code=True,
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            spec.model_name,
            torch_dtype=dtype,           # legacy API
            device_map=spec.device_map,
            trust_remote_code=True,
        )

    if adapter_path and os.path.isdir(adapter_path):
        # Continue from prior SFT LoRA checkpoint
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        return tok, model

    # Fresh LoRA setup
    targets = _default_targets_if_needed(base, spec.target_modules)
    peft_cfg = LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, peft_cfg)
    return tok, model


def has_chat_template(tokenizer) -> bool:
    tmpl = getattr(tokenizer, "chat_template", None)
    return isinstance(tmpl, str) and len(tmpl) > 0
