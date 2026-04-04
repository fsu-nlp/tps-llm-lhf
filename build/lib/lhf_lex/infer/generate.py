from __future__ import annotations

"""
Generation utilities for continuation-only decoding with HF models.

Guarantees:
- Returns continuation only (prompt stripped) in "text".
- The "prompt" field stores the full realised prompt string.
- Anti-loop defaults applied unless overridden.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple, Union
import hashlib, os, time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, pipeline
from peft import PeftModel

from .prompts import render_prompt, PROMPT_SCHEMAS, SchemaName


@dataclass

@dataclass
@dataclass
class LengthWindowConfig:
    """Fixed word-window policy → derived min/max new tokens."""
    window_words: int = 60          # K
    tokens_per_word: float = 1.33   # f
    min_multiplier: float = 1.5     # ceil(K * f * min_multiplier)
    max_multiplier: float = 2.5     # ceil(K * f * max_multiplier)

    def compute_token_bounds(self) -> tuple[int, int]:
        import math
        min_tokens = int(math.ceil(self.window_words * self.tokens_per_word * self.min_multiplier))
        max_tokens = int(math.ceil(self.window_words * self.tokens_per_word * self.max_multiplier))
        # Safety: ensure min<=max and both >=1
        min_tokens = max(1, min_tokens)
        max_tokens = max(min_tokens, max_tokens)
        return min_tokens, max_tokens

@dataclass
class DecodingConfig:
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    # If None, will be computed from length_window (if enabled).
    min_new_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 6
    include_prompt_in_output: bool = False  # for TGI/vLLM parity
    # Length window policy (config-only). If set to None, no automatic bounds will be applied.
    length_window: LengthWindowConfig | None = field(default_factory=LengthWindowConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "include_prompt_in_output": self.include_prompt_in_output,
        }

    
    def gen_kwargs(self) -> Dict[str, Any]:
        """Only include sampling-only args when sampling is enabled.
        Also apply config-only length window to set min/max new tokens if not explicitly provided.
        """
        # Determine min/max tokens
        min_nt, max_nt = self.min_new_tokens, self.max_new_tokens
        if self.length_window is not None:
            mn, mx = self.length_window.compute_token_bounds()
            if min_nt is None:
                min_nt = mn
            if max_nt is None:
                max_nt = mx

        kw: Dict[str, Any] = {
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }
        if max_nt is not None:
            kw["max_new_tokens"] = max_nt
        if min_nt is not None:
            kw["min_new_tokens"] = min_nt
        if self.do_sample:
            kw["temperature"] = self.temperature
            kw["top_p"] = self.top_p
        return kw


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _select_dtype(bf16: bool) -> torch.dtype:
    # Prefer bf16 if requested and supported; avoid fp16 on CPU.
    if bf16:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    bf16: bool = False,
    device_map: Union[str, Dict[str, int], None] = "auto",
):
    """
    Load tokenizer + model with modern dtype handling.

    - Tries the new Transformers kwarg `dtype=` first.
    - Falls back to deprecated `torch_dtype=` for older versions.
    - Uses fp32 on CPU to avoid half-precision CPU issues.
    """
    dtype = _select_dtype(bf16)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,              # new API
            device_map=device_map,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,        # legacy API
            device_map=device_map,
        )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tok, model


def build_prompt_and_meta(tok, schema: SchemaName, raw_user_prompt: str) -> Tuple[str, Dict[str, Any]]:
    full_prompt = render_prompt(schema, raw_user_prompt)
    # Token accounting
    sys_tokens = 0
    user_tokens = 0
    if schema == "chat_v1":
        # Rough accounting by separately tokenising system and user segments
        sys_tokens = len(
            tok.encode(
                render_prompt("chat_v1", "").replace("<|user|>\n\n<|assistant|>\n", "")
            )
        )  # system+tags baseline
        user_tokens = len(tok.encode(raw_user_prompt))
    elif schema == "none":
        user_tokens = len(tok.encode(raw_user_prompt))
    meta = {
        "prompt_schema": schema,
        "template_hash": _sha256(full_prompt.replace(raw_user_prompt, "")),
        "n_sys_tokens": int(sys_tokens),
        "n_user_tokens": int(user_tokens),
    }
    return full_prompt, meta


def generate_hf_pipeline(tok, model, prompts: List[str], dec: DecodingConfig, seed: int) -> List[str]:
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        return_full_text=False,  # critical: continuation-only
        device=0 if torch.cuda.is_available() else -1,
    )
    torch.manual_seed(seed)
    out = generator(prompts, **dec.gen_kwargs())
    # pipeline returns list[dict] or list[list[dict]] depending on batching; normalise
    texts: List[str] = []
    for item in out:
        if isinstance(item, list):
            item = item[0]
        texts.append(item["generated_text"])
    return texts


def _prune_generate_inputs(enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Keep only fields accepted by most causal LMs' generate(): input_ids (+ optional attention_mask).
    Drop BERT-line artefacts (token_type_ids) and other tokenizer extras.
    """
    enc = dict(enc)  # shallow copy
    enc.pop("token_type_ids", None)
    enc.pop("special_tokens_mask", None)
    enc.pop("offset_mapping", None)
    out: Dict[str, torch.Tensor] = {"input_ids": enc["input_ids"]}
    if "attention_mask" in enc:
        out["attention_mask"] = enc["attention_mask"]
    return out


def generate_hf_generate(tok, model, prompts: List[str], dec: DecodingConfig, seed: int) -> List[str]:
    torch.manual_seed(seed)
    device = model.device
    texts: List[str] = []
    for p in prompts:
        enc = tok(p, return_tensors="pt")              # let tokenizer add specials
        inputs = _prune_generate_inputs(enc)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]
        gen = model.generate(
            **inputs,
            **dec.gen_kwargs(),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        cont_ids = gen[:, input_len:]
        text = tok.batch_decode(cont_ids, skip_special_tokens=True)[0]
        texts.append(text)
    return texts


def generate_records(
    model_id: str,
    adapter_path: Optional[str],
    raw_prompts: List[str],
    schema: SchemaName,
    decoding: DecodingConfig,
    seed: int,
    backend: str = "hf-generate",  # or "hf-pipeline"
    stage: Optional[str] = None,
    bf16: bool = False,
    dose: Optional[float] = None,
) -> List[Dict[str, Any]]:
    tok, model = load_model(model_id, adapter_path, bf16=bf16)
    realised_prompts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for rp in raw_prompts:
        pr, meta = build_prompt_and_meta(tok, schema, rp)
        realised_prompts.append(pr)
        metas.append(meta)

    if backend == "hf-pipeline":
        texts = generate_hf_pipeline(tok, model, realised_prompts, decoding, seed)
    else:
        texts = generate_hf_generate(tok, model, realised_prompts, decoding, seed)

    # Build records
    records: List[Dict[str, Any]] = []
    dec_hash = _sha256(json_dumps_sorted(decoding.to_dict()))
    for i, (rp, m, raw, text) in enumerate(zip(realised_prompts, metas, raw_prompts, texts)):
        rec = {
            "idx": i,
            "prompt": rp,
            "text": text,  # continuation only
            "prompt_meta": m,
            "model_name": model_id,
            "adapter_path": adapter_path,
            "decoding": decoding.to_dict(),
            "prompt_id": i,
            "prompt_sha": _sha256(raw),
            "model_id": model_id,
            "stage": stage,
            "dose": dose,
            "seed": seed,
            "decode_hash": dec_hash,
        }
        records.append(rec)
    return records


def json_dumps_sorted(obj: Dict[str, Any]) -> str:
    import json
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
