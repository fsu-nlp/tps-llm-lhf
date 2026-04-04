# src/lhf_lex/train/sft.py
# TRL ≥ 0.21 compatible SFT with LoRA.
# - We pre-format to a single 'text' column and hand it to SFTTrainer.
# - No dataset_text_field / tokenizer / max_seq_length kwargs in the constructor.
# - LoRA passed via peft_config; target modules normalised.

from __future__ import annotations
from typing import Dict, Any, List

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from lhf_lex.train.datasets import load_sft_dataset


def _normalise_targets(target_modules: List[str] | str | None) -> List[str]:
    """Return a clean list of module names for LoRA to target."""
    if isinstance(target_modules, str):
        items = [m.strip() for m in target_modules.split(",") if m.strip()]
    elif isinstance(target_modules, list):
        items = target_modules
    else:
        items = []
    # Safe default for many decoder-only LMs incl. HF-style OLMo:
    return items or ["q_proj", "k_proj", "v_proj", "o_proj"]


def _format_one(tokenizer, ex: Dict[str, Any]) -> str:
    """
    Build a single training string from one example.

    Preference order:
      1) If example has chat `messages` and tokenizer has `chat_template`, use it.
      2) If `messages` exist but no template, use deterministic "ROLE: content" lines.
      3) If plain fields exist, prefer {prompt, completion} else {prompt, response}.
    """
    if "messages" in ex and ex["messages"]:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        parts = [f"{m.get('role','').upper()}: {m.get('content','')}" for m in ex["messages"]]
        return "\n".join(parts)

    if "prompt" in ex and "completion" in ex:
        return f"USER: {ex['prompt']}\nASSISTANT: {ex['completion']}"
    if "prompt" in ex and "response" in ex:
        return f"USER: {ex['prompt']}\nASSISTANT: {ex['response']}"

    raise ValueError("example lacks messages or prompt/(completion|response)")


def train_sft(
    data_path: str,
    output_dir: str,
    model_name: str,
    max_steps: int = 1000,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bf16: bool = True,
    target_modules: List[str] | str | None = None,
):
    """
    Instruction tuning with TRL 0.21+.

    data_path: JSONL with either 'messages' or 'prompt'+['response'|'completion'].
    output_dir: where to save LoRA adapters and config.
    model_name: HF repo id or local path.
    """
    targets = _normalise_targets(target_modules)

    # 1) Load dataset (HF Dataset with dict-like rows)
    ds = load_sft_dataset(data_path)

    # 2) Load tokenizer only for text formatting (trainer loads model/tokenizer internally)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    # 3) Map dataset → single 'text' column (so the trainer can auto-detect)
    def map_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        n = len(next(iter(batch.values())))
        texts: List[str] = []
        keys = list(batch.keys())
        for i in range(n):
            ex = {k: batch[k][i] for k in keys}
            texts.append(_format_one(tok, ex))
        return {"text": texts}

    ds = ds.map(
        map_batch,
        batched=True,
        remove_columns=ds.column_names,
        desc="Formatting training texts",
    )
    # NOTE: In TRL 0.21 the trainer will see the "text" column and tokenise it.
    # Supported dataset formats are documented; no dataset_text_field kw exists. :contentReference[oaicite:2]{index=2}

    # 4) Trainer config (SFTConfig wraps Transformers' TrainingArguments)
    cfg = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        logging_steps=10,
        save_steps=200,
        bf16=bf16,
        report_to="none",
        # If your build supports it and you want explicit truncation:
        # max_length=2048,
        # packing=False,
    )

    # 5) LoRA adapter configuration
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )

    # 6) Construct trainer: no dataset_text_field / tokenizer kwargs in 0.21
    trainer = SFTTrainer(
        model=model_name,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
