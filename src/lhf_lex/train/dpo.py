from __future__ import annotations
import os, json, time, inspect, re
from typing import Optional

from trl import DPOTrainer
try:
    from trl import DPOConfig  # newer TRL
    HAS_DPO_CONFIG = True
except Exception:
    HAS_DPO_CONFIG = False

from transformers import TrainingArguments

from lhf_lex.train.utils import load_lora_model, ModelSpec
from lhf_lex.train.datasets import load_pref_dataset


def _filter_kwargs(callable_or_cls, kwargs: dict) -> dict:
    """
    Keep only kwargs accepted by the target's __init__ (or by the function itself).
    Robust against TRL version drift.
    """
    try:
        sig = inspect.signature(callable_or_cls.__init__)  # classes
    except (TypeError, ValueError, AttributeError):
        sig = inspect.signature(callable_or_cls)  # functions/callables
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


class DPOTrainingArguments(TrainingArguments):
    """
    Fallback for older TRL (no DPOConfig): HF TrainingArguments + TRL-compat fields.
    """
    def __init__(
        self,
        *args,
        padding_value: int | None = 0,
        label_pad_token_id: int = -100,
        truncation_mode: str = "keep_end",
        **kwargs,
    ):
        # Pairwise datasets: avoid column pruning
        kwargs.setdefault("remove_unused_columns", False)
        super().__init__(*args, **kwargs)

        # Commonly inspected attributes
        self.padding_value = padding_value
        self.label_pad_token_id = label_pad_token_id
        self.truncation_mode = truncation_mode

        # Fields various TRL builds probe
        self.model_init_kwargs = None
        self.ref_model_init_kwargs = None
        self.tokenizer_init_kwargs = None
        self.ref_tokenizer_init_kwargs = None
        self.dataset_num_proc = None

        # Eval/logging toggles (some builds)
        self.generate_during_eval = False
        self.predict_with_generate = False
        self.include_inputs_for_metrics = False
        self.evaluation_strategy = "no"
        self.eval_strategy = "no"
        self.do_eval = False

        # Adapter/ref toggles seen in some builds
        self.model_adapter_name = None
        self.ref_adapter_name = None
        self.reference_free = False
        self.disable_dropout = False
        self.use_liger_loss = False


def train_dpo(
    data_path: str,
    output_dir: str,
    model_name: str,
    beta: float = 0.1,
    max_steps: int = 1000,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bf16: bool = True,
    *,
    adapter_path: Optional[str] = None,
) -> str:
    """
    DPO with LoRA adapters.
    Prefers TRL DPOConfig if present (kwargs filtered to match your version);
    otherwise uses a compat TrainingArguments subclass and heals AttributeErrors.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Data
    ds = load_pref_dataset(data_path)
    print(f"[DPO] Loaded dataset: {len(ds)} preference pairs from {data_path}")
    with open(os.path.join(output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"data_path": data_path, "n_pairs": len(ds), "beta": beta, "time": time.time()},
            f,
            indent=2,
        )

    # ---- Model (+ optional SFT adapters)
    tokenizer, model = load_lora_model(
        ModelSpec(
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bf16=bf16,
        ),
        adapter_path=adapter_path,
    )

    # ---- Args / Config
    if HAS_DPO_CONFIG:
        # Prepare a superset; filter to your TRL's accepted fields
        cfg = dict(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            bf16=bf16,
            optim="adamw_torch",
            report_to=[],
            disable_tqdm=False,
            remove_unused_columns=False,

            # Pairwise / padding controls
            padding_value=0,
            label_pad_token_id=-100,
            truncation_mode="keep_end",
            dataset_num_proc=None,

            # Adapter/refs
            model_adapter_name=None,
            ref_adapter_name=None,
            reference_free=False,
            disable_dropout=False,
            use_liger_loss=False,

            # Eval/logging toggles (filter will drop any unsupported)
            generate_during_eval=False,
            predict_with_generate=False,
            include_inputs_for_metrics=False,
            evaluation_strategy="no",
            do_eval=False,
        )
        args = DPOConfig(**_filter_kwargs(DPOConfig, cfg))  # type: ignore[arg-type]
    else:
        args = DPOTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            bf16=bf16,
            optim="adamw_torch",
            report_to=[],
            disable_tqdm=False,
            padding_value=0,
            label_pad_token_id=-100,
            truncation_mode="keep_end",
            remove_unused_columns=False,
        )

    # ---- Trainer
    # Build a superset of kwargs, then filter to what your TRL's DPOTrainer accepts.
    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        beta=beta,
    )
    filtered = _filter_kwargs(DPOTrainer, trainer_kwargs)

    # Older TRL builds may still access missing attributes on args; try to heal a few rounds.
    tries, last_err = 0, None
    while True:
        try:
            trainer = DPOTrainer(**filtered)  # type: ignore[arg-type]
            break
        except AttributeError as e:
            last_err = e
            # Catch "'...TrainingArguments' object has no attribute 'foo'"
            m = re.search(r"has no attribute '([^']+)'", str(e))
            if not m or HAS_DPO_CONFIG:
                # If using DPOConfig, AttributeErrors are unlikely to be fixable here.
                raise
            missing = m.group(1)
            # Default: None for *_kwargs/names; False for toggles; 0 for padding-like.
            if missing.endswith("_kwargs") or missing.endswith("_name"):
                setattr(args, missing, None)
            elif "padding" in missing:
                setattr(args, missing, 0)
            else:
                setattr(args, missing, False)
            print(f"[DPO] Added missing args.{missing} and retrying …")
            tries += 1
            if tries > 8:
                raise last_err

    # Some builds need beta set after init
    try:
        trainer.beta = beta  # type: ignore[attr-defined]
    except Exception:
        pass

    print("[DPO] Starting training …")
    try:
        trainer.train()
        print("[DPO] Training finished.")
    finally:
        print("[DPO] Saving adapters …")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[DPO] Saved to: {output_dir}")

    return output_dir
