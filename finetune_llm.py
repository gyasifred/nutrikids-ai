#!/usr/bin/env python3
"""
Fine‑tune a Llama‑3‑8B‑Instruct model (via Unsloth) to detect malnutrition in
clinical notes with LoRA + SFT.

Key fixes vs. the original script
---------------------------------
* Keep the raw **text** column so `trl.SFTTrainer` can build its own batches, or
  explicitly pass `dataset_text_field=None` when that column is removed.
* Provide an option to **mask the prompt** tokens so that only the answer
  segment is optimised (set `--mask_prompt_tokens`).
* Use the model’s *advertised* maximum context length and fall back to the
  90‑th percentile of the training set if a lower sequence length is needed.
* Wire `gradient_accumulation_steps` directly into the `TrainingArguments` that
  SFTTrainer consumes.

Run
---
python malnutrition_finetune_fixed.py \
  --train_data train.csv \
  --val_data val.csv \
  --examples_data few_shots.csv \
  --output_dir ./runs/llm \
  --model_output ./artifacts/llm_adapter
"""
import argparse
import datetime
import glob
import json
import os
import shutil
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import BitsAndBytesConfig, TrainerCallback, TrainerControl, TrainerState
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

# ---------------------------------------------------------------------------
# Domain‑specific helpers -----------------------------------------------------
# ---------------------------------------------------------------------------
from models.llm_models import (
    MalnutritionDataset,
    MalnutritionPromptBuilder,
    evaluate_predictions,  # noqa: F401 – used downstream if you extend the script
    is_bfloat16_supported,
    plot_evaluation_metrics,  # noqa: F401
    print_metrics_report,  # noqa: F401
    save_metrics_to_csv,  # noqa: F401
    set_seed,
)

# ---------------------------------------------------------------------------
# Early stopping --------------------------------------------------------------
# ---------------------------------------------------------------------------
class EarlyStoppingCallback(TrainerCallback):
    """Stop training when eval loss stops improving."""

    def __init__(self, patience: int = 5, threshold: float = 0.005):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = np.inf
        self.no_improvement = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # noqa: D401,E501
        # Extract the most recent eval loss
        for item in reversed(state.log_history):
            if "eval_loss" in item:
                current_loss = item["eval_loss"]
                break
        else:
            return control  # no eval loss yet

        if self.best_loss - current_loss > self.threshold:
            self.best_loss = current_loss
            self.no_improvement = 0
        else:
            self.no_improvement += 1
        print(
            f"[early‑stop] {self.no_improvement}/{self.patience} | "
            f"best={self.best_loss:.4f} current={current_loss:.4f}"
        )
        if self.no_improvement >= self.patience:
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# CLI ------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Fine‑tune LLaMA‑3 for malnutrition detection")

    # Data -------------------------------------------------------------------
    p.add_argument("--train_data", required=True)
    p.add_argument("--val_data")
    p.add_argument("--examples_data")
    p.add_argument("--text_column", default="txt")
    p.add_argument("--label_column", default="Label")

    # Model / output ---------------------------------------------------------
    p.add_argument(
        "--model_name",
        default="unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit",
    )
    p.add_argument("--output_dir", default="./llm_runs")
    p.add_argument("--model_output", default="./llm_adapter")

    # Train hyper‑params -----------------------------------------------------
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # LoRA -------------------------------------------------------------------
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules")

    # Precision / quantisation ---------------------------------------------
    g = p.add_mutually_exclusive_group()
    g.add_argument("--load_in_4bit", action="store_true", default=True)
    g.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--force_fp16", action="store_true")
    p.add_argument("--force_bf16", action="store_true")

    # Misc ------------------------------------------------------------------
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", choices=["none", "tensorboard", "wandb"], default="none")
    p.add_argument("--use_flash_attention", action="store_true")
    p.add_argument("--max_seq_length", type=int)  # optional manual cap
    p.add_argument("--mask_prompt_tokens", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quant config ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def quant_config(args) -> Optional[BitsAndBytesConfig]:
    if args.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    if args.load_in_4bit:
        dtype = torch.bfloat16 if (args.force_bf16 or is_bfloat16_supported()) else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return None


# ---------------------------------------------------------------------------
# Utilities ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def resolve_precision(args) -> Tuple[bool, bool]:
    if args.force_fp16:
        return True, False
    if args.force_bf16 and is_bfloat16_supported():
        return False, True
    # auto
    return (False, True) if is_bfloat16_supported() else (True, False)


def target_modules_for(model_name: str, user_override: Optional[str]) -> list[str]:
    if user_override:
        return user_override.split(",")
    name = model_name.lower()
    if any(t in name for t in ("llama", "mistral", "mixtral", "deepseek")):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    if "phi" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "qwen" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


# ---------------------------------------------------------------------------
# Data prep ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_tokeniser_fn(tokenizer, mask_prompt: bool, prompt_builder: MalnutritionPromptBuilder):
    prompt_prefix_len_cache: dict[str, int] = {}

    def _tok(example):
        text = example["text"]
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=_tok.max_length,
        )
        if mask_prompt:
            # Compute number of tokens up to (and incl.) the answer tag once per prompt pattern.
            if text not in prompt_prefix_len_cache:
                prompt_prefix = prompt_builder.extract_prompt_part(text)
                prompt_prefix_len_cache[text] = len(tokenizer(prompt_prefix).input_ids)
            n_prefix = prompt_prefix_len_cache[text]
            labels = enc["input_ids"].copy()
            labels[:n_prefix] = [-100] * n_prefix
            enc["labels"] = labels
        else:
            enc["labels"] = enc["input_ids"].copy()
        return enc

    _tok.max_length = 0  # will be set later
    return _tok


def tokenise_dataframe(df: pd.DataFrame, tok_fn, max_len: int) -> Dataset:
    tok_fn.max_length = max_len
    ds = Dataset.from_pandas(df)
    return ds.map(tok_fn, batched=True, remove_columns=[])  # keep "text"


# ---------------------------------------------------------------------------
# Trainer factory ------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_trainer(model, tokenizer, train_ds, eval_ds, sft_cfg) -> SFTTrainer:
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field=None,  # we pass ready‑to‑use tensors
        args=sft_cfg,
    )


# ---------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_output, exist_ok=True)

    # Save run config --------------------------------------------------------
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Precision & quant ------------------------------------------------------
    q_cfg = quant_config(args)
    fp16, bf16 = resolve_precision(args)
    dtype = torch.bfloat16 if bf16 else torch.float16

    attn_impl = "flash_attention_2" if args.use_flash_attention else "eager"

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        dtype=dtype,
        device_map="auto",
        attn_implementation=attn_impl,
        quantization_config=q_cfg,
    )
    model_max_ctx = base_model.config.max_position_embeddings

    # Sequence length to *process* notes ------------------------------------
    prompt_builder = MalnutritionPromptBuilder(args.examples_data)

    def calc_needed_len(path: str) -> int:
        ds = MalnutritionDataset(path, args.text_column, args.label_column)
        lengths = [len(tokenizer.encode(r["text"])) for r in ds.prepare_training_data(prompt_builder)]
        p90 = sorted(lengths)[int(0.9 * len(lengths))]
        return min(model_max_ctx, int(p90 * 1.1))

    seq_len = (
        min(args.max_seq_length, model_max_ctx)
        if args.max_seq_length
        else calc_needed_len(args.train_data)
    )

    # Data ------------------------------------------------------------------
    tok_fn = build_tokeniser_fn(tokenizer, args.mask_prompt_tokens, prompt_builder)

    train_df = MalnutritionDataset(args.train_data, args.text_column, args.label_column).prepare_training_data(
        prompt_builder
    )
    train_ds = tokenise_dataframe(pd.DataFrame(train_df), tok_fn, seq_len)

    eval_ds = None
    if args.val_data:
        val_df = MalnutritionDataset(args.val_data, args.text_column, args.label_column).prepare_training_data(
            prompt_builder
        )
        eval_ds = tokenise_dataframe(pd.DataFrame(val_df), tok_fn, seq_len)

    # LoRA ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model=base_model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules_for(args.model_name, args.target_modules),
        use_gradient_checkpointing=True,
        random_state=args.seed,
        use_rslora=True,
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # SFT config ------------------------------------------------------------
    sft_cfg = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        save_strategy="steps",
        save_steps=10,
        max_seq_length=model_max_ctx,  # how long *model* can see
        num_train_epochs=args.epochs,
        evaluation_strategy="steps" if eval_ds else "no",
        eval_steps=10 if eval_ds else None,
        load_best_model_at_end=bool(eval_ds),
        metric_for_best_model="eval_loss" if eval_ds else None,
        greater_is_better=False,
    )

    # Trainer ---------------------------------------------------------------
    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, sft_cfg)

    if eval_ds is not None:
        trainer.add_callback(EarlyStoppingCallback(patience=5, threshold=0.005))

    print(
        f"[train] {len(train_ds)} examples | seq_len={seq_len} | epochs={args.epochs} "
        f"| grad_acc={args.gradient_accumulation_steps}"
    )
    trainer.train()

    # Save adapter & tokenizer ---------------------------------------------
    model.save_pretrained(args.model_output)
    tokenizer.save_pretrained(args.model_output)

    # Clean checkpoints -----------------------------------------------------
    for ckpt in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        shutil.rmtree(ckpt, ignore_errors=True)

    # README ---------------------------------------------------------------
    with open(os.path.join(args.model_output, "README.md"), "w") as f:
        f.write(
            f"""# Malnutrition LoRA Adapter\n\n"
            f"Base model     : {args.model_name}\n"
            f"Trained on     : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"Epochs         : {args.epochs}\n"
            f"Batch size     : {args.batch_size}\n"
            f"LR             : {args.learning_rate}\n"
            f"LoRA (r/α/↓)   : {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}\n"
            f"Sequence length: {seq_len}\n"
            f"Gradient accum.: {args.gradient_accumulation_steps}\n"""
        )

    print(f"[done] adapter + tokenizer saved to → {args.model_output}")


if __name__ == "__main__":
    main()
