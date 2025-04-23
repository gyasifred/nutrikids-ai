#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune LLaMA-style model using Unsloth + LoRA for malnutrition assessment.
"""

import os
import argparse
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for pediatric malnutrition classification")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B", help="Base model to use")
    parser.add_argument("--output_dir", type=str, default="./malnutrition_model", help="Where to save the model")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--epochs", type=float, default=10, help="Epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--save_steps", type=int, default=50, help="Steps between checkpoints")
    parser.add_argument("--max_steps", type=int, default=None, help="Total steps (overrides epochs)")
    return parser.parse_args()


def create_malnutrition_prompt(note, label=""):
    """Your custom expert-level clinical prompt for fine-tuning."""
    prompt = """You are a pediatric dietitian evaluating malnutrition status in children based on clinical notes.
MALNUTRITION CRITERIA
* Mild: z-score -1 to -1.9 SD (weight-for-height, BMI-for-age)
* Moderate: z-score -2 to -2.9 SD (weight-for-height, BMI-for-age)
* Severe: z-score ≤ -3 SD or severe stunting (length/height-for-age ≤ -3 SD)
* Physical signs: muscle wasting, reduced subcutaneous fat, edema
* Growth trajectory: declining percentiles, weight loss, poor weight gain
IMPORTANT GUIDELINES
* Weight is primarily affected during acute undernutrition
* Chronic undernutrition typically manifests as stunting
* Severe acute undernutrition (ages 6–60 months): very low weight-for-height (< -3 SD z-scores), visible severe wasting (MUAC ≤115 mm), or nutritional edema
* Chronic undernutrition/stunting: height-for-age < -2 SD z-score
* Growth monitoring is the primary outcome measure of nutritional status
CLASSIFICATION GUIDANCE
* Mild malnutrition: usually from acute events (economic circumstances or illness) with unintentional weight loss
* Moderate malnutrition: undernutrition of significant duration with below-normal weight-for-height/BMI-for-age
* Severe malnutrition: prolonged undernutrition with stunting

Based on the clinical note below, determine if the child is malnourished. Answer with only 'yes' or 'no'.

### Clinical Note:
{}

### Assessment:
{}"""
    return prompt.format(note, label)


def prepare_dataset(data_path, tokenizer, max_seq_length):
    df = pd.read_csv(data_path)
    if not {"DEID", "txt", "label"}.issubset(df.columns):
        raise ValueError("CSV must include 'DEID', 'txt', and 'label' columns.")

    prompts = []
    for _, row in df.iterrows():
        label_text = "yes" if str(row["label"]).lower() in {"1", "yes", "true"} else "no"
        prompt = create_malnutrition_prompt(row["txt"], label_text)
        prompts.append(prompt + tokenizer.eos_token)

    return Dataset.from_dict({"text": prompts})


def main():
    args = parse_arguments()

    print(f"[INFO] Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    dataset = prepare_dataset(args.data_path, tokenizer, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs if args.max_steps is None else None,
        report_to="none",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=10,
    )

    # Updated SFTTrainer initialization without dataset_text_field
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
    )

    print("[INFO] Starting training...")
    trainer_stats = trainer.train()

    print(f"[INFO] Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("[INFO] Training complete!")


if __name__ == "__main__":
    main()
