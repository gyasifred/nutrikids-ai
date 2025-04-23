#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Llama model for pediatric malnutrition assessment based on clinical notes.
"""

import os
import pandas as pd
import torch
import argparse
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a model for pediatric malnutrition classification")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B", help="Base model to use for fine-tuning")
    parser.add_argument("--output_dir", type=str, default="./malnutrition_model", help="Directory to save the model")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--epochs", type=float, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X steps")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (overrides epochs if set)")
    return parser.parse_args()


def create_malnutrition_prompt(note, label=""):
    """Create a formatted prompt for malnutrition assessment."""
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


def prepare_dataset(data_path):
    """Load and prepare the dataset from CSV file."""
    df = pd.read_csv(data_path)
    required_cols = ["DEID", "txt", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data file")
    return Dataset.from_dict({
        "id": df["DEID"].tolist(),
        "note": df["txt"].tolist(),
        "label": df["label"].tolist()
    })


def main():
    args = parse_arguments()

    print(f"Loading base model: {args.model_name}")
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

    print(f"Loading dataset from: {args.data_path}")
    dataset = prepare_dataset(args.data_path)

    # Tokenization + truncation formatting function
    def formatting_func(example):
        prompt = create_malnutrition_prompt(example["note"], example["label"]) + tokenizer.eos_token
        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
        }

    response_template = "\n### Assessment:\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=args.save_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=10,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs if args.max_steps is None else None,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        data_collator=data_collator,
        packing=False,
    )

    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    print("Starting training...")
    trainer_stats = trainer.train()

    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
