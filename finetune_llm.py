#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized fine-tuning script for malnutrition assessment with clinical reasoning.
"""

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'  # Enable logits return for probability extraction
import argparse
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported, FastLanguageModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for pediatric malnutrition classification")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-8b", 
                       help="Base model (e.g., 'unsloth/llama-3-8b')")
    parser.add_argument("--output_dir", type=str, default="./malnutrition_model", 
                       help="Output directory for saved model")
    parser.add_argument("--max_seq_length", type=int, default=None, 
                       help="Max sequence length (None for model default)")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=float, default=3.0,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--preprocess_tokens", action="store_true",
                       help="Preprocess special tokens in clinical notes")
    return parser.parse_args()

def preprocess_clinical_note(note_text):
    """Clean clinical notes by handling special tokens."""
    if not note_text:
        return note_text
    
    # Standardize section breaks and special tokens
    replacements = {
        '</s>': '\n\n[SECTION] ',
        '<s>': '[START]',
        '<pad>': '',
        '</pad>': '',
        '<eos>': '[END]',
        '<bos>': '[BEGIN]'
    }
    
    for token, replacement in replacements.items():
        note_text = note_text.replace(token, replacement)
    
    return note_text.strip()

def create_malnutrition_prompt(note, label="", reasoning=""):
    """Optimized prompt template for fine-tuning with probability support."""
    return f"""<|begin_of_text|>[CLINICAL ASSESSMENT PROTOCOL]

[WHO DIAGNOSTIC CRITERIA]
1. SEVERE (SAM): 
   - WFH/BMI < -3 SD OR
   - MUAC < 115mm (<5yrs) OR
   - Edema
2. MODERATE (MAM):
   - WFH/BMI -2 to -3 SD
3. CHRONIC:
   - HFA < -2 SD + declining trend

[REQUIRED OUTPUT FORMAT]
### Assessment: <yes/no>  # Must be first token
### Confidence: <high/medium/low>
### Evidence: <z-scores, MUAC, clinical signs>

[CLINICAL NOTE]
{note}

[EXPERT ANALYSIS]
### Assessment: {label}
### Confidence: {'high' if label else 'medium'}
### Evidence: {reasoning if reasoning else 'See anthropometrics above'}<|end_of_text|>"""

def prepare_dataset(data_path, tokenizer, max_seq_length, preprocess_tokens=False):
    """Prepare dataset with proper formatting for probability-aware training."""
    df = pd.read_csv(data_path)
    required_cols = {"DEID", "txt", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")

    # Convert labels to consistent format
    df["label"] = df["label"].apply(lambda x: "yes" if str(x).lower() in ("1", "yes", "true") else "no")
    
    # Handle optional reasoning column
    has_reasoning = "reasoning" in df.columns
    if not has_reasoning:
        print("[WARNING] No 'reasoning' column found - using empty strings")

    texts = []
    for _, row in df.iterrows():
        note = preprocess_clinical_note(row["txt"]) if preprocess_tokens else row["txt"]
        reasoning = row.get("reasoning", "") if has_reasoning else ""
        
        prompt = create_malnutrition_prompt(
            note=note,
            label=row["label"],
            reasoning=reasoning
        )
        texts.append(prompt + tokenizer.eos_token)  # Add EOS token

    return Dataset.from_dict({"text": texts})

def main():
    args = parse_arguments()
    
    # Load model with optimized settings
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length or 4096,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        token=os.getenv("HF_TOKEN"),  # For gated models
    )
    
    # Configure LoRA with probability-aware training
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0
        bias="none",
        use_gradient_checkpointing=True,
    )
    
    # Prepare dataset
    train_dataset = prepare_dataset(
        args.data_path,
        tokenizer,
        args.max_seq_length,
        args.preprocess_tokens
    )
    
    # # Training configuration
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     learning_rate=args.learning_rate,
    #     num_train_epochs=args.epochs,
    #     logging_steps=10,
    #     save_strategy="steps",
    #     save_steps=200,
    #     fp16=not is_bfloat16_supported(),
    #     bf16=is_bfloat16_supported(),
    #     optim="adamw_8bit",
    #     weight_decay=0.01,
    #     warmup_ratio=0.1,
    #     max_grad_norm=1.0,
    #     report_to="none",
    # )
    
    # Create trainer with probability-aware formatting
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args= TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        report_to="none",
    )
    )
    
    # Start training
    print(f"Training model for {args.epochs} epochs...")
    trainer.train()
    
    # Save final model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
