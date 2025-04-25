#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune LLaMA-style model using Unsloth + LoRA for malnutrition assessment with reasoning.
"""

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'  # Critical for probability extraction
import argparse
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for pediatric malnutrition classification with reasoning")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B", help="Base model to use")
    parser.add_argument("--output_dir", type=str, default="./malnutrition_model", help="Where to save the model")
    parser.add_argument("--max_seq_length", type=int, default=None, 
                        help="Max sequence length (if None, uses model's native max length)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--epochs", type=float, default=10, help="Epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--save_steps", type=int, default=50, help="Steps between checkpoints")
    parser.add_argument("--preprocess_tokens", action="store_true", help="Preprocess </s> tokens in clinical notes")
    parser.add_argument("--use_native_max_len", action="store_true", 
                        help="Use the model's native maximum sequence length")
    return parser.parse_args()

def preprocess_clinical_note(note_text):
    """Preprocess clinical notes to handle special tokens."""
    if not note_text:
        return note_text
    
    # Your original replacement logic
    processed_text = note_text.replace('</s>', '\n\n')
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")
    
    return processed_text

def create_malnutrition_prompt(note, label="", reasoning=""):
    """Your exact prompt structure with added probability support markers"""
    prompt = """[Role] Pediatric Nutrition Specialist
[Task] Malnutrition assessment using WHO criteria

[Critical Diagnostics]
1. SEVERE: 
   - WFH/BMI < -3 SD
   - MUAC < 115mm (<5yrs)
   - Edema
2. MODERATE:
   - WFH/BMI -2 to -3 SD
3. CHRONIC:
   - HFA < -2 SD + decline

[Output Format]
### Assessment: <yes/no>  # MUST be first token
### Evidence: <z-scores, MUAC, signs>

Clinical Note:
{note}

### Assessment: {label}
### Evidence: {reasoning}"""

    return prompt.format(
        note=note,
        label=label,
        reasoning=reasoning
    )

def prepare_dataset(data_path, tokenizer, max_seq_length, preprocess_tokens=False):
    """Your original dataset prep with enhanced validation"""
    df = pd.read_csv(data_path)
    if not {"DEID", "txt", "label"}.issubset(df.columns):
        raise ValueError("CSV must include 'DEID', 'txt', and 'label' columns.")

    # Your original processing flow
    has_reasoning = "reasoning" in df.columns
    prompts = []
    for _, row in df.iterrows():
        note_text = preprocess_clinical_note(row["txt"]) if preprocess_tokens else row["txt"]
        label_text = "yes" if str(row["label"]).lower() in {"1", "yes", "true"} else "no"
        reasoning_text = row.get("reasoning", "") if has_reasoning else ""
        
        prompt = create_malnutrition_prompt(note_text, label_text, reasoning_text)
        prompts.append(prompt + tokenizer.eos_token)  # Your EOS addition

    return Dataset.from_dict({"text": prompts})

def get_model_max_length(model_name):
    """Your original max length detection"""
    try:
        from unsloth.models.llama import auto_get_max_seq_length
        return auto_get_max_seq_length(model_name)
    except (ImportError, AttributeError):
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_name)
            return getattr(config, "max_position_embeddings", 4096)
        except Exception as e:
            print(f"Failed to get model's max length: {e}")
            return 4096

def main():
    args = parse_arguments()

    # Your original max length logic
    if args.use_native_max_len or args.max_seq_length is None:
        max_seq_length = get_model_max_length(args.model_name)
        print(f"Using model's max length: {max_seq_length}")
    else:
        max_seq_length = args.max_seq_length

    # Your original model loading
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # Your original LoRA config
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

    # Your dataset prep
    dataset = prepare_dataset(args.data_path, tokenizer, max_seq_length, args.preprocess_tokens)

    # Your original trainer setup
    def formatting_prompts_func(examples):
        return examples["text"]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func, 
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_steps=args.save_steps,
            num_train_epochs=args.epochs,
            report_to="none",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=10,
        )
    )

    # Your training execution
    print("Starting training...")
    trainer_stats = trainer.train()

    # Your model saving
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
