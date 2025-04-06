#!/usr/bin/env python3
import os
import unsloth
# Set the environment variable for Unsloth logits before any imports
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

import torch
import argparse
from datasets import Dataset
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
import pandas as pd
import datetime
import json
from models.llm_models import (
    MalnutritionDataset,
    MalnutritionPromptBuilder,
    is_bfloat16_supported,
    set_seed,
    evaluate_predictions,
    plot_evaluation_metrics,
    save_metrics_to_csv,
    print_metrics_report,
    WeightedSFTTrainer  # Import the WeightedSFTTrainer from llm_models
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a malnutrition detection model")

    # Model and data arguments
    parser.add_argument("--model_name", type=str,
                        default="unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit",
                        help="Base model to use for fine-tuning")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training CSV data")
    parser.add_argument("--val_data", type=str, required=False, default=None,
                        help="Path to validation CSV data (optional)")
    parser.add_argument("--examples_data", type=str, default=None,
                        help="Path to few-shot examples CSV data (optional)")
    parser.add_argument('--text_column', type=str, default="txt",
                        help='Name of the text column in the CSV')
    parser.add_argument('--label_column', type=str, default="Label",
                        help='Name of the label column in the CSV')

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./llm",
                        help="Directory for saving training outputs")
    parser.add_argument("--model_output", type=str, default="./llm_models",
                        help="Path to save the final model")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for training")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    
    # Class weighting argument
    parser.add_argument("--pos_weight", type=float, default=3.0,
                        help="Weight for positive class (higher values penalize false positives more)")

    # Precision arguments
    parser.add_argument("--force_fp16", action="store_true",
                        help="Force using FP16 precision for training")
    parser.add_argument("--force_bf16", action="store_true",
                        help="Force using BF16 precision for training")

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use Flash Attention 2 if available")
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["none", "tensorboard", "wandb"],
                        help="Where to report training metrics")
    parser.add_argument("--quantize", action="store_true", default=False,
                        help="Enable quantization (not recommended for full fine-tuning)")
    
    args = parser.parse_args()
    
    return args


def get_quantization_config(args):
    """Define quantization configuration for the model based on arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        BitsAndBytesConfig: Quantization configuration or None for full precision
    """
    # For full fine-tuning, we don't want quantization by default
    if not args.quantize:
        return None
    
    # Determine if we should use 4-bit quantization if specifically requested
    # Determine compute dtype based on available hardware and args
    if args.force_bf16 and is_bfloat16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
        
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True
    )


def determine_model_precision(args):
    """Determine appropriate precision settings for model training.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (fp16, bf16) boolean flags
    """
    # Check if user explicitly specified precision
    if args.force_fp16:
        return True, False
    
    if args.force_bf16:
        if is_bfloat16_supported():
            return False, True
        else:
            print("Warning: BF16 requested but not supported by hardware. Falling back to FP16.")
            return True, False
    
    # Auto-detect best precision
    if is_bfloat16_supported():
        return False, True
    else:
        return True, False


def load_model_and_tokenizer(args, quantization_config):
    """Load base model and tokenizer with appropriate settings."""
    print(f"Loading base model and tokenizer: {args.model_name}")
    
    # Determine precision based on hardware and user preferences
    fp16, bf16 = determine_model_precision(args)
    dtype = torch.bfloat16 if bf16 else torch.float16
    
    try:
        print(f"Loading model with settings: precision={'bf16' if bf16 else 'fp16'}, "
              f"quantization={'enabled' if quantization_config else 'disabled (full precision)'}")
        
        # Set attention implementation based on flash attention flag
        attn_implementation = "flash_attention_2" if args.use_flash_attention else "eager"
        
        # Create kwargs for model loading
        model_kwargs = {
            "model_name": args.model_name,
            "dtype": dtype,
            "device_map": "auto",
            "attn_implementation": attn_implementation,
        }
        
        # If quantization_config is provided, use it
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        
        # Load the model with the appropriate parameters
        base_model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        
        print("Model and tokenizer loaded successfully.")
        return base_model, tokenizer, fp16, bf16
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def prepare_model_for_full_finetuning(model):
    """Prepare the model for full fine-tuning by unfreezing all parameters."""
    print("Preparing model for full fine-tuning...")
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    
    print(f"Model prepared for full fine-tuning with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    return model


def get_sft_config(args, fp16, bf16):
    """Configure SFT training arguments.

    Args:
        args: Command line arguments
        fp16: Whether to use FP16 precision
        bf16: Whether to use BF16 precision

    Returns:
        SFTConfig: Configuration for SFT training
    """
    config_kwargs = {
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "warmup_steps": 5,
        "learning_rate": args.learning_rate,
        "fp16": fp16,
        "bf16": bf16,
        "logging_steps": 1,
        "optim": "adamw_torch",  # Using standard AdamW optimizer for full fine-tuning
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",  # Using cosine schedule for full fine-tuning
        "seed": args.seed,
        "output_dir": args.output_dir,
        "report_to": args.report_to,
        "save_strategy": "steps",
        "save_steps": 10,
        "max_seq_length": args.max_seq_length,
        "dataset_num_proc": 4,
        "packing": False,
        "num_train_epochs": args.epochs,
        "gradient_checkpointing": True,  # Enable gradient checkpointing for full fine-tuning
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }
    
    # Add evaluation parameters only if validation data is provided
    if args.val_data is not None:
        config_kwargs.update({
            "eval_strategy": "steps",
            "eval_steps": 10,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
        })
    
    print(f"Training with precision: fp16={fp16}, bf16={bf16}")
    return SFTConfig(**config_kwargs)


def prepare_datasets(train_data_path, val_data_path, prompt_builder, tokenizer, note_col, label_col, max_seq_length):
    """Prepare training and validation datasets.

    Args:
        train_data_path (str): Path to training data CSV
        val_data_path (str): Path to validation data CSV
        prompt_builder: MalnutritionPromptBuilder instance
        tokenizer: Tokenizer for the model
        note_col (str): Name of the text column
        label_col (str): Name of the label column
        max_seq_length (int): Maximum sequence length for tokenization

    Returns:
        Tuple: (train_dataset, eval_dataset)
    """
    print("Preparing datasets...")

    # Load and prepare training data
    train_data = MalnutritionDataset(train_data_path, note_col, label_col)
    train_formatted = train_data.prepare_training_data(prompt_builder)
    
    # Pre-tokenize the data to ensure consistent format
    def tokenize_function(examples):
        # Make sure 'text' field exists and is not empty
        if not examples.get('text'):
            return {"input_ids": [], "attention_mask": []}
            
        # Tokenize the examples
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors=None,  
        )
        # Add labels for supervised fine-tuning
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Convert to Dataset and tokenize
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_formatted))
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=False,
        remove_columns=["text"] if "text" in train_dataset.column_names else [],
    )
    
    # Handle validation data if provided
    eval_tokenized = None
    if val_data_path is not None:
        val_data = MalnutritionDataset(val_data_path, note_col, label_col)
        val_formatted = val_data.prepare_training_data(prompt_builder)
        eval_dataset = Dataset.from_pandas(pd.DataFrame(val_formatted))
        eval_tokenized = eval_dataset.map(
            tokenize_function,
            batched=False,
            remove_columns=["text"] if "text" in eval_dataset.column_names else [],
        )
        print(f"Prepared {len(train_tokenized)} training examples and {len(eval_tokenized)} validation examples")
    else:
        print(f"Prepared {len(train_tokenized)} training examples")
    
    return train_tokenized, eval_tokenized


def main():
    """Main function to run the training pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_output, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Get quantization config - for full fine-tuning, you typically don't want to use quantization
    quantization_config = get_quantization_config(args)

    # Load model and tokenizer with precision detection
    base_model, tokenizer, fp16, bf16 = load_model_and_tokenizer(
        args, quantization_config
    )

    # Prepare model for full fine-tuning
    model = prepare_model_for_full_finetuning(base_model)

    # Initialize prompt builder
    prompt_builder = MalnutritionPromptBuilder(args.examples_data)

    # Load and prepare datasets with tokenization
    train_dataset, eval_dataset = prepare_datasets(
        args.train_data, 
        args.val_data, 
        prompt_builder, 
        tokenizer,
        args.text_column, 
        args.label_column,
        args.max_seq_length
    )

    # Get SFT config with correct precision settings
    sft_config = get_sft_config(args, fp16, bf16)

    # Initialize SFT trainer with weighted loss
    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "args": sft_config,
        "pos_weight": args.pos_weight,  # Pass the positive class weight
    }
    
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = WeightedSFTTrainer(**trainer_kwargs)  # Use the weighted trainer

    # Train the model
    print(f"Starting full fine-tuning with {len(train_dataset)} examples for {args.epochs} epoch(s)...")
    print(f"Using positive class weight: {args.pos_weight}")
    trainer.train()

    # Save the trained model properly
    print(f"Training completed. Saving model to {args.model_output}")
    
    # Save the fully fine-tuned model
    print("Saving fully fine-tuned model...")
    trainer.model.save_pretrained(args.model_output)
    
    # Save the tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(args.model_output)
    
    # Clean up checkpoint files
    print("Cleaning up checkpoint files...")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint-*")
    import glob
    checkpoint_paths = glob.glob(checkpoint_dir)
    
    for checkpoint_path in checkpoint_paths:
        if os.path.isdir(checkpoint_path):
            print(f"Removing checkpoint: {checkpoint_path}")
            import shutil
            try:
                shutil.rmtree(checkpoint_path)
            except Exception as e:
                print(f"Warning: Could not remove checkpoint directory {checkpoint_path}: {e}")
    
    # Remove checkpoint files from model directory
    checkpoint_files = [f for f in os.listdir(args.model_output) if "checkpoint" in f]
    for file in checkpoint_files:
        file_path = os.path.join(args.model_output, file)
        if os.path.isdir(file_path):
            print(f"Removing checkpoint directory from model output: {file_path}")
            import shutil
            try:
                shutil.rmtree(file_path)
            except Exception as e:
                print(f"Warning: Could not remove checkpoint directory {file_path}: {e}")
        elif os.path.isfile(file_path):
            print(f"Removing checkpoint file from model output: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove checkpoint file {file_path}: {e}")

    # Save a small README with information about the model
    readme_content = f"""# Malnutrition Detection Model (Full Fine-tuning)

Base model: {args.model_name}
Training date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Training parameters:
- Learning rate: {args.learning_rate}
- Batch size: {args.batch_size}
- Gradient accumulation steps: {args.gradient_accumulation}
- Epochs: {args.epochs}
- Positive class weight: {args.pos_weight}
- Quantization: {'Enabled' if args.quantize else 'Disabled (full precision)'}

## Usage
This directory contains a fully fine-tuned model (no LoRA).

For inference, you can load this model directly without needing a separate adapter.
"""
    
    with open(os.path.join(args.model_output, "README.md"), "w") as f:
        f.write(readme_content)

    print("Full fine-tuning complete!")
    print(f"Model saved to: {args.model_output}")


if __name__ == "__main__":
    main()
