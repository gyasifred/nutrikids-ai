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
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    
    # Class weighting argument
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="Weight for positive class (higher values penalize false positives more)")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA r parameter (rank)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter (scaling)")
    parser.add_argument("--target_modules", type=str, default=None,
                        help="Comma-separated list of target modules for LoRA")

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
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit precision")
    
    args = parser.parse_args()
    
    if args.load_in_8bit:
        args.load_in_4bit = False
        
    return args


def get_quantization_config(args):
    """Define quantization configuration for the model based on arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        BitsAndBytesConfig: Quantization configuration
    """
    # Determine if we should use 8-bit or 4-bit quantization (but not both)
    if args.load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False,  
            llm_int8_enable_fp32_cpu_offload=True
        )
    elif args.load_in_4bit:
        # Determine compute dtype based on available hardware and args
        if args.force_bf16 and is_bfloat16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
            
        return BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,  # Explicitly set to False
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )
    else:
        # No quantization
        return None


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
        # Ensure we're not using both 4-bit and 8-bit
        load_in_4bit = args.load_in_4bit and not args.load_in_8bit
        load_in_8bit = args.load_in_8bit and not args.load_in_4bit
        
        print(f"Loading model with settings: precision={'bf16' if bf16 else 'fp16'}, "
              f"load_in_4bit={load_in_4bit}, load_in_8bit={load_in_8bit}")
        
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
        else:
            # Otherwise use the direct parameters
            model_kwargs["load_in_4bit"] = load_in_4bit
            model_kwargs["load_in_8bit"] = load_in_8bit
        
        # Load the model with the appropriate parameters
        base_model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
        
        print("Model and tokenizer loaded successfully.")
        return base_model, tokenizer, fp16, bf16
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def get_target_modules(args, model_name):
    """Determine appropriate target modules for LoRA based on model architecture.
    
    Args:
        args: Command line arguments
        model_name: Name of the model
        
    Returns:
        list: List of target module names
    """
    # If user specified target modules, use those
    if args.target_modules:
        return args.target_modules.split(',')
    
    # Default target modules based on model architecture
    model_name_lower = model_name.lower()
    
    if any(name in model_name_lower for name in ["llama", "mistral", "mixtral"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "phi" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "qwen" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2"]
    elif "deepseek" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Default fallback
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def create_peft_model(base_model, args):
    """Create PEFT/LoRA model for fine-tuning with appropriate settings.

    Args:
        base_model: The base language model
        args: Command line arguments

    Returns:
        model: The PEFT model ready for training
    """
    print("Creating PEFT/LoRA model...")
    
    # Get appropriate target modules for this model architecture
    target_modules = get_target_modules(args, args.model_name)
    print(f"Using target modules: {target_modules}")
    
    model = FastLanguageModel.get_peft_model(
        model=base_model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=target_modules,
        use_gradient_checkpointing=True,
        random_state=args.seed,
        use_rslora=True,
        loftq_config=None
    )

    # Enable gradient checkpointing for efficient training
    model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

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
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "output_dir": args.output_dir,
        "report_to": args.report_to,
        "save_strategy": "steps",
        "save_steps": 10,
        "max_seq_length": args.max_seq_length,
        "dataset_num_proc": 4,
        "packing": False,
        "num_train_epochs": args.epochs
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

    # Get quantization config
    quantization_config = get_quantization_config(args)

    # Load model and tokenizer with precision detection
    base_model, tokenizer, fp16, bf16 = load_model_and_tokenizer(
        args, quantization_config
    )

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

    # Create PEFT/LoRA model
    model = create_peft_model(base_model, args)

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
    print(f"Starting training with {len(train_dataset)} examples for {args.epochs} epoch(s)...")
    print(f"Using positive class weight: {args.pos_weight}")
    trainer.train()

    # Save the trained model properly
    print(f"Training completed. Saving model to {args.model_output}")
    
    # Save the LoRA adapter
    print("Saving LoRA adapter weights...")
    trainer.model.save_pretrained(args.model_output)
    
    # Save the tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(args.model_output)
    
    # # Save a merged model version (LoRA weights merged into base model)
    # merged_model_path = os.path.join(args.model_output, "merged")
    # os.makedirs(merged_model_path, exist_ok=True)
    
    # print(f"Creating merged model (LoRA + base) at {merged_model_path}...")
    # try:
    #     # Comment out the merging code
    #     """
    #     # Merge LoRA weights into base model
    #     merged_model = trainer.model.merge_and_unload()
        
    #     # Save the merged model
    #     merged_model.save_pretrained(merged_model_path)
    #     tokenizer.save_pretrained(merged_model_path)
    #     print("Merged model saved successfully.")
    #     """
    #     print("Merged model creation skipped.")
    # except Exception as e:
    #     print(f"Warning: Could not create merged model: {e}")
    #     print("Continuing with adapter-only model.")
    
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
    readme_content = f"""# Malnutrition Detection Model

Base model: {args.model_name}
Training date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Training parameters:
- Learning rate: {args.learning_rate}
- Batch size: {args.batch_size}
- Gradient accumulation steps: {args.gradient_accumulation}
- Epochs: {args.epochs}
- LoRA rank: {args.lora_r}
- LoRA alpha: {args.lora_alpha}
- Positive class weight: {args.pos_weight}

## Usage
This directory contains two versions of the model:
1. LoRA adapter only (main directory)
2. Merged model (in the 'merged' subdirectory)

For inference, the merged model is typically easier to use as it doesn't require loading a separate base model.
"""
    
    with open(os.path.join(args.model_output, "README.md"), "w") as f:
        f.write(readme_content)

    print("Fine-tuning complete!")
    print(f"Model saved to: {args.model_output}")
    print(f"Merged model saved to: {merged_model_path}")


if __name__ == "__main__":
    import datetime  # Add this import at the top of your file
    main()
