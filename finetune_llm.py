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
    print_metrics_report
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a malnutrition detection model")

    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="unsloth/Phi-3-mini-4k-instruct",
                        help="Base model to use for fine-tuning")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training CSV data")
    parser.add_argument("--val_data", type=str, required=False, default=None,
                        help="Path to validation CSV data (optional)")
    parser.add_argument("--examples_data", type=str, default=None,
                        help="Path to few-shot examples CSV data (optional)")
    parser.add_argument('--text_column', type=str, default="Note_Column",
                        help='Name of the text column in the CSV')
    parser.add_argument('--label_column', type=str, default="Malnutrition_Label",
                        help='Name of the label column in the CSV')

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./llm_train_output",
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
    parser.add_argument("--max_steps", type=int, default=60,
                        help="Maximum number of training steps")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length for tokenization")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA r parameter (rank)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter (scaling)")

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use Flash Attention 2 if available")
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["none", "tensorboard", "wandb"],
                        help="Where to report training metrics")

    return parser.parse_args()


def get_quantization_config():
    """Define quantization configuration for the model."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True
    )


def load_model_and_tokenizer(model_name, quantization_config, use_flash_attention=False):
    """Load base model and tokenizer.

    Args:
        model_name (str): Name or path of the model to load
        quantization_config: Quantization configuration
        use_flash_attention (bool): Whether to use Flash Attention 2

    Returns:
        Tuple: (model, tokenizer)
    """
    print(f"Loading base model and tokenizer: {model_name}")
    try:
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
            dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
            use_flash_attention_2=use_flash_attention,
            use_cache=False
        )
        print("Model and tokenizer loaded successfully.")
        return base_model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def create_peft_model(base_model, r=8, lora_alpha=32):
    """Create PEFT/LoRA model for fine-tuning.

    Args:
        base_model: The base language model
        r (int): LoRA rank parameter
        lora_alpha (int): LoRA alpha parameter

    Returns:
        model: The PEFT model ready for training
    """
    print("Creating PEFT/LoRA model...")
    model = FastLanguageModel.get_peft_model(
        model=base_model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=True,
        loftq_config=None
    )

    # Enable gradient checkpointing for efficient training
    model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    return model


def get_sft_config(args):
    """Configure SFT training arguments.

    Args:
        args: Command line arguments

    Returns:
        SFTConfig: Configuration for SFT training
    """
    config_kwargs = {
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "warmup_steps": 5,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
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
    }
    
    # Add evaluation parameters only if validation data is provided
    if args.val_data is not None:
        config_kwargs.update({
            "eval_strategy": "steps",
            "eval_steps": 10,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
        })
    
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
            return_tensors=None,  # Return lists instead of tensors
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
    quantization_config = get_quantization_config()

    # Load model and tokenizer
    base_model, tokenizer = load_model_and_tokenizer(
        args.model_name, quantization_config, args.use_flash_attention
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
    model = create_peft_model(base_model, args.lora_r, args.lora_alpha)

    # Get SFT config
    sft_config = get_sft_config(args)

    # Initialize SFT trainer without custom data collator
    # Using the tokenized datasets directly
    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "args": sft_config,
    }
    
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = SFTTrainer(**trainer_kwargs)

    # Train the model
    print(f"Starting training with {len(train_dataset)} examples...")
    # Note: We already set the environment variable at the top of the file
    trainer.train()

    # Save the final model
    print(f"Training completed. Saving model to {args.model_output}")
    trainer.save_model(args.model_output)

    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()