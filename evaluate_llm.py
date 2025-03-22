#!/usr/bin/env python3
import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig

# Import utilities from the existing implementation
from models.llm_models import (
    MalnutritionPromptBuilder,
    extract_malnutrition_decision,
    set_seed,
    evaluate_predictions,
    plot_evaluation_metrics,
    save_metrics_to_csv,
    print_metrics_report,
    is_bfloat16_supported
)


def parse_arguments():
    """Parse command line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate malnutrition detection model on a test dataset"
    )

    # Model arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the fine-tuned model adapter weights (optional). "
                             "If not provided, the base model is used for inference.")
    parser.add_argument("--base_model", type=str, default="unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit",
                        help="Base model that was fine-tuned or to be used for inference")

    # Test data arguments
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to CSV file with test data (patient notes and labels)")
    parser.add_argument("--text_column", type=str, default="txt",
                        help="Column name in CSV containing patient notes")
    parser.add_argument("--id_column", type=str, default="DEID",
                        help="Column name in CSV containing patient IDs")
    parser.add_argument("--label_column", type=str, default="label",
                        help="Column name in CSV containing true labels")

    # Few-shot settings
    parser.add_argument("--examples_data", type=str, default=None,
                        help="Path to few-shot examples CSV data (optional)")
    parser.add_argument("--few_shot_count", type=int, default=0,
                        help="Number of few-shot examples to use (default: 0 for zero-shot)")
    parser.add_argument("--balanced_examples", action="store_true",
                        help="Whether to balance positive/negative few-shot examples")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./LLM_pre_evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--print_report", action="store_true",
                        help="Print evaluation report to terminal")

    # Quantization options
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit precision")

    # Model settings
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use Flash Attention 2 if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing (default: 1)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for binary classification (default: 0.5)")
    
    # Force precision if needed
    parser.add_argument("--force_fp16", action="store_true",
                        help="Force using FP16 precision")
    parser.add_argument("--force_bf16", action="store_true",
                        help="Force using BF16 precision if supported")

    args = parser.parse_args()
    
    if args.load_in_8bit:
        args.load_in_4bit = False
        
    return args


def get_quantization_config(args):
    """Define quantization configuration for the model based on arguments."""
    # Determine compute dtype based on hardware and user preferences
    compute_dtype = torch.bfloat16 if is_bfloat16_supported() and not args.force_fp16 else torch.float16
    
    # Handle 8-bit quantization
    if args.load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_4bit=False,
            llm_int8_enable_fp32_cpu_offload=True
        )
    # Handle 4-bit quantization
    elif args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )
    # No quantization (full precision)
    else:
        return None


def load_model_and_tokenizer(base_model, model_path, args):
    """
    Load model and tokenizer for evaluation with appropriate quantization settings.

    Args:
        base_model (str): Base model name or path
        model_path (str): Path to fine-tuned model adapter weights (optional)
        args: Command line arguments with quantization settings

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading {'fine-tuned' if model_path else 'base'} model: {base_model}")
    # Determine compute dtype based on hardware and preferences
    compute_dtype = torch.bfloat16 if is_bfloat16_supported() and not args.force_fp16 else torch.float16
    # Get appropriate quantization config
    quantization_config = get_quantization_config(args)
    try:
        # Set up model loading kwargs with common parameters
        model_kwargs = {
            "device_map": "auto",
            "use_flash_attention_2": args.use_flash_attention,
            "use_cache": True  # Enable caching for inference
        }
        # Add quantization settings
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            # Direct quantization flags if no config is provided
            model_kwargs["load_in_4bit"] = args.load_in_4bit
            model_kwargs["load_in_8bit"] = args.load_in_8bit 
        # Set dtype for the model 
        model_kwargs["dtype"] = compute_dtype
        if model_path:
            # Load base model with adapter weights (fine-tuned model)
            model_kwargs["model_name"] = base_model
            model_kwargs["adapter_name"] = model_path
            model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
            print(f"Model loaded successfully with adapter weights from {model_path}")
        else:
            # Load base model only
            model_kwargs["model_name"] = base_model
            model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
            print(f"Base model loaded successfully: {base_model}")
        # Enable native 2x faster inference
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def get_probability_from_logits(logits, tokenizer):
    """
    Extract probability of the 'yes' answer from model logits.
    Args:
        logits (torch.Tensor): The output logits from the model's forward pass
        tokenizer: The tokenizer used with the model
    Returns:
        float: Probability score for "yes" class (0-1)
    """
    # Get token IDs for "yes" and "no"
    yes_tokens = tokenizer("yes", add_special_tokens=False).input_ids
    no_tokens = tokenizer("no", add_special_tokens=False).input_ids
    # Take the first token ID as representative
    yes_token_id = yes_tokens[0]
    no_token_id = no_tokens[0]
    # Extract the relevant logits
    yes_logit = logits[0, yes_token_id].item()
    no_logit = logits[0, no_token_id].item()
    # Convert to probabilities using softmax
    probs = np.exp([yes_logit, no_logit]) / np.sum(np.exp([yes_logit, no_logit]))
    yes_prob = probs[0]
    return yes_prob


def process_single_text_with_logits(text, model, tokenizer, prompt_builder, args):
    """
    Process a single patient note and extract both the prediction and probability.
    Args:
        text (str): Patient note text
        model: The language model
        tokenizer: The tokenizer
        prompt_builder: MalnutritionPromptBuilder instance
        args: Command line arguments
    Returns:
        tuple: (decision, explanation, probability)
    """
    # Prepare prompt with few-shot examples if specified
    prompt = prompt_builder.get_inference_prompt(
        patient_notes=text,
        note_col=args.text_column,
        label_col=args.label_column,
        num_examples=args.few_shot_count,
        balanced=args.balanced_examples
    )
    # Prepare chat format messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    # Apply chat template to get input tokens
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    # Step 1: Run forward pass to get logits for the first token prediction
    with torch.no_grad():
        # Get the logits for the first generated token
        logits = model(inputs).logits[:, -1, :]
        # Get probability for "yes" class
        yes_probability = get_probability_from_logits(logits, tokenizer)
        # Now generate the full response
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode the output - only the generated part
    input_length = inputs.shape[1]
    response_tokens = output[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    # Extract decision and explanation
    decision, explanation = extract_malnutrition_decision(response)
    # Apply threshold to convert probability to binary label (1/0)
    binary_decision = 1 if yes_probability >= args.threshold else 0
    return binary_decision, explanation, yes_probability


def evaluate_model(args, model, tokenizer, prompt_builder):
    """
    Evaluate model on test dataset.
    Args:
        args: Command line arguments
        model: The language model
        tokenizer: The tokenizer
        prompt_builder: MalnutritionPromptBuilder instance
    Returns:
        DataFrame: Results with predictions and probabilities
    """
    print(f"Loading test data from {args.test_csv}")
    # Load test data
    df = pd.read_csv(args.test_csv)
    # Validate columns exist
    required_columns = [args.text_column, args.label_column]
    if args.id_column not in df.columns:
        df[args.id_column] = [f"patient_{i}" for i in range(len(df))]
    else:
        required_columns.append(args.id_column)
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in test CSV")
    # Initialize results storage
    results = []
    # Process each example
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        patient_text = row[args.text_column]
        patient_id = row[args.id_column]
        # Convert true label to binary (1/0)
        true_label_str = str(row[args.label_column]).lower()
        true_label = 1 if true_label_str in ["1", "yes", "true"] else 0
        # Get prediction, explanation, and probability
        predicted_label, explanation, probability = process_single_text_with_logits(
            patient_text, model, tokenizer, prompt_builder, args
        )
        # Store result
        results.append({
            "patient_id": patient_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "probability": probability,
            "explanation": explanation
        })
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def main():
    """Main function to run the evaluation pipeline."""
    args = parse_arguments()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize prompt builder
    prompt_builder = MalnutritionPromptBuilder(args.examples_data)
    
    # Load model and tokenizer with appropriate quantization settings
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.model_path, args
    )
    
    # Evaluate model on test data
    results_df = evaluate_model(args, model, tokenizer, prompt_builder)
    
    # Save predictions to CSV
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Evaluate model performance
    y_true = results_df["true_label"].tolist()
    y_pred = results_df["predicted_label"].tolist()
    y_prob = results_df["probability"].tolist()
    
    print("Computing evaluation metrics...")
    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    
    # Generate and save plots
    print("Generating evaluation plots...")
    plot_evaluation_metrics(metrics, args.output_dir)
    
    # Save metrics to CSV
    metrics_csv_path = os.path.join(args.output_dir, "metrics.csv")
    save_metrics_to_csv(metrics, metrics_csv_path)
    print(f"Metrics saved to {metrics_csv_path}")
    
    # Save threshold used
    metrics_json = {
        'threshold': args.threshold,
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'auc': float(metrics['auc']),
        'avg_precision': float(metrics['avg_precision']),
        'confusion_matrix': metrics['confusion_matrix'],
        'classification_report': metrics['classification_report']
    }
    
    metrics_json_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Detailed metrics saved to {metrics_json_path}")
    
    # Print evaluation report
    if args.print_report:
        print_metrics_report(metrics)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
