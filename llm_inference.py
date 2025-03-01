#!/usr/bin/env python3
import os
import unsloth
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

import torch
import argparse
import pandas as pd
import numpy as np
import re
from transformers import BitsAndBytesConfig
from unsloth import FastLanguageModel
from tqdm import tqdm
import csv
from models.llm_models import (
    MalnutritionPromptBuilder,
    set_seed,
    evaluate_predictions,
    plot_evaluation_metrics,
    save_metrics_to_csv,
    print_metrics_report,
    is_bfloat16_supported
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a malnutrition detection model (base or fine-tuned)"
    )
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the fine-tuned model adapter weights (optional). "
                             "If not provided, the base model is used for inference.")
    parser.add_argument("--base_model", type=str, default="unsloth/Phi-3-mini-4k-instruct",
                        help="Base model that was fine-tuned or to be used for inference")
    
    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_csv", type=str,
                       help="Path to CSV file with patient notes")
    group.add_argument("--input_text", type=str,
                       help="Single patient note as text string")
    
    parser.add_argument("--text_column", type=str, default="Note_Column",
                        help="Column name in CSV containing patient notes")
    parser.add_argument("--id_column", type=str, default="Patient_ID",
                        help="Column name in CSV containing sample IDs")
    parser.add_argument("--label_column", type=str, default='Malnutrition_Label',
                        help="Column name in CSV containing true labels (optional)")
    
    # Few-shot settings
    parser.add_argument("--examples_data", type=str, default=None,
                        help="Path to few-shot examples CSV data (optional)")
    parser.add_argument("--few_shot_count", type=int, default=0,
                        help="Number of few-shot examples to use (default: 0 for zero-shot)")
    parser.add_argument("--balanced_examples", action="store_true",
                        help="Whether to balance positive/negative few-shot examples")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./llm_inference_results",
                        help="Directory to save inference results")
    parser.add_argument("--output_csv", type=str, default="malnutrition_predictions.csv",
                        help="Name of output CSV file")
    parser.add_argument("--print_report", action="store_true",
                        help="Print evaluation report to terminal (if labels available)")
    
    # Model settings
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Use Flash Attention 2 if available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for sampling")
    
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


def load_model_and_tokenizer(base_model, model_path, quantization_config, use_flash_attention=False):
    """
    Load base model and tokenizer.
    
    If a fine-tuned model path (adapter weights) is provided, it loads the adapter alongside the base model.
    Otherwise, it loads the base model alone for inference.
    
    Args:
        base_model (str): Base model name.
        model_path (str): Path to fine-tuned model adapter weights (optional).
        quantization_config: Quantization configuration.
        use_flash_attention (bool): Whether to use Flash Attention 2.
        
    Returns:
        Tuple: (model, tokenizer)
    """
    print(f"Loading base model and tokenizer: {base_model}")
    try:
        if model_path:
            # Load base model with adapter weights (fine-tuned model)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                adapter_name=model_path,
                load_in_4bit=True,
                dtype=torch.float16,
                quantization_config=quantization_config,
                device_map="auto",
                use_flash_attention_2=use_flash_attention,
                use_cache=True  # Enable caching for inference
            )
            FastLanguageModel.for_inference(model)
            print("Model and adapter weights loaded successfully")
        else:
            # Load base model only
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                load_in_4bit=True,
                dtype=torch.float16,
                quantization_config=quantization_config,
                device_map="auto",
                use_flash_attention_2=use_flash_attention,
                use_cache=True  # Enable caching for inference
            )
            FastLanguageModel.for_inference(model)
            print("Base model loaded successfully")

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def extract_malnutrition_decision(response):
    """Extract malnutrition=yes/no decision from model response.
    
    Args:
        response (str): Model response text
        
    Returns:
        Tuple[str, str]: (malnutrition decision, explanation)
    """
    decision_pattern = r'malnutrition=(yes|no)'
    match = re.search(decision_pattern, response, re.IGNORECASE)
    
    decision = "unknown"
    if match:
        decision = match.group(1).lower()
    
    explanation = response
    if match:
        explanation_parts = response.split('malnutrition=', 1)
        if len(explanation_parts) > 0:
            explanation = explanation_parts[0].strip()
    
    return decision, explanation


def process_single_text(text, model, tokenizer, prompt_builder, args):
    """Process a single patient note.
    
    Args:
        text (str): Patient note text.
        model: The language model.
        tokenizer: The tokenizer.
        prompt_builder: MalnutritionPromptBuilder instance.
        args: Command line arguments.
        
    Returns:
        Tuple[str, str]: (decision, explanation)
    """
    # Define default column names if processing a single text
    # These are only used for few-shot example formatting
    text_col = args.text_column if hasattr(args, 'text_column') else "Note_Column"
    label_col = args.label_column if hasattr(args, 'label_column') else "Malnutrition_Label"
    
    prompt = prompt_builder.get_inference_prompt(
        patient_notes=text,
        note_col=text_col,
        label_col=label_col,
        num_examples=args.few_shot_count,
        specific_example_indices=None,
        balanced=args.balanced_examples
    )
    
    # Fix: Ensure inputs is a proper tensor dictionary with input_ids
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move all tensor inputs to the model's device
    for k, v in inputs.items():
        if hasattr(v, 'to'):
            inputs[k] = v.to(model.device)
    
    # Generate with properly formatted inputs
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output - make sure to get only the generated part
    input_length = inputs["input_ids"].shape[1]
    response_tokens = output[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    decision, explanation = extract_malnutrition_decision(response)
    
    return decision, explanation

def process_csv_input(args, model, tokenizer, prompt_builder):
    """Process CSV input file and save results.
    
    Args:
        args: Command line arguments.
        model: The language model.
        tokenizer: The tokenizer.
        prompt_builder: MalnutritionPromptBuilder instance.
        
    Returns:
        pd.DataFrame: DataFrame with prediction results.
    """
    print(f"Processing input CSV: {args.input_csv}")
    
    df = pd.read_csv(args.input_csv)
    
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found in CSV file")
    
    if args.id_column not in df.columns:
        df[args.id_column] = [f"sample_{i}" for i in range(len(df))]
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = row[args.text_column]
        sample_id = row[args.id_column]
        
        true_label = None
        if args.label_column and args.label_column in df.columns:
            true_label = str(row[args.label_column]).lower()
            true_label = "yes" if true_label in ["1", "yes", "true"] else "no"
        
        # Now pass the correct parameters to process_single_text
        decision, explanation = process_single_text(text, model, tokenizer, prompt_builder, args)
        
        result = {
            args.id_column: sample_id,
            args.text_column: text,
            "explanation": explanation,
            "prediction": decision
        }
        
        if true_label is not None:
            result["true_label"] = true_label
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    return results_df


def main():
    """Main function to run the inference pipeline."""
    args = parse_arguments()
    
    # Set environment variable for Unsloth logits (already done at the top of the file)
    # os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize prompt builder
    prompt_builder = MalnutritionPromptBuilder(args.examples_data)
    
    # Get quantization configuration
    quantization_config = get_quantization_config()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.model_path, quantization_config, args.use_flash_attention
    )
    
    if args.input_csv:
        # Process CSV input
        results_df = process_csv_input(args, model, tokenizer, prompt_builder)
        output_path = os.path.join(args.output_dir, args.output_csv)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Evaluate if labels are available
        if args.label_column and "true_label" in results_df.columns:
            y_true = results_df["true_label"].tolist()
            y_pred = results_df["prediction"].tolist()
            metrics = evaluate_predictions(y_true, y_pred)
            plot_evaluation_metrics(metrics, args.output_dir)
            metrics_path = os.path.join(args.output_dir, "metrics.csv")
            save_metrics_to_csv(metrics, metrics_path)
            print(f"Evaluation metrics saved to {metrics_path}")
            if args.print_report:
                print_metrics_report(metrics)
    else:  # Single text input
        # Process single text input
        decision, explanation = process_single_text(
            args.input_text, model, tokenizer, prompt_builder, args
        )
        result = {
            "sample_id": "sample_1",
            "text": args.input_text,
            "explanation": explanation,
            "prediction": decision
        }
        output_path = os.path.join(args.output_dir, args.output_csv)
        pd.DataFrame([result]).to_csv(output_path, index=False)
        
        print("\n" + "="*50)
        print("MALNUTRITION DETECTION RESULT")
        print("="*50)
        print(f"\nPATIENT NOTES:\n{args.input_text[:500]}...")
        print(f"\nEXPLANATION:\n{explanation}")
        print(f"\nDECISION: malnutrition={decision}")
        print("="*50)
        print(f"Result saved to {output_path}")
    
    print("Inference complete!")


if __name__ == "__main__":
    main()