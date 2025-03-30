import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig

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


def get_quantization_config(args):
    """Define quantization configuration for the model based on arguments."""
    # Determine compute dtype based on hardware and user preferences
    compute_dtype = torch.bfloat16 if is_bfloat16_supported(
    ) and not args.force_fp16 else torch.float16

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


def get_device():
    """
    Get the appropriate device, prioritizing GPU usage.

    Returns:
        torch.device: The device to use for model inference
    """
    if torch.cuda.is_available():
        # Use CUDA GPU
        device = torch.device("cuda")
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return device
    else:
        print("No GPU detected, falling back to CPU. This may significantly slow down inference.")
        return torch.device("cpu")


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
    # Get device
    device = get_device()
    print(f"Using device: {device}")

    print(
        f"Loading {'fine-tuned' if model_path else 'base'} model: {base_model}")
    # Determine compute dtype based on hardware and preferences
    compute_dtype = torch.bfloat16 if is_bfloat16_supported(
    ) and not args.force_fp16 else torch.float16
    # Get appropriate quantization config
    quantization_config = get_quantization_config(args)
    try:
        # Set up model loading kwargs with common parameters
        model_kwargs = {
            # Force GPU usage if available instead of "auto" mapping
            "device_map": "cuda" if torch.cuda.is_available() and not args.force_cpu else "auto",
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
            model, tokenizer = FastLanguageModel.from_pretrained(
                **model_kwargs)
            print(
                f"Model loaded successfully with adapter weights from {model_path}")
        else:
            # Load base model only
            model_kwargs["model_name"] = base_model
            model, tokenizer = FastLanguageModel.from_pretrained(
                **model_kwargs)
            print(f"Base model loaded successfully: {base_model}")
        # Enable native 2x faster inference
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def process_batch(batch_texts, model, tokenizer, prompt_builder, args):
    """
    Process a batch of patient notes and extract predictions based on text response.
    
    Args:
        batch_texts (list): List of patient note texts
        model: The language model
        tokenizer: The tokenizer
        prompt_builder: MalnutritionPromptBuilder instance
        args: Command line arguments
        
    Returns:
        list: List of tuples (decision, explanation) for each input text
    """
    # Explicitly use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() and not args.force_cpu else torch.device("cpu")
    
    batch_results = []
    
    for text in batch_texts:
        try:
            # Prepare prompt with few-shot examples if specified
            prompt = prompt_builder.get_balanced_inference_prompt(
                patient_notes=text,
                text_col=args.text_column,
                label_col=args.label_column,
                num_examples=args.few_shot_count
            ) if args.balanced_examples else prompt_builder.get_inference_prompt(
                patient_notes=text,
                note_col=args.text_column,
                label_col=args.label_column,
                num_examples=args.few_shot_count
            )
            
            # Prepare chat format messages
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template to get input tokens and move to device
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            
            # Generate the full response
            with torch.no_grad():
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
            
            # Extract decision and explanation from text response
            decision, explanation = extract_malnutrition_decision(response)
            
            # Convert text decision to binary (1/0), keeping consistent with extract_malnutrition_decision
            binary_decision = 1 if decision.lower() == "yes" else 0
            
            batch_results.append((binary_decision, explanation))
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Return default values in case of error
            batch_results.append((0, f"Error processing text: {str(e)}"))
    
    return batch_results


def evaluate_model(args, model, tokenizer, prompt_builder):
    """
    Evaluate model on test dataset using batch processing.
    
    Args:
        args: Command line arguments
        model: The language model
        tokenizer: The tokenizer
        prompt_builder: MalnutritionPromptBuilder instance
        
    Returns:
        DataFrame: Results with predictions
    """
    print(f"Loading test data from {args.test_csv}")
    # Load test data
    df = pd.read_csv(args.test_csv)
    
    # Handle ID column
    if args.id_column not in df.columns:
        print(f"ID column '{args.id_column}' not found, creating it.")
        df[args.id_column] = [f"patient_{i}" for i in range(len(df))]
    
    # Validate required columns exist
    required_columns = [args.text_column, args.label_column, args.id_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Required columns {missing_columns} not found in test CSV")
    
    # Initialize results storage
    results = []
    
    # Prepare batches for processing
    batch_size = min(args.batch_size, len(df))  # Ensure batch size doesn't exceed dataset size
    print(f"Processing dataset in batches of {batch_size}")
    
    # Create batches
    num_batches = (len(df) + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        
        batch_df = df.iloc[start_idx:end_idx]
        
        # Get texts and other information for the batch
        batch_texts = []
        batch_ids = []
        batch_true_labels = []
        
        for _, row in batch_df.iterrows():
            patient_text = row[args.text_column]
            patient_id = row[args.id_column]
            
            # Handle missing text
            if pd.isna(patient_text) or patient_text == "":
                print(f"Warning: Empty text for patient {patient_id}, skipping.")
                continue
                
            # Convert true label to binary (1/0)
            true_label_str = str(row[args.label_column]).lower()
            true_label = 1 if true_label_str in ["1", "yes", "true"] else 0
            
            batch_texts.append(patient_text)
            batch_ids.append(patient_id)
            batch_true_labels.append(true_label)
        
        # Process the batch if there are valid texts
        if batch_texts:
            batch_results = process_batch(batch_texts, model, tokenizer, prompt_builder, args)
            
            # Store results
            for i in range(len(batch_texts)):
                try:
                    predicted_label, explanation = batch_results[i]
                    
                    results.append({
                        "patient_id": batch_ids[i],
                        "true_label": batch_true_labels[i],
                        "predicted_label": predicted_label,
                        "explanation": explanation
                    })
                except Exception as e:
                    print(f"Error processing result for item {i} in batch {batch_idx}: {e}")
                    # Add error record
                    results.append({
                        "patient_id": batch_ids[i],
                        "true_label": batch_true_labels[i],
                        "predicted_label": -1,
                        "explanation": f"Error: {str(e)}"
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out error rows for evaluation
    eval_df = results_df[results_df["predicted_label"] != -1].copy()
    
    if len(eval_df) == 0:
        print("Warning: No valid results to evaluate!")
    elif len(eval_df) < len(df):
        print(f"Warning: Only {len(eval_df)} out of {len(df)} records were processed successfully.")
    
    return results_df


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
                        help="Batch size for processing (default: 16)")

    # Force precision if needed
    parser.add_argument("--force_fp16", action="store_true",
                        help="Force using FP16 precision")
    parser.add_argument("--force_bf16", action="store_true",
                        help="Force using BF16 precision if supported")

    # Device arguments
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use if multiple GPUs are available (default: 0)")

    args = parser.parse_args()

    if args.load_in_8bit:
        args.load_in_4bit = False
    
    if args.force_fp16 and args.force_bf16:
        print("Warning: Both --force_fp16 and --force_bf16 specified. Using --force_fp16.")
        args.force_bf16 = False

    return args


def main():
    """Main function to run the evaluation pipeline."""
    args = parse_arguments()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Print CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and not args.force_cpu:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  - CUDA Capability: {props.major}.{props.minor}")

        # Set specific GPU to be the current device
        selected_gpu = min(args.gpu_id, torch.cuda.device_count() - 1)
        torch.cuda.set_device(selected_gpu)
        print(
            f"Current CUDA device set to: {selected_gpu} ({torch.cuda.get_device_name(selected_gpu)})")

        # Clear GPU cache to maximize available memory
        torch.cuda.empty_cache()
        print("GPU cache cleared")

        # Verify GPU is being used
        device = torch.device("cuda", selected_gpu)
        print(f"Using device: {device}")

        # Optimize CUDA operations - set these for maximum performance
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark mode enabled for faster training")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print("Note: GPU is available but not being used due to --force_cpu flag")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize prompt builder
    try:
        prompt_builder = MalnutritionPromptBuilder(args.examples_data)
    except Exception as e:
        print(f"Error initializing prompt builder: {e}")
        raise

    # Load model and tokenizer with appropriate quantization settings
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.model_path, args
        )
    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        raise

    # Evaluate model on test data
    try:
        results_df = evaluate_model(args, model, tokenizer, prompt_builder)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        raise

    # Save predictions to CSV
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # Filter out error rows for evaluation
    eval_df = results_df[results_df["predicted_label"] != -1].copy()
    
    if len(eval_df) == 0:
        print("ERROR: No valid results to evaluate! Check logs for details.")
        return
    
    # Evaluate model performance
    y_true = eval_df["true_label"].tolist()
    y_pred = eval_df["predicted_label"].tolist()

    print("Computing evaluation metrics...")
    try:
        # Only pass true and predicted labels
        metrics = evaluate_predictions(y_true, y_pred)
    except Exception as e:
        print(f"Error computing evaluation metrics: {e}")
        raise

    # Generate and save plots
    print("Generating evaluation plots...")
    try:
        plot_evaluation_metrics(metrics, args.output_dir)
    except Exception as e:
        print(f"Error generating evaluation plots: {e}")
        # Continue even if plotting fails

    # Save metrics to CSV
    try:
        metrics_csv_path = os.path.join(args.output_dir, "metrics.csv")
        save_metrics_to_csv(metrics, metrics_csv_path)
        print(f"Metrics saved to {metrics_csv_path}")
    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")

    # Save metrics to JSON - Remove probability-based metrics
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'confusion_matrix': metrics['confusion_matrix'],
        'classification_report': metrics['classification_report']
    }

    try:
        metrics_json_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"Detailed metrics saved to {metrics_json_path}")
    except Exception as e:
        print(f"Error saving metrics to JSON: {e}")

    # Print evaluation report
    if args.print_report:
        print_metrics_report(metrics)
    
    print("Evaluation complete!")



if __name__ == "__main__":
    main()
