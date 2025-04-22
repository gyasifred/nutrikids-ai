#!/usr/bin/env python3
import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import unsloth  # Import unsloth first
# Set the environment variable for Unsloth logits before any imports if needed
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
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


def get_device(args):
    """
    Get the appropriate device, prioritizing GPU usage.

    Returns:
        torch.device: The device to use for model inference
    """
    if torch.cuda.is_available() and not args.force_cpu:
        # Use CUDA GPU
        selected_gpu = min(args.gpu_id, torch.cuda.device_count() - 1)
        device = torch.device("cuda", selected_gpu)
        print(f"GPU detected: {torch.cuda.get_device_name(selected_gpu)}")
        props = torch.cuda.get_device_properties(selected_gpu)
        print(f"GPU memory: {props.total_memory / 1024**3:.2f} GB")
        
        # Optimize CUDA operations
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark mode enabled for faster inference")
        
        return device
    else:
        print("No GPU detected or forced CPU, using CPU. This may significantly slow down inference.")
        return torch.device("cpu")


def determine_model_precision(args):
    """Determine appropriate precision settings for model inference."""
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

def load_model_and_tokenizer(base_model=None, model_path=None, args=None):
    """
    Load model and tokenizer for evaluation with appropriate quantization settings.
    
    - If base_model is provided: Loads the specified base model
    - If model_path is provided: Loads the adapter model (Unsloth handles base model internally)
    - If both are provided: Loads base_model first, then applies adapter weights from model_path

    Args:
        base_model (str, optional): Base model name or path
        model_path (str, optional): Path to fine-tuned model adapter weights
        args: Command line arguments with quantization settings

    Returns:
        tuple: (model, tokenizer)
    """
    # Check if at least one model source is provided
    if not base_model and not model_path:
        raise ValueError("Either base_model or model_path must be provided")

    # Get device
    device = get_device(args)
    print(f"Using device: {device}")
    
    # Determine precision based on hardware and user preferences
    fp16, bf16 = determine_model_precision(args)
    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"Using compute dtype: {dtype}")
    
    # Get appropriate quantization config
    quantization_config = get_quantization_config(args)
    if quantization_config:
        print(f"Quantization config: {quantization_config}")
    else:
        print("No quantization config created, using direct quantization flags")
    
    try:
        # Set attention implementation based on flash attention flag
        attn_implementation = "flash_attention_2" if args.use_flash_attention else "eager"
        
        # Set up common model loading kwargs
        common_kwargs = {
            "dtype": dtype,
            "device_map": "cuda" if torch.cuda.is_available() and not args.force_cpu else "auto",
            "attn_implementation": attn_implementation
        }
        
        # Add quantization settings
        if quantization_config is not None:
            common_kwargs["quantization_config"] = quantization_config
        else:
            # Direct quantization flags if no config is provided
            common_kwargs["load_in_4bit"] = args.load_in_4bit
            common_kwargs["load_in_8bit"] = args.load_in_8bit
        
        # Case 1: Only base_model is provided - load base model directly
        if base_model:
            print(f"Loading base model: {base_model}")
            print(f"Loading with kwargs: {common_kwargs}")
            model, tokenizer = FastLanguageModel.from_pretrained(base_model, **common_kwargs)
        
        # Case 2: Only model_path is provided - load adapter model directly
        if model_path and base_model:
            print(f"Loading adapter model: {model_path}")
            print(f"Loading with kwargs: {common_kwargs}")
            model, tokenizer = FastLanguageModel.from_pretrained(base_model, model_path, **common_kwargs)
        
        
        # Make sure max_seq_length is explicitly set in model config
        model_max_seq_length = get_model_max_length(model)
        print(f"Model's maximum sequence length: {model_max_seq_length}")
        
        if args.max_seq_length and args.max_seq_length < model_max_seq_length:
            print(f"Note: Using user-specified max_seq_length={args.max_seq_length}, "
                  f"which is less than model's native {model_max_seq_length}")
        
        # Enable native faster inference
        print("Enabling faster inference...")
        FastLanguageModel.for_inference(model)
        print("Model ready for inference")
        
        return model, tokenizer
    except Exception as e:
        print(f"Detailed error loading model: {str(e)}")
        import traceback
        traceback.print_exc()  # This will print the full stack trace
        raise
    
def get_model_max_length(model):
    """
    Get the maximum context length for the model.
    
    Args:
        model: The language model
        
    Returns:
        int: Maximum sequence length the model can handle
    """
    # Consistent handling of max sequence length
    if hasattr(model.config, 'max_position_embeddings'):
        return model.config.max_position_embeddings
    elif hasattr(model.config, 'max_seq_length'):
        return model.config.max_seq_length
    elif hasattr(model.config, 'context_length'):
        return model.config.context_length
    elif hasattr(model.config, 'max_length'):
        return model.config.max_length
    else:
        # Default fallback
        return 4096


def truncate_text(text, tokenizer, max_tokens, keep_beginning=True):
    """
    Truncate text to fit within max_tokens, with option to keep beginning or end.
    
    Args:
        text (str): The text to truncate
        tokenizer: The tokenizer
        max_tokens (int): Maximum number of tokens allowed
        keep_beginning (bool): Whether to keep the beginning (True) or end (False)
        
    Returns:
        str: Truncated text
    """
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Check if truncation is needed
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate to keep the beginning or end
    if keep_beginning:
        truncated_tokens = tokens[:max_tokens]
    else:
        truncated_tokens = tokens[-max_tokens:]
    
    # Decode back to text
    truncated_text = tokenizer.decode(truncated_tokens)
    
    return truncated_text


def process_batch(batch_texts, model, tokenizer, prompt_builder, args, device=None):
    """
    Process a batch of patient notes and extract predictions with confidence scores
    for malnutrition classification.
    """
    # Use provided device or get it if not provided (should be called only once)
    if device is None:
        device = get_device(args)
    
    # Get model's maximum sequence length
    model_max_length = get_model_max_length(model)
    actual_max_length = min(args.max_seq_length, model_max_length) if args.max_seq_length else model_max_length
    
    # Reserve tokens for generation and prompt formatting
    generation_reserve = min(args.max_length + 50, actual_max_length // 4)  # Reserve for generated response
    prompt_formatting_reserve = 256  # Reserve for prompt templates and instructions
    
    # Max tokens available for patient note
    max_note_tokens = actual_max_length - generation_reserve - prompt_formatting_reserve
    max_note_tokens = max(128, max_note_tokens)  # Ensure we have at least 128 tokens for the note
    
    batch_results = []
    
    for idx, text in enumerate(batch_texts):
        try:
            # Use sliding window approach consistent with training
            text_token_count = len(tokenizer.encode(text, add_special_tokens=False))
            
            # Truncate text if too long, before building prompt (keep beginning)
            if text_token_count > max_note_tokens:
                text = truncate_text(text, tokenizer, max_note_tokens, keep_beginning=True)
            
            # Dynamically adjust few-shot examples based on available context
            current_few_shot_count = args.few_shot_count
            
            while True:
                # Build prompt with current few-shot count
                if args.balanced_examples:
                    prompt = prompt_builder.get_balanced_inference_prompt(
                        patient_notes=text,
                        text_col=args.text_column,
                        label_col=args.label_column,
                        num_examples=current_few_shot_count
                    )
                else:
                    prompt = prompt_builder.get_inference_prompt(
                        patient_notes=text,
                        note_col=args.text_column,
                        label_col=args.label_column,
                        num_examples=current_few_shot_count
                    )
                
                # Check prompt length
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
                prompt_token_length = len(prompt_tokens)
                total_required_length = prompt_token_length + generation_reserve
                
                # If prompt fits, proceed
                if total_required_length <= actual_max_length:
                    break
                
                # If still too long, reduce few-shot examples
                if current_few_shot_count > 0:
                    current_few_shot_count -= 1
                    continue
                
                # If zero few-shot and still too long, truncate text further
                current_text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                new_max_tokens = max(64, current_text_tokens - (total_required_length - actual_max_length) - 50)
                
                if new_max_tokens < current_text_tokens:
                    text = truncate_text(text, tokenizer, new_max_tokens, keep_beginning=True)
                    continue
                
                # Emergency truncation of prompt if still too long
                prompt_template = "Analyze this patient note for malnutrition: "
                text_tokens = tokenizer.encode(text, add_special_tokens=False)
                available_tokens = actual_max_length - len(tokenizer.encode(prompt_template, add_special_tokens=True)) - generation_reserve
                available_tokens = max(64, available_tokens)  # Ensure we have at least some space
                truncated_text = truncate_text(text, tokenizer, available_tokens, keep_beginning=True)
                prompt = prompt_template + truncated_text
                break
            
            # Prepare chat format messages
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            # Final safety check - truncate if still too long
            if inputs.shape[1] > actual_max_length - generation_reserve:
                inputs = inputs[:, :actual_max_length - generation_reserve]
            
            # Move to device
            inputs = inputs.to(device)
            
            # Generate response
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs,
                    max_new_tokens=args.max_length,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True 
                )
            
            # Get the generated tokens and their scores
            sequences = output.sequences
            scores = output.scores
            
            # Decode the output - only the generated part
            input_length = inputs.shape[1]
            response_tokens = sequences[0][input_length:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Extract decision and explanation
            decision, explanation = extract_malnutrition_decision(response)
            
            # Convert text decision to binary
            binary_decision = 1 if decision.lower() == "yes" else 0
            
            # Improved confidence score calculation
            confidence_score = calculate_confidence_score(binary_decision, response, response_tokens, scores, tokenizer)
            
            # Print decision and explanation for monitoring
            decision_text = "Yes" if binary_decision == 1 else "No"
            explanation_preview = explanation[:100] + "..." if len(explanation) > 100 else explanation
            print(f"Patient {idx+1}: Decision: {decision_text}, Confidence: {confidence_score:.2f}, Explanation: {explanation_preview}")
            
            # Add result to batch results
            batch_results.append((binary_decision, explanation, confidence_score))
            
        except Exception as e:
            print(f"Error processing text {idx+1}: {str(e)}")
            # Include a reasonable fallback
            batch_results.append((0, f"Error: {str(e)}", 0.5))
    
    return batch_results


def calculate_confidence_score(binary_decision, response, response_tokens, scores, tokenizer):
    """
    Calculate confidence score for the prediction.
    
    Args:
        binary_decision: The binary decision (0 or 1)
        response: The full text response
        response_tokens: The tokens of the response
        scores: The scores from the model generation
        tokenizer: The tokenizer
        
    Returns:
        float: Confidence score between 0 and 1
    """
    # Start with a moderate confidence based on the model's decision
    base_confidence = 0.65 if binary_decision == 1 else 0.35
    
    # Look for decision indicators that would strengthen confidence
    response_lower = response.lower()
    confidence_modifiers = 0.0
    
    # Check for strong indicators in the response
    if binary_decision == 1:  # Yes to malnutrition
        if "definite" in response_lower or "clear evidence" in response_lower:
            confidence_modifiers += 0.2
        elif "likely" in response_lower or "probable" in response_lower:
            confidence_modifiers += 0.1
        elif "possible" in response_lower or "may" in response_lower:
            confidence_modifiers -= 0.1
    else:  # No to malnutrition
        if "no evidence" in response_lower or "normal nutritional" in response_lower:
            confidence_modifiers += 0.2
        elif "unlikely" in response_lower or "not likely" in response_lower:
            confidence_modifiers += 0.1
        elif "can't determine" in response_lower or "insufficient" in response_lower:
            confidence_modifiers -= 0.1
    
    # Use token probabilities for the decision section if we can identify it
    decision_keywords = ["malnutrition", "malnourished", "nutritional"]
    token_confidence = 0.0
    token_count = 0
    
    try:
        # Convert token IDs to tokens for analysis
        tokens = [tokenizer.decode([token_id.item()]) for token_id in response_tokens]
        
        # Find segments related to the decision
        for idx, token in enumerate(tokens):
            if any(keyword in token.lower() for keyword in decision_keywords) and idx < len(scores):
                # Get probability for this token from scores
                token_id = response_tokens[idx].item()
                token_prob = torch.softmax(scores[idx][0], dim=-1)[token_id].item()
                token_confidence += token_prob
                token_count += 1
        
        # If we found relevant tokens, use their average probability
        if token_count > 0:
            # Convert to a confidence score and scale it
            avg_token_confidence = token_confidence / token_count
            
            # Combine base confidence with modifiers and token confidence
            confidence_score = base_confidence + confidence_modifiers + (avg_token_confidence - 0.5) * 0.2
        else:
            confidence_score = base_confidence + confidence_modifiers
    except Exception as e:
        # If token confidence calculation fails, fall back to base+modifiers
        print(f"Warning: Token confidence calculation failed: {e}")
        confidence_score = base_confidence + confidence_modifiers
    
    # Ensure confidence is between 0.05 and 0.95
    confidence_score = max(0.05, min(0.95, confidence_score))
    
    return float(confidence_score)


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
    
    # Get device once for all batch processing
    device = get_device(args)
    print(f"Using device for all batches: {device}")
    
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
        
        # Process the batch if there are valid texts - pass device to avoid repeated detection
        if batch_texts:
            print(f"\n--- Processing batch {batch_idx+1}/{num_batches} ---")
            batch_results = process_batch(batch_texts, model, tokenizer, prompt_builder, args, device)
            
            # Store results
            for i in range(len(batch_texts)):
                try:
                    predicted_label, explanation, pred_score = batch_results[i]
                    
                    results.append({
                        "patient_id": batch_ids[i],
                        "true_label": batch_true_labels[i],
                        "predicted_label": predicted_label,
                        "prediction_score": pred_score,
                        "explanation": explanation,
                        "original_note": batch_texts[i][:500] + "..." if len(batch_texts[i]) > 500 else batch_texts[i]
                    })
                except Exception as e:
                    print(f"Error processing result for item {i} in batch {batch_idx}: {e}")
                    # Add error record
                    results.append({
                        "patient_id": batch_ids[i],
                        "true_label": batch_true_labels[i],
                        "predicted_label": -1,
                        "prediction_score": -1,
                        "explanation": f"Error: {str(e)}",
                        "original_note": batch_texts[i][:500] + "..." if len(batch_texts[i]) > 500 else batch_texts[i]
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter out error rows for evaluation
    eval_df = results_df[results_df["predicted_label"] != -1].copy()
    
    if len(eval_df) == 0:
        print("Warning: No valid results to evaluate!")
    elif len(eval_df) < len(df):
        print(f"Warning: Only {len(eval_df)} out of {len(df)} records were processed successfully.")
    
    # Evaluate model performance
    if len(eval_df) > 0:
        y_true = eval_df["true_label"].tolist()
        y_pred = eval_df["predicted_label"].tolist()
        y_scores = eval_df["prediction_score"].tolist()  # Get probability scores
        
        print("Computing evaluation metrics...")
        try:
            metrics = evaluate_predictions(y_true, y_pred, y_scores)  # Pass scores
            # Add metrics to results_df as attributes or save separately as needed
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, str, bool, np.number)) or metric_value is None:
                    results_df.attrs[metric_name] = metric_value
        except Exception as e:
            print(f"Error computing evaluation metrics: {e}")
    
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
    parser.add_argument("--base_model", type=str, default=None,
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
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum number of tokens to generate (default: 200)")
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

    # Maximum sequence length the model can handle
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Maximum sequence length the model can handle (default: 4096)")

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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration for reproducibility
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

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
    
    # Include prediction scores if available
    if "prediction_score" in eval_df.columns:
        y_scores = eval_df["prediction_score"].tolist()
        print("Computing evaluation metrics with confidence scores...")
        try:
            # Pass scores for ROC and PR curve calculations
            metrics = evaluate_predictions(y_true, y_pred, y_scores)
        except Exception as e:
            print(f"Error computing evaluation metrics with scores: {e}")
            # Fallback to basic metrics without scores
            metrics = evaluate_predictions(y_true, y_pred)
    else:
        print("Computing basic evaluation metrics...")
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

    # Save metrics to JSON - Remove probability-based metrics that can't be serialized
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'confusion_matrix': metrics['confusion_matrix'].tolist() if hasattr(metrics['confusion_matrix'], 'tolist') else metrics['confusion_matrix'],
        'classification_report': metrics['classification_report']
    }
    
    # Add ROC AUC and PR AUC if available
    if 'roc_auc' in metrics:
        metrics_json['roc_auc'] = float(metrics['roc_auc'])
    if 'pr_auc' in metrics:
        metrics_json['pr_auc'] = float(metrics['pr_auc'])

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
