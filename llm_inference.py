#!/usr/bin/env python3
from models.llm_models import (
    MalnutritionPromptBuilder,
    extract_malnutrition_decision,
    is_bfloat16_supported,
    set_seed
)
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig, TextStreamer
import pandas as pd
import argparse
import torch
import os
import unsloth
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a malnutrition detection model (base or fine-tuned)"
    )

    # Model arguments
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the fine-tuned model adapter weights (optional). "
                             "If not provided, the base model is used for inference.")
    parser.add_argument("--base_model", type=str, default="unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit",
                        help="Base model that was fine-tuned or to be used for inference")

    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_csv", type=str,
                       help="Path to CSV file with patient notes")
    group.add_argument("--input_text", type=str,
                       help="Single patient note as text string")

    parser.add_argument("--text_column", type=str, default="txt",
                        help="Column name in CSV containing patient notes")
    parser.add_argument("--id_column", type=str, default="DEID",
                        help="Column name in CSV containing sample IDs")
    parser.add_argument("--label_column", type=str, default='Label',
                        help="Column name in CSV containing true labels (optional)")

    # Few-shot settings
    parser.add_argument("--examples_data", type=str, default=None,
                        help="Path to few-shot examples CSV data (optional)")
    parser.add_argument("--few_shot_count", type=int, default=0,
                        help="Number of few-shot examples to use (default: 0 for zero-shot)")
    parser.add_argument("--balanced_examples", action="store_true",
                        help="Whether to balance positive/negative few-shot examples")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./llm_inference",
                        help="Directory to save inference results")
    parser.add_argument("--output_csv", type=str, default="predictions.csv",
                        help="Name of output CSV file")
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
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for sampling")
    parser.add_argument("--min_p", type=float, default=0.0,
                        help="Min-P sampling parameter (optional)")
    parser.add_argument("--stream_output", action="store_true",
                        help="Stream model output to console during generation")
    parser.add_argument("--force_fp16", action="store_true", default=False,
                        help="Force FP16 precision even if BF16 is supported")
    parser.add_argument("--force_cpu", action="store_true", default=False,
                        help="Force CPU usage even if GPU is available")

    args = parser.parse_args()

    if args.load_in_8bit:
        args.load_in_4bit = False

    return args


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
            "use_cache": True  
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
    text_col = args.text_column if hasattr(args, 'text_column') else "txt"
    label_col = args.label_column if hasattr(args, 'label_column') else "label"

    prompt = prompt_builder.get_inference_prompt(
        patient_notes=text,
        note_col=text_col,
        label_col=label_col,
        num_examples=args.few_shot_count,
        specific_example_indices=None,
        balanced=args.balanced_examples
    )

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
    ).to(model.device)

    # Set up text streamer if requested
    if args.stream_output:
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        streamer = text_streamer
        print("\n--- Streaming model output ---")
        print("PROMPT:", prompt[:100], "...\n")
        print("RESPONSE:")
    else:
        streamer = None

    # Generate with properly formatted inputs
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p if args.min_p > 0 else None,
            do_sample=args.temperature > 0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the output - make sure to get only the generated part
    input_length = inputs.shape[1]
    response_tokens = output[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)

    if args.stream_output:
        print("\n--- End of streaming output ---\n")

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
        raise ValueError(
            f"Text column '{args.text_column}' not found in CSV file")

    if args.id_column not in df.columns:
        df[args.id_column] = [f"sample_{i}" for i in range(len(df))]

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = row[args.text_column]
        patient_id = row[args.id_column]

        true_label = None
        if args.label_column and args.label_column in df.columns:
            true_label = str(row[args.label_column]).lower()
            true_label = "yes" if true_label in ["1", "yes", "true"] else "no"

        predicted_label, explanation = process_single_text(text, model, tokenizer,
                                                          prompt_builder, args)

        result = {
            "patient_id": patient_id,
            "explanation": explanation,
            "predicted_label": predicted_label,
        }

        if true_label is not None:
            result["true_label"] = true_label

        results.append(result)

    results_df = pd.DataFrame(results)

    return results_df


def main():
    """Main function to run the inference pipeline."""
    args = parse_arguments()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize prompt builder
    prompt_builder = MalnutritionPromptBuilder(args.examples_data)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.model_path, args
    )

    if args.input_csv:
        # Process CSV input
        results_df = process_csv_input(args, model, tokenizer, prompt_builder)
        output_path = os.path.join(args.output_dir, args.output_csv)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        # Show a sample of results
        if not results_df.empty:
            sample = results_df.iloc[0]
            print("\n" + "="*50)
            print("SAMPLE RESULT:")
            print("="*50)
            print(f"PATIENT ID: {sample['patient_id']}")
            print(f"EXPLANATION: {sample['explanation']}")
            print(f"DECISION: malnutrition={sample['predicted_label']}")
            print("="*50)
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