#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference on clinical notes using the trained malnutrition assessment model.
Simplified version for classification only.
"""

import os
import pandas as pd
import numpy as np
import torch
import argparse
import time
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from transformers import TextStreamer
from sklearn.metrics import classification_report, confusion_matrix
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference for pediatric malnutrition classification")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the CSV file with clinical notes")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save all results and metrics")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=256, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--load_in_4bit", action="store_true", 
                        help="Load model in 4-bit quantization")
    parser.add_argument("--stream_output", action="store_true", 
                        help="Stream output tokens during inference")
    parser.add_argument("--test_mode", action="store_true", 
                        help="Test mode: infer on a small subset of data")
    parser.add_argument("--note_column", type=str, default="txt", 
                        help="Column name for clinical notes")
    parser.add_argument("--id_column", type=str, default="DEID", 
                        help="Column name for patient IDs")
    parser.add_argument("--label_column", type=str, default="label", 
                        help="Column name for true labels (if available)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature for sampling during generation (default: 0.1)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling parameter (default: 0.95)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Penalty for token repetition (default: 1.0, higher = less repetition)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2,
                        help="Size of n-grams that shouldn't be repeated (default: 3)")
    parser.add_argument("--preprocess_tokens", action="store_true", 
                        help="Preprocess </s> tokens in clinical notes")
    return parser.parse_args()


def preprocess_clinical_note(note_text):
    """
    Preprocess clinical notes to handle special tokens like </s> that might interfere with model training.
    """
    if not note_text:
        return note_text

    # Replace '</s>' tokens with a more appropriate separator
    processed_text = note_text.replace('</s>', '\n\n')

    # Check for any remaining special tokens that might interfere
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")  # Escape them

    return processed_text


def create_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """Create improved malnutrition assessment prompt with clearer output format and structured analysis."""
    base_prompt = """[Role] Read the patient's clinical note and determine if the patient has malnutrition based on the criteria below.

MALNUTRITION CRITERIA TABLE:
### Mild Malnutrition
- Weight-for-height: −1 to −1.9 z score
- BMI-for-age: −1 to −1.9 z score
- Length/height-for-age: No specific criteria
- Mid–upper arm circumference (MUAC): Greater than or equal to −1 to −1.9 z score

### Moderate Malnutrition
- Weight-for-height: −2 to −2.9 z score
- BMI-for-age: −2 to −2.9 z score
- Length/height-for-age: No specific criteria
- Mid–upper arm circumference (MUAC): Greater than or equal to −2 to −2.9 z score

### Severe Malnutrition
- Weight-for-height: −3 or lower z score
- BMI-for-age: −3 or lower z score
- Length/height-for-age: −3 or lower z score
- Mid–upper arm circumference (MUAC): Greater than or equal to −3 z score

[Instructions]
1. Carefully analyze the clinical note to find ANY evidence of z-scores, BMI, weight-for-height, length/height-for-age, or MUAC measurements.
2. Compare ANY found measurements directly against the criteria table above.
3. A patient is malnourished if they meet ANY ONE of the criteria for mild, moderate, or severe malnutrition.
4. If NO anthropometric measurements are found OR if measurements are all within normal range, the patient is NOT malnourished.
5. Your response must be strictly "YES" or "NO" only, with no additional explanation.

[Output]
Is the patient malnourished? 
Answer with ONLY "YES" or "NO". Do not provide any explanation.

Clinical note for analysis:
"""

    # Token-aware note truncation
    if tokenizer and max_tokens:
        # Calculate token budgets
        base_tokens = len(tokenizer.encode(base_prompt))
        safety_margin = 128  # For generation space
        max_note_tokens = max_tokens - base_tokens - safety_margin

        if max_note_tokens > 0:
            note_tokens = tokenizer.encode(note)
            if len(note_tokens) > max_note_tokens:
                note = tokenizer.decode(note_tokens[:max_note_tokens])
                print(f"Truncated note from {len(note_tokens)} to {max_note_tokens} tokens")

    return base_prompt + note + "\n\n[Output]\nIs the patient malnourished? "


def extract_yes_no(output_text):
    """
    Extract the yes/no classification from the model's output with improved pattern matching.
    Prioritizes extracting the final YES/NO answer, which is what the prompt specifically asks for.
    """
    if not output_text:
        return "error"
    
    # Convert to lowercase for case-insensitive matching
    output_lower = output_text.lower().strip()
    
    # First, check if the text ends with a clear yes/no response (highest priority)
    # This matches patterns like "Answer: YES" or just "YES" at the end
    ending_match = re.search(r'(?:^|.*\W)(yes|no)[\s\.\,]*$', output_lower)
    if ending_match:
        return ending_match.group(1)
    
    # Look for standalone "yes" or "no" surrounded by whitespace or punctuation
    standalone_pattern = r'(?:^|\W)(yes|no)(?:$|\W)'
    standalone_matches = re.findall(standalone_pattern, output_lower)
    
    if standalone_matches:
        # Get the last standalone match (most likely to be the final answer)
        return standalone_matches[-1]
    
    # Context-specific phrases with more precise matching
    yes_phrases = [
        r"patient\s+is\s+malnourished", 
        r"has\s+malnutrition",
        r"malnutrition:\s*yes",
        r"assessment:\s*yes",
        r"conclusion:\s*yes"
    ]
    
    no_phrases = [
        r"patient\s+is\s+not\s+malnourished",
        r"no\s+malnutrition",
        r"does\s+not\s+have\s+malnutrition",
        r"malnutrition:\s*no",
        r"assessment:\s*no",
        r"conclusion:\s*no"
    ]
    
    # Check for precise yes phrases
    for pattern in yes_phrases:
        if re.search(pattern, output_lower):
            return "yes"
    
    # Check for precise no phrases
    for pattern in no_phrases:
        if re.search(pattern, output_lower):
            return "no"
    
    # If still undetermined, look for the basic "yes" or "no" as a last resort
    # This is less reliable but better than returning "error" if nothing else works
    if "yes" in output_lower and "no" not in output_lower:
        return "yes"
    elif "no" in output_lower and "yes" not in output_lower:
        return "no"
    
    # If we can't determine clearly
    return "error"


def load_model(model_path, load_in_4bit):
    """Load the fine-tuned model with native sequence length."""
    try:
        # Try loading with Unsloth first
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            dtype=None,  # Auto detect
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)  # Enable faster inference

        # Get model's native max sequence length
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)
        print(f"Using model's native max sequence length: {max_seq_length}")

    except Exception as e:
        print(f"Error loading with Unsloth: {e}")
        print("Trying to load with Hugging Face PEFT...")
        # Fallback to Hugging Face PEFT
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Get model's native max sequence length
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)
        print(f"Using model's native max sequence length: {max_seq_length}")

    return model, tokenizer, max_seq_length


def generate_text(model, tokenizer, prompt, max_new_tokens=100, streamer=None, max_seq_length=None, temperature=0.1, top_p=0.95, repetition_penalty=1.0, no_repeat_ngram_size=2):
    """Generate text from the model with simplified parameters."""
    # Tokenize the prompt
    inputs = tokenizer([prompt], return_tensors="pt")

    # Check if the input is too long
    if max_seq_length and inputs.input_ids.shape[1] > max_seq_length - max_new_tokens:
        print(f"Warning: Input length {inputs.input_ids.shape[1]} exceeds maximum allowed when combined with max_new_tokens.")
        # Truncate to leave room for generated tokens
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
        print(f"Input was truncated to {inputs.input_ids.shape[1]} tokens to fit within model context")

    # Move input to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate output with simplified parameters
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }

    if streamer:
        generation_kwargs["streamer"] = streamer

    outputs = model.generate(**generation_kwargs)

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return prediction


def run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length):
    """Run inference on the provided clinical notes."""
    results = []

    # Create a text streamer if streaming is enabled
    streamer = TextStreamer(tokenizer) if args.stream_output else None

    for i in tqdm(range(0, len(notes), args.batch_size), desc="Running inference"):
        batch_notes = notes[i:i+args.batch_size]
        batch_ids = patient_ids[i:i+args.batch_size]
        batch_labels = true_labels[i:i+args.batch_size] if true_labels else [None] * len(batch_notes)

        for note, patient_id, true_label in zip(batch_notes, batch_ids, batch_labels):
            # Preprocess the clinical note if enabled
            if args.preprocess_tokens:
                note = preprocess_clinical_note(note)

            # Create prompt for the current note with length control
            prompt = create_malnutrition_prompt(
                note, 
                tokenizer=tokenizer, 
                max_tokens=max_seq_length - args.max_new_tokens
            )

            # Generate prediction
            start_time = time.time()

            if args.stream_output:
                print(f"\nGenerating assessment for patient {patient_id}...")
                prediction = generate_text(
                    model, 
                    tokenizer, 
                    prompt, 
                    args.max_new_tokens, 
                    streamer,
                    max_seq_length=max_seq_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size
                )
            else:
                prediction = generate_text(
                    model, 
                    tokenizer, 
                    prompt, 
                    args.max_new_tokens,
                    max_seq_length=max_seq_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size
                )

            inference_time = time.time() - start_time

            # Extract the output part from the prediction
            output_part = prediction.split("[Output]")[-1].strip() if "[Output]" in prediction else prediction
            
            # Extract yes/no using improved logic
            assessment = extract_yes_no(output_part)

            # Print the true label and predicted label to terminal
            true_label_str = str(true_label) if true_label is not None else "unknown"
            print(f"Patient ID: {patient_id} | True label: {true_label_str} | Predicted label: {assessment}")

            result = {
                "DEID": patient_id,
                "true_label": true_label if true_label is not None else "unknown",
                "predicted_label": assessment,
                "full_output": output_part,
                "original_note": note,
                "inference_time": inference_time
            }

            results.append(result)

    return results


def calculate_metrics(results, args):
    """Calculate and save performance metrics."""
    # Create metrics directory if it doesn't exist
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Update args to include metrics_dir for use elsewhere
    args.metrics_dir = metrics_dir

    # Check if we have true labels to calculate metrics
    if all(r["true_label"] == "unknown" for r in results):
        print("No true labels available - skipping metrics calculation")
        return None

    # Filter out results with unknown true labels or error predictions
    valid_results = [r for r in results if r["true_label"] != "unknown" and r["predicted_label"] != "error"]

    if not valid_results:
        print("No valid results with true labels for metrics calculation")
        return None

    # Convert labels to binary format for metrics calculation
    # Check if true_label is already numeric
    if all(isinstance(r["true_label"], (int, float)) for r in valid_results):
        y_true = [int(r["true_label"]) for r in valid_results]
    else:
        # If string, convert using string comparison
        y_true = [1 if str(r["true_label"]).lower() == "yes" else 0 for r in valid_results]

    # For predicted labels (these are strings from model output)
    y_pred = [1 if r["predicted_label"].lower() == "yes" else 0 for r in valid_results]

    # Save raw prediction data for future analysis
    metrics_df = pd.DataFrame({
        "DEID": [r["DEID"] for r in valid_results],
        "true_label": [r["true_label"] for r in valid_results],
        "predicted_label": [r["predicted_label"] for r in valid_results],
        "full_output": [r["full_output"] for r in valid_results],
        "true_binary": y_true,
        "pred_binary": y_pred
    })
    metrics_df.to_csv(os.path.join(metrics_dir, "raw_predictions.csv"), index=False)

    # Classification Report based on binary predictions
    report = classification_report(y_true, y_pred, output_dict=True, target_names=["Non-malnourished", "Malnourished"])
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(metrics_dir, "classification_report.csv"))

    # Save classification report as text
    with open(os.path.join(metrics_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["Non-malnourished", "Malnourished"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm, 
        index=["Actual Non-malnourished", "Actual Malnourished"],
        columns=["Predicted Non-malnourished", "Predicted Malnourished"]
    )
    cm_df.to_csv(os.path.join(metrics_dir, "confusion_matrix.csv"))

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    plt.matshow(cm, cmap=plt.cm.Blues, fignum=1)
    plt.colorbar()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")

    # Save metrics summary
    metrics_summary = {
        "accuracy": report["accuracy"],
        "precision_malnourished": report["Malnourished"]["precision"],
        "recall_malnourished": report["Malnourished"]["recall"],
        "f1_malnourished": report["Malnourished"]["f1-score"],
        "precision_nonmalnourished": report["Non-malnourished"]["precision"],
        "recall_nonmalnourished": report["Non-malnourished"]["recall"],
        "f1_nonmalnourished": report["Non-malnourished"]["f1-score"],
        "total_samples": len(valid_results),
        "malnourished_count": sum(y_true),
        "nonmalnourished_count": len(y_true) - sum(y_true)
    }

    with open(os.path.join(metrics_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


def main():
    args = parse_arguments()

    # Create output directory and subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "inference_results.csv")

    # Load the data
    print(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)

    # Check for required columns
    if args.note_column not in df.columns:
        raise ValueError(f"Note column '{args.note_column}' not found in the input file")
    if args.id_column not in df.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in the input file")

    # Extract notes and patient IDs
    notes = df[args.note_column].tolist()
    patient_ids = df[args.id_column].tolist()

    # Extract true labels if available
    true_labels = None
    if args.label_column and args.label_column in df.columns:
        print(f"Found label column '{args.label_column}' - metrics will be calculated")
        true_labels = df[args.label_column].tolist()
        # Print the first few labels to debug type issues
        print(f"First 5 labels (type: {type(true_labels[0]).__name__}): {true_labels[:5]}")
    else:
        print("No label column specified or not found - metrics will not be calculated")

    # For test mode, use only a small subset
    if args.test_mode:
        print("Running in test mode with 5 samples")
        test_size = min(5, len(notes))
        notes = notes[:test_size]
        patient_ids = patient_ids[:test_size]
        if true_labels:
            true_labels = true_labels[:test_size]

    # Load the model with native sequence length
    print(f"Loading model from: {args.model_path}")
    model, tokenizer, max_seq_length = load_model(args.model_path, args.load_in_4bit)
    print(f"Model loaded with native max sequence length: {max_seq_length}")

    # Run inference
    print("Starting inference...")
    if args.preprocess_tokens:
        print("Note preprocessing enabled - </s> tokens will be handled appropriately")
    results = run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    # Calculate and save metrics if true labels are available
    if true_labels:
        print("Calculating performance metrics...")
        metrics = calculate_metrics(results, args)
        if metrics:
            print("\nPerformance Metrics Summary:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Malnourished cases - Precision: {metrics['precision_malnourished']:.4f}, Recall: {metrics['recall_malnourished']:.4f}, F1: {metrics['f1_malnourished']:.4f}")
            print(f"Non-malnourished cases - Precision: {metrics['precision_nonmalnourished']:.4f}, Recall: {metrics['recall_nonmalnourished']:.4f}, F1: {metrics['f1_nonmalnourished']:.4f}")
            print(f"Metrics saved to: {args.metrics_dir}/")

    # Show summary statistics
    yes_count = sum(1 for r in results if r["predicted_label"] == "yes")
    no_count = sum(1 for r in results if r["predicted_label"] == "no")
    error_count = sum(1 for r in results if r["predicted_label"] == "error")

    print("\nInference Summary:")
    print(f"Total cases processed: {len(results)}")
    print(f"Malnourished cases: {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"Non-malnourished cases: {no_count} ({no_count/len(results)*100:.1f}%)")
    if error_count > 0:
        print(f"Error cases: {error_count} ({error_count/len(results)*100:.1f}%)")

    # Calculate average inference time
    avg_time = sum(r["inference_time"] for r in results) / len(results)
    print(f"Average inference time per note: {avg_time:.3f} seconds")


if __name__ == "__main__":
    main()
