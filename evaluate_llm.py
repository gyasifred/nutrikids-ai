#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed inference script with proper JSON serialization and metric handling,
addressing the "all yes predictions" issue.
"""

import os
import re
import argparse
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel

def load_model(model_path, load_in_4bit):
    """Load model with proper device handling."""
    try:
        # Try Unsloth first
        print("Attempting to load model with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)
    except Exception as e:
        print(f"Error loading with Unsloth: {e}")
        print("Falling back to standard HuggingFace loading...")
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)

    # Add safety margin for token length
    effective_max_length = max_seq_length - 50  # Reserve 50 tokens for generation
    print(f"Model loaded with max sequence length: {max_seq_length}")
    print(f"Using effective max input length: {effective_max_length} (reserving 50 tokens for generation)")
    
    return model, tokenizer, effective_max_length

def generate_assessment(model, tokenizer, prompt, max_new_tokens=100, 
                      max_seq_length=None, temperature=0.2, top_p=0.95):
    """Generate assessment from model using chat-style generation.
    
    Modified to use slightly non-deterministic settings to avoid bias.
    """
    inputs = tokenizer([prompt], return_tensors="pt")
    
    # Handle sequence length constraints
    if max_seq_length and inputs.input_ids.shape[1] > max_seq_length - max_new_tokens:
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with modified settings to reduce deterministic bias
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # Increased from 50 to 100 to allow more complete reasoning
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Enable sampling to reduce deterministic bias
        num_return_sequences=1,
        repetition_penalty=1.1  # Add repetition penalty to avoid getting stuck in patterns
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def preprocess_clinical_note(note_text):
    """Preprocess clinical notes identically to training."""
    if not isinstance(note_text, str):
        return ""
    
    processed_text = note_text.replace('</s>', '\n\n')
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        processed_text = processed_text.replace(token, f"[{token}]")
    
    processed_text = processed_text.replace('\r\n', '\n').replace('\r', '\n')
    processed_text = ' '.join(processed_text.split())
    return processed_text.strip()

def create_balanced_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """Improved malnutrition assessment prompt with balanced language to avoid bias."""
    prompt = """Read the patient's notes carefully and objectively determine if the patient has malnutrition:

Consider these factors when assessing malnutrition:
- Anthropometric measurements like weight-for-height, BMI-for-age, height-for-age, MUAC
- Growth trajectory and percentile changes
- Clinical signs like edema, muscle wasting, decreased energy
- Nutritional intake pattern and history
- Medical conditions affecting nutrition
- Social or environmental factors impacting food security
- Recent weight changes or growth concerns

IMPORTANT: There is NO DEFAULT ANSWER. Evaluate objectively based only on evidence in the notes.

REQUIRED OUTPUT FORMAT:
1. First methodically analyze the available evidence in the clinical note
2. Identify what data points support OR contradict a malnutrition diagnosis
3. Explicitly state the z-scores or growth patterns if present
4. Conclude with exactly one of these formatted responses:
   malnutrition=yes (ONLY if clear evidence supports this diagnosis)
   OR
   malnutrition=no (if evidence is absent or insufficient for diagnosis)

CLINICAL NOTE FOR ANALYSIS:
{note}"""

    # Safety buffer to allow for maximum new tokens and avoid context overflow
    safe_max_tokens = max_tokens - 50 if max_tokens else None
    
    if tokenizer and safe_max_tokens:
        # Calculate available space for clinical note
        template = prompt.format(note="")
        template_tokens = tokenizer.encode(template)
        
        # Reserve space for the template plus a small buffer
        available_tokens = safe_max_tokens - len(template_tokens)
        
        if available_tokens <= 0:
            available_tokens = safe_max_tokens // 2
        
        # Check if the note needs truncation
        note_tokens = tokenizer.encode(note)
        if len(note_tokens) > available_tokens:
            # Truncate note to fit within available tokens
            truncated_note_tokens = note_tokens[:available_tokens]
            truncated_note = tokenizer.decode(truncated_note_tokens, skip_special_tokens=True)
            formatted_prompt = prompt.format(note=truncated_note)
            print(f"Note truncated from {len(note_tokens)} tokens to {len(truncated_note_tokens)} tokens")
        else:
            formatted_prompt = prompt.format(note=note)
    else:
        formatted_prompt = prompt.format(note=note)
    
    return formatted_prompt

def parse_model_output(output_text):
    """Improved output parsing with more specific pattern matching to avoid false positives."""
    output_text = output_text.lower().strip()
    
    # First try exact formatted matches (highly specific)
    if "malnutrition=yes" in output_text:
        return 1
    elif "malnutrition=no" in output_text:
        return 0
    
    # Search for specific patterns with boundaries to avoid partial matches
    lines = output_text.split('\n')
    for line in reversed(lines):
        line = line.strip().lower()
        
        # Check for conclusion statements with clear boundaries
        if any(kw in line for kw in ["conclusion:", "assessment:", "diagnosis:", "malnutrition:"]):
            # More specific pattern matching to avoid false positives
            if re.search(r'\byes\b|\bpositive\b|\bpresent\b|\bhas malnutrition\b', line):
                return 1
            elif re.search(r'\bno\b|\bnegative\b|\babsent\b|\bdoes not have malnutrition\b|\bno malnutrition\b', line):
                return 0
    
    # Look for specific final statements with word boundaries
    for line in reversed(lines[-5:]):  # Check more lines for context
        line = line.strip().lower()
        
        # Using word boundaries to avoid matching words like "yesterday" or "noyes"
        if re.search(r'\b(malnutrition\s*(is|:|=)\s*yes)\b', line) or re.search(r'\bpatient has malnutrition\b', line):
            return 1
        elif re.search(r'\b(malnutrition\s*(is|:|=)\s*no)\b', line) or re.search(r'\bpatient does not have malnutrition\b', line):
            return 0
    
    # If still undetermined, look for keywords in the last section
    last_section = " ".join(lines[-10:]).lower()
    
    # Count yes vs no indicators in the conclusion section
    yes_indicators = len(re.findall(r'\byes\b|\bpositive\b|\bpresent\b|\bconfirmed\b', last_section))
    no_indicators = len(re.findall(r'\bno\b|\bnegative\b|\babsent\b|\bruled out\b', last_section))
    
    if yes_indicators > no_indicators:
        return 1
    elif no_indicators > yes_indicators:
        return 0
    
    return -1  # Truly undetermined

def generate_predictions(model, tokenizer, data_path, max_seq_length, output_dir):
    """Generate predictions with proper device handling."""
    # Import regex at the top level for parsing
    import re
    
    df = pd.read_csv(data_path)
    required_columns = {"txt", "label", "DEID"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    results = []
    true_labels = []
    pred_labels = []
    deids = []
    
    print("\nStarting inference...")
    print("-" * 50)
    
    # Get device from model (respects accelerate offloading)
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # Calculate max tokens available for inputs (accounting for generation headroom)
    max_input_length = max_seq_length - 100  # Reserve 100 tokens for generation (increased)
    print(f"Using maximum input length of {max_input_length} tokens (from {max_seq_length} total)")
    
    # Enable debug mode for the first few examples
    debug_count = min(5, len(df))
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
        try:
            # Preprocess identically to training
            note_text = preprocess_clinical_note(row["txt"])
            
            # Create prompt with explicit max tokens to ensure we don't exceed limits
            prompt = create_balanced_malnutrition_prompt(
                note=note_text,
                tokenizer=tokenizer,
                max_tokens=max_input_length
            )
            
            # Double-check length and truncate if necessary
            if len(tokenizer.encode(prompt)) > max_input_length:
                print(f"Warning: Prompt for case {row['DEID']} still exceeds max length. Applying hard truncation.")
                tokens = tokenizer.encode(prompt)[:max_input_length]
                prompt = tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Generate assessment using the chat-style generation function
            output_text = generate_assessment(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=100,  # Increased
                max_seq_length=max_seq_length,
                temperature=0.2  # Slightly non-deterministic
            )
            
            # Run the improved parsing function
            pred = parse_model_output(output_text)
            true_label = int(row["label"])
            
            # Store results
            results.append({
                "DEID": row["DEID"],
                "TRUE_LABEL": true_label,
                "PREDICTED_LABEL": pred,
                "MODEL_OUTPUT": output_text,
                "INPUT_TEXT": note_text,
                "PROMPT": prompt
            })
            
            if pred != -1:  # Only include determinate predictions in metrics
                true_labels.append(true_label)
                pred_labels.append(pred)
                deids.append(row["DEID"])
            
            # Debug output for the first few examples
            if idx < debug_count:
                print(f"\n--- DEBUG: Case {idx + 1} ---")
                print(f"DEID: {row['DEID']}")
                print(f"TRUE: {'Malnutrition (1)' if true_label == 1 else 'No Malnutrition (0)'}")
                print(f"PRED: {'Malnutrition (1)' if pred == 1 else 'No Malnutrition (0)' if pred == 0 else 'Undetermined (-1)'}")
                print("\nModel output excerpt (last 500 chars):")
                print(output_text[-500:] if len(output_text) > 500 else output_text)
                print("-" * 40)
            else:
                # Regular output
                print(f"\nCase {idx + 1}/{len(df)} - DEID: {row['DEID']}")
                print(f"TRUE: {'Malnutrition (1)' if true_label == 1 else 'No Malnutrition (0)'}")
                print(f"PRED: {'Malnutrition (1)' if pred == 1 else 'No Malnutrition (0)' if pred == 0 else 'Undetermined (-1)'}")
        
        except Exception as e:
            print(f"\nError processing case {idx + 1} - DEID: {row['DEID']}")
            print(f"Error: {str(e)}")
            results.append({
                "DEID": row["DEID"],
                "TRUE_LABEL": int(row["label"]),
                "PREDICTED_LABEL": -1,
                "MODEL_OUTPUT": f"Error: {str(e)}",
                "INPUT_TEXT": note_text if 'note_text' in locals() else "",
                "PROMPT": prompt if 'prompt' in locals() else ""
            })
    
    # Calculate metrics with zero_division parameter to avoid warnings
    metrics = {}
    if true_labels:
        # Convert to Python native types for JSON serialization
        true_labels_native = [int(x) for x in true_labels]
        pred_labels_native = [int(x) for x in pred_labels]
        
        # Add prediction distribution analysis
        pred_distribution = {
            "positive_predictions": sum(1 for p in pred_labels_native if p == 1),
            "negative_predictions": sum(1 for p in pred_labels_native if p == 0),
            "positive_percentage": sum(1 for p in pred_labels_native if p == 1) / len(pred_labels_native) * 100 if pred_labels_native else 0,
            "negative_percentage": sum(1 for p in pred_labels_native if p == 0) / len(pred_labels_native) * 100 if pred_labels_native else 0
        }
        
        metrics = {
            "accuracy": float(accuracy_score(true_labels_native, pred_labels_native)),
            "f1": float(f1_score(true_labels_native, pred_labels_native, zero_division=0)),
            "recall": float(recall_score(true_labels_native, pred_labels_native, zero_division=0)),
            "roc_auc": float(roc_auc_score(true_labels_native, pred_labels_native)),
            "classification_report": classification_report(
                true_labels_native, 
                pred_labels_native, 
                output_dict=True,
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(true_labels_native, pred_labels_native).tolist(),
            "n_samples": len(true_labels_native),
            "n_indeterminate": len(df) - len(true_labels_native),
            "prediction_distribution": pred_distribution,
            "class_distribution": {
                "true_positives": int(sum((np.array(true_labels_native) == 1) & (np.array(pred_labels_native) == 1))),
                "true_negatives": int(sum((np.array(true_labels_native) == 0) & (np.array(pred_labels_native) == 0))),
                "false_positives": int(sum((np.array(true_labels_native) == 0) & (np.array(pred_labels_native) == 1))),
                "false_negatives": int(sum((np.array(true_labels_native) == 1) & (np.array(pred_labels_native) == 0))),
            }
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Full predictions
    full_path = os.path.join(output_dir, f"full_results_{timestamp}.csv")
    pd.DataFrame(results).to_csv(full_path, index=False)
    
    # 2. Metrics
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 3. Simplified (DEID, TRUE, PREDICTED)
    simple_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    pd.DataFrame({
        "DEID": deids,
        "TRUE_LABEL": true_labels,
        "PREDICTED_LABEL": pred_labels
    }).to_csv(simple_path, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("Inference Complete")
    print(f"Processed {len(df)} cases")
    if true_labels:
        print(f"\nMetrics (on {len(true_labels)} determinate predictions):")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"AUC: {metrics['roc_auc']:.4f}")
        
        print("\nPrediction Distribution:")
        print(f"Positive predictions: {metrics['prediction_distribution']['positive_predictions']} ({metrics['prediction_distribution']['positive_percentage']:.1f}%)")
        print(f"Negative predictions: {metrics['prediction_distribution']['negative_predictions']} ({metrics['prediction_distribution']['negative_percentage']:.1f}%)")
        
        print("\nConfusion Matrix:")
        print(f"TP: {metrics['class_distribution']['true_positives']}")
        print(f"TN: {metrics['class_distribution']['true_negatives']}")
        print(f"FP: {metrics['class_distribution']['false_positives']}")
        print(f"FN: {metrics['class_distribution']['false_negatives']}")
    print(f"\nResults saved to {output_dir}")
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description="Run inference with proper device handling")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    print("\nLoading model...")
    model, tokenizer, max_seq_length = load_model(args.model_path, args.load_in_4bit)
    
    print("\nGenerating predictions...")
    metrics, predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        max_seq_length=max_seq_length,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
