#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized inference script with proper Unsloth integration and robust output handling.
"""

import os
import gc
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
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_model(model_path, load_in_4bit):
    """Load model with optimized approach based on model type."""
    has_adapters = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    try:
        # Prioritize Unsloth for faster inference
        from unsloth import FastLanguageModel
        print("Loading model with Unsloth for 2x faster inference...")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            dtype=None,  # Auto-detect optimal dtype
            load_in_4bit=load_in_4bit,
        )
        # Enable native 2x faster inference
        FastLanguageModel.for_inference(model)
        
    except Exception as e:
        print(f"Unsloth loading error: {e}")
        print("Falling back to standard HuggingFace loading...")
        
        if has_adapters:
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer
            
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=load_in_4bit,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=load_in_4bit,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get model context length
    max_seq_length = getattr(model.config, "max_position_embeddings", 4096)
    
    # Add safety margin for token length
    effective_max_length = max_seq_length - 50  # Reserve 50 tokens for generation
    print(f"Model loaded with max sequence length: {max_seq_length}")
    print(f"Using effective max input length: {effective_max_length} (reserving 50 tokens for generation)")
    
    return model, tokenizer, effective_max_length

def smart_truncate_text(text, tokenizer, max_tokens):
    """Smart truncation that preserves beginning and end of clinical notes."""
    if not text:
        return ""
        
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    # Keep first 60% and last 40% of available tokens
    # This preserves both context from beginning and recent findings
    first_part = int(max_tokens * 0.6)
    last_part = max_tokens - first_part
    
    truncated_tokens = tokens[:first_part] + tokens[-last_part:]
    result = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # Add marker to indicate truncation happened
    truncation_marker = "\n[...Note truncated due to length...]\n"
    return result[:first_part*4] + truncation_marker + result[-last_part*4:]

def generate_assessment(model, tokenizer, prompt, max_new_tokens=120, 
                      max_seq_length=None, temperature=0.0):
    """Generate assessment from model with optimal settings."""
    inputs = tokenizer([prompt], return_tensors="pt")
    
    # Handle sequence length constraints
    if max_seq_length and inputs.input_ids.shape[1] > max_seq_length - max_new_tokens:
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with appropriate settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Increased max tokens for fuller reasoning
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=(temperature > 0),  # Only sample if temperature > 0
            repetition_penalty=1.2,  # Prevent repetition
            no_repeat_ngram_size=3,  # Prevent repetitive phrases
        )
    
    # Extract full response but trim prompt portion
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # If the prompt is found in the response, return only the part after the prompt
    if prompt in full_response:
        return full_response[len(prompt):].strip()
    
    return full_response

def preprocess_clinical_note(note_text):
    """Clean clinical notes for consistent processing."""
    if not isinstance(note_text, str):
        return ""
    
    processed_text = note_text.replace('</s>', '\n\n')
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        processed_text = processed_text.replace(token, f"[{token}]")
    
    processed_text = processed_text.replace('\r\n', '\n').replace('\r', '\n')
    processed_text = ' '.join(processed_text.split())
    return processed_text.strip()

def create_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """Create optimized prompt for malnutrition assessment."""
    prompt = """Read the patient's notes and determine if the patient is likely to have malnutrition:
    Consider these factors when assessing malnutrition:
    - Anthropometric measurements like weight-for-height, BMI-for-age, height-for-age, MUAC
    - Growth trajectory and percentile changes
    - Clinical signs like edema, muscle wasting, decreased energy
    - Nutritional intake pattern and history
    - Medical conditions affecting nutrition
    - Social or environmental factors impacting food security
    - Recent weight changes or growth concerns

IMPORTANT: You must first analyze the evidence for AND against malnutrition before deciding.

REQUIRED OUTPUT FORMAT:
1. First analyze what evidence exists for BOTH possibilities
2. Explicitly state any relevant measurements, z-scores, or growth patterns
3. Weigh the evidence for malnutrition against evidence for normal nutritional status
4. Conclude with EXACTLY one of these formatted responses only:
   malnutrition=yes
   OR
   malnutrition=no

The default assumption is that a patient does NOT have malnutrition unless there is clear evidence.

CLINICAL NOTE FOR ANALYSIS:
{note}"""

    if tokenizer and max_tokens:
        # Calculate available tokens for the clinical note
        template = prompt.format(note="")
        template_tokens = len(tokenizer.encode(template))
        available_tokens = max_tokens - template_tokens
        
        if available_tokens <= 0:
            available_tokens = max_tokens // 2
            print(f"Warning: Template size exceeds limit. Allocating {available_tokens} tokens for note.")
        
        # Apply smart truncation if needed
        processed_note = smart_truncate_text(note, tokenizer, available_tokens)
        return prompt.format(note=processed_note)
    
    return prompt.format(note=note)

def parse_model_output(output_text):
    """Robust output parsing with structured approach."""
    if not output_text:
        return -1
        
    output_text = output_text.lower().strip()
    
    # Direct format match - most strict matching
    if "malnutrition=yes" in output_text:
        return 1
    if "malnutrition=no" in output_text:
        return 0
    
    # Check for conclusion sections with careful parsing
    conclusion_indicators = ["conclusion:", "assessment:", "impression:", "malnutrition:"]
    lines = output_text.split('\n')
    
    # Search last 5 lines for conclusion indicators
    for line in reversed(lines[-5:]):
        line_lower = line.strip().lower()
        for indicator in conclusion_indicators:
            if indicator in line_lower:
                # More careful parsing to prevent false positives
                if "no malnutrition" in line_lower or "not malnourished" in line_lower:
                    return 0
                elif "yes malnutrition" in line_lower or "is malnourished" in line_lower:
                    return 1
                
                # Parse individual words with context
                words = line_lower.split()
                word_idx = 0
                while word_idx < len(words):
                    if words[word_idx] == "no" and word_idx + 1 < len(words):
                        # Check that "no" isn't part of "not" followed by negative
                        if words[word_idx + 1] not in ["evidence", "signs", "indication", "malnutrition"]:
                            word_idx += 1
                            continue
                        return 0
                    elif words[word_idx] == "yes" and word_idx + 1 < len(words):
                        if "not" not in words[max(0, word_idx-1):min(len(words), word_idx+2)]:
                            return 1
                    word_idx += 1
    
    # Look for strongest indicators throughout the text
    if "no evidence of malnutrition" in output_text or "does not have malnutrition" in output_text:
        return 0
    if "has malnutrition" in output_text or "patient is malnourished" in output_text:
        return 1
    
    # Final check - if we made it here, be more cautious
    yes_indicators = sum(1 for indicator in ["malnutrition=yes", "has malnutrition", "is malnourished"] 
                        if indicator in output_text)
    no_indicators = sum(1 for indicator in ["malnutrition=no", "no malnutrition", "not malnourished"]
                       if indicator in output_text)
    
    if yes_indicators > no_indicators:
        return 1
    elif no_indicators > yes_indicators:
        return 0
    
    # Unable to determine with confidence
    print("WARNING: Could not determine prediction with confidence - Output text:")
    print(output_text[:200] + "..." if len(output_text) > 200 else output_text)
    return -1

def generate_predictions(model, tokenizer, data_path, max_seq_length, output_dir, temperature=0.0, max_new_tokens=120):
    """Generate predictions with optimized memory usage and error handling."""
    # Load and validate dataset
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
    
    # Log device information
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # Process data with memory management
    batch_size = 1  # Process one at a time for clinical notes
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
        try:
            # Preprocess note
            note_text = preprocess_clinical_note(row["txt"])
            
            # Create prompt with appropriate truncation
            prompt = create_malnutrition_prompt(
                note=note_text,
                tokenizer=tokenizer,
                max_tokens=max_seq_length
            )
            
            # Generate assessment
            output_text = generate_assessment(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                max_seq_length=max_seq_length,
                temperature=temperature
            )
            
            # Parse result
            pred = parse_model_output(output_text)
            true_label = int(row["label"])
            
            # Store results
            results.append({
                "DEID": row["DEID"],
                "TRUE_LABEL": true_label,
                "PREDICTED_LABEL": pred,
                "MODEL_OUTPUT": output_text,
                "INPUT_LENGTH": len(tokenizer.encode(prompt)),
                "PROMPT": prompt
            })
            
            if pred != -1:  # Only include determinate predictions in metrics
                true_labels.append(true_label)
                pred_labels.append(pred)
                deids.append(row["DEID"])
                
            # Print current case
            print(f"\nCase {idx + 1}/{len(df)} - DEID: {row['DEID']}")
            print(f"TRUE: {'Malnutrition (1)' if true_label == 1 else 'No Malnutrition (0)'}")
            print(f"PRED: {'Malnutrition (1)' if pred == 1 else 'No Malnutrition (0)' if pred == 0 else 'Undetermined (-1)'}")
            print("-" * 40)
            
            # Periodic memory cleanup
            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"\nError processing case {idx + 1} - DEID: {row['DEID']}")
            print(f"Error: {str(e)}")
            results.append({
                "DEID": row["DEID"],
                "TRUE_LABEL": int(row["label"]),
                "PREDICTED_LABEL": -1,
                "MODEL_OUTPUT": f"Error: {str(e)}",
                "PROMPT": "Error generating prompt"
            })
    
    # Clean up memory before metrics calculation
    torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate metrics
    metrics = {}
    if true_labels:
        # Convert to Python native types for JSON serialization
        true_labels_native = [int(x) for x in true_labels]
        pred_labels_native = [int(x) for x in pred_labels]
        
        try:
            metrics = {
                "accuracy": float(accuracy_score(true_labels_native, pred_labels_native)),
                "f1": float(f1_score(true_labels_native, pred_labels_native, zero_division=0)),
                "recall": float(recall_score(true_labels_native, pred_labels_native, zero_division=0)),
                "roc_auc": float(roc_auc_score(true_labels_native, pred_labels_native)) if len(set(true_labels_native)) > 1 and len(set(pred_labels_native)) > 1 else float('nan'),
                "classification_report": classification_report(
                    true_labels_native, 
                    pred_labels_native, 
                    output_dict=True,
                    zero_division=0
                ),
                "confusion_matrix": confusion_matrix(true_labels_native, pred_labels_native).tolist(),
                "n_samples": len(true_labels_native),
                "n_indeterminate": len(df) - len(true_labels_native),
                "class_distribution": {
                    "true_positives": int(sum((np.array(true_labels_native) == 1) & (np.array(pred_labels_native) == 1))),
                    "true_negatives": int(sum((np.array(true_labels_native) == 0) & (np.array(pred_labels_native) == 0))),
                    "false_positives": int(sum((np.array(true_labels_native) == 0) & (np.array(pred_labels_native) == 1))),
                    "false_negatives": int(sum((np.array(true_labels_native) == 1) & (np.array(pred_labels_native) == 0))),
                }
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {"error": str(e)}
    
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
    if deids:
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
        print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"F1 Score: {metrics.get('f1', 'N/A'):.4f}")
        print(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
        if not np.isnan(metrics.get('roc_auc', float('nan'))):
            print(f"AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        print("\nConfusion Matrix:")
        print(f"TP: {metrics.get('class_distribution', {}).get('true_positives', 'N/A')}")
        print(f"TN: {metrics.get('class_distribution', {}).get('true_negatives', 'N/A')}")
        print(f"FP: {metrics.get('class_distribution', {}).get('false_positives', 'N/A')}")
        print(f"FN: {metrics.get('class_distribution', {}).get('false_negatives', 'N/A')}")
    print(f"\nResults saved to {output_dir}")
    
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description="Optimized inference with Unsloth integration")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation (0.0 for deterministic)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with more verbose output")
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Maximum new tokens for generation")
    args = parser.parse_args()
    
    print("\nLoading model...")
    model, tokenizer, max_seq_length = load_model(args.model_path, args.load_in_4bit)
    
    # Run a quick test case if in debug mode
    if args.debug:
        print("\nRunning test case for debugging...")
        test_prompt = create_malnutrition_prompt("Patient is a 5-year-old male with height below the 3rd percentile and weight below the 5th percentile. Recent weight loss of 2kg in past 3 months. No other significant findings.")
        test_output = generate_assessment(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_new_tokens=args.max_new_tokens,
            max_seq_length=max_seq_length,
            temperature=args.temperature
        )
        print("\nTEST PROMPT:")
        print(test_prompt)
        print("\nTEST OUTPUT:")
        print(test_output)
        print("\nPARSED RESULT:")
        test_result = parse_model_output(test_output)
        print(f"malnutrition={'yes' if test_result == 1 else 'no' if test_result == 0 else 'indeterminate'}")
        
        # Ask user if they want to continue
        response = input("\nContinue with full dataset? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    print("\nGenerating predictions...")
    metrics, predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        max_seq_length=max_seq_length,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )

if __name__ == "__main__":
    main()
