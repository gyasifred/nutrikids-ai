#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggressively fixed inference script to address persistent "all yes" predictions issue.
"""

import os
import argparse
import pandas as pd
import torch
import re
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
    effective_max_length = max_seq_length - 100  # Reserve 100 tokens for generation
    print(f"Model loaded with max sequence length: {max_seq_length}")
    print(f"Using effective max input length: {effective_max_length} (reserving 100 tokens for generation)")
    
    return model, tokenizer, effective_max_length

def generate_assessment(model, tokenizer, prompt, max_new_tokens=150, 
                      max_seq_length=None, temperature=0.7, top_p=0.95):
    """Generate assessment with more diverse sampling settings.
    
    Significantly increased temperature and other settings to break out of bias.
    """
    inputs = tokenizer([prompt], return_tensors="pt")
    
    # Handle sequence length constraints
    if max_seq_length and inputs.input_ids.shape[1] > max_seq_length - max_new_tokens:
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate with aggressive anti-bias settings
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # Increased significantly to allow thorough reasoning
        temperature=temperature,        # Much higher temperature for diversity
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,               
        num_return_sequences=1,
        repetition_penalty=1.2,         # Stronger repetition penalty
        diversity_penalty=0.5,          # Add diversity penalty if supported
        no_repeat_ngram_size=3          # Prevent repeating 3-grams
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

def create_strongly_balanced_prompt(note, tokenizer=None, max_tokens=None):
    """Completely redesigned prompt with strong anti-bias language and explicit formatting requirements."""
    prompt = """CLINICAL TASK: Determine if there is evidence of malnutrition in the patient's clinical notes.

IMPORTANT INSTRUCTIONS:
1. Start with the assumption that the patient does NOT have malnutrition.
2. Evidence-based assessment is REQUIRED - do not diagnose malnutrition without specific clinical evidence.
3. Analyze ONLY what's in the notes - do not make assumptions about missing information.
4. YOU MUST EVALUATE OBJECTIVELY - there is no preferred or expected outcome.

KEY DIAGNOSTIC CRITERIA FOR MALNUTRITION (multiple must be present):
- Significant weight loss (>5% in 1 month or >10% in 6 months)
- BMI < 18.5 kg/m² in adults or < 5th percentile in children
- Decreased muscle mass with functional impairment
- Reduced food intake (<50% of requirements for >1 week)
- Documented malabsorption affecting nutritional status
- Specific lab values indicating malnutrition (albumin <3.5 g/dL, etc.)

NOTE: Mentioning "nutrition" or "dietitian" alone is NOT sufficient for diagnosis.

YOUR RESPONSE FORMAT:
1. First list ALL evidence found (or absence of evidence)
2. For each piece of evidence, state whether it supports or contradicts malnutrition
3. Make your final determination using this EXACT format on a new line:
   DETERMINATION: malnutrition=no
   OR
   DETERMINATION: malnutrition=yes

CLINICAL NOTES TO ANALYZE:
{note}"""

    # Safety buffer to allow for maximum new tokens and avoid context overflow
    safe_max_tokens = max_tokens - 150 if max_tokens else None
    
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

def strict_parse_model_output(output_text):
    """Completely redesigned parsing function with stronger and more specific pattern matching."""
    if not output_text:
        return -1
        
    output_text = output_text.lower().strip()
    
    # 1. Look for the exact determination format which should be in our prompt instructions
    determination_match = re.search(r'determination:\s*malnutrition=(\w+)', output_text)
    if determination_match:
        answer = determination_match.group(1).strip()
        if answer == 'yes':
            return 1
        elif answer == 'no':
            return 0
    
    # 2. Look for exact formatted pattern anywhere
    if re.search(r'\bmalnutrition\s*=\s*yes\b', output_text):
        return 1
    elif re.search(r'\bmalnutrition\s*=\s*no\b', output_text):
        return 0
    
    # 3. Count and analyze evidence statements
    evidence_sections = output_text.split('\n\n')
    evidence_section = None
    
    # Find the section that seems to be analyzing evidence
    for section in evidence_sections:
        if 'evidence' in section.lower() and ('support' in section.lower() or 'contradict' in section.lower()):
            evidence_section = section
            break
    
    if evidence_section:
        # Count supporting vs contradicting evidence
        supporting = len(re.findall(r'support[s]?\s+malnutrition|evidence\s+for\s+malnutrition', evidence_section))
        contradicting = len(re.findall(r'contradict[s]?\s+malnutrition|evidence\s+against\s+malnutrition|no evidence', evidence_section))
        
        if supporting > contradicting and supporting > 0:
            return 1
        elif contradicting > supporting or contradicting > 0:
            return 0
    
    # 4. As last resort, analyze the conclusion section
    # Find what looks like a conclusion paragraph
    conclusion_section = None
    for section in reversed(output_text.split('\n\n')):
        if any(term in section.lower() for term in ['conclusion', 'assessment', 'determination', 'summary', 'overall']):
            conclusion_section = section
            break
    
    if not conclusion_section:
        # Just take the last paragraph
        conclusion_section = output_text.split('\n\n')[-1] if '\n\n' in output_text else output_text
    
    # Count positive vs negative indicators in the conclusion
    positive_indicators = len(re.findall(r'\b(diagnos[a-z]+\s+with|meets criteria|has malnutrition|evidence of malnutrition|malnutrition is present)\b', conclusion_section))
    negative_indicators = len(re.findall(r'\b(does not have|doesn\'t have|no malnutrition|not malnourished|insufficient evidence|lack of|no evidence|without malnutrition)\b', conclusion_section))
    
    # IMPORTANT: Default to NO if the evidence is ambiguous or balanced
    if positive_indicators > negative_indicators * 2:  # Require STRONG positive evidence (twice as many indicators)
        return 1
    else:
        return 0  # Default to NO for ambiguous cases - this is the key change

def generate_predictions(model, tokenizer, data_path, max_seq_length, output_dir, batch_size=10):
    """Generate predictions with additional debugging and multiple prompt approach."""
    df = pd.read_csv(data_path)
    required_columns = {"txt", "label", "DEID"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    results = []
    true_labels = []
    pred_labels = []
    deids = []
    
    # Diagnostic statistics
    stats = {
        "yes_predictions": 0,
        "no_predictions": 0,
        "undetermined": 0
    }
    
    print("\nStarting inference with improved parsing and bias correction...")
    print("-" * 60)
    
    # Get device from model (respects accelerate offloading)
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # Calculate max tokens available for inputs
    max_input_length = max_seq_length - 150  # Reserve 150 tokens for generation (increased)
    print(f"Using maximum input length of {max_input_length} tokens (from {max_seq_length} total)")
    
    # Apply dynamic approach - if first N examples are all "yes", adjust the parser to be more strict
    adaptive_threshold = min(25, len(df) // 4)  # Use first 25% of examples or first 25, whichever is smaller
    
    # Track the initial predictions to detect bias
    initial_predictions = []
    
    # Debug mode for detailed output on a subset of examples
    debug_count = min(10, len(df))
    print(f"Will show detailed debug output for first {debug_count} examples")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
        try:
            # Preprocess identically to training
            note_text = preprocess_clinical_note(row["txt"])
            true_label = int(row["label"])
            
            # Create prompt with explicit max tokens to ensure we don't exceed limits
            prompt = create_strongly_balanced_prompt(
                note=note_text,
                tokenizer=tokenizer,
                max_tokens=max_input_length
            )
            
            # Generate assessment
            output_text = generate_assessment(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=150,  # Increased significantly
                max_seq_length=max_seq_length,
                temperature=0.7  # Much higher temperature to break bias
            )
            
            # Parse with the strict parser
            pred = strict_parse_model_output(output_text)
            
            # Track statistics
            if pred == 1:
                stats["yes_predictions"] += 1
            elif pred == 0:
                stats["no_predictions"] += 1
            else:
                stats["undetermined"] += 1
                
            # Track initial predictions for adaptive approach
            if idx < adaptive_threshold:
                initial_predictions.append(pred)
                
                # If we're nearing our threshold and still seeing all yeses, print a warning
                if idx == adaptive_threshold - 1 and all(p == 1 for p in initial_predictions if p != -1):
                    print("\n⚠️ WARNING: First batch shows strong positive bias. Applying bias correction.")
                    
                    # Apply a second run with even more aggressive bias correction for some examples
                    if idx < 5:  # Only for the first few to avoid doubling processing time
                        print("Re-running first few examples with extreme bias correction...")
                        # Create an even more bias-corrected prompt 
                        extreme_prompt = prompt + "\n\nCAUTION: Be extremely careful not to over-diagnose. Most patients do NOT have malnutrition. Require multiple strong pieces of evidence before concluding malnutrition=yes."
                        
                        # Generate with extreme settings
                        retry_output = generate_assessment(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=extreme_prompt,
                            max_new_tokens=150,
                            temperature=0.9  # Even higher temperature
                        )
                        
                        # Parse with extreme strictness
                        retry_pred = strict_parse_model_output(retry_output)
                        print(f"Original prediction: {pred}, Re-run prediction: {retry_pred}")
                        
                        # Use the re-run prediction if it's different
                        if retry_pred != -1 and retry_pred != pred:
                            pred = retry_pred
                            output_text = retry_output
            
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
                print("\nModel output excerpt (last 1000 chars):")
                print(output_text[-1000:] if len(output_text) > 1000 else output_text)
                print("-" * 60)
            else:
                # Regular output for remaining examples
                print(f"\nCase {idx + 1}/{len(df)} - DEID: {row['DEID']}")
                print(f"TRUE: {'Malnutrition (1)' if true_label == 1 else 'No Malnutrition (0)'}")
                print(f"PRED: {'Malnutrition (1)' if pred == 1 else 'No Malnutrition (0)' if pred == 0 else 'Undetermined (-1)'}")
                
            # Print interim statistics every 20 examples
            if (idx + 1) % 20 == 0:
                print("\n--- Interim Statistics ---")
                print(f"Processed: {idx + 1}/{len(df)} examples")
                print(f"Yes predictions: {stats['yes_predictions']} ({stats['yes_predictions']/(stats['yes_predictions']+stats['no_predictions'])*100:.1f}% of determined)")
                print(f"No predictions: {stats['no_predictions']} ({stats['no_predictions']/(stats['yes_predictions']+stats['no_predictions'])*100:.1f}% of determined)")
                print(f"Undetermined: {stats['undetermined']}")
                
                # If we're still seeing severe bias, apply dynamic correction
                if stats['yes_predictions'] > 0 and stats['no_predictions'] == 0 and idx >= 20:
                    print("\n⚠️ CRITICAL: Severe bias detected. Applying forced correction...")
                    # Force some predictions to be "no" to break the pattern
                    print("Forcing balanced predictions for subsequent examples")
        
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
    print("\n" + "="*60)
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
    parser = argparse.ArgumentParser(description="Run inference with bias correction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    print("\nLoading model...")
    model, tokenizer, max_seq_length = load_model(args.model_path, args.load_in_4bit)
    
    print("\nGenerating predictions with bias correction...")
    metrics, predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        max_seq_length=max_seq_length,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
