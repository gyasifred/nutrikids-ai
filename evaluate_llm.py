#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete fixed inference script for malnutrition assessment model.
Handles device placement properly and includes all metrics tracking.
"""

import os
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
from unsloth import FastLanguageModel
from tqdm import tqdm

def load_model(model_path, load_in_4bit):
    """Load model with proper device handling."""
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)
    except Exception as e:
        print(f"Error loading with Unsloth: {e}")
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)

    print(f"Model loaded with max sequence length: {max_seq_length}")
    return model, tokenizer, max_seq_length

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

def create_simplified_malnutrition_prompt(note, label="", tokenizer=None, max_tokens=None):
    """
    Complete malnutrition assessment prompt identical to training.
    """
    prompt = """Read the patient's notes and determine if the patient is likely to have malnutrition: Criteria list.
Mild malnutrition related to undernutrition is usually the result of an acute event, either due to economic circumstances or acute illness, and presents with unintentional weight loss or weight gain velocity less than expected. Moderate malnutrition related to undernutrition occurs due to undernutrition of a significant duration that results in weight-for-length/height values or BMI-for-age values that are below the normal range. Severe malnutrition related to undernutrition occurs as a result of prolonged undernutrition and is most frequently quantified by declines in rates of linear growth that result in stunting.

You should use z scores (also called z for short) for weight-for-height/length, BMI-for-age, length/height-for-age or MUAC criteria. When a child has only one data point in the records (single z score present) use the table below:

Table 1. Single data point present.
Mild Malnutrition
Weight-for-height: −1 to −1.9 z score
BMI-for-age: −1 to −1.9 z score
Length/height-for-age: No Data
Mid–upper arm circumference: Greater than or equal to −1 to −1.9 z score

Moderate Malnutrition
Weight-for-height: −2 to −2.9 z score
BMI-for-age: −2 to −2.9 z score
Length/height-for-age: No Data
Mid–upper arm circumference: Greater than or equal to −2 to −2.9 z score

Severe Malnutrition
Weight-for-height:−3 or greater z score
BMI-for-age: −3 or greater z score
Length/height-for-age: −3 z score
Mid–upper arm circumference: Greater than or equal to −3 z score

When the child has 2 or more data points (multiple z scores over time) use this table:

Table 2. Multiple data points available.
Mild Malnutrition
Weight gain velocity (<2 years of age): Less than 75% of the norm for expected weight gain
Weight loss (2–20 years of age): 5% usual body weigh
Deceleration in weight for length/height: Decline of 1 z score
Inadequate nutrient intake: 51%−75% estimated energy/protein need

Moderate Malnutrition
Weight gain velocity (<2 years of age): Less than 50% of the norm for expected weight gain
Weight loss (2–20 years of age): 7.5% usual body weight
Deceleration in weight for length/height: Decline of 2 z score
Inadequate nutrient intake: 26%−50% estimated energy/protein need

Severe Malnutrition
Weight gain velocity (<2 years of age): Less than 25% of the normb for expected weight gain
Weight loss (2–20 years of age): 10% usual body weight
Deceleration in weight for length/height: Decline of 3 z score
Inadequate nutrient intake: less than 25% estimated energy/protein need

Follow this format:
1) First provide some explanations about your decision. In your explanation mention did you use single or multiple data points, and list z scores you used.
2) Then format your output as follows, strictly follow this format: malnutrition=yes or malnutrition=no

Clinical note for analysis:
{note}

{label_part}"""

    label_part = ""
    formatted_prompt = prompt.format(note=note, label_part=label_part)
    
    if tokenizer and max_tokens:
        tokens = tokenizer.encode(formatted_prompt)
        if len(tokens) > max_tokens:
            # Calculate available space for note
            template = prompt.format(note="", label_part=label_part)
            template_tokens = tokenizer.encode(template)
            available_tokens = max_tokens - len(template_tokens)
            
            if available_tokens <= 0:
                available_tokens = max_tokens // 2
            
            # Tokenize note separately and truncate
            note_tokens = tokenizer.encode(note)
            truncated_note_tokens = note_tokens[:available_tokens]
            truncated_note = tokenizer.decode(truncated_note_tokens, skip_special_tokens=True)
            
            # Rebuild prompt with truncated note
            formatted_prompt = prompt.format(note=truncated_note, label_part=label_part)
    
    return formatted_prompt

def parse_model_output(output_text):
    """Robust output parsing with multiple fallbacks."""
    output_text = output_text.lower().strip()
    
    # First try exact matches
    if "malnutrition=yes" in output_text:
        return 1
    elif "malnutrition=no" in output_text:
        return 0
    
    # Search for last occurrence of yes/no
    lines = output_text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if any(kw in line for kw in ["malnutrition:", "conclusion:", "assessment:"]):
            if "yes" in line:
                return 1
            elif "no" in line:
                return 0
    
    # Final fallback
    for line in reversed(lines[-3:]):
        if "yes" in line.split():
            return 1
        elif "no" in line.split():
            return 0
    
    return -1  # Undetermined

def generate_predictions(model, tokenizer, data_path, max_seq_length, output_dir):
    """Generate predictions with proper device handling."""
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
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
        try:
            # Preprocess identically to training
            note_text = preprocess_clinical_note(row["txt"])
            prompt = create_simplified_malnutrition_prompt(
                note=note_text,
                tokenizer=tokenizer,
                max_tokens=max_seq_length - 20
            )
            
            # Move inputs to model's device
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=0.0
                )
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
                
            # Print current case
            print(f"\nCase {idx + 1}/{len(df)} - DEID: {row['DEID']}")
            print(f"TRUE: {'Malnutrition (1)' if true_label == 1 else 'No Malnutrition (0)'}")
            print(f"PRED: {'Malnutrition (1)' if pred == 1 else 'No Malnutrition (0)' if pred == 0 else 'Undetermined (-1)'}")
            print("-" * 40)
            
        except Exception as e:
            print(f"\nError processing case {idx + 1} - DEID: {row['DEID']}")
            print(f"Error: {str(e)}")
            results.append({
                "DEID": row["DEID"],
                "TRUE_LABEL": int(row["label"]),
                "PREDICTED_LABEL": -1,
                "MODEL_OUTPUT": f"Error: {str(e)}",
                "INPUT_TEXT": note_text,
                "PROMPT": prompt
            })
    
    # Calculate metrics
    metrics = {}
    if true_labels:
        metrics = {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "roc_auc": roc_auc_score(true_labels, pred_labels),
            "classification_report": classification_report(true_labels, pred_labels, output_dict=True),
            "confusion_matrix": confusion_matrix(true_labels, pred_labels).tolist(),
            "n_samples": len(true_labels),
            "n_indeterminate": len(df) - len(true_labels),
            "class_distribution": {
                "true_positives": sum((np.array(true_labels) == 1) & (np.array(pred_labels) == 1),
                "true_negatives": sum((np.array(true_labels) == 0) & (np.array(pred_labels) == 0),
                "false_positives": sum((np.array(true_labels) == 0) & (np.array(pred_labels) == 1),
                "false_negatives": sum((np.array(true_labels) == 1) & (np.array(pred_labels) == 0),
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
