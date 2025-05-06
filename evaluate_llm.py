#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Malnutrition Assessment Tool with Binary Classification Focus
"""

import os
import pandas as pd
import numpy as np
import torch
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from transformers import TextStreamer
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import json
import re

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run malnutrition assessment inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input_file", type=str, required=True, help="CSV file with clinical notes")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--stream_output", action="store_true", help="Stream output during inference")
    parser.add_argument("--test_mode", action="store_true", help="Run on small subset for testing")
    parser.add_argument("--note_column", type=str, default="txt", help="Column with clinical notes")
    parser.add_argument("--id_column", type=str, default="DEID", help="Column with patient IDs")
    parser.add_argument("--label_column", type=str, default="label", help="Column with true labels")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--preprocess_tokens", action="store_true", help="Preprocess special tokens")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for generation in seconds")
    return parser.parse_args()

def preprocess_clinical_note(note_text):
    """Preprocess clinical notes."""
    if not note_text or not isinstance(note_text, str):
        return "" if note_text is None else str(note_text)
    
    processed_text = note_text.replace('</s>', '\n\n')
    
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")
    
    return processed_text

def create_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """Create simplified prompt focused on binary classification."""
    prompt = """Read the patient's notes and determine if the patient is likely to have malnutrition: Criteria list.
Malnutrition related to undernutrition is characterized by:
- Unintentional weight loss
- Weight-for-height or BMI-for-age values below normal
- Inadequate nutrient intake
- Decline in z-scores
- Clinical signs like edema, muscle wasting, decreased energy

Analyze the patient's note and provide ONLY the binary classification result in this exact format:
malnutrition=[yes/no]

Clinical note for analysis:
{note}

Assessment:"""
    
    if tokenizer and max_tokens:
        base_tokens = len(tokenizer.encode(prompt.format(note="")))
        safety_margin = 128
        max_note_tokens = max_tokens - base_tokens - safety_margin
        
        if max_note_tokens > 0:
            note_tokens = tokenizer.encode(note)
            if len(note_tokens) > max_note_tokens:
                note = tokenizer.decode(note_tokens[:max_note_tokens])
                print(f"Truncated note from {len(note_tokens)} to {max_note_tokens} tokens")

    return prompt.format(note=note)

def load_model(model_path, load_in_4bit):
    """Load model with native sequence length."""
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

def generate_assessment(model, tokenizer, prompt, max_new_tokens, streamer=None, 
                      max_seq_length=None, temperature=0.1, top_p=0.9, timeout=60):
    """Generate assessment from model with timeout protection."""
    inputs = tokenizer([prompt], return_tensors="pt")
    
    if max_seq_length and inputs.input_ids.shape[1] > max_seq_length - max_new_tokens:
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Set a timer for generation
    start_time = time.time()
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print(f"Generation timed out after {elapsed_time:.2f} seconds")
            return prompt + "\n[TIMEOUT: Generation took too long]"
        else:
            print(f"Generation error: {e}")
            return prompt + f"\n[ERROR: {str(e)}]"

def extract_decision(prediction):
    """Extract binary decision from model output."""
    prediction = prediction.replace('\r', '\n').strip()
    
    # Simple regex to extract yes/no classification
    malnutrition_match = re.search(r'malnutrition\s*=\s*(yes|no)', prediction, re.IGNORECASE)
    if malnutrition_match:
        return malnutrition_match.group(1).lower()
    else:
        # Fallback - look for yes/no in the text
        if "yes" in prediction.lower() and "no" not in prediction.lower():
            return "yes"
        elif "no" in prediction.lower() and "yes" not in prediction.lower():
            return "no"
        else:
            return "error"  # Cannot determine

def run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length):
    """Run inference with simplified output focused on binary classification."""
    results = []
    streamer = TextStreamer(tokenizer) if args.stream_output else None
    
    # Use tqdm for progress tracking
    for i in tqdm(range(0, len(notes), args.batch_size), desc="Assessing patients"):
        batch_notes = notes[i:i+args.batch_size]
        batch_ids = patient_ids[i:i+args.batch_size]
        batch_labels = true_labels[i:i+args.batch_size] if true_labels else [None]*len(batch_notes)
        
        for note, patient_id, true_label in zip(batch_notes, batch_ids, batch_labels):
            try:
                if args.preprocess_tokens:
                    note = preprocess_clinical_note(note)
                    
                prompt = create_malnutrition_prompt(
                    note, 
                    tokenizer=tokenizer, 
                    max_tokens=max_seq_length - args.max_new_tokens
                )
                
                start_time = time.time()
                prediction = generate_assessment(
                    model, tokenizer, prompt,
                    args.max_new_tokens, streamer,
                    max_seq_length, args.temperature, args.top_p,
                    args.timeout
                )
                inference_time = time.time() - start_time
                
                predicted_label = extract_decision(prediction)
                
                result = {
                    "DEID": patient_id,
                    "true_label": true_label if true_label is not None else "unknown",
                    "predicted_label": predicted_label,
                    "inference_time": inference_time,
                    "raw_prediction": prediction if args.test_mode else ""
                }
                
                results.append(result)
                
                # Print true and predicted labels for every input
                print(f"Patient ID: {patient_id} | True: {true_label} | Pred: {predicted_label} | Time: {inference_time:.2f}s")
                    
            except Exception as e:
                print(f"Error processing patient {patient_id}: {str(e)}")
                results.append({
                    "DEID": patient_id,
                    "true_label": true_label if true_label is not None else "unknown",
                    "predicted_label": "error",
                    "inference_time": time.time() - start_time if 'start_time' in locals() else 0,
                    "raw_prediction": f"ERROR: {str(e)}"
                })
    
    return results

def calculate_metrics(results, args):
    """Calculate comprehensive performance metrics with focus on AUC."""
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    valid_results = [r for r in results if r["true_label"] != "unknown" and r["predicted_label"] != "error"]
    if not valid_results:
        print("No valid results for metrics calculation")
        return None
    
    # Convert labels to binary
    y_true = []
    for r in valid_results:
        if isinstance(r["true_label"], (int, float)):
            y_true.append(int(r["true_label"]))
        else:
            y_true.append(1 if str(r["true_label"]).lower() == "yes" else 0)
    
    y_pred = [1 if r["predicted_label"].lower() == "yes" else 0 for r in valid_results]
    
    # For AUC calculation, create a simple probability score
    # Since we're focusing only on binary classification, use a fixed probability
    # 0.9 for predicted yes, 0.1 for predicted no
    y_prob = [0.9 if pred == 1 else 0.1 for pred in y_pred]
    
    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True,
                                  target_names=["Non-malnourished", "Malnourished"])
    pd.DataFrame(report).transpose().to_csv(os.path.join(metrics_dir, "classification_report.csv"))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=["Actual Non-malnourished", "Actual Malnourished"],
                        columns=["Predicted Non-malnourished", "Predicted Malnourished"])
    cm_df.to_csv(os.path.join(metrics_dir, "confusion_matrix.csv"))
    
    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Binary Classification)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(metrics_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics summary
    metrics_summary = {
        "accuracy": report["accuracy"],
        "precision": report["Malnourished"]["precision"],
        "recall": report["Malnourished"]["recall"],
        "f1": report["Malnourished"]["f1-score"],
        "auc": roc_auc,
        "error_rate": sum(1 for r in results if r["predicted_label"] == "error") / len(results),
    }
    
    with open(os.path.join(metrics_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    print("\n==== METRICS SUMMARY ====")
    print(f"Accuracy: {metrics_summary['accuracy']:.3f}")
    print(f"Precision: {metrics_summary['precision']:.3f}")
    print(f"Recall: {metrics_summary['recall']:.3f}")
    print(f"F1 Score: {metrics_summary['f1']:.3f}")
    print(f"AUC: {metrics_summary['auc']:.3f}")
    print(f"Error Rate: {metrics_summary['error_rate']:.3f}")
    print("=========================\n")
    
    return metrics_summary

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Handle missing columns gracefully
    if args.note_column not in df.columns:
        print(f"Error: Column '{args.note_column}' not found in input file")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    if args.id_column not in df.columns:
        print(f"Warning: Column '{args.id_column}' not found in input file. Using index as ID.")
        patient_ids = df.index.astype(str).tolist()
    else:
        patient_ids = df[args.id_column].astype(str).tolist()
    
    notes = df[args.note_column].tolist()
    true_labels = df[args.label_column].tolist() if args.label_column in df.columns else None
    
    if args.test_mode:
        test_size = min(5, len(notes))
        notes = notes[:test_size]
        patient_ids = patient_ids[:test_size]
        true_labels = true_labels[:test_size] if true_labels else None
        print(f"Test mode: using first {test_size} records")
    
    print(f"Loading model from {args.model_path}")
    model, tokenizer, max_seq_length = load_model(args.model_path, args.load_in_4bit)
    
    print("Starting inference...")
    results = run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "assessment_results.csv"), index=False)
    
    if true_labels:
        metrics = calculate_metrics(results, args)
        if metrics:
            print("\nAUC Value:", metrics['auc'])
    
    print("\nInference completed successfully")

if __name__ == "__main__":
    main()
