#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference on clinical notes using the trained malnutrition assessment model
with enhanced explanation extraction.
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
    parser.add_argument("--max_new_tokens", type=int, default=512, 
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
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for sampling during generation (default: 0.1)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter (default: 0.9)")
    parser.add_argument("--preprocess_tokens", action="store_true", 
                       help="Preprocess </s> tokens in clinical notes")
    parser.add_argument("--generate_explanations", action="store_true",
                       help="Generate explanations for malnutrition assessments")
    return parser.parse_args()

def preprocess_clinical_note(note_text):
    """Preprocess clinical notes to handle special tokens."""
    if not note_text:
        return note_text
    
    processed_text = note_text.replace('</s>', '\n\n')
    
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")
    
    return processed_text

def create_simplified_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """Create malnutrition assessment prompt with clinical criteria."""
    prompt = """Read the patient's notes and determine if the patient is likely to have malnutrition: Criteria list.
Mild malnutrition related to undernutrition is usually the result of an acute event, either due to economic circumstances or acute illness, and presents with unintentional weight loss or weight gain velocity less than expected. Moderate malnutrition related to undernutrition occurs due to undernutrition of a significant duration that results in weight-for-length/height values or BMI-for-age values that are below the normal range. Severe malnutrition related to undernutrition occurs as a result of prolonged undernutrition and is most frequently quantified by declines in rates of linear growth that result in stunting.

Follow this format:
1) First provide some explanations about your decision. In your explanation mention did you use single or multiple data points, and list z scores you used.
2) Then format your output as follows, strictly follow this format: malnutrition=yes or malnutrition=no

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
    """Load the fine-tuned model."""
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

def generate_text(model, tokenizer, prompt, max_new_tokens=100, streamer=None, 
                 max_seq_length=None, temperature=0.1, top_p=0.9):
    """Generate text with the model."""
    inputs = tokenizer([prompt], return_tensors="pt")
    
    if max_seq_length and inputs.input_ids.shape[1] > max_seq_length - max_new_tokens:
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
        print(f"Input truncated to {inputs.input_ids.shape[1]} tokens")
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_explanation(prediction):
    """Enhanced explanation extraction that handles clinical content better."""
    try:
        # Normalize the prediction text
        normalized = prediction.replace('\r', '\n').strip()
        
        # Find the assessment section
        if "Assessment:" in normalized:
            assessment_part = normalized.split("Assessment:")[-1]
        else:
            assessment_part = normalized
        
        # Remove any final assessment markers
        for marker in ["malnutrition=yes", "malnutrition=no", "malnutrition: yes", "malnutrition: no"]:
            if marker in assessment_part.lower():
                assessment_part = assessment_part[:assessment_part.lower().find(marker)].strip()
        
        # Split into lines and clean up
        lines = [line.strip() for line in assessment_part.split('\n') if line.strip()]
        
        # Case 1: Standard malnutrition criteria description
        if any("malnutrition related to undernutrition" in line.lower() for line in lines):
            return lines[0]
        
        # Case 2: Single word "Malnutrition"
        elif len(lines) == 1 and lines[0].lower() == "malnutrition":
            return "Evidence of malnutrition found in clinical notes"
        
        # Case 3: Specific malnutrition criteria listed
        elif any(("weight gain velocity" in line.lower() or 
                 "weight loss" in line.lower() or 
                 "z score" in line.lower()) for line in lines):
            return '\n'.join(lines)
        
        # Case 4: Empty explanation - try to extract from raw prediction
        elif not lines:
            clinical_terms = ["weight loss", "underweight", "failure to thrive", 
                             "z score", "nutritional deficiency"]
            found_terms = [term for term in clinical_terms if term in normalized.lower()]
            if found_terms:
                return f"Clinical indicators found: {', '.join(found_terms)}"
            return "No malnutrition indicators found in clinical notes"
        
        # Default case
        return '\n'.join(lines) if lines else "No explanation extracted"
    
    except Exception as e:
        print(f"Error extracting explanation: {str(e)}")
        return "Error extracting explanation"

def extract_assessment(prediction):
    """More rigorous assessment extraction that checks explanation content."""
    normalized = prediction.lower().replace('\n', ' ').replace('\r', ' ')
    
    # First check strict format
    if "malnutrition=yes" in normalized:
        return "yes"
    elif "malnutrition=no" in normalized:
        # Verify this isn't contradicted by explanation
        explanation = extract_explanation(prediction).lower()
        if any(term in explanation for term in ["malnutrition", "underweight", "weight loss"]):
            return "error"  # Flag for manual review
        return "no"
    
    # Check explanation content if format isn't strict
    explanation = extract_explanation(prediction).lower()
    
    # Positive indicators
    if any(term in explanation for term in ["malnutrition", "underweight", 
                                          "weight loss", "z score", 
                                          "nutritional deficiency"]):
        return "yes"
    
    # Negative indicators
    if any(term in explanation for term in ["no malnutrition", "well-nourished", 
                                          "adequate nutrition"]):
        return "no"
    
    return "error"  # Default to error if unclear

def run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length):
    """Run inference with enhanced explanation extraction."""
    results = []
    streamer = TextStreamer(tokenizer) if args.stream_output else None
    
    for i in tqdm(range(0, len(notes), args.batch_size), desc="Processing notes"):
        batch_notes = notes[i:i+args.batch_size]
        batch_ids = patient_ids[i:i+args.batch_size]
        batch_labels = true_labels[i:i+args.batch_size] if true_labels else [None]*len(batch_notes)
        
        for note, patient_id, true_label in zip(batch_notes, batch_ids, batch_labels):
            if args.preprocess_tokens:
                note = preprocess_clinical_note(note)
                
            prompt = create_simplified_malnutrition_prompt(
                note, 
                tokenizer=tokenizer, 
                max_tokens=max_seq_length - args.max_new_tokens
            )
            
            start_time = time.time()
            
            if args.stream_output:
                print(f"\nAssessing patient {patient_id}...")
                prediction = generate_text(
                    model, tokenizer, prompt, 
                    args.max_new_tokens, streamer,
                    max_seq_length, args.temperature, args.top_p
                )
            else:
                prediction = generate_text(
                    model, tokenizer, prompt,
                    args.max_new_tokens, None,
                    max_seq_length, args.temperature, args.top_p
                )
            
            inference_time = time.time() - start_time
            
            assessment = extract_assessment(prediction)
            explanation = extract_explanation(prediction) if args.generate_explanations else ""
            
            result = {
                "DEID": patient_id,
                "true_label": true_label if true_label is not None else "unknown",
                "predicted_label": assessment,
                "explanation": explanation,
                "inference_time": inference_time,
                "raw_prediction": prediction if args.test_mode else ""
            }
            
            if args.generate_explanations:
                print(f"\n----- Patient {patient_id} -----")
                print(f"Assessment: {assessment}")
                print(f"Explanation: {explanation}")
                print("-" * 80)
            
            results.append(result)
            
    return results

def calculate_metrics(results, args):
    """Calculate performance metrics."""
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    if all(r["true_label"] == "unknown" for r in results):
        print("No true labels available - skipping metrics")
        return None
    
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
    
    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True, 
                                 target_names=["Non-malnourished", "Malnourished"])
    pd.DataFrame(report).transpose().to_csv(os.path.join(metrics_dir, "classification_report.csv"))
    
    with open(os.path.join(metrics_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["Non-malnourished", "Malnourished"]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=["Actual Non-malnourished", "Actual Malnourished"],
                        columns=["Predicted Non-malnourished", "Predicted Malnourished"])
    cm_df.to_csv(os.path.join(metrics_dir, "confusion_matrix.csv"))
    
    plt.figure(figsize=(8, 6))
    plt.matshow(cm, cmap=plt.cm.Blues, fignum=1)
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    
    # Metrics summary
    metrics_summary = {
        "accuracy": report["accuracy"],
        "precision_malnourished": report["Malnourished"]["precision"],
        "recall_malnourished": report["Malnourished"]["recall"],
        "f1_malnourished": report["Malnourished"]["f1-score"],
        "error_assessments": sum(1 for r in results if r["predicted_label"] == "error"),
        "total_samples": len(valid_results)
    }
    
    with open(os.path.join(metrics_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    return metrics_summary

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    notes = df[args.note_column].tolist()
    patient_ids = df[args.id_column].tolist()
    true_labels = df[args.label_column].tolist() if args.label_column in df.columns else None
    
    if args.test_mode:
        test_size = min(5, len(notes))
        notes = notes[:test_size]
        patient_ids = patient_ids[:test_size]
        if true_labels:
            true_labels = true_labels[:test_size]
    
    print(f"Loading model from {args.model_path}")
    model, tokenizer, max_seq_length = load_model(args.model_path, args.load_in_4bit)
    
    print("Starting inference...")
    results = run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length)
    
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "inference_results.csv"), index=False)
    
    if true_labels:
        metrics = calculate_metrics(results, args)
        if metrics:
            print("\nMetrics Summary:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Malnourished Precision/Recall: {metrics['precision_malnourished']:.3f}/{metrics['recall_malnourished']:.3f}")
            print(f"Error Assessments: {metrics['error_assessments']}")
    
    print("\nInference Complete")
    if args.test_mode:
        print("\nTest Mode Samples:")
        for r in results[:3]:
            print(f"\nPatient {r['DEID']}:")
            print(f"Assessment: {r['predicted_label']}")
            print(f"Explanation: {r['explanation'][:200]}...")

if __name__ == "__main__":
    main()
