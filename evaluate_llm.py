#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference on clinical notes using the trained malnutrition assessment model
with detailed criteria for malnutrition classification.
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
    parser.add_argument("--retry_errors", action="store_true",
                       help="Retry cases with error or inconsistent assessments")
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

def create_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """Create detailed malnutrition assessment prompt with strict output format."""
    prompt = """Analyze this clinical note for malnutrition using these criteria:

1. Weight Patterns:
   - Significant weight loss (>5% in 1 month, >10% in 6 months)
   - BMI/weight-for-height z-scores <-2
   - Documented growth failure

2. Nutritional Intake:
   - Inadequate caloric/protein intake
   - Prolonged NPO status
   - Frequent vomiting/diarrhea

3. Clinical Signs:
   - Muscle wasting
   - Subcutaneous fat loss
   - Edema (in severe cases)

Assessment Guidelines:
- Mild: 1-2 criteria present
- Moderate: 3-4 criteria present
- Severe: 5+ criteria or life-threatening

Output Format STRICTLY:
1) Criteria Analysis: [List which criteria are met with evidence from note]
2) Explanation: [Detailed rationale connecting findings to assessment]
3) Final Assessment: malnutrition=[yes/no] 
   Severity: [none/mild/moderate/severe]
   Confidence: [low/medium/high]

Clinical Note:
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
    """Generate text with strict output format."""
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

def extract_assessment(prediction):
    """Robust assessment extraction with severity detection."""
    prediction = prediction.lower()
    
    # First check for strict format
    if "final assessment:" in prediction:
        assessment_part = prediction.split("final assessment:")[-1].strip()
        if "malnutrition=yes" in assessment_part:
            return "yes"
        elif "malnutrition=no" in assessment_part:
            return "no"
    
    # Check for severity indicators
    severity_map = {
        "severe": "yes",
        "moderate": "yes",
        "mild": "yes",
        "none": "no"
    }
    
    for severity, assessment in severity_map.items():
        if f"severity: {severity}" in prediction:
            return assessment
    
    # Check for weight/nutrition indicators
    positive_indicators = [
        "weight loss", "underweight", "failure to thrive",
        "z-score <-2", "growth faltering", "malnourished"
    ]
    
    if any(indicator in prediction for indicator in positive_indicators):
        return "yes"
    
    return "error"

def extract_severity(prediction):
    """Extract severity level from prediction."""
    prediction = prediction.lower()
    for severity in ["severe", "moderate", "mild"]:
        if f"severity: {severity}" in prediction:
            return severity
    return "none"

def extract_explanation(prediction):
    """Extract detailed explanation from prediction."""
    try:
        if "criteria analysis:" in prediction.lower() and "explanation:" in prediction.lower():
            analysis_part = prediction.lower().split("criteria analysis:")[-1]
            explanation_part = analysis_part.split("explanation:")[-1]
            return explanation_part.split("final assessment:")[0].strip()
        return prediction.split("Assessment:")[-1].strip()
    except:
        return "Unable to extract explanation"

def validate_assessment(explanation, assessment):
    """Thorough validation of explanation-assessment consistency."""
    if not explanation or explanation.strip() == "":
        return False
        
    explanation = explanation.lower()
    
    positive_terms = [
        "weight loss", "underweight", "failure to thrive",
        "z-score", "growth delay", "malnutrition", "nutritional deficiency"
    ]
    
    negative_terms = [
        "no malnutrition", "well-nourished", "adequate growth",
        "normal nutrition", "no evidence of malnutrition"
    ]
    
    if assessment == "yes":
        if any(term in explanation for term in negative_terms):
            return False
        if not any(term in explanation for term in positive_terms):
            return False
    elif assessment == "no":
        if any(term in explanation for term in positive_terms):
            if not any(term in explanation for term in negative_terms):
                return False
                
    return True

def run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length):
    """Run inference with enhanced validation."""
    results = []
    streamer = TextStreamer(tokenizer) if args.stream_output else None
    
    for i in tqdm(range(0, len(notes), args.batch_size), desc="Processing notes"):
        batch_notes = notes[i:i+args.batch_size]
        batch_ids = patient_ids[i:i+args.batch_size]
        batch_labels = true_labels[i:i+args.batch_size] if true_labels else [None]*len(batch_notes)
        
        for note, patient_id, true_label in zip(batch_notes, batch_ids, batch_labels):
            if args.preprocess_tokens:
                note = preprocess_clinical_note(note)
                
            prompt = create_malnutrition_prompt(
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
            severity = extract_severity(prediction)
            explanation = extract_explanation(prediction)
            is_consistent = validate_assessment(explanation, assessment)
            
            # Retry logic for errors/inconsistencies
            if args.retry_errors and (assessment == "error" or not is_consistent):
                print(f"\nRetrying patient {patient_id} due to {'inconsistency' if not is_consistent else 'error'}")
                prediction = generate_text(
                    model, tokenizer, prompt + "\nPlease reformat your answer strictly using the specified format.",
                    args.max_new_tokens, None, max_seq_length, 0.1, 0.9
                )
                assessment = extract_assessment(prediction)
                severity = extract_severity(prediction)
                explanation = extract_explanation(prediction)
                is_consistent = validate_assessment(explanation, assessment)
            
            result = {
                "DEID": patient_id,
                "true_label": true_label if true_label is not None else "unknown",
                "predicted_label": assessment,
                "severity": severity,
                "explanation": explanation,
                "is_consistent": is_consistent,
                "inference_time": inference_time,
                "raw_prediction": prediction if args.test_mode else ""
            }
            
            if not is_consistent and assessment != "error":
                print(f"\nWARNING: Inconsistent assessment for {patient_id}")
                print(f"Assessment: {assessment}")
                print(f"Explanation: {explanation[:200]}...")
                print("-" * 80)
            
            results.append(result)
            
    return results

def calculate_metrics(results, args):
    """Calculate performance metrics with enhanced reporting."""
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
    
    # Save full results with severity
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, "detailed_results.csv"), index=False)
    
    # Metrics summary
    metrics_summary = {
        "accuracy": report["accuracy"],
        "precision_malnourished": report["Malnourished"]["precision"],
        "recall_malnourished": report["Malnourished"]["recall"],
        "f1_malnourished": report["Malnourished"]["f1-score"],
        "inconsistent_assessments": sum(1 for r in results if not r["is_consistent"] and r["predicted_label"] != "error"),
        "error_assessments": sum(1 for r in results if r["predicted_label"] == "error"),
        "severity_distribution": {
            "severe": sum(1 for r in results if r["severity"] == "severe"),
            "moderate": sum(1 for r in results if r["severity"] == "moderate"),
            "mild": sum(1 for r in results if r["severity"] == "mild"),
            "none": sum(1 for r in results if r["severity"] == "none")
        }
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
            print(f"Inconsistent Assessments: {metrics['inconsistent_assessments']}")
            print(f"Error Assessments: {metrics['error_assessments']}")
            print("Severity Distribution:")
            for severity, count in metrics['severity_distribution'].items():
                print(f"  {severity.title()}: {count}")
    
    print("\nInference Complete")
    if args.test_mode:
        print("\nTest Mode Samples:")
        for r in results[:3]:
            print(f"\nPatient {r['DEID']}:")
            print(f"Assessment: {r['predicted_label']} (Severity: {r['severity']})")
            print(f"Explanation: {r['explanation'][:200]}...")

if __name__ == "__main__":
    main()
