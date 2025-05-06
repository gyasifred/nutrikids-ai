#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Malnutrition Assessment Tool
- Maintains original prompt and criteria tables
- Improved analysis steps with comprehensive explanation
- Includes all original metrics (AUC, etc.)
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
    return parser.parse_args()

def preprocess_clinical_note(note_text):
    """Preprocess clinical notes (original implementation)."""
    if not note_text:
        return note_text
    
    processed_text = note_text.replace('</s>', '\n\n')
    
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")
    
    return processed_text

def create_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """Your original prompt with enhanced analysis steps."""
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

Analysis Steps:
1) First determine malnutrition status using:
   - The criteria tables above (primary)
   - Any other relevant clinical factors (secondary)
   - Overall clinical judgment (tertiary)

2) Explain your decision by listing:
   - Which specific criteria from tables were met (if any)
   - Any other relevant clinical factors considered
   - Your overall clinical reasoning

3) Provide your final assessment in this exact format: 
   malnutrition=[yes/no]
   confidence=[high/medium/low]
   severity=[none/mild/moderate/severe]

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
    """Load model with native sequence length (original implementation)."""
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
                      max_seq_length=None, temperature=0.1, top_p=0.9):
    """Generate assessment from model."""
    inputs = tokenizer([prompt], return_tensors="pt")
    
    if max_seq_length and inputs.input_ids.shape[1] > max_seq_length - max_new_tokens:
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
    
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

def extract_decision(prediction):
    """Enhanced decision extraction with comprehensive explanation."""
    def clean_text(text):
        return text.replace('\r', '\n').strip()
    
    prediction = clean_text(prediction)
    result = {
        'assessment': 'error',
        'confidence': 'unknown',
        'severity': 'unknown',
        'criteria_used': [],
        'other_factors': [],
        'explanation': '',
        'raw_prediction': prediction
    }
    
    # Extract assessment components
    assessment_part = prediction.split("Assessment:")[-1] if "Assessment:" in prediction else prediction
    assessment_part_lower = assessment_part.lower()
    
    # Extract final assessment
    if "malnutrition=yes" in assessment_part_lower:
        result['assessment'] = "yes"
    elif "malnutrition=no" in assessment_part_lower:
        result['assessment'] = "no"
    
    # Extract confidence
    for conf in ["high", "medium", "low"]:
        if f"confidence={conf}" in assessment_part_lower:
            result['confidence'] = conf
            break
    
    # Extract severity
    for sev in ["severe", "moderate", "mild", "none"]:
        if f"severity={sev}" in assessment_part_lower:
            result['severity'] = sev
            break
    
    # Extract explanation (everything before final assessment)
    explanation_end = min(
        assessment_part_lower.find("malnutrition=yes"),
        assessment_part_lower.find("malnutrition=no"),
        len(assessment_part)
    )
    if explanation_end == -1:
        explanation_end = len(assessment_part)
    
    result['explanation'] = clean_text(assessment_part[:explanation_end])
    
    # Extract criteria used from tables
    table_criteria = [
        "weight-for-height", "bmi-for-age", "length/height-for-age",
        "mid-upper arm circumference", "weight gain velocity",
        "weight loss", "deceleration in weight", "inadequate nutrient intake"
    ]
    result['criteria_used'] = [
        crit for crit in table_criteria 
        if crit in result['explanation'].lower()
    ]
    
    # Extract other relevant factors
    other_factors = [
        "edema", "muscle wasting", "fat loss", "albumin", "prealbumin",
        "vitamin deficiency", "feeding difficulty", "chronic illness"
    ]
    result['other_factors'] = [
        factor for factor in other_factors 
        if factor in result['explanation'].lower()
    ]
    
    return result

def run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length):
    """Run inference with comprehensive decision extraction."""
    results = []
    streamer = TextStreamer(tokenizer) if args.stream_output else None
    
    for i in tqdm(range(0, len(notes), args.batch_size), desc="Assessing patients"):
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
            prediction = generate_assessment(
                model, tokenizer, prompt,
                args.max_new_tokens, streamer,
                max_seq_length, args.temperature, args.top_p
            )
            inference_time = time.time() - start_time
            
            decision = extract_decision(prediction)
            
            results.append({
                "DEID": patient_id,
                "true_label": true_label if true_label is not None else "unknown",
                "predicted_label": decision['assessment'],
                "confidence": decision['confidence'],
                "severity": decision['severity'],
                "criteria_used": ", ".join(decision['criteria_used']) if decision['criteria_used'] else "None",
                "other_factors": ", ".join(decision['other_factors']) if decision['other_factors'] else "None",
                "explanation": decision['explanation'],
                "inference_time": inference_time,
                "raw_prediction": prediction if args.test_mode else ""
            })
            
            if args.stream_output:
                print(f"\nPatient {patient_id}:")
                print(f"Assessment: {decision['assessment']} (Confidence: {decision['confidence']}, Severity: {decision['severity']})")
                print(f"Criteria Used: {decision['criteria_used']}")
                print(f"Other Factors: {decision['other_factors']}")
                print(f"Explanation: {decision['explanation'][:200]}...")
    
    return results

def calculate_metrics(results, args):
    """Calculate comprehensive performance metrics including AUC."""
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
    
    # For AUC calculation - use confidence as proxy for probability
    conf_to_prob = {"low": 0.3, "medium": 0.6, "high": 0.9}
    probas = [conf_to_prob.get(r["confidence"], 0.5) if r["predicted_label"] == "yes" 
             else 1-conf_to_prob.get(r["confidence"], 0.5) for r in valid_results]
    
    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True,
                                 target_names=["Non-malnourished", "Malnourished"])
    pd.DataFrame(report).transpose().to_csv(os.path.join(metrics_dir, "classification_report.csv"))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, 
                        index=["Actual Non-malnourished", "Actual Malnourished"],
                        columns=["Predicted Non-malnourished", "Predicted Malnourished"]
    ).to_csv(os.path.join(metrics_dir, "metrics/confusion_matrix.csv"))
    
    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true, probas)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
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
        notes = notes[:5]
        patient_ids = patient_ids[:5]
        true_labels = true_labels[:5] if true_labels else None
    
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
            print("\nPerformance Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1 Score: {metrics['f1']:.3f}")
            print(f"AUC: {metrics['auc']:.3f}")
            print(f"Error Rate: {metrics['error_rate']:.3f}")
            print("\nSeverity Distribution:")
            for sev, count in metrics['severity_distribution'].items():
                print(f"{sev.title()}: {count}")
    
    print("\nInference completed successfully")

if __name__ == "__main__":
    main()
