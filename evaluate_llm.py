#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Malnutrition Assessment Tool with Strict Output Formatting and Metrics
"""

import os
import pandas as pd
import argparse
import time
import json
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import TextStreamer
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run malnutrition assessment with metrics")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input_file", type=str, required=True, help="CSV file with clinical notes")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--note_column", type=str, default="txt", help="Column with clinical notes")
    parser.add_argument("--id_column", type=str, default="DEID", help="Column with patient IDs")
    parser.add_argument("--label_column", type=str, default="label", help="Column with true labels")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()

def preprocess_clinical_note(note_text):
    """Preprocess clinical notes according to specifications."""
    if not note_text:
        return note_text
    
    processed_text = note_text.replace('</s>', '\n\n')
    
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")
    
    return processed_text

def create_malnutrition_prompt(note):
    """Create prompt with original criteria tables and strict output formatting."""
    prompt = f"""Read the patient's notes and determine if the patient is likely to have malnutrition: 
Criteria list.

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
Weight-for-height:	−3 or greater z score
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
3) Do not explain, just output malnutrition=yes or malnutrition=no

Clinical note for analysis:
{note}

Assessment:"""
    return prompt

def load_model(model_path, load_in_4bit):
    """Load model with native sequence length."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def extract_final_assessment(prediction):
    """Extract the final assessment from model output."""
    # Find the last occurrence of the assessment pattern
    prediction = prediction.lower()
    last_yes = prediction.rfind('malnutrition=yes')
    last_no = prediction.rfind('malnutrition=no')
    
    if last_yes > last_no:
        return "yes"
    elif last_no > last_yes:
        return "no"
    else:
        # If pattern not found, look for yes/no in the last 20 characters
        last_part = prediction[-20:]
        if 'yes' in last_part:
            return "yes"
        elif 'no' in last_part:
            return "no"
    return "no"  # Default to no if unclear

def generate_assessment(model, tokenizer, prompt, max_new_tokens, temperature):
    """Generate assessment with strict output formatting."""
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output

def calculate_metrics(results, args):
    """Calculate comprehensive performance metrics with focus on AUC."""
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    valid_results = [r for r in results if r["true_label"] != "unknown" and r["assessment"] in ["yes", "no"]]
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
    
    y_pred = [1 if r["assessment"].lower() == "yes" else 0 for r in valid_results]
    
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
        "error_rate": sum(1 for r in results if r["assessment"] not in ["yes", "no"]) / len(results),
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

def run_inference(model, tokenizer, notes, patient_ids, true_labels, args):
    """Run inference with strict output formatting."""
    results = []
    
    for i in tqdm(range(0, len(notes)), desc="Assessing patients"):
        note = notes[i]
        patient_id = patient_ids[i]
        true_label = true_labels[i] if true_labels else "unknown"
        
        processed_note = preprocess_clinical_note(note)
        prompt = create_malnutrition_prompt(processed_note)
        
        start_time = time.time()
        raw_output = generate_assessment(
            model, tokenizer, prompt,
            args.max_new_tokens, args.temperature
        )
        inference_time = time.time() - start_time
        
        assessment = extract_final_assessment(raw_output)
        
        results.append({
            "patient_id": patient_id,
            "true_label": true_label,
            "assessment": assessment,
            "raw_output": raw_output if args.debug else "",
            "inference_time": inference_time
        })
    
    return results

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    notes = df[args.note_column].fillna("").tolist()
    patient_ids = df[args.id_column].tolist()
    true_labels = df[args.label_column].tolist() if args.label_column in df.columns else ["unknown"]*len(notes)
    
    print(f"Loading model from {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.load_in_4bit)
    
    print("Starting inference with strict output formatting...")
    results = run_inference(model, tokenizer, notes, patient_ids, true_labels, args)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(args.output_dir, "malnutrition_assessments.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")
    print("Assessment summary:")
    print(results_df['assessment'].value_counts())
    
    if "unknown" not in true_labels:
        calculate_metrics(results, args)

if __name__ == "__main__":
    main()
