#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Malnutrition Assessment Tool with Enhanced Error Handling and Prediction Robustness
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
import signal

class TimeoutException(Exception): pass

def timeout_handler(signum, frame): raise TimeoutException()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run malnutrition assessment inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--stream_output", action="store_true")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--note_column", type=str, default="txt")
    parser.add_argument("--id_column", type=str, default="DEID")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--preprocess_tokens", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()

def preprocess_clinical_note(note_text):
    if not note_text or not isinstance(note_text, str):
        return "" if note_text is None else str(note_text)
    processed_text = note_text.replace('</s>', '\n\n')
    for token in ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']:
        processed_text = processed_text.replace(token, f"[{token}]")
    return processed_text

def create_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    prompt = """You are a pediatric dietitian evaluating malnutrition status in children based on clinical notes.
MALNUTRITION CRITERIA
* Mild: z-score -1 to -1.9 SD (weight-for-height, BMI-for-age)
* Moderate: z-score -2 to -2.9 SD (weight-for-height, BMI-for-age)
* Severe: z-score ≤ -3 SD or severe stunting (length/height-for-age ≤ -3 SD)
* Physical signs: muscle wasting, reduced subcutaneous fat, edema
* Growth trajectory: declining percentiles, weight loss, poor weight gain
IMPORTANT GUIDELINES
* Weight is primarily affected during acute undernutrition
* Chronic undernutrition typically manifests as stunting
* Severe acute undernutrition (ages 6–60 months): very low weight-for-height (< -3 SD z-scores), visible severe wasting (MUAC ≤115 mm), or nutritional edema
* Chronic undernutrition/stunting: height-for-age < -2 SD z-score

Analyze the clinical note below and provide ONLY the binary classification in the following format:
malnutrition=[yes/no]

Clinical note:
{note}

Assessment:"""
    if tokenizer and max_tokens:
        base_tokens = len(tokenizer.encode(prompt.format(note="")))
        safety_margin = 128
        max_note_tokens = max_tokens - base_tokens - safety_margin
        if max_note_tokens > 0:
            note_tokens = tokenizer.encode(note)
            if len(note_tokens) > max_note_tokens:
                note = tokenizer.decode(note_tokens[:max_note_tokens], skip_special_tokens=True)
                print(f"Truncated note from {len(note_tokens)} to {max_note_tokens} tokens")
    return prompt.format(note=note)

def extract_decision(prediction):
    prediction = prediction.replace('\r', '\n').strip()
    match = re.search(r'malnutrition\s*[:=]?\s*(yes|no)', prediction, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    prediction_lower = prediction.lower()
    if "yes" in prediction_lower and "no" not in prediction_lower:
        return "yes"
    elif "no" in prediction_lower and "yes" not in prediction_lower:
        return "no"
    return "error"

def load_model(model_path, load_in_4bit):
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_path, dtype=None, load_in_4bit=load_in_4bit)
        FastLanguageModel.for_inference(model)
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)
    except Exception as e:
        print(f"Error loading with Unsloth: {e}")
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, load_in_4bit=load_in_4bit, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)
    print(f"Model loaded with max sequence length: {max_seq_length}")
    return model, tokenizer, max_seq_length

def generate_assessment(model, tokenizer, prompt, max_new_tokens, streamer=None, 
                        max_seq_length=None, temperature=0.1, top_p=0.9, timeout=60):
    inputs = tokenizer([prompt], return_tensors="pt")
    if max_seq_length and inputs.input_ids.shape[1] > max_seq_length - max_new_tokens:
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id
        )
        signal.alarm(0)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except TimeoutException:
        return prompt + "\n[TIMEOUT: Exceeded generation time]"
    except Exception as e:
        return prompt + f"\n[ERROR: {str(e)}]"

def run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length):
    results = []
    streamer = TextStreamer(tokenizer) if args.stream_output else None
    for i in tqdm(range(0, len(notes), args.batch_size), desc="Assessing patients"):
        batch_notes = notes[i:i+args.batch_size]
        batch_ids = patient_ids[i:i+args.batch_size]
        batch_labels = true_labels[i:i+args.batch_size] if true_labels else [None]*len(batch_notes)
        for note, pid, label in zip(batch_notes, batch_ids, batch_labels):
            try:
                if args.preprocess_tokens:
                    note = preprocess_clinical_note(note)
                prompt = create_malnutrition_prompt(note, tokenizer, max_tokens=max_seq_length - args.max_new_tokens)
                start_time = time.time()
                prediction = generate_assessment(model, tokenizer, prompt, args.max_new_tokens, streamer,
                                                 max_seq_length, args.temperature, args.top_p, args.timeout)
                elapsed = time.time() - start_time
                pred_label = extract_decision(prediction)
                results.append({"DEID": pid, "true_label": label if label is not None else "unknown",
                                "predicted_label": pred_label, "inference_time": elapsed,
                                "raw_prediction": prediction if args.test_mode else ""})
                print(f"Patient ID: {pid} | True: {label} | Pred: {pred_label} | Time: {elapsed:.2f}s")
                if pred_label == "error":
                    with open(os.path.join(args.output_dir, "error_cases.txt"), "a") as f:
                        f.write(f"\nPatient {pid}\n---\n{prediction}\n\n")
            except Exception as e:
                print(f"Error processing patient {pid}: {str(e)}")
                results.append({"DEID": pid, "true_label": label if label is not None else "unknown",
                                "predicted_label": "error", "inference_time": 0,
                                "raw_prediction": f"ERROR: {str(e)}"})
    return results

def calculate_metrics(results, args):
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    valid = [r for r in results if r["true_label"] != "unknown" and r["predicted_label"] != "error"]
    if not valid:
        print("No valid results for metrics calculation")
        return None
    y_true = [int(r["true_label"]) if isinstance(r["true_label"], (int, float)) else 1 if str(r["true_label"]).lower() == "yes" else 0 for r in valid]
    y_pred = [1 if r["predicted_label"] == "yes" else 0 for r in valid]
    y_prob = [0.9 if p == 1 else 0.1 for p in y_pred]
    report = classification_report(y_true, y_pred, output_dict=True,
                                   target_names=["Non-malnourished", "Malnourished"])
    pd.DataFrame(report).transpose().to_csv(os.path.join(metrics_dir, "classification_report.csv"))
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=["Actual Non-malnourished", "Actual Malnourished"],
                 columns=["Predicted Non-malnourished", "Predicted Malnourished"]).to_csv(os.path.join(metrics_dir, "confusion_matrix.csv"))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
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
    summary = {"accuracy": report["accuracy"],
               "precision": report["Malnourished"]["precision"],
               "recall": report["Malnourished"]["recall"],
               "f1": report["Malnourished"]["f1-score"],
               "auc": roc_auc,
               "error_rate": sum(1 for r in results if r["predicted_label"] == "error") / len(results)}
    with open(os.path.join(metrics_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n==== METRICS SUMMARY ====")
    for k, v in summary.items():
        print(f"{k.capitalize()}: {v:.3f}")
    return summary

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    if args.note_column not in df.columns:
        print(f"Error: Column '{args.note_column}' not found")
        return
    patient_ids = df[args.id_column].astype(str).tolist() if args.id_column in df.columns else df.index.astype(str).tolist()
    notes = df[args.note_column].tolist()
    true_labels = df[args.label_column].tolist() if args.label_column in df.columns else None
    if args.test_mode:
        notes, patient_ids = notes[:5], patient_ids[:5]
        true_labels = true_labels[:5] if true_labels else None
    print(f"Loading model from {args.model_path}")
    model, tokenizer, max_seq_length = load_model(args.model_path, args.load_in_4bit)
    print("Starting inference...")
    results = run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length)
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "assessment_results.csv"), index=False)
    if true_labels:
        calculate_metrics(results, args)
    print("\nInference completed successfully")

if __name__ == "__main__":
    main()
