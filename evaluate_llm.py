#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized malnutrition classification inference script with:
- Better prediction parsing
- Batch processing
- Error recovery
- Performance optimizations
"""

import os
import pandas as pd
import numpy as np
import torch
import argparse
import time
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from transformers import TextStreamer, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import json
from typing import List, Dict, Union, Optional

def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimized malnutrition classification inference")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to the trained model")
    parser.add_argument("--input_file", type=str, required=True,
                      help="Path to the CSV file with clinical notes")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for inference (default: 4)")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                      help="Maximum new tokens to generate (default: 32)")
    parser.add_argument("--min_new_tokens", type=int, default=2,
                      help="Minimum new tokens to generate (default: 2)")
    parser.add_argument("--load_in_4bit", action="store_true",
                      help="Load model in 4-bit quantization")
    parser.add_argument("--stream_output", action="store_true",
                      help="Stream output during generation")
    parser.add_argument("--test_mode", action="store_true",
                      help="Run on small subset for testing")
    parser.add_argument("--note_column", type=str, default="txt",
                      help="Column name for clinical notes")
    parser.add_argument("--id_column", type=str, default="DEID",
                      help="Column name for patient IDs")
    parser.add_argument("--label_column", type=str, default="label",
                      help="Column name for true labels")
    parser.add_argument("--temperature", type=float, default=0.1,
                      help="Sampling temperature (default: 0.1)")
    parser.add_argument("--top_p", type=float, default=0.9,
                      help="Top-p sampling (default: 0.9)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                      help="Repetition penalty (default: 1.1)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2,
                      help="No repeat n-gram size (default: 2)")
    parser.add_argument("--num_beams", type=int, default=2,
                      help="Number of beams for search (default: 2)")
    parser.add_argument("--early_stopping", action="store_true",
                      help="Enable early stopping")
    parser.add_argument("--preprocess_tokens", action="store_true",
                      help="Preprocess special tokens in notes")
    parser.add_argument("--max_seq_length", type=int, default=None,
                      help="Override model max sequence length")
    return parser.parse_args()

def preprocess_clinical_note(note_text: str) -> str:
    """Clean clinical notes by handling special tokens and formatting"""
    if not note_text:
        return note_text

    processed_text = note_text.replace('</s>', '\n\n')
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        processed_text = processed_text.replace(token, f"[{token[1:-1]}]")
    
    # Remove excessive whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text

def create_malnutrition_prompt_alpaca(note: str, tokenizer=None, max_tokens=None) -> str:
    """Create standardized prompt for malnutrition assessment"""
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Analyze the clinical note and determine if the patient has malnutrition based on standard criteria:

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

Respond with exactly "YES" for malnourished or "NO" for not malnourished.

### Input:
{note}

### Response:"""
    
    prompt = prompt_template.format(note=note)
    
    if tokenizer and max_tokens:
        tokens = tokenizer.encode(prompt)
        if len(tokens) > max_tokens:
            available_tokens = max_tokens - len(tokenizer.encode(prompt_template.format(note="")))
            note_tokens = tokenizer.encode(note)
            truncated_note = tokenizer.decode(note_tokens[:available_tokens])
            prompt = prompt_template.format(note=truncated_note)
    
    return prompt

def extract_yes_no(output_text: str) -> str:
    """Robust extraction of yes/no classification with multiple validation layers"""
    if not output_text:
        return "error"
    
    output_lower = output_text.lower().strip()
    
    # Layer 1: Direct answer patterns (highest confidence)
    direct_patterns = [
        r'(?:final[-\s]?answer|classification|malnutrition)\s*:\s*(yes|no)\b',
        r'^\s*(yes|no)\s*$',
        r'\banswer\s*:\s*(yes|no)\b',
        r'\bresponse\s*:\s*(yes|no)\b'
    ]
    
    for pattern in direct_patterns:
        match = re.search(pattern, output_lower)
        if match:
            return match.group(1)
    
    # Layer 2: Strong affirmative/negative phrases
    yes_phrases = [
        r'\bmalnourished\b',
        r'\bhas malnutrition\b',
        r'\bpositive for malnutrition\b',
        r'\bmalnutrition present\b',
        r'\bdiagnosed with malnutrition\b'
    ]
    
    no_phrases = [
        r'\bnot malnourished\b',
        r'\bno malnutrition\b',
        r'\bnegative for malnutrition\b',
        r'\bmalnutrition not present\b',
        r'\bno evidence of malnutrition\b'
    ]
    
    yes_count = sum(bool(re.search(p, output_lower)) for p in yes_phrases)
    no_count = sum(bool(re.search(p, output_lower)) for p in no_phrases)
    
    if yes_count > 0 and no_count == 0:
        return "yes"
    elif no_count > 0 and yes_count == 0:
        return "no"
    
    # Layer 3: Standalone yes/no with context validation
    standalone_match = re.search(r'\b(yes|no)\b', output_lower)
    if standalone_match:
        # Verify context around the standalone match
        context = output_lower[max(0, standalone_match.start()-20):standalone_match.end()+20]
        if ('malnutrition' in context or 'nutrition' in context or 
            'nourish' in context or 'diet' in context):
            return standalone_match.group(1)
    
    # Layer 4: Final fallback with strict checking
    if "yes" in output_lower and "no" not in output_lower:
        return "yes"
    elif "no" in output_lower and "yes" not in output_lower:
        return "no"
    
    return "error"

def load_model(model_path: str, load_in_4bit: bool, max_seq_length: Optional[int] = None):
    """Load model with proper configuration and error handling"""
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            dtype=None,
            load_in_4bit=load_in_4bit,
            max_seq_length=max_seq_length,
        )
        FastLanguageModel.for_inference(model)
        final_seq_length = max_seq_length if max_seq_length else model.config.max_position_embeddings
        print(f"Model loaded with max sequence length: {final_seq_length}")
        return model, tokenizer, final_seq_length
    except Exception as e:
        print(f"Error loading with Unsloth: {e}")
        print("Falling back to standard transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            torch_dtype=torch.float16
        )
        final_seq_length = max_seq_length if max_seq_length else tokenizer.model_max_length
        return model, tokenizer, final_seq_length

def batch_inference(
    model, 
    tokenizer, 
    prompts: List[str], 
    max_new_tokens: int,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 2,
    num_beams: int = 2,
    early_stopping: bool = True
) -> List[str]:
    """Efficient batch processing of prompts"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length - max_new_tokens
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            early_stopping=early_stopping,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

def run_inference(
    model, 
    tokenizer, 
    notes: List[str], 
    patient_ids: List[str], 
    true_labels: Optional[List[Union[int, str]]], 
    args, 
    max_seq_length: int
) -> List[Dict]:
    """Run optimized inference pipeline with error recovery"""
    results = []
    failed_indices = []
    
    # Preprocess all notes
    processed_notes = [
        preprocess_clinical_note(note) if args.preprocess_tokens else note 
        for note in notes
    ]
    
    # Create all prompts
    prompts = [
        create_malnutrition_prompt_alpaca(
            note,
            tokenizer=tokenizer,
            max_tokens=max_seq_length - args.max_new_tokens
        )
        for note in processed_notes
    ]
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Running inference"):
        batch_prompts = prompts[i:i+args.batch_size]
        batch_ids = patient_ids[i:i+args.batch_size]
        batch_labels = true_labels[i:i+args.batch_size] if true_labels else [None]*len(batch_prompts)
        
        try:
            start_time = time.time()
            batch_outputs = batch_inference(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                num_beams=args.num_beams,
                early_stopping=args.early_stopping
            )
            batch_time = time.time() - start_time
            
            for j, (output, pid, true_label) in enumerate(zip(batch_outputs, batch_ids, batch_labels)):
                try:
                    output_part = output.split("### Response:")[-1].strip() if "### Response:" in output else output
                    assessment = extract_yes_no(output_part)
                    
                    results.append({
                        "DEID": pid,
                        "true_label": true_label,
                        "predicted_label": assessment,
                        "full_output": output_part,
                        "original_note": processed_notes[i+j],
                        "inference_time": batch_time/len(batch_prompts)
                    })
                    
                    print(f"Patient {pid} | True: {true_label or 'unknown'} | Pred: {assessment}")
                
                except Exception as e:
                    print(f"Error processing patient {pid}: {str(e)}")
                    failed_indices.append(i+j)
        
        except Exception as e:
            print(f"Batch failed (patients {batch_ids[0]} to {batch_ids[-1]}): {str(e)}")
            failed_indices.extend(range(i, min(i+args.batch_size, len(prompts))))
    
    # Retry failed cases individually
    if failed_indices:
        print(f"\nRetrying {len(failed_indices)} failed cases individually...")
        for idx in tqdm(failed_indices, desc="Retrying failed cases"):
            try:
                start_time = time.time()
                output = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompts[idx],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    num_beams=args.num_beams,
                    early_stopping=args.early_stopping
                )
                inference_time = time.time() - start_time
                
                output_part = output.split("### Response:")[-1].strip() if "### Response:" in output else output
                assessment = extract_yes_no(output_part)
                
                results.append({
                    "DEID": patient_ids[idx],
                    "true_label": true_labels[idx] if true_labels else None,
                    "predicted_label": assessment,
                    "full_output": output_part,
                    "original_note": processed_notes[idx],
                    "inference_time": inference_time
                })
                
                print(f"[Retry] Patient {patient_ids[idx]} | Pred: {assessment}")
            
            except Exception as e:
                print(f"Failed again on patient {patient_ids[idx]}: {str(e)}")
                results.append({
                    "DEID": patient_ids[idx],
                    "true_label": true_labels[idx] if true_labels else None,
                    "predicted_label": "error",
                    "full_output": str(e),
                    "original_note": processed_notes[idx],
                    "inference_time": 0
                })
    
    return results

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 2,
    num_beams: int = 2,
    early_stopping: bool = True,
    streamer=None
) -> str:
    """Generate text with comprehensive generation parameters"""
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True).to(model.device)
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "num_beams": num_beams,
        "early_stopping": early_stopping,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    if streamer:
        generation_kwargs["streamer"] = streamer
    
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_metrics(results: List[Dict], args) -> Optional[Dict]:
    """Calculate and save comprehensive performance metrics"""
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Filter valid results
    valid_results = [r for r in results if r["true_label"] not in [None, "unknown"] and r["predicted_label"] != "error"]
    
    if not valid_results:
        print("No valid results for metrics calculation")
        return None
    
    # Convert labels to binary
    y_true = []
    y_pred = []
    
    for r in valid_results:
        # Handle true label conversion
        if isinstance(r["true_label"], (int, float)):
            y_true.append(int(r["true_label"]))
        else:
            y_true.append(1 if str(r["true_label"]).lower() in ["1", "yes"] else 0)
        
        # Handle predicted label
        y_pred.append(1 if r["predicted_label"].lower() == "yes" else 0)
    
    # Classification Report
    report = classification_report(
        y_true, y_pred, 
        output_dict=True, 
        target_names=["Non-malnourished", "Malnourished"]
    )
    
    # Save reports
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(metrics_dir, "classification_report.csv"))
    
    with open(os.path.join(metrics_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=["Non-malnourished", "Malnourished"]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Non-malnourished", "Actual Malnourished"],
        columns=["Predicted Non-malnourished", "Predicted Malnourished"]
    )
    cm_df.to_csv(os.path.join(metrics_dir, "confusion_matrix.csv"))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Non-malnourished", "Malnourished"])
    plt.yticks(tick_marks, ["Non-malnourished", "Malnourished"])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    
    # Metrics summary
    metrics_summary = {
        "accuracy": report["accuracy"],
        "precision_malnourished": report["Malnourished"]["precision"],
        "recall_malnourished": report["Malnourished"]["recall"],
        "f1_malnourished": report["Malnourished"]["f1-score"],
        "precision_nonmalnourished": report["Non-malnourished"]["precision"],
        "recall_nonmalnourished": report["Non-malnourished"]["recall"],
        "f1_nonmalnourished": report["Non-malnourished"]["f1-score"],
        "total_samples": len(valid_results),
        "malnourished_count": sum(y_true),
        "nonmalnourished_count": len(y_true) - sum(y_true),
        "error_rate": len([r for r in results if r["predicted_label"] == "error"]) / len(results),
        "avg_inference_time": np.mean([r["inference_time"] for r in results if r["inference_time"] > 0])
    }
    
    with open(os.path.join(metrics_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    return metrics_summary

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Validate columns
    for col in [args.note_column, args.id_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input file")
    
    notes = df[args.note_column].tolist()
    patient_ids = df[args.id_column].astype(str).tolist()
    
    true_labels = None
    if args.label_column and args.label_column in df.columns:
        print(f"Found label column '{args.label_column}'")
        true_labels = df[args.label_column].tolist()
        print(f"First 5 labels: {true_labels[:5]}")
    else:
        print("No label column found - running in prediction-only mode")
    
    if args.test_mode:
        print("Running in test mode with first 5 samples")
        notes = notes[:5]
        patient_ids = patient_ids[:5]
        if true_labels:
            true_labels = true_labels[:5]
    
    print(f"Loading model from: {args.model_path}")
    model, tokenizer, max_seq_length = load_model(
        args.model_path,
        args.load_in_4bit,
        args.max_seq_length
    )
    
    print("Starting inference...")
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        notes=notes,
        patient_ids=patient_ids,
        true_labels=true_labels,
        args=args,
        max_seq_length=max_seq_length
    )
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(args.output_dir, "inference_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Calculate metrics if labels available
    if true_labels:
        print("Calculating metrics...")
        metrics = calculate_metrics(results, args)
        
        if metrics:
            print("\n=== Metrics Summary ===")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Malnourished - Precision: {metrics['precision_malnourished']:.4f}, Recall: {metrics['recall_malnourished']:.4f}, F1: {metrics['f1_malnourished']:.4f}")
            print(f"Non-malnourished - Precision: {metrics['precision_nonmalnourished']:.4f}, Recall: {metrics['recall_nonmalnourished']:.4f}, F1: {metrics['f1_nonmalnourished']:.4f}")
            print(f"Error Rate: {metrics['error_rate']:.4f}")
            print(f"Avg Inference Time: {metrics['avg_inference_time']:.4f}s")
    
    # Final summary
    pred_counts = results_df["predicted_label"].value_counts().to_dict()
    print("\n=== Inference Summary ===")
    print(f"Total processed: {len(results)}")
    print(f"Malnourished predictions: {pred_counts.get('yes', 0)}")
    print(f"Non-malnourished predictions: {pred_counts.get('no', 0)}")
    print(f"Errors: {pred_counts.get('error', 0)}")

if __name__ == "__main__":
    main()
