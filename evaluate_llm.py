#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized malnutrition classification inference script with:
1. Strict yes/no output requirements
2. No prompt contamination in predictions
3. Removed redundant code
4. Improved confidence scoring
"""

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
import argparse
import pandas as pd
import numpy as np
import torch
import re
from datasets import Dataset
from transformers import pipeline
from unsloth import FastLanguageModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimized malnutrition classifier")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV test data")
    parser.add_argument("--output_path", type=str, default="./results.csv", 
                       help="Output path for results")
    parser.add_argument("--max_seq_length", type=int, default=8192, 
                       help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--temperature", type=float, default=0.1, 
                       help="Generation temperature")
    parser.add_argument("--max_new_tokens", type=int, default=64, 
                       help="Max new tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.9, 
                       help="Top-p sampling value")
    return parser.parse_args()

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def create_prompt(note):
    """Create strict classification prompt"""
    return f"""Analyze this clinical note and respond ONLY with "malnutrition=yes" or "malnutrition=no":
    
{note}

Response:"""

def extract_prediction(text):
    """Strict extraction of yes/no prediction from generated text"""
    if not text:
        return None
    
    # Remove any prompt remnants
    cleaned = text.split("Response:")[-1].strip()
    
    # Strict matching
    if re.match(r'^\s*malnutrition\s*=\s*yes\s*$', cleaned, re.I):
        return 1
    if re.match(r'^\s*malnutrition\s*=\s*no\s*$', cleaned, re.I):
        return 0
    
    return None

def calculate_confidence(text):
    """Calculate confidence score based on response characteristics"""
    if not text:
        return 0.5
    
    # Penalize long responses that didn't follow instructions
    if len(text) > 50:
        return 0.6
    
    # Reward concise correct format
    if re.match(r'^\s*malnutrition\s*=\s*(yes|no)\s*$', text, re.I):
        return 0.95
    
    return 0.75  # Default medium confidence

def load_model(args):
    """Load model with optimized settings"""
    print(f"Loading model from {args.model_path}...")
    return FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

def prepare_dataset(data_path):
    """Load and prepare dataset"""
    df = pd.read_csv(data_path)
    if "txt" not in df.columns:
        raise ValueError("CSV must contain 'txt' column")
    
    df['cleaned_text'] = df['txt'].apply(clean_text)
    df['prompt'] = df['cleaned_text'].apply(create_prompt)
    
    has_labels = "label" in df.columns
    if has_labels:
        df['label'] = df['label'].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true'] else 0)
    
    return Dataset.from_pandas(df)

def generate_predictions(model, tokenizer, dataset, args):
    """Run batch inference with strict output control"""
    model.eval()
    predictions = []
    confidences = []
    generated_texts = []
    
    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i:i+args.batch_size]
        inputs = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length - args.max_new_tokens
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        batch_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for text in batch_texts:
            # Remove prompt from generated text
            generated = text.split("Response:")[-1].strip()
            pred = extract_prediction(generated)
            conf = calculate_confidence(generated)
            
            predictions.append(pred)
            confidences.append(conf)
            generated_texts.append(generated)
            
    return predictions, confidences, generated_texts

def evaluate_results(true_labels, predictions, confidences, output_dir="results"):
    """Comprehensive evaluation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out None predictions
    valid_idx = [i for i, p in enumerate(predictions) if p is not None]
    if not valid_idx:
        print("No valid predictions!")
        return {}
    
    y_true = [true_labels[i] for i in valid_idx]
    y_pred = [predictions[i] for i in valid_idx]
    y_prob = [confidences[i] for i in valid_idx]
    
    # Binary metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()
    
    return metrics

def save_results(dataset, predictions, confidences, generated_texts, args):
    """Save all results to CSV"""
    results = pd.DataFrame({
        'original_text': dataset['txt'],
        'generated_text': generated_texts,
        'prediction': predictions,
        'confidence': confidences
    })
    
    if 'label' in dataset.column_names:
        results['true_label'] = dataset['label']
    
    results.to_csv(args.output_path, index=False)

def main():
    args = parse_arguments()
    
    # Load components
    model, tokenizer = load_model(args)
    dataset = prepare_dataset(args.data_path)
    
    # Run inference
    predictions, confidences, generated_texts = generate_predictions(
        model, tokenizer, dataset, args
    )
    
    # Evaluate if labels available
    if 'label' in dataset.column_names:
        metrics = evaluate_results(
            dataset['label'],
            predictions,
            confidences
        )
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    
    # Save results
    save_results(dataset, predictions, confidences, generated_texts, args)
    print(f"\nResults saved to {args.output_path}")

if __name__ == "__main__":
    main()
