#!/usr/bin/env python
import pandas as pd
import numpy as np
import json
import torch
import re
import random
import os
import argparse
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel
from tqdm.auto import tqdm

# Import your prompt builder
# This assumes the MalnutritionPromptBuilder is in a file called prompt_builder.py
# If not, adjust the import statement accordingly
from models.llm_models import MalnutritionPromptBuilder, preprocess_text

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run malnutrition prediction using a fine-tuned LLM.')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, default="lora_model",
                        help='Path to your saved LoRA model')
    parser.add_argument('--input_csv', type=str, default="input_data.csv",
                        help='Path to your input CSV file')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default="outputs",
                        help='Directory to save output files')
    parser.add_argument('--max_length', type=int, default=4096,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--num_examples', type=int, default=2,
                        help='Number of few-shot examples to use (0 for zero-shot)')
    parser.add_argument('--examples_csv', type=str, default=None,
                        help='Path to examples CSV if using few-shot (optional)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Temperature for text generation')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p value for text generation')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed for reproducibility')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load model in 4-bit precision')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed debug output')
    parser.add_argument('--sample_limit', type=int, default=None,
                        help='Limit the number of samples to process (for debugging)')
    
    return parser.parse_args()

def extract_json_from_output(text):
    """Extract JSON from model output."""
    try:
        # Debug: Print the text we're trying to parse
        print("\nDebug - Attempting to extract JSON from:")
        print(f"Text length: {len(text)} characters")
        print(f"First 100 chars: {text[:100]}...")
        print(f"Last 100 chars: ...{text[-100:]}")
        
        # Find JSON pattern between curly braces
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(0)
            print(f"Debug - Found potential JSON: {json_str[:100]}...")
            
            # Parse JSON
            parsed_json = json.loads(json_str)
            print("Debug - Successfully parsed JSON:")
            print(json.dumps(parsed_json, indent=2)[:200] + "..." if len(json.dumps(parsed_json, indent=2)) > 200 else json.dumps(parsed_json, indent=2))
            return parsed_json
        else:
            print("Debug - No JSON pattern found in text")
            return None
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw text excerpt: {text[:200]}...")
        return None

def get_model_prediction(model, tokenizer, prompt, device, temperature, top_p,max_seq_length):
    """Get model prediction for a single prompt."""
    # Debug: Print prompt length and first/last part
    print(f"\nDebug - Prompt length: {len(prompt)} characters")
    print(f"Prompt starts with: {prompt[:100]}...")
    print(f"Prompt ends with: ...{prompt[-100:]}")
    
    inputs = tokenizer(prompt,truncation=True, max_length=max_seq_length, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=temperature,
        top_p=top_p,
        do_sample=False
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the part after the prompt
    result = response[len(prompt):]
    
    # Debug: Print response length and first part
    print(f"Debug - Raw response length: {len(result)} characters")
    print(f"Raw response starts with: {result[:150]}...")
    
    return result

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Print debug information about arguments
    print("\n=== Script Configuration ===")
    print(f"Model path: {args.model_path}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of few-shot examples: {args.num_examples}")
    print(f"Examples CSV: {args.examples_csv}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"Sample limit: {args.sample_limit if args.sample_limit else 'No limit'}")
    print("============================")
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_length,
        dtype=None,  # Auto-detect
        load_in_4bit=args.load_in_4bit,
    )
    
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    model.eval()
    
    # Load the dataset
    print(f"Loading dataset from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    required_columns = ["DEID", "label", "txt"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV")
    
    # Initialize prompt builder
    prompt_builder = MalnutritionPromptBuilder(examples_csv_path=args.examples_csv)
    
    # Process the dataset and generate predictions
    print("Generating predictions...")
    results = []
    binary_preds = []
    binary_true = []
    confidences = []
    
    # Apply sample limit if specified (for debugging)
    if args.sample_limit:
        print(f"Debug: Limiting analysis to first {args.sample_limit} samples")
        df = df.head(args.sample_limit)
    
    # Process in batches
    for i in tqdm(range(0, len(df), args.batch_size)):
        batch_df = df.iloc[i:i+args.batch_size]
        
        for _, row in batch_df.iterrows():
            patient_id = row["DEID"]
            true_label = row["label"]
            patient_notes = row["txt"]
            
            # Convert true label to binary (0/1)
            true_binary = 1 if str(true_label).lower() in ["1", "yes", "true"] else 0
            
            # Generate prompt
            prompt = prompt_builder.get_inference_prompt(
                patient_notes=patient_notes,
                note_col="txt",
                label_col="label",
                num_examples=args.num_examples,
                balanced=True
            )
            
            # Debug: Print patient info
            if args.debug:
                print(f"\n==== Processing Patient ID: {patient_id} ====")
                print(f"Patient notes length: {len(patient_notes)} characters")
                print(f"Notes excerpt: {patient_notes[:150]}...")
                print(f"True label: {true_label}")
            
            # Get model prediction
            raw_output = get_model_prediction(
                model, tokenizer, prompt, device, args.temperature, args.top_p,args.max_seq_length
            )
            
            # Parse JSON from output
            parsed_output = extract_json_from_output(raw_output)
            
            if parsed_output is None or "malnutrition" not in parsed_output:
                print(f"Failed to parse output for patient {patient_id}")
                predicted_label = "no"
                confidence = 0.5
                explanation = "Error in prediction"
                prediction_binary = 0
            else:
                predicted_label = parsed_output.get("malnutrition", "no").lower()
                confidence = parsed_output.get("confidence", 0.5)
                explanation = parsed_output.get("explanation", "")
                prediction_binary = 1 if predicted_label == "yes" else 0
                
            # Debug: Print prediction details
            print(f"\n--- Debug: Patient {patient_id} ---")
            print(f"True label: {true_label}")
            print(f"Predicted label: {predicted_label} (confidence: {confidence:.2f})")
            print(f"Explanation: {explanation[:100]}..." if len(explanation) > 100 else f"Explanation: {explanation}")
            if parsed_output is None:
                print(f"Raw output (first 200 chars): {raw_output[:200]}...")
            
            # Store results
            results.append({
                "DEID": patient_id,
                "txt": preprocess_text(patient_notes),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "explanation": explanation,
                "confidence": confidence,
                "original_notes": patient_notes
            })
            
            # Store binary predictions for ROC curve
            binary_preds.append(confidence if prediction_binary == 1 else 1 - confidence)
            binary_true.append(true_binary)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save predictions to CSV
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    results_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(binary_true, binary_preds)
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Generate ROC curve data
        fpr, tpr, thresholds = roc_curve(binary_true, binary_preds)
        
        # Save ROC data to CSV
        roc_data = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        })
        roc_data_path = os.path.join(args.output_dir, "roc_auc_data.csv")
        roc_data.to_csv(roc_data_path, index=False)
        print(f"ROC curve data saved to {roc_data_path}")
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Malnutrition Prediction')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        roc_plot_path = os.path.join(args.output_dir, "roc_curve.png")
        plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to {roc_plot_path}")
        
    except Exception as e:
        print(f"Error calculating ROC-AUC: {e}")
    
    # Generate summary metrics
    total = len(results_df)
    correctly_predicted = sum(
        (results_df['true_label'].astype(str).str.lower().isin(['1', 'yes', 'true']) & 
         results_df['predicted_label'].str.lower() == 'yes') |
        (~results_df['true_label'].astype(str).str.lower().isin(['1', 'yes', 'true']) & 
         results_df['predicted_label'].str.lower() == 'no')
    )
    accuracy = correctly_predicted / total
    
    # Calculate precision, recall, and F1 score for "yes" class
    true_positive = sum(results_df['true_label'].astype(str).str.lower().isin(['1', 'yes', 'true']) & 
                        (results_df['predicted_label'].str.lower() == 'yes'))
    false_positive = sum(~results_df['true_label'].astype(str).str.lower().isin(['1', 'yes', 'true']) & 
                         (results_df['predicted_label'].str.lower() == 'yes'))
    false_negative = sum(results_df['true_label'].astype(str).str.lower().isin(['1', 'yes', 'true']) & 
                         (results_df['predicted_label'].str.lower() == 'no'))
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Save summary metrics
    summary = {
        'total_samples': total,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc if 'roc_auc' in locals() else 'N/A'
    }
    
    summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['value'])
    summary_path = os.path.join(args.output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path)
    print(f"Summary metrics saved to {summary_path}")
    
    # Print summary
    print("\nPerformance Summary:")
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if 'roc_auc' in locals():
        print(f"ROC-AUC: {roc_auc:.4f}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
