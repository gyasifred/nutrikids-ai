#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference on clinical notes using the trained malnutrition assessment model.
Includes enhanced metrics tracking, classification reports, ROC curves with probability scores,
model explanations, and preprocessing for special tokens like </s>.
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
from transformers import TextStreamer, LogitsProcessor, LogitsProcessorList
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import shap
import pickle
import json
import torch.nn.functional as F


class YesNoLogitsProcessor(LogitsProcessor):
    """
    Custom logits processor to extract probability scores for yes/no classification.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Find token IDs for yes and no (considering potential casing)
        self.yes_token_ids = []
        self.no_token_ids = []
        
        # Encode all possible variations with and without leading space
        for prefix in ['', ' ', '\n']:
            for variant in ['yes', 'Yes', 'YES']:
                tokens = tokenizer.encode(prefix + variant, add_special_tokens=False)
                if tokens and tokens[0] not in self.yes_token_ids:
                    self.yes_token_ids.append(tokens[0])
            
            for variant in ['no', 'No', 'NO']:
                tokens = tokenizer.encode(prefix + variant, add_special_tokens=False)
                if tokens and tokens[0] not in self.no_token_ids:
                    self.no_token_ids.append(tokens[0])
        
        self.probabilities = {"yes": 0.0, "no": 0.0}
        self.step = 0  # Track generation step
        
    def __call__(self, input_ids, scores):
        # Only process the first generation step
        if self.step == 0:
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(scores, dim=-1)
            
            # Extract yes/no probabilities
            yes_prob = sum(probs[0, yes_id].item() for yes_id in self.yes_token_ids)
            no_prob = sum(probs[0, no_id].item() for no_id in self.no_token_ids)
            
            # Normalize to sum to 1
            total = yes_prob + no_prob
            if total > 0:
                self.probabilities["yes"] = yes_prob / total
                self.probabilities["no"] = no_prob / total
            else:
                # Handle case where neither token was found
                self.probabilities["yes"] = 0.0
                self.probabilities["no"] = 0.0
        
        self.step += 1
        return scores
    
    def get_probabilities(self):
        return self.probabilities

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
    parser.add_argument("--generate_explanations", action="store_true", 
                        help="Generate explanations for each prediction")
    parser.add_argument("--explanation_prompt", type=str, default="explain",
                        help="Prompt style for explanations: 'explain', 'evidence', or 'detailed'")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Temperature for sampling during generation (default: 0.3)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter (default: 0.9)")
    parser.add_argument("--preprocess_tokens", action="store_true", 
                        help="Preprocess </s> tokens in clinical notes")
    return parser.parse_args()


def preprocess_clinical_note(note_text):
    """
    Preprocess clinical notes to handle special tokens like </s> that might interfere with model training.
    
    Args:
        note_text (str): The raw clinical note text
        
    Returns:
        str: Processed clinical note text with special tokens handled
    """
    if not note_text:
        return note_text
    
    # Replace '</s>' tokens with a more appropriate separator that won't be interpreted as special
    # Using double newlines to maintain document structure
    processed_text = note_text.replace('</s>', '\n\n')
    
    # Check for any remaining special tokens that might interfere
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")  # Escape them
    
    return processed_text

def create_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """Create balanced malnutrition assessment prompt with clear criteria for both positive and negative determinations."""
    base_prompt = """[Role] Pediatric nutrition specialist analyzing growth charts and clinical documentation.

[WHO Diagnostic Framework]
<<CRITERIA FOR MALNUTRITION>>
1. **Severe Acute Malnutrition (SAM):**
   - Weight-for-height/BMI-for-age < -3 SD z-score **OR**
   - MUAC < 115 mm (6-59mo) **OR**
   - Bilateral pitting edema
2. **Moderate Acute Malnutrition (MAM):**
   - Weight-for-height/BMI-for-age -2 to -3 SD z-score
3. **Stunting (Chronic):**
   - Height-for-age < -2 SD z-score
4. **Supportive Indicators:**
   ▸ Weight loss >5% in 30 days
   ▸ Inadequate intake >5 days
   ▸ Documented growth decline >2 percentiles
<</CRITERIA FOR MALNUTRITION>>

<<CRITERIA FOR NORMAL NUTRITIONAL STATUS>>
1. **Normal Growth Parameters:**
   - Weight-for-height/BMI-for-age ≥ -2 SD z-score
   - Height-for-age ≥ -2 SD z-score
   - MUAC ≥ 115 mm (for children 6-59mo)
2. **Normal Clinical Assessment:**
   - No bilateral pitting edema
   - No significant recent weight loss (<5% in 30 days)
   - Adequate dietary intake
   - Stable or appropriate growth trajectory
3. **Note:** Growth measurements should be evaluated in context of the child's overall clinical picture and medical history
<</CRITERIA FOR NORMAL NUTRITIONAL STATUS>>

[Assessment Protocol]
1. Primary reliance on anthropometrics and clinical data
2. Differentiate acute vs chronic patterns when present
3. Confirm with clinical correlates
4. Determine "yes" ONLY when criteria for malnutrition are definitively met
5. Determine "no" when criteria for normal nutritional status are met

[Output Format]
Strictly follow this structure:

### Assessment:
<yes/no>  # FIRST TOKEN MUST BE yes/no, based ONLY on evidence meeting criteria

### Severity:
<sam/mam/stunting/none>

### Basis:
- Maximum 3 bullet points
- Cite specific z-scores or measurements when available
- Note key clinical findings
- For "no" assessments, document normal growth parameters

Clinical note for analysis:
"""

    # Token-aware note truncation
    if tokenizer and max_tokens:
        # Calculate token budgets
        base_tokens = len(tokenizer.encode(base_prompt))
        safety_margin = 128  # For generation space
        max_note_tokens = max_tokens - base_tokens - safety_margin
        
        if max_note_tokens > 0:
            note_tokens = tokenizer.encode(note)
            if len(note_tokens) > max_note_tokens:
                note = tokenizer.decode(note_tokens[:max_note_tokens])
                print(f"Truncated note from {len(note_tokens)} to {max_note_tokens} tokens")

    return base_prompt + note + "\n\n### Assessment:\n"

def create_explanation_prompt(note, assessment, prompt_style="explain", tokenizer=None, max_tokens=None):
    """Create a prompt for generating explanations of the model's prediction with optional length control."""
    if prompt_style == "explain":
        base_prompt = """You are a pediatric dietitian who has assessed that a child is {}malnourished based on their clinical notes.
Provide a brief explanation (2-3 sentences) for this assessment, focusing on the key factors that led to this conclusion.

### Clinical Note:
"""
        base_prompt = base_prompt.format("" if assessment.lower() == "yes" else "not ")
    
    elif prompt_style == "evidence":
        base_prompt = """You are a pediatric dietitian who has assessed that a child is {}malnourished based on their clinical notes.
List the specific evidence from the clinical note that supports this assessment.

### Clinical Note:
"""
        base_prompt = base_prompt.format("" if assessment.lower() == "yes" else "not ")
    
    elif prompt_style == "detailed":
        base_prompt = """You are a pediatric dietitian who has assessed that a child is {}malnourished based on their clinical notes.
Provide a detailed explanation of this assessment, addressing:
1. Key anthropometric data that influenced the decision
2. Growth trajectory indicators noted in the chart
3. Physical signs or symptoms supporting the assessment
4. Other relevant nutritional factors

### Clinical Note:
"""
        base_prompt = base_prompt.format("" if assessment.lower() == "yes" else "not ")
    
    else:
        # Default to the simple explain prompt
        return create_explanation_prompt(note, assessment, "explain", tokenizer, max_tokens)
    
    # If tokenizer and max_tokens are provided, truncate the note if needed
    if tokenizer and max_tokens:
        # Calculate approximate token count for the base prompt and ending
        if prompt_style == "explain":
            ending = "\n\n### Explanation:\n"
        elif prompt_style == "evidence":
            ending = "\n\n### Evidence:\n"
        elif prompt_style == "detailed":
            ending = "\n\n### Detailed Explanation:\n"
            
        base_tokens = len(tokenizer.encode(base_prompt))
        ending_tokens = len(tokenizer.encode(ending))
        
        # Calculate how many tokens we can allocate to the note
        available_tokens = max_tokens - base_tokens - ending_tokens - 50  # 50 tokens buffer
        
        # Tokenize and truncate the note if needed
        note_tokens = tokenizer.encode(note)
        if len(note_tokens) > available_tokens:
            truncated_note_tokens = note_tokens[:available_tokens]
            note = tokenizer.decode(truncated_note_tokens)
            print(f"Note truncated from {len(note_tokens)} to {available_tokens} tokens for explanation")
    
    # Complete the prompt
    if prompt_style == "explain":
        prompt = base_prompt + note + "\n\n### Explanation:\n"
    elif prompt_style == "evidence":
        prompt = base_prompt + note + "\n\n### Evidence:\n"
    elif prompt_style == "detailed":
        prompt = base_prompt + note + "\n\n### Detailed Explanation:\n"
        
    return prompt


def load_model(model_path, load_in_4bit):
    """Load the fine-tuned model with native sequence length."""
    try:
        # Try loading with Unsloth first - using model's native max_seq_length
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            dtype=None,  # Auto detect
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)  # Enable faster inference
        
        # Get model's native max sequence length
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)
        print(f"Using model's native max sequence length: {max_seq_length}")
        
    except Exception as e:
        print(f"Error loading with Unsloth: {e}")
        print("Trying to load with Hugging Face PEFT...")
        # Fallback to Hugging Face PEFT
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Get model's native max sequence length
        max_seq_length = getattr(model.config, "max_position_embeddings", 4096)
        print(f"Using model's native max sequence length: {max_seq_length}")

    return model, tokenizer, max_seq_length


def generate_text_with_probabilities(model, tokenizer, prompt, max_new_tokens=100, streamer=None, max_seq_length=None, temperature=0.3, top_p=0.9):
    """Generate text and extract yes/no probabilities."""
    # Create logits processor for probability extraction
    yes_no_processor = YesNoLogitsProcessor(tokenizer)
    logits_processors = LogitsProcessorList([yes_no_processor])
    
    # Tokenize the prompt
    inputs = tokenizer([prompt], return_tensors="pt")
    
    # Check if the input is too long
    if max_seq_length and inputs.input_ids.shape[1] > max_seq_length - max_new_tokens:
        print(f"Warning: Input length {inputs.input_ids.shape[1]} exceeds maximum allowed when combined with max_new_tokens.")
        # Truncate to leave room for generated tokens
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
        print(f"Input was truncated to {inputs.input_ids.shape[1]} tokens to fit within model context")
    
    # Move input to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate output with temperature and top_p parameters
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "temperature": temperature,
        "top_p": top_p,
        "logits_processor": logits_processors
    }
    
    if streamer:
        generation_kwargs["streamer"] = streamer
    
    outputs = model.generate(**generation_kwargs)
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Get the probabilities from the logits processor
    probabilities = yes_no_processor.get_probabilities()
    
    return prediction, probabilities


def generate_explanation(model, tokenizer, note, assessment, args, max_seq_length):
    """Generate an explanation for the model's prediction."""
    explanation_prompt = create_explanation_prompt(
        note, 
        assessment, 
        args.explanation_prompt,
        tokenizer=tokenizer,
        max_tokens=max_seq_length - 256  # Reserve tokens for the explanation
    )
    
    # Generate a longer response for explanation
    streamer = None
    if args.stream_output:
        streamer = TextStreamer(tokenizer)
        print("\nGenerating explanation...")
    
    explanation_text, _ = generate_text_with_probabilities(
        model, 
        tokenizer, 
        explanation_prompt, 
        max_new_tokens=256,
        streamer=streamer,
        max_seq_length=max_seq_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Extract just the explanation part
    try:
        if "### Explanation:" in explanation_text:
            explanation = explanation_text.split("### Explanation:")[-1].strip()
        elif "### Evidence:" in explanation_text:
            explanation = explanation_text.split("### Evidence:")[-1].strip()
        elif "### Detailed Explanation:" in explanation_text:
            explanation = explanation_text.split("### Detailed Explanation:")[-1].strip()
        else:
            # Fallback - get everything after the prompt
            explanation = explanation_text.replace(explanation_prompt, "").strip()
    except:
        explanation = "Error generating explanation."
    
    return explanation


def run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length):
    """Run inference on the provided clinical notes."""
    results = []

    # Create a text streamer if streaming is enabled
    streamer = TextStreamer(tokenizer) if args.stream_output else None
    
    for i in tqdm(range(0, len(notes), args.batch_size), desc="Running inference"):
        batch_notes = notes[i:i+args.batch_size]
        batch_ids = patient_ids[i:i+args.batch_size]
        batch_labels = true_labels[i:i+args.batch_size] if true_labels else [None] * len(batch_notes)
        
        for note, patient_id, true_label in zip(batch_notes, batch_ids, batch_labels):
            # Preprocess the clinical note if enabled
            if args.preprocess_tokens:
                note = preprocess_clinical_note(note)
                
            # Create prompt for the current note with length control
            prompt = create_malnutrition_prompt(
                note, 
                tokenizer=tokenizer, 
                max_tokens=max_seq_length - args.max_new_tokens
            )
            
            # Generate prediction with probabilities
            start_time = time.time()
            
            if args.stream_output:
                print(f"\nGenerating assessment for patient {patient_id}...")
                prediction, probabilities = generate_text_with_probabilities(
                    model, 
                    tokenizer, 
                    prompt, 
                    args.max_new_tokens, 
                    streamer,
                    max_seq_length=max_seq_length,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
            else:
                prediction, probabilities = generate_text_with_probabilities(
                    model, 
                    tokenizer, 
                    prompt, 
                    args.max_new_tokens,
                    max_seq_length=max_seq_length,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
            
            inference_time = time.time() - start_time
            
            # Extract just the assessment (yes/no) from the prediction
            try:
                assessment_part = prediction.split("### Assessment:")[-1].strip()
                # Clean up to get just yes or no
                assessment = assessment_part.split()[0].lower()
                if assessment not in ['yes', 'no']:
                    # Handle case where model gives more than yes/no
                    assessment = 'yes' if 'yes' in assessment_part.lower() else 'no'
            except:
                assessment = "error"
            
            # Get the probability for the positive class (yes)
            positive_prob = probabilities.get("yes", 0.0)
            
            # Generate explanation if requested
            explanation = None
            if args.generate_explanations and assessment in ['yes', 'no']:
                explanation = generate_explanation(model, tokenizer, note, assessment, args, max_seq_length)
                # Print explanation to terminal
                print(f"\n----- Explanation for patient {patient_id} (Assessment: {assessment}, Prob: {positive_prob:.2f}) -----")
                print(explanation)
                print("-" * 80)
            
            result = {
                "DEID": patient_id,
                "true_label": true_label if true_label is not None else "unknown",
                "predicted_label": assessment,
                "positive_probability": positive_prob,
                "explanation": explanation if explanation else "",
                "original_note": note,
                "inference_time": inference_time
            }
            
            results.append(result)
            
    return results


def calculate_metrics(results, args):
    """Calculate and save performance metrics using probability scores for AUC."""
    # Create metrics directory if it doesn't exist
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Update args to include metrics_dir for use elsewhere
    args.metrics_dir = metrics_dir
    
    # Check if we have true labels to calculate metrics
    if all(r["true_label"] == "unknown" for r in results):
        print("No true labels available - skipping metrics calculation")
        return None
    
    # Filter out results with unknown true labels or error predictions
    valid_results = [r for r in results if r["true_label"] != "unknown" and r["predicted_label"] != "error"]
    
    if not valid_results:
        print("No valid results with true labels for metrics calculation")
        return None
    
    # Convert labels to binary format for metrics calculation
    # Check if true_label is already numeric
    if all(isinstance(r["true_label"], (int, float)) for r in valid_results):
        y_true = [int(r["true_label"]) for r in valid_results]
    else:
        # If string, convert using string comparison
        y_true = [1 if str(r["true_label"]).lower() == "yes" else 0 for r in valid_results]
    
    # For predicted labels (these are strings from model output)
    y_pred = [1 if r["predicted_label"].lower() == "yes" else 0 for r in valid_results]
    
    # Extract probabilities for ROC curve calculation
    # For positive class (probability of yes)
    probabilities = [r["positive_probability"] for r in valid_results]
    
    # Save raw prediction data for future analysis
    metrics_df = pd.DataFrame({
        "DEID": [r["DEID"] for r in valid_results],
        "true_label": [r["true_label"] for r in valid_results],
        "predicted_label": [r["predicted_label"] for r in valid_results],
        "true_binary": y_true,
        "pred_binary": y_pred,
        "positive_probability": probabilities
    })
    metrics_df.to_csv(os.path.join(metrics_dir, "raw_predictions.csv"), index=False)
    
    # Classification Report based on binary predictions
    report = classification_report(y_true, y_pred, output_dict=True, target_names=["Non-malnourished", "Malnourished"])
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(metrics_dir, "classification_report.csv"))
    
    # Save classification report as text
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
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    plt.matshow(cm, cmap=plt.cm.Blues, fignum=1)
    plt.colorbar()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(metrics_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    
    # ROC Curve and AUC using probabilities
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)
    
    # Save ROC data
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(os.path.join(metrics_dir, "roc_data.csv"), index=False)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (Using Probabilities)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(metrics_dir, "roc_curve_probabilities.png"), dpi=300, bbox_inches="tight")
    
    # For comparison, also calculate the ROC using binary predictions
    fpr_binary, tpr_binary, _ = roc_curve(y_true, y_pred)
    roc_auc_binary = auc(fpr_binary, tpr_binary)
    
    # Plot comparative ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Probability-based (AUC = {roc_auc:.2f})")
    plt.plot(fpr_binary, tpr_binary, color="green", lw=2, label=f"Binary-based (AUC = {roc_auc_binary:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison: Probability vs Binary Classification")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(metrics_dir, "roc_curves_comparison.png"), dpi=300, bbox_inches="tight")
    
    # Save metrics summary
    metrics_summary = {
        "accuracy": report["accuracy"],
        "precision_malnourished": report["Malnourished"]["precision"],
        "recall_malnourished": report["Malnourished"]["recall"],
        "f1_malnourished": report["Malnourished"]["f1-score"],
        "precision_nonmalnourished": report["Non-malnourished"]["precision"],
        "recall_nonmalnourished": report["Non-malnourished"]["recall"],
        "f1_nonmalnourished": report["Non-malnourished"]["f1-score"],
        "auc_probability": roc_auc,
        "auc_binary": roc_auc_binary,
        "total_samples": len(valid_results),
        "malnourished_count": sum(y_true),
        "nonmalnourished_count": len(y_true) - sum(y_true)
    }
    
    with open(os.path.join(metrics_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    return metrics_summary


def main():
    args = parse_arguments()
    
    # Create output directory and subdirectories
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "inference_results.csv")
    
    # Load the data
    print(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    # Check for required columns
    if args.note_column not in df.columns:
        raise ValueError(f"Note column '{args.note_column}' not found in the input file")
    if args.id_column not in df.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in the input file")
    
    # Extract notes and patient IDs
    notes = df[args.note_column].tolist()
    patient_ids = df[args.id_column].tolist()
    
    # Extract true labels if available
    true_labels = None
    if args.label_column and args.label_column in df.columns:
        print(f"Found label column '{args.label_column}' - metrics will be calculated")
        true_labels = df[args.label_column].tolist()
        # Print the first few labels to debug type issues
        print(f"First 5 labels (type: {type(true_labels[0]).__name__}): {true_labels[:5]}")
    else:
        print("No label column specified or not found - metrics will not be calculated")
    
    # For test mode, use only a small subset
    if args.test_mode:
        print("Running in test mode with 5 samples")
        test_size = min(5, len(notes))
        notes = notes[:test_size]
        patient_ids = patient_ids[:test_size]
        if true_labels:
            true_labels = true_labels[:test_size]
    
    # Load the model with native sequence length
    print(f"Loading model from: {args.model_path}")
    model, tokenizer, max_seq_length = load_model(args.model_path, args.load_in_4bit)
    print(f"Model loaded with native max sequence length: {max_seq_length}")
    
    # Run inference
    print("Starting inference...")
    if args.preprocess_tokens:
        print("Note preprocessing enabled - </s> tokens will be handled appropriately")
    results = run_inference(model, tokenizer, notes, patient_ids, true_labels, args, max_seq_length)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Calculate and save metrics if true labels are available
    if true_labels:
        print("Calculating performance metrics...")
        metrics = calculate_metrics(results, args)
        if metrics:
            print("\nPerformance Metrics Summary:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Malnourished cases - Precision: {metrics['precision_malnourished']:.4f}, Recall: {metrics['recall_malnourished']:.4f}, F1: {metrics['f1_malnourished']:.4f}")
            print(f"Non-malnourished cases - Precision: {metrics['precision_nonmalnourished']:.4f}, Recall: {metrics['recall_nonmalnourished']:.4f}, F1: {metrics['f1_nonmalnourished']:.4f}")
            print(f"AUC-ROC (using probabilities): {metrics['auc_probability']:.4f}")
            print(f"AUC-ROC (using binary decisions): {metrics['auc_binary']:.4f}")
            print(f"Metrics saved to: {args.metrics_dir}/")
    
    # Show summary statistics
    yes_count = sum(1 for r in results if r["predicted_label"] == "yes")
    no_count = sum(1 for r in results if r["predicted_label"] == "no")
    error_count = sum(1 for r in results if r["predicted_label"] == "error")
    
    print("\nInference Summary:")
    print(f"Total cases processed: {len(results)}")
    print(f"Malnourished cases: {yes_count} ({yes_count/len(results)*100:.1f}%)")
    print(f"Non-malnourished cases: {no_count} ({no_count/len(results)*100:.1f}%)")
    if error_count > 0:
        print(f"Error cases: {error_count} ({error_count/len(results)*100:.1f}%)")
    
    # Calculate average inference time
    avg_time = sum(r["inference_time"] for r in results) / len(results)
    print(f"Average inference time per note: {avg_time:.3f} seconds")
    print(f"Sampling parameters: temperature={args.temperature}, top_p={args.top_p}")
    
    if args.generate_explanations:
        print(f"Explanations generated and saved to: {output_file}")


if __name__ == "__main__":
    main()
