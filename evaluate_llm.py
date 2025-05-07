#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for LLaMA-style model fine-tuned for malnutrition assessment
using Unsloth + LoRA. Includes comprehensive evaluation metrics for classification performance.

This script handles:
1. Clinical notes preprocessing using the same approach as training
2. Native max sequence length for model consistency
3. Proper truncation for inputs exceeding max sequence length
4. Classification metrics including AUC and AUC-ROC
5. Enhanced malnutrition detection from model outputs
"""

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'  # Critical for probability extraction
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
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned LLM for malnutrition classification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV test data file")
    parser.add_argument("--output_path", type=str, default="./inference_results.csv", 
                        help="Path to save inference results")
    parser.add_argument("--max_seq_length", type=int, default=8192, 
                        help="Max sequence length (default: 8192)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--preprocess_tokens", action="store_true", 
                        help="Preprocess </s> tokens in clinical notes")
    parser.add_argument("--plot_metrics", action="store_true", 
                        help="Generate and save performance metric plots")
    parser.add_argument("--temperature", type=float, default=0.1, 
                        help="Temperature for generation (lower = more deterministic)")
    parser.add_argument("--max_new_tokens", type=int, default=256, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling value for generation")
    parser.add_argument("--stream", action="store_true", 
                        help="Enable text streaming during generation")
    return parser.parse_args()

def preprocess_clinical_note(note_text):
    """Preprocess clinical notes to handle special tokens."""
    if not note_text:
        return note_text
    
    # Replace problematic tokens
    processed_text = note_text.replace('</s>', '\n\n')
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")
    
    return processed_text

def create_improved_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """
    Create a specialized malnutrition assessment prompt with detailed clinical criteria.
    Designed to force the model to provide prediction first, then explanation.
    """
    # Define a structured prompt with comprehensive malnutrition criteria
    prompt = """[Task] Read the patient's notes and determine if the patient is likely to have malnutrition based on the following criteria.

YOUR RESPONSE MUST BEGIN WITH EITHER "malnutrition=yes" OR "malnutrition=no" ON THE FIRST LINE.

EXAMPLE RESPONSE FORMAT:
malnutrition=yes
Based on the patient's notes, I identified evidence of malnutrition because...

OR

malnutrition=no
Based on the patient's notes, there is no evidence of malnutrition because...

[Assessment Guidelines]
Malnutrition Criteria:
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
Weight loss (2–20 years of age): 5% usual body weight
Deceleration in weight for length/height: Decline of 1 z score
Inadequate nutrient intake: 51%−75% estimated energy/protein need

Moderate Malnutrition
Weight gain velocity (<2 years of age): Less than 50% of the norm for expected weight gain
Weight loss (2–20 years of age): 7.5% usual body weight
Deceleration in weight for length/height: Decline of 2 z score
Inadequate nutrient intake: 26%−50% estimated energy/protein need

Severe Malnutrition
Weight gain velocity (<2 years of age): Less than 25% of the norm for expected weight gain
Weight loss (2–20 years of age): 10% usual body weight
Deceleration in weight for length/height: Decline of 3 z score
Inadequate nutrient intake: less than 25% estimated energy/protein need

REMINDER: Your response MUST begin with either "malnutrition=yes" or "malnutrition=no" on the first line.

Clinical note for analysis:
{note}"""

    # Apply token truncation if needed
    formatted_prompt = prompt.format(note=note)
    
    if tokenizer and max_tokens:
        tokens = tokenizer.encode(formatted_prompt)
        
        # Check if tokens is too long
        if len(tokens) > max_tokens:
            # Get template without the note to determine available tokens
            template = prompt.format(note="")
            template_tokens = tokenizer.encode(template)
            
            # Calculate available tokens for the note
            available_tokens = max_tokens - len(template_tokens)
            
            # Ensure at least some tokens are available
            if available_tokens <= 0:
                available_tokens = max_tokens // 2  # Fallback if template is too long
            
            # Tokenize the note separately
            note_tokens = tokenizer.encode(note)
            
            # Truncate the note tokens
            truncated_note_tokens = note_tokens[:available_tokens]
            
            # Decode the truncated tokens back to text
            truncated_note = tokenizer.decode(truncated_note_tokens)
            
            # Recreate the prompt with the truncated note
            formatted_prompt = prompt.format(note=truncated_note)
    
    return formatted_prompt

def extract_malnutrition_decision(text):
    """Extract the malnutrition classification from the model output with balanced yes/no detection."""
    # First, check if the text is empty or None
    if not text or len(text.strip()) == 0:
        return None
    
    text_lower = text.lower().strip()
    
    # Look for the exact format we requested (this should be at the beginning of the response)
    primary_pattern = r'(?:^|\n)\s*malnutrition\s*=\s*(yes|no)'
    primary_match = re.search(primary_pattern, text_lower)
    
    if primary_match:
        decision = primary_match.group(1)
        return 1 if decision.lower() == 'yes' else 0
    
    # If the exact format isn't found, check for variations
    # Make these patterns more explicit and strict
    yes_patterns = [
        r'\bmalnutrition\s*[:=]?\s*yes\b',
        r'\bpatient has malnutrition\b',
        r'\bdiagnosis:?\s*malnutrition\b',
        r'\bmalnutrition is present\b',
        r'\bclassified as malnourished\b',
        r'\bmalnutrition is confirmed\b',
        r'\bpatient is malnourished\b',
        r'\bpatient does have malnutrition\b',
        r'\bevidence of malnutrition\b'
    ]
    
    no_patterns = [
        r'\bmalnutrition\s*[:=]?\s*no\b',
        r'\bno malnutrition\b',
        r'\bpatient does not have malnutrition\b',
        r'\bno evidence of malnutrition\b',
        r'\bno indication of malnutrition\b',
        r'\bpatient is not malnourished\b',
        r'\bnot classified as malnourished\b',
        r'\bmalnutrition is not present\b',
        r'\babsence of malnutrition\b'
    ]
    
    # Look for "yes" patterns in the first few lines (more weight on beginning of response)
    first_chunk = '\n'.join(text_lower.split('\n')[:5])
    for pattern in yes_patterns:
        if re.search(pattern, first_chunk):
            return 1
    
    for pattern in no_patterns:
        if re.search(pattern, first_chunk):
            return 0
    
    # If not found in the first few lines, search the entire text
    for pattern in yes_patterns:
        if re.search(pattern, text_lower):
            return 1
    
    for pattern in no_patterns:
        if re.search(pattern, text_lower):
            return 0
    
    # Handle potential ambiguity with balanced approach
    # Count positive vs negative indicators in text
    positive_indicators = ["mild malnutrition", "moderate malnutrition", "severe malnutrition", 
                           "meets criteria", "consistent with malnutrition", "z score below"]
    negative_indicators = ["no malnutrition", "not malnourished", "doesn't meet criteria", 
                           "normal nutritional", "z score within normal", "normal range"]
    
    pos_count = sum(1 for term in positive_indicators if term in text_lower)
    neg_count = sum(1 for term in negative_indicators if term in text_lower)
    
    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return 0
    
    # Check for sentences ending with clear decisions (clinical assessment style)
    sentence_patterns = [
        (r'patient (has|shows|demonstrates|exhibits|presents with) malnutrition', 1),
        (r'patient (does not have|does not show|does not demonstrate|does not exhibit|does not present with) malnutrition', 0),
        (r'assessment[:]?\s*(mild|moderate|severe) malnutrition', 1),
        (r'(mild|moderate|severe) malnutrition [a-z\s]+ (identified|diagnosed|present)', 1),
        (r'no malnutrition [a-z\s]+ (identified|diagnosed|present)', 0),
    ]
    
    for pattern, value in sentence_patterns:
        if re.search(pattern, text_lower):
            return value
    
    # If still unsure, look for some final keyword patterns
    if any(kw in text_lower for kw in ["z score < -2", "z-score < -2", "z score less than -2"]):
        return 1
    
    # If we still can't determine, default to None (undetermined)
    # This is safer than defaulting to either yes or no
    return None

def get_model_max_length(model_path):
    """Get the model's maximum sequence length."""
    try:
        # First try to get it from the model's config
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_path)
            return getattr(config, "max_position_embeddings", 8192)  # Default to 8192 if not found
        except Exception as e:
            print(f"Failed to get model's max length from config: {e}")
            
        # Fall back to unsloth's auto getter if available
        try:
            from unsloth.models.llama import auto_get_max_seq_length
            return auto_get_max_seq_length(model_path)
        except (ImportError, AttributeError) as e:
            print(f"Failed to get model's max length from unsloth: {e}")
            
        # Default fallback
        return 8192  # Increased default from 4096 to 8192
    except Exception as e:
        print(f"Failed to get model's max length: {e}")
        return 8192  # Increased default from 4096 to 8192

def evaluate_classification(true_labels, predictions, probabilities, output_dir="./metrics"):
    """Evaluate classification performance with comprehensive metrics."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out None values
    valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
    if not valid_indices:
        print("No valid predictions found.")
        return {}
    
    filtered_true = [true_labels[i] for i in valid_indices]
    filtered_pred = [predictions[i] for i in valid_indices]
    filtered_prob = [probabilities[i] for i in valid_indices]
    
    # Calculate basic classification metrics
    accuracy = accuracy_score(filtered_true, filtered_pred)
    precision = precision_score(filtered_true, filtered_pred, zero_division=0)
    recall = recall_score(filtered_true, filtered_pred, zero_division=0)
    f1 = f1_score(filtered_true, filtered_pred, zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(filtered_true, filtered_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Malnutrition', 'Malnutrition'],
                yticklabels=['No Malnutrition', 'Malnutrition'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate and plot ROC curve if probability estimates are available
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    try:
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(filtered_true, filtered_prob)
        roc_auc = roc_auc_score(filtered_true, filtered_prob)
        metrics['roc_auc'] = roc_auc
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(filtered_true, filtered_prob)
        avg_precision = average_precision_score(filtered_true, filtered_prob)
        metrics['average_precision'] = avg_precision
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
                 label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not calculate ROC/PR metrics: {e}")
    
    # Generate classification report for more detailed metrics by class
    report = classification_report(filtered_true, filtered_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    return metrics

def prepare_dataset(data_path, tokenizer, max_seq_length, preprocess_tokens=False):
    """Prepare dataset for inference with specialized malnutrition prompt structure"""
    df = pd.read_csv(data_path)
    if "txt" not in df.columns:
        raise ValueError("CSV must include 'txt' column.")
        
    # Check if label column exists for evaluation
    has_labels = "label" in df.columns
    
    prompts = []
    original_texts = []
    true_labels = []
    
    for _, row in df.iterrows():
        note_text = preprocess_clinical_note(row["txt"]) if preprocess_tokens else row["txt"]
        original_texts.append(note_text)
        
        if has_labels:
            # Convert label format if needed
            if isinstance(row["label"], (int, float)):
                label = 1 if row["label"] > 0.5 else 0
            elif isinstance(row["label"], str):
                label = 1 if row["label"].lower() in {"1", "yes", "true"} else 0
            else:
                label = None
            true_labels.append(label)
        
        # Calculate max tokens available for input
        # Reserve space for output tokens
        available_tokens = max_seq_length - 256  # Reserve space for output tokens
        
        prompt = create_improved_malnutrition_prompt(
            note=note_text,
            tokenizer=tokenizer,
            max_tokens=available_tokens
        )
        prompts.append(prompt)
    
    result = {
        "text": original_texts,
        "prompt": prompts
    }
    
    if has_labels:
        result["label"] = true_labels
        
    return Dataset.from_dict(result)

def generate_assessment(model, tokenizer, prompt, max_new_tokens, temperature=0.1, top_p=0.9, max_seq_length=None, stream=False):
    """Generate assessment from model with optional streaming support."""
    # Ensure model is in inference mode
    FastLanguageModel.for_inference(model)
    
    # Prepare input
    inputs = tokenizer([prompt], return_tensors="pt")
    
    # Check if input length exceeds max_seq_length
    input_length = inputs.input_ids.shape[1]
    if max_seq_length and input_length > max_seq_length - max_new_tokens:
        print(f"Warning: Input length {input_length} exceeds max available length. Truncating...")
        # Truncate from the left side of the input to preserve the most recent/relevant information
        inputs.input_ids = inputs.input_ids[:, -(max_seq_length - max_new_tokens):]
        if hasattr(inputs, 'attention_mask'):
            inputs.attention_mask = inputs.attention_mask[:, -(max_seq_length - max_new_tokens):]
    
    # Move inputs to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Handle streaming if requested
    if stream:
        from transformers import TextStreamer
        text_streamer = TextStreamer(tokenizer)
        
        # Generate text with streaming
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            streamer=text_streamer,
            use_cache=True
             )
    else:
        # Generate text without streaming
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    # Decode output
    generated_text = tokenizer.decode(outputs)
    
    # Extract the generated part (not the prompt)
    if prompt in generated_text:
        response = generated_text[len(prompt):]
    else:
        response = generated_text  # Return full text if we can't find the prompt
    
    return response, outputs  # Return both text and token IDs

def run_inference(model, tokenizer, dataset, batch_size=4, temperature=0.1, top_p=0.9, max_new_tokens=256, max_seq_length=None, stream=False):
    """Run inference on the dataset and return predictions, with optional streaming support"""
    outputs = []
    token_outputs = []
    probabilities = []
    decisions = []
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    # Process samples
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:min(i+batch_size, len(dataset))]
        prompts = batch["prompt"]
        
        # Log progress
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}")
        
        # Process each prompt in the batch
        for j, prompt in enumerate(prompts):
            idx = i + j  # Calculate actual index in the dataset
            
            # Get true label if available
            true_label = dataset[idx].get("label") if "label" in dataset.column_names else None
            
            try:
                # Generate text
                generated_text, token_ids = generate_assessment(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    max_seq_length=max_seq_length,
                    stream=stream  # Pass streaming option
                )
                
                # Store generated text and token IDs
                outputs.append(generated_text)
                token_outputs.append(token_ids)
                
                # Extract malnutrition decision
                decision = extract_malnutrition_decision(generated_text)
                decisions.append(decision)
                
                # Print true label and predicted label
                print(f"\n--- Sample {idx + 1} ---")
                if not stream:  # Only print result if not already streamed
                    print(f"Generated text: {generated_text[:100]}...")  # Print first 100 chars
                print(f"True label: {true_label if true_label is not None else 'N/A'}")
                print(f"Predicted label: {decision if decision is not None else 'UNKNOWN'}")
                
                # Try to extract probability
                confidence = None
                
                # Look for explicit confidence mentions
                confidence_pattern = r'(with|at)?\s*(\d+)%\s*confidence'
                conf_match = re.search(confidence_pattern, generated_text.lower())
                if conf_match:
                    try:
                        confidence = float(conf_match.group(2)) / 100
                    except:
                        confidence = None
                
                # Look for confidence words
                confidence_words = {
                    'certain': 0.95, 'definite': 0.95, 'definitive': 0.95,
                    'clear': 0.9, 'strong': 0.9, 'evident': 0.9,
                    'likely': 0.8, 'probable': 0.8, 'appears': 0.7,
                    'possible': 0.6, 'suggests': 0.6, 'may': 0.5,
                    'uncertain': 0.4, 'unclear': 0.3, 'unlikely': 0.2
                }
                
                if not confidence:
                    for word, value in confidence_words.items():
                        if word in generated_text.lower():
                            distance_to_decision = 100  # Large initial value
                            for match in re.finditer(word, generated_text.lower()):
                                # Find closest occurrence to the malnutrition decision
                                mal_pos = generated_text.lower().find('malnutrition')
                                if mal_pos >= 0:
                                    curr_distance = abs(match.start() - mal_pos)
                                    if curr_distance < distance_to_decision:
                                        distance_to_decision = curr_distance
                                        confidence = value
                
                # Set probability based on decision + confidence
                if decision is not None:
                    if confidence:
                        if decision == 1:
                            prob = confidence
                        else:
                            prob = 1 - confidence
                    else:
                        # Without explicit confidence, use default confidence based on decision
                        prob = 0.9 if decision == 1 else 0.1
                else:
                    prob = 0.5  # Uncertain
                
                # Print probability
                print(f"Confidence: {prob:.2f}")
                probabilities.append(prob)
                
            except Exception as e:
                print(f"Error processing sample {idx+1}: {e}")
                outputs.append("Error during generation")
                token_outputs.append(None)
                decisions.append(None)
                probabilities.append(0.5)
                
                print(f"\n--- Sample {idx + 1} ---")
                print(f"True label: {true_label if true_label is not None else 'N/A'}")
                print(f"Predicted label: ERROR")
                print(f"Confidence: N/A")
    
    return outputs, token_outputs, decisions, probabilities

def main():
    args = parse_arguments()
    
    # Determine max sequence length
    max_seq_length = args.max_seq_length
    print(f"Using max sequence length: {max_seq_length}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    
    # Prepare dataset
    print(f"Preparing dataset from {args.data_path}...")
    dataset = prepare_dataset(
        args.data_path, 
        tokenizer, 
        max_seq_length, 
        args.preprocess_tokens
    )

    # Run inference
    print("Running inference...")
    outputs, token_outputs, decisions, probabilities = run_inference(
        model, 
        tokenizer, 
        dataset,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=max_seq_length,
        stream=args.stream  # Pass streaming option
    )
    
    # Create results dataframe
    results = {
        "original_text": dataset["text"],
        "model_output": outputs,
        "prediction": decisions,
        "probability": probabilities
    }
    
    # Add true labels if they exist
    if "label" in dataset.column_names:
        results["true_label"] = dataset["label"]
        
        # Calculate metrics
        print("Calculating evaluation metrics...")
        metrics_dir = os.path.join(os.path.dirname(args.output_path), "metrics")
        metrics = evaluate_classification(
            dataset["label"], 
            decisions, 
            probabilities,
            output_dir=metrics_dir
        )
        
        # Print overall summary
        print("\n===== OVERALL CLASSIFICATION RESULTS =====")
        print(f"Total samples: {len(dataset)}")
        print(f"Correctly predicted: {sum(1 for i, d in enumerate(decisions) if d == dataset['label'][i])}")
        print(f"Incorrectly predicted: {sum(1 for i, d in enumerate(decisions) if d is not None and d != dataset['label'][i])}")
        print(f"Undetermined: {sum(1 for d in decisions if d is None)}")
        print("==========================================\n")
        
        # Add evaluation info to results file
        with open(os.path.join(metrics_dir, "metrics_summary.txt"), "w") as f:
            f.write("===== Classification Metrics =====\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            if "roc_auc" in metrics:
                f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
            if "average_precision" in metrics:
                f.write(f"Average Precision (AUPRC): {metrics['average_precision']:.4f}\n")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
