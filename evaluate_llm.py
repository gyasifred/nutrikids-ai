#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for LLaMA-style model fine-tuned for malnutrition assessment
using Unsloth + LoRA. Includes comprehensive evaluation metrics for classification performance.

This script handles:
1. Clinical notes preprocessing using the same approach as training
2. Native max sequence length for model consistency
3. Truncation for inputs exceeding max sequence length
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
    parser.add_argument("--max_seq_length", type=int, default=None, 
                        help="Max sequence length (if None, uses model's native max length)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use_native_max_len", action="store_true", 
                        help="Use the model's native maximum sequence length")
    parser.add_argument("--preprocess_tokens", action="store_true", 
                        help="Preprocess </s> tokens in clinical notes")
    parser.add_argument("--plot_metrics", action="store_true", 
                        help="Generate and save performance metric plots")
    parser.add_argument("--temperature", type=float, default=0.1, 
                        help="Temperature for generation (lower = more deterministic)")
    parser.add_argument("--max_new_tokens", type=int, default=256, 
                        help="Maximum number of new tokens to generate")
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

def create_simplified_malnutrition_prompt(note, tokenizer=None, max_tokens=None):
    """
    Create a specialized malnutrition assessment prompt with detailed clinical criteria.
    """
    # Define a structured prompt with comprehensive malnutrition criteria
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

Follow this format:
1) First provide some explanations about your decision. In your explanation mention did you use single or multiple data points, and list z scores you used.
2) Then format your output as follows, strictly follow this format: malnutrition=yes or malnutrition=no

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

def get_model_max_length(model_path):
    """Get the model's maximum sequence length."""
    try:
        # First try to get it from the model's config
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_path)
            return getattr(config, "max_position_embeddings", 4096)
        except Exception as e:
            print(f"Failed to get model's max length from config: {e}")
            
        # Fall back to unsloth's auto getter if available
        try:
            from unsloth.models.llama import auto_get_max_seq_length
            return auto_get_max_seq_length(model_path)
        except (ImportError, AttributeError) as e:
            print(f"Failed to get model's max length from unsloth: {e}")
            
        # Default fallback
        return 4096
    except Exception as e:
        print(f"Failed to get model's max length: {e}")
        return 4096

def extract_malnutrition_decision(text):
    """Extract the malnutrition classification from the model output."""
    # Find the malnutrition=yes/no pattern (the most explicit format we requested)
    pattern = r'malnutrition\s*=\s*(yes|no)'
    match = re.search(pattern, text.lower())
    
    if match:
        decision = match.group(1)
        return 1 if decision.lower() == 'yes' else 0
    
    # Look for variations on the format that might appear
    yes_pattern = r'malnutrition\s*[:=]?\s*yes'
    no_pattern = r'malnutrition\s*[:=]?\s*no'
    
    if re.search(yes_pattern, text.lower()):
        return 1
    if re.search(no_pattern, text.lower()):
        return 0
    
    # Fallback: Check for yes/no mentions near malnutrition
    if 'malnutrition' in text.lower():
        text_lower = text.lower()
        # Check if "no malnutrition" or similar phrases appear
        no_patterns = ['no malnutrition', 'not malnourished', 'does not have malnutrition',
                      'unlikely to have malnutrition', 'no evidence of malnutrition',
                      'without malnutrition', 'absence of malnutrition']
        for pattern in no_patterns:
            if pattern in text_lower:
                return 0
        
        # Check if "has malnutrition" or similar phrases appear
        yes_patterns = ['has malnutrition', 'is malnourished', 'does have malnutrition',
                       'likely to have malnutrition', 'evidence of malnutrition',
                       'presence of malnutrition', 'diagnosed with malnutrition',
                       'meets criteria for malnutrition']
        for pattern in yes_patterns:
            if pattern in text_lower:
                return 1
    
    # Check for sentences ending with clear decisions (clinical assessment style)
    sentence_patterns = [
        r'patient (has|shows|demonstrates|exhibits|presents with) malnutrition',
        r'patient (does not have|does not show|does not demonstrate|does not exhibit|does not present with) malnutrition',
        r'assessment[:]?\s*(mild|moderate|severe) malnutrition',
        r'(mild|moderate|severe) malnutrition [a-z\s]+ (identified|diagnosed|present)',
        r'no malnutrition [a-z\s]+ (identified|diagnosed|present)',
    ]
    
    for pattern in sentence_patterns:
        match = re.search(pattern, text.lower())
        if match:
            if 'does not' in match.group(0) or 'no malnutrition' in match.group(0):
                return 0
            else:
                return 1
    
    # If still unsure, check for simple yes/no at the end of the text
    if text.lower().strip().endswith('yes'):
        return 1
    if text.lower().strip().endswith('no'):
        return 0
    
    # If we can't determine, return None
    return None

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
        
        prompt = create_simplified_malnutrition_prompt(
            note=note_text,
            tokenizer=tokenizer,
            max_tokens=max_seq_length - 20  # Leave room for EOS token
        )
        prompts.append(prompt)
    
    result = {
        "text": original_texts,
        "prompt": prompts
    }
    
    if has_labels:
        result["label"] = true_labels
        
    return Dataset.from_dict(result)

def run_inference(model, tokenizer, dataset, batch_size=4, temperature=0.1, max_new_tokens=256):
    """Run inference on the dataset and return predictions"""
    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
    )
    
    # Run inference
    outputs = []
    probabilities = []
    decisions = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:min(i+batch_size, len(dataset))]
        prompts = batch["prompt"]
        
        # Log progress
        print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}")
        
        # Generate predictions
        with torch.no_grad():
            try:
                results = pipe(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    return_full_text=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                for j, result in enumerate(results):
                    idx = i + j  # Calculate actual index in the dataset
                    generated_text = result[0]["generated_text"]
                    outputs.append(generated_text)
                    
                    # Extract malnutrition decision
                    decision = extract_malnutrition_decision(generated_text)
                    decisions.append(decision)
                    
                    # Get true label if available
                    true_label = dataset[idx].get("label") if "label" in dataset.column_names else None
                    
                    # Print true label and predicted label for each input
                    print(f"\n--- Sample {idx + 1} ---")
                    print(f"True label: {true_label if true_label is not None else 'N/A'}")
                    print(f"Predicted label: {decision if decision is not None else 'UNKNOWN'}")
                    
                    # Try to extract probability from the content (better heuristic)
                    # For malnutrition=yes/no format, use a confidence value
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
                    
                    # Print probability for better tracking
                    print(f"Confidence: {prob:.2f}")
                    
                    probabilities.append(prob)
            
            except Exception as e:
                print(f"Error in batch processing: {e}")
                # Handle the error by adding placeholders
                for j in range(len(batch)):
                    idx = i + j
                    outputs.append("Error during generation")
                    decisions.append(None)
                    probabilities.append(0.5)
                    
                    # Get true label if available
                    true_label = dataset[idx].get("label") if "label" in dataset.column_names else None
                    
                    # Print error for this sample
                    print(f"\n--- Sample {idx + 1} ---")
                    print(f"True label: {true_label if true_label is not None else 'N/A'}")
                    print(f"Predicted label: ERROR")
                    print(f"Confidence: N/A")
    
    return outputs, decisions, probabilities

def evaluate_classification(true_labels, predicted_labels, predicted_probs, output_dir="./metrics"):
    """Evaluate classification performance with multiple metrics"""
    # Filter out None values (cases where prediction couldn't be determined)
    valid_indices = [i for i, pred in enumerate(predicted_labels) if pred is not None]
    
    if not valid_indices:
        print("No valid predictions to evaluate")
        return {}
    
    # Count and report how many predictions couldn't be determined
    invalid_count = len(predicted_labels) - len(valid_indices)
    if invalid_count > 0:
        print(f"Warning: {invalid_count} out of {len(predicted_labels)} predictions ({(invalid_count/len(predicted_labels))*100:.1f}%) couldn't be determined.")
    
    true_filtered = [true_labels[i] for i in valid_indices]
    pred_filtered = [predicted_labels[i] for i in valid_indices]
    prob_filtered = [predicted_probs[i] for i in valid_indices]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = {}
    metrics["accuracy"] = accuracy_score(true_filtered, pred_filtered)
    metrics["precision"] = precision_score(true_filtered, pred_filtered, zero_division=0)
    metrics["recall"] = recall_score(true_filtered, pred_filtered, zero_division=0)
    metrics["f1"] = f1_score(true_filtered, pred_filtered, zero_division=0)
    
    # Calculate AUC and ROC curve if we have probabilities
    try:
        metrics["roc_auc"] = roc_auc_score(true_filtered, prob_filtered)
        fpr, tpr, _ = roc_curve(true_filtered, prob_filtered)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        
        # Precision-Recall curve and AUPRC
        precision, recall, _ = precision_recall_curve(true_filtered, prob_filtered)
        metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
        metrics["average_precision"] = average_precision_score(true_filtered, prob_filtered)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {metrics["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'PR curve (AP = {metrics["average_precision"]:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC - {e}")
    
    # Get confusion matrix
    cm = confusion_matrix(true_filtered, pred_filtered)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Malnutrition', 'Malnutrition'],
                yticklabels=['No Malnutrition', 'Malnutrition'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Get detailed classification report
    class_report = classification_report(true_filtered, pred_filtered, output_dict=True)
    metrics["classification_report"] = class_report
    
    # Print metrics summary
    print("\n===== Classification Metrics =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    if "average_precision" in metrics:
        print(f"Average Precision (AUPRC): {metrics['average_precision']:.4f}")
    print("==================================\n")
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print("                  Predicted")
    print("                  No    Yes")
    print(f"True No Malnutrition:  {cm[0][0]}    {cm[0][1]}")
    print(f"True Malnutrition:     {cm[1][0]}    {cm[1][1]}")
    print("\n")
    
    return metrics

def main():
    args = parse_arguments()
    
    # Determine max sequence length
    if args.use_native_max_len or args.max_seq_length is None:
        max_seq_length = get_model_max_length(args.model_path)
        print(f"Using model's native max length: {max_seq_length}")
    else:
        max_seq_length = args.max_seq_length
    
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
    outputs, decisions, probabilities = run_inference(
        model, 
        tokenizer, 
        dataset,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
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

def try_extract_prediction_probabilities(model, tokenizer, text, malnutrition_token_ids=None):
    """
    Advanced function to try extracting actual prediction probabilities from the model's logits.
    This is an experimental feature that might provide better probability estimates than heuristics.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        text: Input text (prompt)
        malnutrition_token_ids: Dictionary mapping token IDs to yes/no tokens
        
    Returns:
        float: Probability between 0-1 for malnutrition=yes
    """
    # This is placeholder code that could be expanded to extract actual probabilities
    # from the model's logits if UNSLOTH_RETURN_LOGITS is enabled
    try:
        # Encode the input
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # Get logits from model (requires UNSLOTH_RETURN_LOGITS=1)
        with torch.no_grad():
            outputs = model(**inputs)
            
        if hasattr(outputs, "logits"):
            # If we have token IDs for yes/no tokens
            if malnutrition_token_ids:
                yes_id = malnutrition_token_ids.get("yes")
                no_id = malnutrition_token_ids.get("no")
                
                if yes_id is not None and no_id is not None:
                    logits = outputs.logits[0, -1]  # Last token prediction
                    yes_logit = logits[yes_id].item()
                    no_logit = logits[no_id].item()
                    
                    # Convert to probability with softmax
                    total = torch.exp(torch.tensor(yes_logit)) + torch.exp(torch.tensor(no_logit))
                    yes_prob = torch.exp(torch.tensor(yes_logit)) / total
                    
                    return yes_prob.item()
    
    except Exception as e:
        print(f"Failed to extract prediction probabilities: {e}")
    
    # Default fallback
    return None

if __name__ == "__main__":
    main()
