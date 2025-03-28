#!/usr/bin/env python3
import os
import re
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from trl import SFTTrainer
import torch.nn.functional as F
from typing import List, Dict, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve)


class MalnutritionDataset:
    """Class to handle malnutrition dataset operations."""

    def __init__(self, data_path: str, note_col: str, label_col: str):
        """Initialize dataset from a CSV file.

        Args:
            data_path (str): Path to the CSV file containing the data
            note_col (str): Name of the text column in the CSV
            label_col (str): Name of the label column in the CSV
        """
        self.df = pd.read_csv(data_path)
        self.text = note_col
        self.label = label_col

    def prepare_training_data(self, prompt_builder) -> List[Dict[str, str]]:
        """Prepare data in the format required for training.

        Args:
            prompt_builder: An instance of MalnutritionPromptBuilder

        Returns:
            List of dictionaries with text and labels formatted for training
        """
        formatted_data = []
        for _, row in self.df.iterrows():
            # Generate prompt for each example
            prompt = prompt_builder.get_training_prompt(row[self.text])

            formatted_data.append({
                "text": prompt,
                "labels": "yes" if str(row[self.label]).lower() in ["1", "yes", "true"] else "no"
            })

        return formatted_data


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def is_bfloat16_supported():
    """Check if bfloat16 is supported by the current device.

    Returns:
        bool: True if bfloat16 is supported, False otherwise
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def evaluate_predictions(
    y_true: List[Any],
    y_pred: List[Any],
    y_prob: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Evaluate model predictions and return comprehensive metrics.

    Args:
        y_true (List[Any]): Ground truth labels (1/0 or "yes"/"no")
        y_pred (List[Any]): Predicted labels (1/0 or "yes"/"no")
        y_prob (Optional[List[float]]): Predicted probabilities for the positive class

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics
    """
    # Convert labels to binary only if they are strings
    def convert_to_binary(label):
        if isinstance(label, str):
            return 1 if label.lower() == "yes" else 0
        return int(label)  # Assume already 0/1 if not a string

    # Convert both y_true and y_pred to binary (0 or 1)
    y_true_binary = [convert_to_binary(label) for label in y_true]
    y_pred_binary = [convert_to_binary(label) for label in y_pred]

    # Calculate basic metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)

    # Get detailed classification report
    cls_report = classification_report(y_true_binary, y_pred_binary, target_names=[
                                       "no", "yes"], output_dict=True)

    # ROC and PR curve metrics (if probabilities are provided)
    auc = avg_precision = 0.0
    fpr = tpr = precision_curve = recall_curve = []

    # Ensure we have both classes and predicted probabilities for ROC/PR curves
    if y_prob is not None and len(set(y_true_binary)) > 1:
        auc = roc_auc_score(y_true_binary, y_prob)
        avg_precision = average_precision_score(y_true_binary, y_prob)

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob)

        # Precision-Recall curve data
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true_binary, y_prob)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'avg_precision': float(avg_precision),
        'confusion_matrix': cm.tolist(),
        'classification_report': cls_report,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve
    }

    return metrics


def plot_evaluation_metrics(metrics: Dict[str, Any], output_dir: str):
    """Create and save visualizations for evaluation metrics.

    Args:
        metrics (Dict[str, Any]): Dictionary containing evaluation metrics
        output_dir (str): Directory to save plots to
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. Plot ROC curve (if available)
    if len(metrics['fpr']) > 0 and len(metrics['tpr']) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(metrics['fpr'], metrics['tpr'],
                 label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()

    # 3. Plot Precision-Recall curve (if available)
    if len(metrics['precision_curve']) > 0 and len(metrics['recall_curve']) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(metrics['recall_curve'], metrics['precision_curve'],
                 label=f'PR Curve (AP = {metrics["avg_precision"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
        plt.close()

    # 4. Plot metrics summary
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    values = [metrics[m] for m in metrics_to_plot]

    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    bars = plt.bar(metrics_to_plot, values, color=colors)
    plt.ylim(0, 1.0)
    plt.title('Classification Metrics Summary')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'))
    plt.close()


def save_metrics_to_csv(metrics: Dict[str, Any], output_path: str):
    """Save main evaluation metrics to CSV file.

    Args:
        metrics (Dict[str, Any]): Dictionary containing evaluation metrics
        output_path (str): Path to save the CSV file
    """
    # Extract main metrics for CSV
    main_metrics = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'avg_precision': metrics['avg_precision']
    }

    # Convert to DataFrame
    df = pd.DataFrame([main_metrics])

    # Save CSV
    df.to_csv(output_path, index=False)


def print_metrics_report(metrics: Dict[str, Any]):
    """Print a formatted report of metrics to the terminal.

    Args:
        metrics (Dict[str, Any]): Dictionary containing evaluation metrics
    """
    print("\n" + "="*50)
    print("MALNUTRITION DETECTION PERFORMANCE METRICS")
    print("="*50)

    print(f"\nACCURACY:      {metrics['accuracy']:.4f}")
    print(f"PRECISION:     {metrics['precision']:.4f}")
    print(f"RECALL:        {metrics['recall']:.4f}")
    print(f"F1 SCORE:      {metrics['f1']:.4f}")

    if metrics['auc'] > 0:
        print(f"AUC:           {metrics['auc']:.4f}")
        print(f"AVG PRECISION: {metrics['avg_precision']:.4f}")

    print("\nCONFUSION MATRIX:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"                  Predicted")
    print(f"                  No    Yes")
    print(f"Actual    No     {cm[0][0]:<5d} {cm[0][1]:<5d}")
    print(f"          Yes    {cm[1][0]:<5d} {cm[1][1]:<5d}")

    print("\nCLASSIFICATION REPORT:")
    # Extract relevant information from classification report
    report = metrics['classification_report']

    # Format as a table
    print(f"              precision    recall  f1-score   support")
    print(
        f"no           {report['no']['precision']:.4f}      {report['no']['recall']:.4f}    {report['no']['f1-score']:.4f}      {int(report['no']['support'])}")
    print(
        f"yes          {report['yes']['precision']:.4f}      {report['yes']['recall']:.4f}    {report['yes']['f1-score']:.4f}      {int(report['yes']['support'])}")
    print(
        f"accuracy                          {report['accuracy']:.4f}      {int(report['macro avg']['support'])}")
    print(f"macro avg    {report['macro avg']['precision']:.4f}      {report['macro avg']['recall']:.4f}    {report['macro avg']['f1-score']:.4f}      {int(report['macro avg']['support'])}")
    print(f"weighted avg {report['weighted avg']['precision']:.4f}      {report['weighted avg']['recall']:.4f}    {report['weighted avg']['f1-score']:.4f}      {int(report['weighted avg']['support'])}")

    print("\n" + "="*50 + "\n")

class MalnutritionPromptBuilder:
    """A class to manage the creation of malnutrition prompts for various scenarios."""

    def __init__(self, examples_csv_path: Optional[str] = None):
        """Initialize the prompt builder with an optional examples dataset.

        Args:
            examples_csv_path (Optional[str]): Path to CSV file with few-shot examples
        """
        self.examples_csv_path = examples_csv_path
        self.examples_cache = None

        # Load examples into cache if path is provided
        if examples_csv_path:
            try:
                self.examples_cache = pd.read_csv(examples_csv_path)
                print(f"Loaded {len(self.examples_cache)} examples from {examples_csv_path}")
            except Exception as e:
                print(f"Error loading examples: {e}")

    def get_training_prompt(self, patient_notes: str) -> str:
        """Get a standard prompt for training without few-shot examples.

        Args:
            patient_notes (str): The patient notes to analyze

        Returns:
            str: A formatted prompt for training
        """
        return self.construct_malnutrition_prompt(patient_notes)

    def get_inference_prompt(
        self,
        patient_notes: str,
        note_col: str,
        label_col: str,
        num_examples: int = 0,
        specific_example_indices: Optional[List[int]] = None,
        balanced: bool = False
    ) -> str:
        """Get an inference prompt with optional few-shot examples.

        Args:
            patient_notes (str): The patient notes to analyze
            note_col (str): Name of the text column
            label_col (str): Name of the label column
            num_examples (int): Number of examples to include (0 for zero-shot)
            specific_example_indices (Optional[List[int]]): Specific indices to use
            balanced (bool): Whether to balance positive/negative examples

        Returns:
            str: A formatted prompt for inference
        """
        if num_examples == 0 or self.examples_cache is None:
            return self.construct_malnutrition_prompt(patient_notes)

        few_shot_examples = []

        if balanced and num_examples >= 2:
            return self.get_balanced_inference_prompt(patient_notes, note_col, label_col, num_examples)

        if specific_example_indices:
            # Use specified examples
            valid_indices = [
                i for i in specific_example_indices if 0 <= i < len(self.examples_cache)]
            for idx in valid_indices[:num_examples]:
                few_shot_examples.append({
                    "text": self.examples_cache.iloc[idx][note_col],
                    "label": self.examples_cache.iloc[idx][label_col]
                })
        else:
            # Randomly select examples
            num_to_select = min(num_examples, len(self.examples_cache))
            selected_indices = random.sample(
                range(len(self.examples_cache)), num_to_select)

            for idx in selected_indices:
                few_shot_examples.append({
                    "text": self.examples_cache.iloc[idx][note_col],
                    "label": self.examples_cache.iloc[idx][label_col]
                })

        return self.construct_malnutrition_prompt(patient_notes, few_shot_examples)

    def get_balanced_inference_prompt(
        self,
        patient_notes: str,
        text_col: str,
        label_col: str,
        num_examples: int = 4

    ) -> str:
        """Get an inference prompt with balanced few-shot examples (equal positive/negative).

        Args:
            patient_notes (str): The patient notes to analyze
            text_col (str): Name of the text column in examples
            label_col (str): Name of the label column in examples
            num_examples (int): Total number of examples to include (should be even)

        Returns:
            str: A formatted prompt with balanced examples
        """
        if self.examples_cache is None or num_examples == 0:
            return self.construct_malnutrition_prompt(patient_notes)

        # Make num_examples even for balance
        if num_examples % 2 != 0:
            num_examples += 1

        # Split examples by label
        positive_examples = self.examples_cache[
            self.examples_cache[label_col].astype(
                str).str.lower().isin(["1", "yes", "true"])]
        negative_examples = self.examples_cache[
            ~self.examples_cache[label_col].astype(
                str).str.lower().isin(["1", "yes", "true"])]

        # Determine how many of each to use
        num_each = num_examples // 2
        num_positive = min(num_each, len(positive_examples))
        num_negative = min(num_each, len(negative_examples))

        # If one category has fewer examples, add more from the other to reach total
        if num_positive < num_each and len(negative_examples) > num_negative:
            extra_needed = num_each - num_positive
            num_negative = min(num_negative + extra_needed,
                               len(negative_examples))
        elif num_negative < num_each and len(positive_examples) > num_positive:
            extra_needed = num_each - num_negative
            num_positive = min(num_positive + extra_needed,
                               len(positive_examples))

        # Select examples
        if len(positive_examples) > 0:
            positive_indices = random.sample(
                range(len(positive_examples)), num_positive)
        else:
            positive_indices = []

        if len(negative_examples) > 0:
            negative_indices = random.sample(
                range(len(negative_examples)), num_negative)
        else:
            negative_indices = []

        few_shot_examples = []

        for idx in positive_indices:
            few_shot_examples.append({
                "text": positive_examples.iloc[idx][text_col],
                "label": positive_examples.iloc[idx][label_col]
            })

        for idx in negative_indices:
            few_shot_examples.append({
                "text": negative_examples.iloc[idx][text_col],
                "label": negative_examples.iloc[idx][label_col]
            })

        # Shuffle examples to avoid label pattern
        random.shuffle(few_shot_examples)

        return self.construct_malnutrition_prompt(patient_notes, few_shot_examples)

    def format_example(self, example: Dict[str, str]) -> str:
        """Format a single example for few-shot learning.

        Args:
            example (Dict[str, str]): Dictionary containing text and label

        Returns:
            str: Formatted example
        """
        patient_notes = example["text"]
        label = "yes" if str(example["label"]).lower() in [
            "1", "yes", "true"] else "no"

        # Create a generic explanation based on the label
        if label == "yes":
            explanation = "This patient shows signs of malnutrition based on the clinical measurements and symptoms described."
        else:
            explanation = "This patient does not show significant signs of malnutrition based on the available measurements."

        return f"""Example:
Patient notes: {patient_notes}

Assessment:
{explanation}
malnutrition={label}
"""

    def construct_malnutrition_prompt(
        self,
        patient_notes: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Constructs a comprehensive prompt for malnutrition assessment with optional few-shot examples.

        Args:
            patient_notes (str): The clinical notes about the patient to be assessed
            few_shot_examples (Optional[List[Dict[str, str]]]): List of examples for few-shot learning

        Returns:
            str: A fully formatted prompt ready for the model
        """
        instructions = """Read the patient's notes and determine if the patient is likely to have malnutrition.
Be sure to make a definitive classification: malnutrition=yes or malnutrition=no.
Along with clinical criteria, consider any additional factors such as:
1) Recent illness or surgeries
2) Socioeconomic factors (e.g., food insecurity, poverty)
3) Symptoms like fatigue, weakness, or poor appetite
4) Family history of malnutrition or chronic disease
5) Laboratory results (e.g., albumin, hemoglobin, vitamin levels)
6) Current medications or side effects that may impact nutrition (e.g., diuretics, chemotherapy)
7) Mental health factors (e.g., depression, eating disorders)
8) Any signs of malabsorption (e.g., diarrhea, malabsorptive diseases) or nutritional deficiencies (e.g., anemia, edema).
"""

        clinical_criteria = """
Standard criteria for assessment of malnutrition are based on weight-for-height, BMI-for-age, and mid-upper arm circumference. 

However, consider the following additional factors that may contribute to malnutrition:
- Recent illness (infection, gastrointestinal issues, etc.), surgery, or trauma that can increase energy expenditure and reduce nutritional intake.
- Socioeconomic factors such as food insecurity or poverty.
- Symptoms like fatigue, weakness, poor appetite, and muscle wasting.
- Family history of chronic diseases that may predispose to malnutrition (e.g., metabolic diseases).
- Laboratory results such as low serum albumin, hemoglobin, and vitamin/mineral deficiencies.
- Medication side effects that interfere with nutrition (e.g., medications causing nausea, poor appetite, or gastrointestinal side effects).
"""

        classification_table = """
Table for assessment:

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
Weight-for-height: −3 or greater z score
BMI-for-age: −3 or greater z score
Length/height-for-age: −3 z score
Mid–upper arm circumference: Greater than or equal to −3 z score
"""

        output_format = """
Output format:

1) First, provide a definitive decision: malnutrition=yes or malnutrition=no
2) Then, provide a clear explanation of your decision based on clinical data and any additional factors such as illness, lab results, medications, and socioeconomic factors.
"""

        # Format few-shot examples if provided
        few_shot_section = ""
        if few_shot_examples and len(few_shot_examples) > 0:
            few_shot_text = "\n\n".join(
                [self.format_example(example) for example in few_shot_examples])
            few_shot_section = f"""
Here are some examples of how to assess malnutrition:

{few_shot_text}

Now, assess the following patient:
"""

        complete_prompt = (
            f"{instructions}\n\n"
            f"{clinical_criteria}\n"
            f"{classification_table}\n\n"
            f"{output_format}\n\n"
            f"{few_shot_section}\n\n"
            f"{patient_notes}"
        )

        return complete_prompt


def extract_malnutrition_decision(response):
    """Extract malnutrition=yes/no decision from model response.
    Args:
        response (str): Model response text

    Returns:
        Tuple[str, str]: (malnutrition decision, explanation)
    """
    decision_pattern = r'malnutrition=(yes|no)'
    match = re.search(decision_pattern, response, re.IGNORECASE)

    decision = "unknown"
    if match:
        decision = match.group(1).lower()

    explanation = response
    if match:
        explanation_parts = response.split('malnutrition=', 1)
        if len(explanation_parts) > 0:
            explanation = explanation_parts[0].strip()

    return decision, explanation


class WeightedSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that supports weighted loss for imbalanced classes.
    This is particularly useful for scenarios where false positives
    should be penalized more heavily than false negatives.
    """
    def __init__(self, pos_weight=3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        print(f"Using custom weighted loss with positive weight: {pos_weight}")
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to implement weighted loss for imbalanced classes.
        This specifically penalizes false positive predictions more heavily.
        
        Args:
            model: The model being trained
            inputs: The inputs to the model
            return_outputs: Whether to return the model outputs along with the loss
            kwargs: Additional keyword arguments passed by Unsloth
            
        Returns:
            The computed loss or a tuple of (loss, outputs) if return_outputs is True
        """
        # Get standard outputs from the model
        outputs = model(**inputs)
        
        # Default loss from the model
        loss = outputs.loss
        
        # If logits are available (UNSLOTH_RETURN_LOGITS=1 should be set)
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            logits = outputs.logits
            
            # Get target labels
            labels = inputs["labels"]
            
            # Create mask for valid positions (non-padding)
            valid_mask = (labels != -100)
            
            if valid_mask.any():  # Only proceed if we have valid positions
                # Extract logits and labels for valid positions
                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]
                
                # For binary classification with higher penalty for false positives:
                if valid_logits.size(-1) > 1:  # Check if we have multiple output classes
                    # Create a weight tensor that's higher for positive samples
                    weights = torch.ones_like(valid_labels, dtype=torch.float)
                    
                    # Apply weights to positive examples (assuming 1 is the positive class)
                    # Higher weight means the model gets penalized more for missing these
                    is_positive = (valid_labels == 1)
                    weights[is_positive] = self.pos_weight
                    
                    # Compute weighted cross-entropy loss
                    log_probs = F.log_softmax(valid_logits, dim=-1)
                    
                    # Get one-hot encoded labels
                    one_hot_labels = F.one_hot(valid_labels, num_classes=logits.size(-1))
                    
                    # Calculate weighted negative log likelihood
                    weighted_nll = -torch.sum(weights.unsqueeze(-1) * one_hot_labels * log_probs)
                    weighted_loss = weighted_nll / torch.sum(weights)
                    
                    # Replace the original loss with our weighted loss
                    loss = weighted_loss
        
        return (loss, outputs) if return_outputs else loss
