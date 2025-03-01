#!/usr/bin/env python3
import os
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve


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


def evaluate_predictions(y_true: List[str], y_pred: List[str], y_prob: Optional[List[float]] = None) -> Dict[str, Any]:
    """Evaluate model predictions and return comprehensive metrics.

    Args:
        y_true (List[str]): Ground truth labels ("yes" or "no")
        y_pred (List[str]): Predicted labels ("yes" or "no")
        y_prob (Optional[List[float]]): Predicted probabilities for the positive class

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics
    """
    # Convert string labels to binary for sklearn metrics
    y_true_binary = [1 if label.lower() == "yes" else 0 for label in y_true]
    y_pred_binary = [1 if label.lower() == "yes" else 0 for label in y_pred]

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
    auc = 0.0
    avg_precision = 0.0
    fpr = []
    tpr = []
    precision_curve = []
    recall_curve = []

    # Ensure we have both classes
    if y_prob is not None and len(set(y_true_binary)) > 1:
        auc = roc_auc_score(y_true_binary, y_prob)
        avg_precision = average_precision_score(y_true_binary, y_prob)

        # ROC curve data
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob)

        # Precision-Recall curve data
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true_binary, y_prob)

    # Combine all metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'avg_precision': float(avg_precision),
        'confusion_matrix': cm.tolist(),
        'classification_report': cls_report,
        'fpr': fpr.tolist() if len(fpr) > 0 else [],
        'tpr': tpr.tolist() if len(tpr) > 0 else [],
        'precision_curve': precision_curve.tolist() if len(precision_curve) > 0 else [],
        'recall_curve': recall_curve.tolist() if len(recall_curve) > 0 else []
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

    # Save to CSV
    pd.DataFrame([main_metrics]).to_csv(output_path, index=False)


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
                print(
                    f"Loaded {len(self.examples_cache)} examples from {examples_csv_path}")
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
                str).str.lower().isin(["1", "yes", "true"])
        ]
        negative_examples = self.examples_cache[
            ~self.examples_cache[label_col].astype(
                str).str.lower().isin(["1", "yes", "true"])
        ]

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
        instructions = """Read the patient's notes and determine if the patient is likely to have malnutrition."""

        clinical_criteria = """
Criteria list.

Weight is primarily affected during periods of acute undernutrition, whereas chronic undernutrition typically manifests as stunting. Severe acute undernutrition, experienced by children ages 6–60 months of age, is defined as a very low weight-for-height (less than −3 standard deviations [SD] [z scores] of the median WHO growth standards), by visible severe wasting (mid–upper arm circumference [MUAC] ≤115 mm), or by the presence of nutritional edema.

Chronic undernutrition or stunting is defined by WHO as having a height-forage (or length-for-age) that is less than −2 SD (z score) of the median of the WHO international reference.

Growth is the primary outcome measure of nutritional status in children. Growth should be monitored at regular intervals throughout childhood and adolescence and should also be measured every time a child presents, in any healthcare setting, for preventive, acute, or chronic care. In children less than 36 months of age, measures of growth include length-for-age, weight-for-age, head circumference-for-age, and weight-for-length. In children ages 2–20 years, standing height-for-age, weight-for-age, and body mass index (BMI)-for-age are typically collected.

Mild malnutrition related to undernutrition is usually the result of an acute event, either due to economic circumstances or acute illness, and presents with unintentional weight loss or weight gain velocity less than expected. Moderate malnutrition related to undernutrition occurs due to undernutrition of a significant duration that results in weight-for-length/height values or BMI-for-age values that are below the normal range. Severe malnutrition related to undernutrition occurs as a result of prolonged undernutrition and is most frequently quantified by declines in rates of linear growth that result in stunting.
"""

        classification_table = """
Table.

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

        single_point_guidance = """
On initial presentation, a child may have only a single data point for use as a criterion for the identification and diagnosis of malnutrition related to undernutrition. When this is the case, the use of z scores for weight-for-height/length, BMI-for-age, length/height-for-age or MUAC criteria as stated in Table below:
"""

        output_format = """
Follow this format:

1) First provide some explanations about your decision.
2) Then format your output as follows, strictly follow this format: malnutrition=yes or malnutrition=no
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
            f"{single_point_guidance}\n"
            f"{classification_table}\n\n"
            f"{output_format}\n\n"
            f"{few_shot_section}\n\n"
            f"{patient_notes}"
        )

        return complete_prompt
