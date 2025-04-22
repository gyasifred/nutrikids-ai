#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
import os
import re
import json
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from trl import SFTTrainer
import torch.nn.functional as F
from typing import List, Dict, Optional, Any, Match, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    auc,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve)


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
    y_scores: Optional[List[float]] = None 
) -> Dict[str, Any]:
    """Evaluate model predictions and return comprehensive metrics.
    
    Args:
        y_true (List[Any]): Ground truth labels (1/0 or "yes"/"no")
        y_pred (List[Any]): Predicted labels (1/0 or "yes"/"no")
        y_scores (Optional[List[float]]): Prediction scores/probabilities (optional)
        
    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics
    """
    # Convert labels to binary only if they are strings
    def convert_to_binary(label):
        if isinstance(label, str):
            return 1 if label.lower() == "yes" else 0
        return int(label)  # Assume already 0/1 if not a string
    
    # Convert both y_true and y_pred to binary (0 or 1)
    y_true_binary = np.array([convert_to_binary(label) for label in y_true])
    y_pred_binary = np.array([convert_to_binary(label) for label in y_pred])
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    # Get detailed classification report
    cls_report = classification_report(y_true_binary, y_pred_binary, 
                                      target_names=["no", "yes"], output_dict=True)
    
    # Initialize metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': cls_report,
        'fpr': [],
        'tpr': [],
        'precision_curve': [],
        'recall_curve': [],
        'auc': 0.0,
        'avg_precision': 0.0,
        # Store the raw data for reuse in other tools
        'raw_data': {
            'y_true': y_true,
            'y_true_binary': y_true_binary.tolist(),
            'y_pred': y_pred,
            'y_pred_binary': y_pred_binary.tolist(),
            'y_scores': y_scores if y_scores is not None else []
        }
    }
    
    # Calculate ROC and PR curves if we have probability scores
    if y_scores is not None:
        try:
            # ROC Curve
            fpr, tpr, thresholds_roc = roc_curve(y_true_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Precision-Recall Curve
            precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true_binary, y_scores)
            avg_precision = average_precision_score(y_true_binary, y_scores)
            
            metrics.update({
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds_roc': thresholds_roc.tolist(),
                'precision_curve': precision_curve.tolist(),
                'recall_curve': recall_curve.tolist(),
                'thresholds_pr': thresholds_pr.tolist() if len(thresholds_pr) > 0 else [],
                'auc': float(roc_auc),
                'avg_precision': float(avg_precision)
            })
        except Exception as e:
            print(f"Error calculating ROC/PR curves: {e}")
    
    return metrics

def save_metrics_to_csv(metrics, metrics_csv_path):
    """Save metrics to CSV file.
    
    Args:
        metrics (Dict[str, Any]): Dictionary containing evaluation metrics
        metrics_csv_path (str): Path to save metrics CSV
    """
    output_dir = os.path.dirname(metrics_csv_path)
    
    # Save basic metrics
    basic_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Average Precision'],
        'Value': [
            metrics['accuracy'], 
            metrics['precision'], 
            metrics['recall'], 
            metrics['f1'], 
            metrics.get('auc', 0.0), 
            metrics.get('avg_precision', 0.0)
        ]
    }
    pd.DataFrame(basic_metrics).to_csv(metrics_csv_path, index=False)
    print(f"Basic metrics saved to {metrics_csv_path}")
    
    # Save raw prediction data for reuse in other tools
    if 'raw_data' in metrics:
        # Create DataFrame with raw predictions
        raw_data = pd.DataFrame({
            'y_true_original': metrics['raw_data']['y_true'],
            'y_true_binary': metrics['raw_data']['y_true_binary'],
            'y_pred_original': metrics['raw_data']['y_pred'],
            'y_pred_binary': metrics['raw_data']['y_pred_binary']
        })
        
        # Add scores if available
        if len(metrics['raw_data']['y_scores']) > 0:
            raw_data['y_scores'] = metrics['raw_data']['y_scores']
        
        raw_data_path = os.path.join(output_dir, 'raw_predictions.csv')
        raw_data.to_csv(raw_data_path, index=False)
        print(f"Raw prediction data saved to {raw_data_path}")
    
    # Save raw ROC curve data
    if 'fpr' in metrics and len(metrics['fpr']) > 0:
        # ROC curve data
        roc_data = pd.DataFrame({
            'False Positive Rate': metrics['fpr'],
            'True Positive Rate': metrics['tpr'],
            'Thresholds': metrics.get('thresholds_roc', [0.0] * len(metrics['fpr']))
        })
        roc_csv_path = os.path.join(output_dir, 'roc_curve_data.csv')
        roc_data.to_csv(roc_csv_path, index=False)
        print(f"ROC curve data saved to {roc_csv_path}")
        
        # Precision-Recall curve data
        pr_data = pd.DataFrame({
            'Precision': metrics['precision_curve'],
            'Recall': metrics['recall_curve'],
            'Thresholds': metrics.get('thresholds_pr', [0.0] * len(metrics['precision_curve'])) + [0.0] if len(metrics['precision_curve']) > len(metrics.get('thresholds_pr', [])) else metrics.get('thresholds_pr', [0.0] * len(metrics['precision_curve']))
        })
        pr_csv_path = os.path.join(output_dir, 'precision_recall_curve_data.csv')
        pr_data.to_csv(pr_csv_path, index=False)
        print(f"Precision-Recall curve data saved to {pr_csv_path}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm_data = pd.DataFrame(
                metrics['confusion_matrix'],
                index=['Actual No', 'Actual Yes'],
                columns=['Predicted No', 'Predicted Yes']
            )
            cm_csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
            cm_data.to_csv(cm_csv_path)
            print(f"Confusion matrix saved to {cm_csv_path}")

def plot_evaluation_metrics(metrics, output_dir):
    """Generate and save evaluation metric plots.
    
    Args:
        metrics (Dict[str, Any]): Dictionary containing evaluation metrics
        output_dir (str): Directory to save the plots
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Plot ROC curve if available
    if 'fpr' in metrics and len(metrics['fpr']) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(metrics['fpr'], metrics['tpr'], label=f'ROC curve (AUC = {metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300)
        plt.close()
        
        # Plot Precision-Recall curve
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(metrics['recall_curve'], metrics['precision_curve'], 
                 label=f'Precision-Recall curve (AP = {metrics["avg_precision"]:.3f})')
        plt.axhline(y=sum(y_true)/len(y_true), color='r', linestyle='--', 
                    label=f'Baseline (Prevalence = {sum(y_true)/len(y_true):.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'precision_recall_curve.png'), dpi=300)
        plt.close()
        
        # Generate a calibration curve if scores are available
        if 'thresholds_roc' in metrics:
            try:
                # Create calibration curve data
                # Group predictions into bins and calculate observed vs expected rates
                bin_count = min(10, len(set(np.array(y_scores) * 100).astype(int)))
                if bin_count > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
                    
                    # sklearn calibration_curve function
                    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=bin_count)
                    plt.plot(prob_pred, prob_true, 's-', label=f'Model calibration (bins={bin_count})')
                    
                    plt.xlabel('Mean predicted probability')
                    plt.ylabel('Fraction of positives')
                    plt.title('Calibration Curve')
                    plt.legend(loc='best')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, 'calibration_curve.png'), dpi=300)
                    plt.close()
                    
                    # Save calibration data to CSV
                    calibration_data = pd.DataFrame({
                        'Mean Predicted Probability': prob_pred,
                        'Observed Fraction of Positives': prob_true
                    })
                    calibration_csv_path = os.path.join(output_dir, 'calibration_curve_data.csv')
                    calibration_data.to_csv(calibration_csv_path, index=False)
                    print(f"Calibration curve data saved to {calibration_csv_path}")
            except Exception as e:
                print(f"Warning: Could not generate calibration curve: {e}")
                
    # Plot metrics distribution with bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    metrics_values = [
        metrics['accuracy'], 
        metrics['precision'], 
        metrics['recall'], 
        metrics['f1'], 
        metrics.get('auc', 0)
    ]
    plt.bar(metrics_labels, metrics_values, color=['steelblue', 'forestgreen', 'indianred', 'mediumpurple', 'goldenrod'])
    plt.ylim([0, 1.05])
    plt.ylabel('Score')
    plt.title('Performance Metrics Overview')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels on top of bars
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_summary.png'), dpi=300)
    plt.close()
    
    print(f"Evaluation plots saved to {plots_dir}")


def print_metrics_report(metrics):
    """Print a comprehensive metrics report to the console.
    
    Args:
        metrics (Dict[str, Any]): Dictionary containing evaluation metrics
    """
    print("\n" + "="*50)
    print(" "*15 + "EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nBasic Metrics:")
    print(f"  - Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  - Precision:   {metrics['precision']:.4f}")
    print(f"  - Recall:      {metrics['recall']:.4f}")
    print(f"  - F1 Score:    {metrics['f1']:.4f}")
    
    if 'auc' in metrics and metrics['auc'] > 0:
        print(f"  - AUC:         {metrics['auc']:.4f}")
        print(f"  - Avg Precision: {metrics['avg_precision']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  [True Neg: {cm[0][0]}, False Pos: {cm[0][1]}]")
    print(f"  [False Neg: {cm[1][0]}, True Pos: {cm[1][1]}]")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.flatten()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    prevalence = (tp + fn) / (tp + fp + tn + fn)
    
    print("\nAdditional Metrics:")
    print(f"  - Specificity: {specificity:.4f}")
    print(f"  - NPV:         {npv:.4f}")
    print(f"  - Prevalence:  {prevalence:.4f}")
    
    print("\nClassification Report by Class:")
    for class_name, metrics_dict in metrics['classification_report'].items():
        if class_name in ['0', '1', 'no', 'yes']:
            class_display = 'No' if class_name in ['0', 'no'] else 'Yes'
            print(f"  - Class: {class_display}")
            print(f"    * Precision: {metrics_dict['precision']:.4f}")
            print(f"    * Recall:    {metrics_dict['recall']:.4f}")
            print(f"    * F1-Score:  {metrics_dict['f1-score']:.4f}")
            print(f"    * Support:   {metrics_dict['support']}")
    
    print("="*50)

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
        self.text_col = note_col
        self.label_col = label_col
    
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.df)

    def prepare_training_data(self, prompt_builder, tokenizer) -> List[Dict[str, str]]:
        """Prepare data in the format required for training, following Alpaca-style formatting.

        Args:
            prompt_builder: An instance of MalnutritionPromptBuilder
            tokenizer: The tokenizer to use for adding EOS token

        Returns:
            List of dictionaries with formatted text for training
        """
        EOS_TOKEN = tokenizer.eos_token
        formatted_data = []

        for _, row in self.df.iterrows():
            # Generate prompt for each example
            prompt = prompt_builder.get_training_prompt(row[self.text_col])

            # Format label as "yes" or "no"
            label_text = "yes" if str(row[self.label_col]).lower() in ["1", "yes", "true"] else "no"

            # JSON format output as per the prompt builder's specifications
            output = json.dumps({"malnutrition": label_text,
                               "explanation": f"Based on the clinical evidence in the patient notes."})

            # Create formatted text with prompt and response, similar to Alpaca format
            formatted_text = f"{prompt}\n{output}{EOS_TOKEN}"

            formatted_data.append({
                "text": formatted_text
            })

        return formatted_data

    def to_huggingface_dataset(self, prompt_builder, tokenizer) -> Dataset:
        """Convert prepared data to a HuggingFace Dataset.

        Args:
            prompt_builder: An instance of MalnutritionPromptBuilder
            tokenizer: Tokenizer for adding EOS token

        Returns:
            HuggingFace Dataset ready for model training
        """
        formatted_data = self.prepare_training_data(prompt_builder, tokenizer)

        return Dataset.from_dict({
            "text": [item["text"] for item in formatted_data]
        })
# ────────────────────────────────────────────────────────────────────────────────
# Enhanced clinical‑note cleaner
# ────────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Light preprocessing to clean clinical text."""
    text = re.sub(r"\s+", " ", text)                    # collapse multiple spaces/newlines
    text = text.replace(" </s> ", "\n- ")               # convert separators to bullets
    text = re.sub(r"_date_|_lgnum_", "[REDACTED]", text)
    return text.strip()


# ────────────────────────────────────────────────────────────────────────────────
# Enhanced Prompt‑builder with improved prompt design
# ────────────────────────────────────────────────────────────────────────────────
class MalnutritionPromptBuilder:
    """Manage creation of malnutrition prompts for training or inference."""

    # ------------------------------------------------------------------ #
    # Initialise (optionally load CSV of few‑shot examples)
    # ------------------------------------------------------------------ #
    def __init__(self, examples_csv_path: Optional[str] = None):
        self.examples_csv_path = examples_csv_path
        self.examples_cache: Optional[pd.DataFrame] = None

        if examples_csv_path:
            try:
                self.examples_cache = pd.read_csv(examples_csv_path)
                print(f"[PromptBuilder] Loaded {len(self.examples_cache)} examples from {examples_csv_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[PromptBuilder] Error loading examples: {exc}")

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def get_training_prompt(self, patient_notes: str) -> str:
        """Return a zero‑shot prompt for supervised fine‑tuning."""
        return self._construct_prompt(patient_notes)

    def get_inference_prompt(
        self,
        patient_notes: str,
        note_col: str,
        label_col: str,
        *,
        num_examples: int = 0,
        specific_example_indices: Optional[List[int]] = None,
        balanced: bool = False,
    ) -> str:
        """
        Build an inference prompt with optional few‑shot examples.

        Parameters
        ----------
        patient_notes : str
            Note you want classified.
        note_col, label_col : str
            Column names in the CSV for the note text and label.
        num_examples : int
            Number of examples to prepend (0 → zero‑shot).
        specific_example_indices : list[int], optional
            Explicit rows to use as few‑shot examples.
        balanced : bool
            If True and `num_examples ≥ 2`, sample a 50/50 yes/no mix.
        """
        if num_examples == 0 or self.examples_cache is None:
            return self._construct_prompt(patient_notes)

        # balanced branch
        if balanced and num_examples >= 2:
            return self._get_balanced_prompt(
                patient_notes, note_col, label_col, num_examples=num_examples
            )

        # generic few‑shot sampling
        few_shot_examples: List[Dict[str, str]] = []
        if specific_example_indices:
            valid = [i for i in specific_example_indices if 0 <= i < len(self.examples_cache)]
            for idx in valid[: num_examples]:
                few_shot_examples.append(
                    {
                        "text": self.examples_cache.at[idx, note_col],
                        "label": self.examples_cache.at[idx, label_col],
                    }
                )
        else:
            chosen = random.sample(
                range(len(self.examples_cache)), k=min(num_examples, len(self.examples_cache))
            )
            for idx in chosen:
                few_shot_examples.append(
                    {
                        "text": self.examples_cache.at[idx, note_col],
                        "label": self.examples_cache.at[idx, label_col],
                    }
                )

        return self._construct_prompt(patient_notes, few_shot_examples)

    def get_balanced_inference_prompt(
        self,
        patient_notes: str,
        text_col: str,
        label_col: str,
        *,
        num_examples: int = 4,
    ) -> str:
        """Return a prompt with a balanced yes/no example mix."""
        return self._get_balanced_prompt(
            patient_notes, text_col, label_col, num_examples=num_examples
        )

    # ------------------------------------------------------------------ #
    # Few‑shot example formatter - enhanced for clarity
    # ------------------------------------------------------------------ #
    def _format_example(self, example: Dict[str, str]) -> str:
        """Return one example block that matches the JSON output spec."""
        notes = preprocess_text(example["text"])
        raw_label = str(example["label"]).lower()
        label = "yes" if raw_label in {"1", "yes", "true"} else "no"

        # Enhanced explanations based on label
        if label == "yes":
            explanations = [
                "Evidence of significant weight loss and reduced dietary intake.",
                "Z-scores below -2 SD with clinical signs of wasting.",
                "Poor nutritional intake with anthropometric measurements in malnutrition range.",
                "Clinical evidence of muscle wasting with documented inadequate intake.",
                "Height-for-age z-score below -3 SD indicating severe chronic malnutrition."
            ]
        else:
            explanations = [
                "Anthropometry within normal limits with adequate intake documented.",
                "No significant weight loss or reduced intake reported.",
                "All nutritional parameters within normal range.",
                "Growth and development on track with no clinical signs of malnutrition.",
                "Z-scores within normal limits and no reported feeding difficulties."
            ]

        explanation = random.choice(explanations)

        return (
            "Patient notes:\n"
            f"{notes}\n"
            "Output:\n"
            f'{{"malnutrition":"{label}","explanation":"{explanation}"}}\n'
        )

    # ------------------------------------------------------------------ #
    # ENHANCED PROMPT CONSTRUCTION - improved readability and effectiveness
    # ------------------------------------------------------------------ #
    def _construct_prompt(
        self,
        patient_notes: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        notes_clean = preprocess_text(patient_notes)

        header = (
            "You are a board‑certified clinical dietitian with expertise in pediatric malnutrition assessment."
        )

        task = (
            "# ASSESSMENT TASK\n"
            "Evaluate the patient notes to determine if there is clinical evidence of malnutrition.\n\n"
            "Classification guidelines:\n"
            "• \"yes\" - Patient meets at least MILD criteria for malnutrition\n"
            "• \"no\" - Patient does not meet ANY malnutrition criteria\n"
            "• IMPORTANT: If evidence is borderline or ambiguous, classify as \"no\"\n"
        )

        checklist = (
            "# MALNUTRITION DIAGNOSTIC CRITERIA\n\n"
            "1. **ANTHROPOMETRY**\n"
            "   ✓ Mild:     z-score -1.0 to -1.9 SD\n"
            "   ✓ Moderate: z-score -2.0 to -2.9 SD\n"
            "   ✓ Severe:   z-score ≤ -3.0 SD\n"
            "   ✓ Weight-for-height, BMI-for-age, or weight-for-age\n"
            "   ✓ Height-for-age z-score ≤ -3 SD indicates severe stunting\n"
            "   ✓ MUAC (Mid-Upper Arm Circumference) follows same cutoffs\n\n"

            "2. **WEIGHT LOSS**\n"
            "   ✓ Documented involuntary weight loss\n"
            "   ✓ Failure to gain expected weight/height in child\n"
            "   ✓ Declining percentile crossing on growth charts\n\n"

            "3. **REDUCED INTAKE/ABSORPTION**\n"
            "   ✓ Decreased appetite or food intake\n"
            "   ✓ Feeding difficulties or dysphagia\n"
            "   ✓ Restricted diet or food insecurity\n"
            "   ✓ Malabsorption conditions\n\n"

            "4. **CLINICAL ASSESSMENT**\n"
            "   ✓ Muscle wasting (temporal, extremities)\n"
            "   ✓ Subcutaneous fat loss\n"
            "   ✓ Edema (can mask weight loss)\n"
            "   ✓ Poor wound healing, skin/hair changes\n\n"

            "5. **COMPLICATING FACTORS**\n"
            "   ✓ Chronic illness/inflammation\n"
            "   ✓ Increased metabolic demand (infection, trauma)\n"
            "   ✓ Medication effects on intake/absorption\n"
            "   ✓ Psychosocial factors\n"
        )

        output_spec = (
            "# OUTPUT REQUIREMENTS\n"
            "Return a valid JSON object with these exact fields:\n"
            "```json\n"
            "{\n"
            '  "malnutrition": "yes" | "no",\n'
            '  "explanation": "<1-2 concise sentences citing specific evidence>"\n'
            "}\n"
            "```\n"
            "- Provide only the JSON object without additional text\n"
            "- Base assessment solely on evidence present in the notes\n"
            "- Include specific metrics/findings that support your conclusion\n"
            "- Do not show your detailed reasoning process\n"
        )

        # few‑shot examples block
        few_shot_block = ""
        if few_shot_examples:
            few_shot_block = (
                "# EXAMPLES\n"
                + "\n".join(self._format_example(ex) for ex in few_shot_examples)
                + "\n---\n"
            )

        # assemble prompt
        return (
            f"{header}\n\n{task}\n{checklist}\n{output_spec}\n"
            f"{few_shot_block}"
            "# PATIENT NOTES\n"
            f"{notes_clean}\n\n"
            "# ASSESSMENT"
        )

    # ------------------------------------------------------------------ #
    # Improved balanced few‑shot sampling helper
    # ------------------------------------------------------------------ #
    def _get_balanced_prompt(
        self,
        patient_notes: str,
        note_col: str,
        label_col: str,
        *,
        num_examples: int = 4,
    ) -> str:
        """Return a prompt with a balanced yes/no example mix."""
        if self.examples_cache is None:
            return self._construct_prompt(patient_notes)

        yes_mask = self.examples_cache[label_col].astype(str).str.lower().isin({"1", "yes", "true"})
        yes_df = self.examples_cache[yes_mask]
        no_df = self.examples_cache[~yes_mask]

        # Calculate how many examples to get from each class
        yes_count = len(yes_df)
        no_count = len(no_df)
        total_available = yes_count + no_count

        if total_available < num_examples:
            # Not enough examples, use all available
            few_shot_examples = []
            for i in range(yes_count):
                few_shot_examples.append({
                    "text": yes_df.iloc[i][note_col],
                    "label": yes_df.iloc[i][label_col]
                })
            for i in range(no_count):
                few_shot_examples.append({
                    "text": no_df.iloc[i][note_col],
                    "label": no_df.iloc[i][label_col]
                })
        else:
            # We have enough examples, try to balance
            half = num_examples // 2
            yes_needed = min(half, yes_count)
            no_needed = min(num_examples - yes_needed, no_count)

            # If we couldn't get enough of one class, get more from the other
            if yes_needed < half and no_count > no_needed:
                no_needed = min(num_examples - yes_needed, no_count)
            elif no_needed < (num_examples - half) and yes_count > yes_needed:
                yes_needed = min(num_examples - no_needed, yes_count)

            # Sample
            few_shot_examples = []
            if yes_needed > 0:
                yes_indices = random.sample(range(yes_count), k=yes_needed)
                for i in yes_indices:
                    few_shot_examples.append({
                        "text": yes_df.iloc[i][note_col],
                        "label": yes_df.iloc[i][label_col]
                    })

            if no_needed > 0:
                no_indices = random.sample(range(no_count), k=no_needed)
                for i in no_indices:
                    few_shot_examples.append({
                        "text": no_df.iloc[i][note_col],
                        "label": no_df.iloc[i][label_col]
                    })

        # Shuffle to avoid order bias
        random.shuffle(few_shot_examples)
        return self._construct_prompt(patient_notes, few_shot_examples)

def extract_malnutrition_decision(response: str) -> Tuple[str, str]:
    """
    Extract the malnutrition decision (yes|no) and explanation from model output.
    Handles both JSON format and fallback extraction methods.

    Returns
    -------
    decision : str  ("yes", "no", or "unknown")
    explanation : str
    """
    # Clean the input string to handle markdown code blocks
    cleaned = response.strip()

    # Extract content from code blocks if present
    if "```" in cleaned:
        pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(pattern, cleaned, re.DOTALL)
        if matches:
            # Use the first code block that looks like valid JSON
            for match in matches:
                try:
                    # Test if parseable as JSON
                    parsed_json = json.loads(match.strip())
                    cleaned = match.strip()
                    break
                except json.JSONDecodeError:
                    continue

    # Try direct JSON parsing first (most reliable method)
    try:
        parsed = json.loads(cleaned)
        # Look for either "malnutrition" key as specified in prompt
        if "malnutrition" in parsed:
            decision = str(parsed.get("malnutrition", "unknown")).lower()
            explanation = str(parsed.get("explanation", "")).strip()
            if decision in ["yes", "no"]:
                return decision, explanation
    except json.JSONDecodeError:
        pass

    # Fallback extraction methods for both JSON and non-JSON formats

    # 1. Extract malnutrition decision with improved patterns
    decision_patterns = [
        # JSON format patterns
        r'"malnutrition"\s*:\s*"(yes|no)"',  # Standard JSON format
        r'"malnutrition"\s*:\s*"([^"]+)"',   # Any string in JSON format

        # Non-JSON format patterns
        r'malnutrition\s*[=:]\s*(yes|no)',   # Key-value pair format
        r'malnutrition.*?(yes|no)',          # Loose format

        # Sentence-based patterns
        r"patient (does|doesn['']t) meet.*?malnutrition",  # Clinical statement
        r'(evidence|signs|indications) of malnutrition',    # Evidence statement
    ]

    decision = "unknown"
    for pattern in decision_patterns:
        decision_match = re.search(pattern, cleaned, flags=re.I)
        if decision_match:
            matched_text = decision_match.group(1).lower()
            # Map various positive/negative expressions to yes/no
            if matched_text in ["yes", "does", "evidence", "signs", "indications"]:
                decision = "yes"
                break
            elif matched_text in ["no", "doesn't", "doesn't", "doesnt"]:
                decision = "no"
                break
            elif matched_text in ["yes", "no"]:
                decision = matched_text
                break

    # 2. Extract explanation with improved patterns
    explanation = ""
    explanation_patterns = [
        # JSON format patterns
        r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"',         # Quoted with possible escapes
        r'"explanation"\s*:\s*\'([^\']*)\'',                # Single-quoted
        r'"explanation"\s*:\s*"([^"]*)"',                   # Double-quoted simple

        # Non-JSON formats
        r'explanation\s*[:=]\s*["\'](.*?)["\']',            # Quoted after key
        r'explanation\s*[:=]\s*([^",\s][^,}]*)',            # Unquoted after key

        # Context-based patterns - capturing more content
        r'malnutrition.*?because\s+(.*?)(?:$|(?=\n\n|\}))', # After "because" until end or delimiter
        r'(due to\s+.*?)(?:$|(?=\n\n|\}))',                 # "Due to" phrase until delimiter
        r'evidence includes\s+(.*?)(?:$|(?=\n\n|\}))',      # Evidence statement until delimiter
    ]

    for pattern in explanation_patterns:
        expl_match = re.search(pattern, cleaned, flags=re.I | re.DOTALL)
        if expl_match:
            # Get the first non-None group
            groups = expl_match.groups()
            explanation = next((g for g in groups if g is not None), "").strip()
            if explanation:
                break

    # As last resort, try to extract relevant consecutive sentences if we have a decision but no explanation
    if decision != "unknown" and not explanation:
        sentences = re.split(r'[.!?]\s+', cleaned)
        relevant_sentences = []
        
        for sentence in sentences:
            # Check if sentence contains malnutrition-related terms
            if re.search(r'malnutrition|nutritional|weight|growth|z-score|BMI|intake|wasting|anthropometric|MUAC', sentence, re.I):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Join up to 3 relevant sentences to form a coherent explanation
            explanation = ". ".join(relevant_sentences[:3])
            if not explanation.endswith((".", "!", "?")):
                explanation += "."

    return decision, explanation
    
    
class WeightedSFTTrainer(SFTTrainer):
    """
    Performance-optimized SFTTrainer that implements a weighted loss for language modeling.
    """
    def __init__(self, pos_weight=3.0, neg_weight=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
        # Cache token IDs for positive and negative classes
        pos_tokens = ["yes", "\"yes\"", " yes", " \"yes\""]
        neg_tokens = ["no", "\"no\"", " no", " \"no\""]
        
        # Get token IDs for positive and negative labels - do this only once
        self.pos_token_ids = set()
        self.neg_token_ids = set()
        
        if hasattr(self, 'tokenizer'):
            for token in pos_tokens:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                self.pos_token_ids.update(ids)
            for token in neg_tokens:
                ids = self.tokenizer.encode(token, add_special_tokens=False)
                self.neg_token_ids.update(ids)
        
        print(f"Using weighted loss with positive weight: {pos_weight}, negative weight: {neg_weight}")
        print(f"Positive token IDs: {self.pos_token_ids}")
        print(f"Negative token IDs: {self.neg_token_ids}")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Performance-optimized loss computation.
        """
        # Only apply weighted loss during training, use standard loss for evaluation
        is_training = model.training
        
        # Fast path for evaluation
        if not is_training:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
            
        # If training, apply the weighted loss
        outputs = model(**inputs)
        
        # Standard language modeling loss as a baseline
        if not (self.pos_token_ids and self.neg_token_ids and 'labels' in inputs):
            return (outputs.loss, outputs) if return_outputs else outputs.loss
            
        # Get logits and labels
        logits = outputs.logits
        labels = inputs['labels']
        
        # Create weight mask
        batch_size, seq_len = labels.shape
        weights = torch.ones_like(labels, dtype=torch.float)
        
        # Create mask for positive and negative tokens
        pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # This loop is unavoidable but we'll optimize it
        for token_id in self.pos_token_ids:
            pos_mask |= (labels == token_id)
            
        for token_id in self.neg_token_ids:
            neg_mask |= (labels == token_id)
            
        # Apply weights using masks (vectorized operations)
        weights[pos_mask] = self.pos_weight
        weights[neg_mask] = self.neg_weight
        
        # Skip masked tokens (-100)
        valid_mask = labels != -100
        weights = weights * valid_mask
            
        # Get predictions
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Apply weights and take mean
        shift_weights_view = shift_weights.view(-1)
        valid_weight_sum = shift_weights_view.sum()
        
        # Handle edge case where all weights are zero
        if valid_weight_sum > 0:
            weighted_loss = (losses * shift_weights_view).sum() / valid_weight_sum
        else:
            weighted_loss = outputs.loss  # Fallback to default loss
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss
