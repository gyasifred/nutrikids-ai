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
    auc,
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

#!/usr/bin/env python3
"""
Malnutrition prompt builder – full version (April 2025).

Features
--------
* Role‑primed, evidence‑checklist prompt with WHO z‑score bands:
      −1 to −1.9 SD  → mild
      −2 to −2.9 SD  → moderate
      ≤ −3 SD        → severe
* JSON‑only output spec for easy parsing.
* Zero‑, one‑, or few‑shot generation with optional balanced class sampling.
* Helper to extract the decision + explanation from model output.

Dependencies
------------
pandas (for loading few‑shot examples)
"""
from __future__ import annotations

import json
import random
import re
from typing import Dict, List, Optional, Tuple, Match

import pandas as pd


# ────────────────────────────────────────────────────────────────────────────────
# Simple clinical‑note cleaner
# ────────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Light preprocessing to clean clinical text."""
    text = re.sub(r"\s+", " ", text)                    # collapse multiple spaces/newlines
    text = text.replace(" </s> ", "\n- ")               # convert separators to bullets
    text = re.sub(r"_date_|_lgnum_", "[REDACTED]", text)
    return text.strip()


# ────────────────────────────────────────────────────────────────────────────────
# Prompt‑builder
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

    # ------------------------------------------------------------------ #
    # Few‑shot example formatter
    # ------------------------------------------------------------------ #
    def _format_example(self, example: Dict[str, str]) -> str:
        """Return one example block that matches the JSON output spec."""
        notes = preprocess_text(example["text"])
        raw_label = str(example["label"]).lower()
        label = "yes" if raw_label in {"1", "yes", "true"} else "no"

        explanation = (
            "Evidence of weight loss and reduced intake."
            if label == "yes"
            else "Anthropometry and intake within normal limits."
        )

        return (
            "Patient notes:\n"
            f"{notes}\n"
            "Output:\n"
            f'{{"malnutrition":"{label}","explanation":"{explanation}"}}\n'
        )

    # ------------------------------------------------------------------ #
    # PROMPT CONSTRUCTION (includes z‑score bands)
    # ------------------------------------------------------------------ #
    def _construct_prompt(
        self,
        patient_notes: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        notes_clean = preprocess_text(patient_notes)

        header = (
            "You are a board‑certified clinical dietitian.  Evaluate the following "
            "patient note for evidence of malnutrition."
        )

        task = (
            "TASK\n"
            'Classify as:\n'
            '• malnutrition = "yes"  → patient meets at least **mild** criteria below or other strong evidence\n'
            '• malnutrition = "no"   → patient meets **none** of the criteria\n'
        )

        checklist = (
            "EVIDENCE CHECKLIST  (use any data present)\n"
            "1. Anthropometry  (z‑scores)\n"
            "       Mild     : −1.0 to −1.9 SD\n"
            "       Moderate : −2.0 to −2.9 SD\n"
            "       Severe   : ≤ −3.0 SD\n"
            "   • Length/height‑for‑age z‑score ≤ −3 SD ⇒ automatically *severe*\n"
            "   • Mid‑upper‑arm‑circumference (MUAC) follows same cut‑offs.\n"
            "2. Clinical signs   – muscle/fat wasting, oedema, brittle hair/skin, fatigue\n"
            "3. Intake          – reduced calories / protein, poor appetite, food insecurity\n"
            "4. Medical factors – chronic disease, malabsorption, infection, cancer, etc.\n"
            "5. Other risks     – polypharmacy, mental‑health issues, low socioeconomic status\n"
        )

        output_spec = (
            "OUTPUT – return valid JSON **only** (no extra prose):\n"
            '{\n'
            '  "malnutrition": "yes" | "no",\n'
            '  "explanation": "<1‑2 concise sentences citing strongest evidence lines>"\n'
            "}\n"
            "If evidence is borderline, default to \"no\" but briefly state why.\n"
            "Think through the checklist internally; do **not** reveal your full reasoning.\n"
        )

        # few‑shot examples block
        few_shot_block = ""
        if few_shot_examples:
            few_shot_block = (
                "HERE ARE FEW‑SHOT EXAMPLES\n"
                + "\n".join(self._format_example(ex) for ex in few_shot_examples)
                + "\n---\n"
            )

        # assemble prompt
        return (
            f"{header}\n\n{task}{checklist}{output_spec}"
            f"{few_shot_block}"
            "PATIENT NOTES\n"
            f"{notes_clean}"
        )

    # ------------------------------------------------------------------ #
    # Balanced few‑shot sampling helper
    # ------------------------------------------------------------------ #
    def _get_balanced_prompt(
        self,
        patient_notes: str,
        note_col: str,
        label_col: str,
        *,
        num_examples: int = 4,
    ) -> str:
        """Return a prompt with a 50/50 yes/no example mix."""
        if self.examples_cache is None:
            return self._construct_prompt(patient_notes)

        yes_mask = self.examples_cache[label_col].astype(str).str.lower().isin({"1", "yes", "true"})
        yes_df = self.examples_cache[yes_mask]
        no_df = self.examples_cache[~yes_mask]

        # Determine how many examples we can actually get from each class
        per_class = max(1, num_examples // 2)
        actual_yes = min(per_class, len(yes_df))
        actual_no = min(per_class, len(no_df))
        
        # Adjust sampling to maintain balance
        if actual_yes < per_class and actual_no > actual_yes:
            # If we have fewer "yes" examples, take more "no" examples to reach target count
            actual_no = min(num_examples - actual_yes, len(no_df))
        elif actual_no < per_class and actual_yes > actual_no:
            # If we have fewer "no" examples, take more "yes" examples to reach target count
            actual_yes = min(num_examples - actual_no, len(yes_df))
            
        few: List[Dict[str, str]] = []

        # sample yes
        if len(yes_df) > 0:
            idx = random.sample(range(len(yes_df)), k=actual_yes)
            for i in idx:
                few.append({"text": yes_df.iloc[i][note_col], "label": yes_df.iloc[i][label_col]})

        # sample no
        if len(no_df) > 0:
            idx = random.sample(range(len(no_df)), k=actual_no)
            for i in idx:
                few.append({"text": no_df.iloc[i][note_col], "label": no_df.iloc[i][label_col]})

        random.shuffle(few)
        return self._construct_prompt(patient_notes, few)


# ────────────────────────────────────────────────────────────────────────────────
# Helper: parse model response
# ────────────────────────────────────────────────────────────────────────────────
def extract_malnutrition_decision(response: str) -> Tuple[str, str]:
    """
    Extract the malnutrition=yes|no decision and explanation from model output.

    Returns
    -------
    decision : str  ("yes", "no", or "unknown")
    explanation : str
    """
    cleaned = response.strip().lstrip("```json").rstrip("```").strip()

    # JSON first
    try:
        parsed = json.loads(cleaned)
        decision = str(parsed.get("malnutrition", "unknown")).lower()
        explanation = str(parsed.get("explanation", "")).strip()
        return decision, explanation
    except Exception:  # noqa: BLE001
        pass

    # fallback regex (legacy formats)
    match: Optional[Match[str]] = re.search(r'"?malnutrition"?\s*[:=]\s*"?\b(yes|no)\b', cleaned, flags=re.I)
    decision = match.group(1).lower() if match else "unknown"

    expl_match: Optional[Match[str]] = re.search(r'"?explanation"?\s*[:=]\s*"?([^"\n}]+)', cleaned, flags=re.I)
    explanation = expl_match.group(1).strip() if expl_match else ""

    return decision, explanation
    
class WeightedSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that supports weighted loss for imbalanced classes.
    This allows separate weighting for both positive and negative class predictions.
    """
    def __init__(self, pos_weight=3.0, neg_weight=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        print(f"Using custom weighted loss with positive weight: {pos_weight}, negative weight: {neg_weight}")
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to implement weighted loss for imbalanced classes.
        This specifically penalizes both false positive and false negative predictions
        with separate weights.
        
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
                
                # For binary classification with separate penalties for both classes:
                if valid_logits.size(-1) > 1:  # Check if we have multiple output classes
                    # Create a weight tensor that's different for each class
                    weights = torch.ones_like(valid_labels, dtype=torch.float)
                    
                    # Apply weights based on the class
                    is_positive = (valid_labels == 1)
                    is_negative = (valid_labels == 0)
                    
                    # Apply the separate weights
                    weights[is_positive] = self.pos_weight
                    weights[is_negative] = self.neg_weight
                    
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
