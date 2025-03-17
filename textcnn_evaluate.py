#!/usr/bin/env python3

from collections import Counter
import json
import numpy as np
import pandas as pd
import torch
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve, auc,
    classification_report
)
from models.text_cnn import predict_batch, load_model_artifacts


def generate_confusion_matrix(y_true, y_pred, classes, output_dir):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create non-normalized confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Create normalized confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'))
    plt.close()

    # Save confusion matrix data
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_labels': classes.tolist()
        if hasattr(classes, 'tolist') else classes
    }

    with open(os.path.join(output_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump(cm_data, f, indent=2)


def generate_roc_curve(y_true, y_proba, classes, output_dir):
    """Generate and save ROC curve and AUC score."""
    plt.figure(figsize=(10, 8))

    roc_data = {}

    if len(classes) == 2:
        # Binary classification
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')

        # Save ROC data
        roc_data['binary'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': roc_auc
        }
    else:
        # Multi-class classification (one-vs-rest)
        roc_data['multiclass'] = {}

        for i, class_name in enumerate(classes):
            y_true_binary = (np.array(y_true) == i).astype(int)
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                     label=f'Class {class_name} (AUC = {roc_auc:.2f})')

            # Save ROC data
            roc_data['multiclass'][class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': roc_auc
            }

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # Save ROC data
    with open(os.path.join(output_dir, 'roc_data.json'), 'w') as f:
        json.dump(roc_data, f, indent=2)


def generate_precision_recall_curve(y_true, y_proba, classes, output_dir):
    """Generate and save Precision-Recall curve and Average Precision score."""
    plt.figure(figsize=(10, 8))

    pr_data = {}

    if len(classes) == 2:
        # Binary classification
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_proba[:, 1])
        plt.plot(recall, precision, lw=2,
                 label=f'PR curve (AP = {avg_precision:.2f})')

        # Save PR data
        pr_data['binary'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'average_precision': avg_precision
        }
    else:
        # Multi-class classification (one-vs-rest)
        pr_data['multiclass'] = {}

        for i, class_name in enumerate(classes):
            y_true_binary = (np.array(y_true) == i).astype(int)
            precision, recall, thresholds = precision_recall_curve(
                y_true_binary, y_proba[:, i])
            avg_precision = average_precision_score(
                y_true_binary, y_proba[:, i])
            plt.plot(recall, precision, lw=2,
                     label=f'Class {class_name} (AP = {avg_precision:.2f})')

            # Save PR data
            pr_data['multiclass'][class_name] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist(),
                'average_precision': avg_precision
            }

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

    # Save PR data
    with open(os.path.join(output_dir,
                           'precision_recall_data.json'), 'w') as f:
        json.dump(pr_data, f, indent=2)


def generate_class_distribution(y_true, y_pred, classes, output_dir):
    """Generate and save class distribution plots."""
    # Actual label distribution
    plt.figure(figsize=(10, 6))
    # Convert numpy arrays to scalar values if needed
    y_true_list = [i.item() if hasattr(i, 'item') else i for i in y_true]
    y_true_counts = Counter([classes[i] for i in y_true_list])
    sns.barplot(x=list(y_true_counts.keys()), y=list(y_true_counts.values()))
    plt.title('Actual Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_class_distribution.png'))
    plt.close()

    # Predicted label distribution
    plt.figure(figsize=(10, 6))
    # Convert numpy arrays to scalar values if needed
    y_pred_list = [i.item() if hasattr(i, 'item') else i for i in y_pred]
    y_pred_counts = Counter([classes[i] for i in y_pred_list])
    sns.barplot(x=list(y_pred_counts.keys()), y=list(y_pred_counts.values()))
    plt.title('Predicted Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_class_distribution.png'))
    plt.close()

    # Save distribution data
    distribution_data = {
        'actual': {k: v for k, v in sorted(y_true_counts.items())},
        'predicted': {k: v for k, v in sorted(y_pred_counts.items())}
    }

    with open(os.path.join(output_dir, 'class_distribution.json'), 'w') as f:
        json.dump(distribution_data, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate TextCNN model on test data')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--text_column', type=str, default='txt',
                        help='Name of the text column in CSV (default: txt)')
    parser.add_argument('--label_column', type=str,
                        default='label',
                        help='Name of the label column in CSV')
    parser.add_argument('--id_column', type=str, default="DEID",
                        help='Name of the column containing IDs')
    parser.add_argument('--model_dir', type=str, default='CNN',
                        help='Directory containing model and artifacts')
    parser.add_argument('--output_dir', type=str, default='CNN/evaluation',
                        help='Directory to save evaluation artifacts')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction (default: 32)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification \
                            (default: 0.5)')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading Test Data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)

    # Check if required columns exist
    if args.text_column not in test_df.columns:
        raise ValueError(
            f"Text column '{args.text_column}' not found in the test file")
    if args.label_column not in test_df.columns:
        raise ValueError(
            f"Label column '{args.label_column}' not found in the test file")

    test_texts = test_df[args.text_column].fillna("").tolist()
    test_labels = test_df[args.label_column].tolist()

    print(f"Test data: {len(test_texts)} examples")

    # Load model and artifacts
    print(f"Loading model and artifacts from {args.model_dir}...")
    model, tokenizer, label_encoder, config = load_model_artifacts(
        args.model_dir)

    # Convert string labels to integers
    y_true = label_encoder.transform(test_labels)

    # Make predictions
    print(f"Making predictions on {len(test_texts)} texts...")
    # Process in batches to handle large datasets
    all_predictions = []
    all_probabilities = []

    for i in range(0, len(test_texts), args.batch_size):
        batch_texts = test_texts[i:i+args.batch_size]
        batch_preds, batch_probs = predict_batch(
            model, tokenizer, batch_texts)
        # Convert tensors to lists if needed
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu().numpy().tolist()
        if isinstance(batch_probs, torch.Tensor):
            batch_probs = batch_probs.cpu().numpy().tolist()
        all_predictions.extend(batch_preds)
        all_probabilities.extend(batch_probs)

    # Convert to numpy arrays for analysis functions
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)

    # Generate classification report
    print("Generating classification report...")
    report = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_)
    print(report)

    # Save classification report
    with open(os.path.join(args.output_dir,
                           'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    # Generate confusion matrix
    print("Generating confusion matrix...")
    generate_confusion_matrix(
        y_true, y_pred, label_encoder.classes_, args.output_dir)

    # Generate ROC curve
    print("Generating ROC curve...")
    generate_roc_curve(
        y_true, y_proba, label_encoder.classes_, args.output_dir)
    # Generate precision-recall curve
    print("Generating precision-recall curve...")
    generate_precision_recall_curve(
        y_true, y_proba, label_encoder.classes_, args.output_dir)
    # Generate class distribution
    print("Generating class distribution plots...")
    generate_class_distribution(
        y_true, y_pred, label_encoder.classes_, args.output_dir)

    print("Saving predictions...")

    # Explicit copy to avoid modification issues
    pred_df = test_df[[args.id_column]].copy()

    # Store true and predicted labels
    pred_df["true_label"] = y_true
    pred_df["predicted_label"] = label_encoder.inverse_transform(y_pred)

    # Add probability scores
    for i, class_name in enumerate(label_encoder.classes_):
        pred_df[f'prob_{class_name}'] = y_proba[:, i]

    # Save to CSV
    output_path = os.path.join(args.output_dir, 'predictions.csv')
    pred_df.to_csv(output_path, index=False)

    print(f"Evaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
