#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import os
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import shap
from captum.attr import Saliency
from models.text_cnn import TextCNN, predict_batch, TextTokenizer
from sklearn.preprocessing import LabelEncoder

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TextCNN model on test data')
    parser.add_argument('--test', type=str, required=True, 
                        help='Path to test CSV file')
    parser.add_argument('--text-column', type=str, default='Note_Column',
                        help='Name of the text column in CSV (default: Note_Column)')
    parser.add_argument('--label-column', type=str, default='Malnutrition_Label',
                        help='Name of the label column in CSV (default: Malnutrition_Label)')
    parser.add_argument('--model-dir', type=str, default='model_output',
                        help='Directory containing model and artifacts (default: model_output)')
    parser.add_argument('--output-dir', type=str, default='evaluation_output',
                        help='Directory to save evaluation artifacts (default: evaluation_output)')
    return parser.parse_args()

def load_model_artifacts(model_dir):
    """Load the model, tokenizer, and label encoder from the model directory."""
    # Load model state dict
    model_path = os.path.join(model_dir, "model.pt")
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load the tokenizer using joblib
    tokenizer_path = os.path.join(model_dir, "tokenizer.joblib")
    tokenizer = joblib.load(tokenizer_path)

    # Load the label encoder using joblib
    label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    label_encoder = joblib.load(label_encoder_path)

    # Load best config to get model parameters
    try:
        config = joblib.load(os.path.join(model_dir, "best_config.joblib"))
    except:
        # If config is not available, use default values
        config = {
            "embed_dim": 100,
            "num_filters": 100,
            "kernel_sizes": [3, 4, 5],
            "dropout_rate": 0.5,
            "max_vocab_size": 10000
        }
    
    vocab_size = tokenizer.vocab_size_
    
    model = TextCNN(
        vocab_size= vocab_size,
        embed_dim=config.get("embed_dim", 100),
        num_filters=config.get("num_filters", 100),
        kernel_sizes=config.get("kernel_sizes", [3, 4, 5]),
        dropout_rate=config.get("dropout_rate", 0.5)
    )
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.eval()
    
    return model, tokenizer, label_encoder, config

def generate_confusion_matrix(y_true, y_pred, classes, output_dir):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def generate_roc_curve(y_true, y_proba, classes, output_dir):
    """Generate and save ROC curve and AUC score."""
    plt.figure(figsize=(10, 8))
    
    if len(classes) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        # Multi-class classification (one-vs-rest)
        for i, class_name in enumerate(classes):
            y_true_binary = (np.array(y_true) == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()



# Updated main() function to use these alternative methods
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading Test Data from {args.test}...")
    test_df = pd.read_csv(args.test)

    test_texts = test_df[args.text_column].tolist()
    test_labels = test_df[args.label_column].tolist()
    
    print(f"Test data: {len(test_texts)} examples")
    
    # Load model and artifacts
    print(f"Loading model and artifacts from {args.model_dir}...")
    model, tokenizer, label_encoder, config = load_model_artifacts(args.model_dir)
    
    # Convert string labels to integers
    y_true = label_encoder.transform(test_labels)
    
    # Make predictions
    print("Making predictions on test data...")
    y_pred, y_proba = predict_batch(model, tokenizer, test_texts)
    
    # Generate classification report
    print("Generating classification report...")
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(report)
    
    # Save classification report
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate confusion matrix
    print("Generating confusion matrix...")
    generate_confusion_matrix(y_true, y_pred, label_encoder.classes_, args.output_dir)
    
    # Generate ROC curve
    print("Generating ROC curve...")
    generate_roc_curve(y_true, y_proba, label_encoder.classes_, args.output_dir)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()