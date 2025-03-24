#!/usr/bin/env python3
"""
XGBoost Model Evaluation Script: Comprehensive model evaluation with flexible label handling.
"""

import os
import glob
import json
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns 
from utils import load_xgbartifacts

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def handle_labels(y_test, label_encoder=None):
    """
    Handle label transformation for test data.
    
    Args:
        y_test (pd.Series): Label series to be transformed
        label_encoder (LabelEncoder, optional): Existing label encoder
    
    Returns:
        tuple: Transformed labels and label mapping (if created)
    """
    # Case 1: Labels are already numeric
    if pd.api.types.is_numeric_dtype(y_test):
        return y_test.values, None

    # Case 2: Label encoder provided
    if label_encoder is not None:
        try:
            # Check if test labels match encoder classes
            unique_test_labels = y_test.unique()
            if set(unique_test_labels).issubset(set(label_encoder.classes_)):
                return label_encoder.transform(y_test), None
            else:
                logging.warning("Test labels do not match trained encoder classes")
        except Exception as e:
            logging.error(f"Error using provided label encoder: {e}")

    # Case 3: No label encoder or mismatch - create a new one
    temp_encoder = LabelEncoder()
    y_transformed = temp_encoder.fit_transform(y_test)
    
    # Create mapping for reference
    mapping = {
        "original_labels": list(map(str, temp_encoder.classes_)),
        "encoded_labels": list(range(len(temp_encoder.classes_)))
    }
    
    logging.info(f"Created new label mapping with {len(mapping['original_labels'])} classes")
    return y_transformed, mapping


def evaluate_xgb_model(
    model, 
    X_test, 
    y_test, 
    feature_names=None, 
    output_dir='xgb_evaluation', 
    model_name='xgb_model',
    id_series=None
):
    """
    Comprehensive model evaluation with multiple metrics and visualizations.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        feature_names: Optional list of feature names
        output_dir: Directory to save evaluation results
        model_name: Name of the model for file naming
        id_series: Optional series of IDs for the test data
    
    Returns:
        dict: Comprehensive evaluation metrics
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluation metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Try computing ROC AUC if binary classification
    try:
        if len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
    except Exception as e:
        logging.warning(f"Could not compute ROC AUC: {e}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Classification Report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature Importance (if available)
    if feature_names is not None:
        plt.figure(figsize=(12, 8))
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, [feature_names[i] for i in sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} Feature Importance')
        importance_path = os.path.join(output_dir, f'{model_name}_feature_importance.png')
        plt.savefig(importance_path, bbox_inches='tight')
        plt.close()
    
    # ROC Curve (for binary classification)
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        roc_path = os.path.join(output_dir, f'{model_name}_roc_curve.png')
        plt.savefig(roc_path)
        plt.close()
    
    # Prepare predictions DataFrame
    if id_series is not None:
        predictions_df = pd.DataFrame({
            'id': id_series,
            'true_label': y_test,
            'predicted_label': y_pred
        })
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logging.info(f"Predictions saved to {predictions_path}")
    
    # Save results
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'confusion_matrix_path': cm_path
    }
    
    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f'{model_name}_evaluation_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logging.info(f"Evaluation results saved to {results_path}")
    
    return results


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='XGBoost Model Evaluation Script')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the directory containing model artifacts')
    parser.add_argument('--data_file', type=str, required=True, 
                        help='Path to the CSV test data file')
    
    # Optional arguments
    parser.add_argument('--model_name', type=str, default='xgboost', 
                        help='Name of the model for file naming')
    parser.add_argument('--text_column', type=str, default='txt', 
                        help='Name of the column containing text data')
    parser.add_argument('--label_column', type=str, default='label', 
                        help='Name of the column containing labels')
    parser.add_argument('--id_column', type=str, default='DEID', 
                        help='Name of the column containing unique identifiers')
    parser.add_argument('--output_dir', type=str, default='xgb_evaluation', 
                        help='Directory to save evaluation results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load model artifacts
    model, label_encoder, pipeline, feature_names = load_xgbartifacts(
        args.model_path, args.model_name)
    
    # Load and preprocess test data
    logging.info(f"Loading test data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    
    # Transform features
    X_test = pipeline.transform(df[args.text_column])
    
    # Handle labels
    y_test, label_mapping = handle_labels(df[args.label_column], label_encoder)
    
    # If a new label mapping was created, save it
    if label_mapping:
        mapping_filename = os.path.join(
            args.output_dir, 
            f"{args.model_name}_label_mapping.json"
        )
        with open(mapping_filename, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        logging.info(f"Label mapping saved to {mapping_filename}")
    
    # Ensure X_test is in correct format
    if hasattr(X_test, 'toarray'):
        X_test = X_test.toarray()
    
    # Evaluate model
    logging.info("Starting model evaluation...")
    results = evaluate_xgb_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=args.output_dir,
        model_name=args.model_name,
        id_series=df[args.id_column] if args.id_column in df.columns else None
    )
    
    logging.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
