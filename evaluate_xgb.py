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
import shap
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
    Now includes SHAP-based feature importance.
    """
    import shap  # Local import to avoid dependency issues
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=1),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=1),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=1)
    }

    unique_true_labels = np.unique(y_test)
    unique_pred_labels = np.unique(y_pred)
    logging.info(f"Unique true labels: {unique_true_labels}")
    logging.info(f"Unique predicted labels: {unique_pred_labels}")

    missing_classes = set(unique_true_labels) - set(unique_pred_labels)
    if missing_classes:
        logging.warning(f"Missing predicted classes: {missing_classes}")

    if len(np.unique(y_test)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        except Exception as e:
            logging.warning(f"Could not compute ROC AUC: {e}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importances (standard XGBoost)
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

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        importance_csv_path = os.path.join(output_dir, f'{model_name}_feature_importance.csv')
        importance_df.to_csv(importance_csv_path, index=False)
        logging.info(f"Feature importances saved to {importance_csv_path}")

        # ---- SHAP Values ----
        logging.info("Computing SHAP values...")
        try:
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)

            shap_df = pd.DataFrame(
                np.abs(shap_values.values).mean(axis=0),
                index=feature_names,
                columns=["mean_abs_shap"]
            ).sort_values(by="mean_abs_shap", ascending=False)

            shap_csv_path = os.path.join(output_dir, f"{model_name}_shap_importance.csv")
            shap_df.to_csv(shap_csv_path)
            logging.info(f"SHAP feature importances saved to {shap_csv_path}")

            # Optional SHAP beeswarm plot
            shap_plot_path = os.path.join(output_dir, f"{model_name}_shap_summary.png")
            shap.plots.beeswarm(shap_values, show=False)
            plt.tight_layout()
            plt.savefig(shap_plot_path)
            plt.close()
            logging.info(f"SHAP summary plot saved to {shap_plot_path}")

        except Exception as e:
            logging.warning(f"SHAP analysis failed: {e}")
            
    # ROC Curve and data (for binary classification)
    if len(np.unique(y_test)) == 2:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
        
        # Save ROC curve data to CSV
        roc_df = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        })
        roc_csv_path = os.path.join(output_dir, f'{model_name}_roc_curve_data.csv')
        roc_df.to_csv(roc_csv_path, index=False)
        logging.info(f"ROC curve data saved to {roc_csv_path}")
        
        # Plot ROC curve
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
        
        # Precision-Recall curve and data
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
        
        # Save Precision-Recall curve data to CSV
        pr_df = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'thresholds': np.append(pr_thresholds, [1.0])  # Add 1.0 to match the length of precision/recall
        })
        pr_csv_path = os.path.join(output_dir, f'{model_name}_precision_recall_curve_data.csv')
        pr_df.to_csv(pr_csv_path, index=False)
        logging.info(f"Precision-Recall curve data saved to {pr_csv_path}")
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='green', label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend(loc="upper right")
        pr_path = os.path.join(output_dir, f'{model_name}_precision_recall_curve.png')
        plt.savefig(pr_path)
        plt.close()
    
    # For multiclass classification, save class-specific metrics
    if len(np.unique(y_test)) > 2:
        # Get class probabilities for each sample
        class_probs = y_pred_proba
        
        # Save all prediction probabilities to CSV
        # Create a DataFrame with predicted probabilities for each class
        probs_df = pd.DataFrame(class_probs)
        
        # Rename columns to match class labels
        probs_df.columns = [f'class_{i}_prob' for i in range(probs_df.shape[1])]
        
        # Add true and predicted labels
        probs_df['true_label'] = y_test
        probs_df['predicted_label'] = y_pred
        
        # Add ID column if available
        if id_series is not None:
            probs_df['id'] = id_series.values
        
        # Save to CSV
        probs_csv_path = os.path.join(output_dir, f'{model_name}_class_probabilities.csv')
        probs_df.to_csv(probs_csv_path, index=False)
        logging.info(f"Class probabilities saved to {probs_csv_path}")
        
        # One-vs-Rest ROC curves for each class
        plt.figure(figsize=(12, 10))
        
        # Create DataFrame to store all OvR ROC data
        ovr_roc_data = []
        
        for i in range(len(np.unique(y_test))):
            # Create binary labels (1 for current class, 0 for all others)
            binary_y = (y_test == i).astype(int)
            
            # Calculate ROC curve for this class
            fpr, tpr, thresholds = roc_curve(binary_y, class_probs[:, i])
            
            # Store data for this class
            for j in range(len(fpr)):
                ovr_roc_data.append({
                    'class': i,
                    'fpr': fpr[j],
                    'tpr': tpr[j],
                    'threshold': thresholds[j] if j < len(thresholds) else None
                })
            
            # Plot this class's ROC curve
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc_score(binary_y, class_probs[:, i]):.2f})')
        
        # Save OvR ROC data to CSV
        ovr_roc_df = pd.DataFrame(ovr_roc_data)
        ovr_roc_csv_path = os.path.join(output_dir, f'{model_name}_ovr_roc_curve_data.csv')
        ovr_roc_df.to_csv(ovr_roc_csv_path, index=False)
        logging.info(f"One-vs-Rest ROC curve data saved to {ovr_roc_csv_path}")
        
        # Complete the OvR ROC plot
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} One-vs-Rest ROC Curves')
        plt.legend(loc='lower right')
        ovr_roc_path = os.path.join(output_dir, f'{model_name}_ovr_roc_curves.png')
        plt.savefig(ovr_roc_path)
        plt.close()
    
    # Prepare predictions DataFrame with all details
    predictions_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred
    })
    
    # Add probabilities to predictions DataFrame
    for i in range(y_pred_proba.shape[1]):
        predictions_df[f'prob_class_{i}'] = y_pred_proba[:, i]
    
    # Add ID column if available
    if id_series is not None:
        predictions_df['id'] = id_series.values
    
    # Save predictions to CSV
    predictions_path = os.path.join(output_dir, f'{model_name}_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logging.info(f"Predictions with probabilities saved to {predictions_path}")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(cm)
    cm_csv_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.csv')
    cm_df.to_csv(cm_csv_path, index=False, header=False)

    # Classification report CSV
    class_report_df = pd.DataFrame(class_report).transpose()
    report_csv_path = os.path.join(output_dir, f'{model_name}_classification_report.csv')
    class_report_df.to_csv(report_csv_path)

    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'confusion_matrix_path': cm_path,
        'exported_files': {
            'confusion_matrix_csv': cm_csv_path,
            'classification_report_csv': report_csv_path,
        }
    }

    if feature_names is not None:
        results['exported_files']['feature_importance_csv'] = importance_csv_path
        results['exported_files']['shap_importance_csv'] = shap_csv_path
        results['exported_files']['shap_summary_plot'] = shap_plot_path

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
    parser.add_argument('--id_column', type=str, default='id', 
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
