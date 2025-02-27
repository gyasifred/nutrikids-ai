#!/usr/bin/env python3
"""
XGBoost Evaluation Script: Evaluates a trained XGBoost model on a test dataset.
Generates and saves comprehensive evaluation metrics and visualizations.
"""

import json
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import logging
import torch
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import xgboost as xgb
from scipy.sparse import csr_matrix


def load_artifacts(model_dir: str, model_name: str):
    """ 
    Load all model artifacts (model, label encoder, pipeline) from the given directory.

    Args:
        model_dir (str): Path to the directory containing model artifacts.
        model_name (str): Name of the model.

    Returns:
        model, label_encoder, pipeline, feature_names
    """

    # Define the file patterns to match the latest files
    model_pattern = os.path.join(model_dir, f"{model_name}_nutrikidai_model.json")
    label_encoder_pattern = os.path.join(model_dir, f"{model_name}_nutrikidai_classifier_label_encoder_*.joblib")
    pipeline_pattern = os.path.join(model_dir, f"{model_name}_nutrikidai_pipeline.joblib")
    # List the files that match the patterns
    model_files = glob.glob(model_pattern)
    label_encoder_files = glob.glob(label_encoder_pattern)
    pipeline_files = glob.glob(pipeline_pattern)

    # Debugging prints to check the found files
    logging.info(f"Found model files: {model_files}")
    logging.info(f"Found Label Encoder files: {label_encoder_files}")
    logging.info(f"Found pipeline files: {pipeline_files}")

    # Ensure that there are files found for each pattern
    if not model_files:
        raise ValueError(f"No model files found matching pattern: {model_pattern}")
    if not label_encoder_files:
        raise ValueError(f"No label encoder files found matching pattern: {label_encoder_pattern}")
    if not pipeline_files:
        raise ValueError(f"No pipeline files found matching pattern: {pipeline_pattern}")

    logging.info(f"Loading model from {model_dir}...")
    model = xgb.XGBClassifier()
    model.load_model(model_pattern)
    # Get the latest label encoder file
    label_encoder_path = max(label_encoder_files, key=os.path.getmtime)
    logging.info(f"Loading label encoder from {label_encoder_path}...")
    label_encoder = joblib.load(label_encoder_path)

    # Get the latest pipeline file
    pipeline_path = max(pipeline_files, key=os.path.getmtime)
    logging.info(f"Loading pipeline from {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)
    
    # Get feature names from pipeline
    feature_names = None
    try:
        vectorizer = pipeline.named_steps.get('vectorizer')  # Update name if different
        if vectorizer and hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out().tolist()
            logging.info(f"Extracted {len(feature_names)} feature names from pipeline")
        else:
            logging.warning("Could not find vectorizer or get_feature_names_out method")
    except Exception as e:
        logging.warning(f"Error extracting feature names: {str(e)}")

    return model, label_encoder, pipeline, feature_names


def ensure_features_match(X_test, feature_names):
    """
    Ensure the test feature matrix has all the features expected by the model,
    in the correct order.
    
    Args:
        X_test: Test features DataFrame or sparse matrix
        feature_names: List of expected feature names
    
    Returns:
        X_test_aligned: A matrix with columns aligned to feature_names
    """
    if feature_names is None:
        logging.warning("No feature names provided. Cannot ensure feature alignment.")
        return X_test
    
    logging.info(f"Ensuring feature alignment (expected {len(feature_names)} features)")
    
    # Convert to DataFrame if it's a sparse matrix
    if isinstance(X_test, csr_matrix):
        X_test_dense = pd.DataFrame.sparse.from_spmatrix(X_test)
    else:
        X_test_dense = pd.DataFrame(X_test)
    
    # Create empty DataFrame with training features
    aligned_df = pd.DataFrame(columns=feature_names)
    
    # Fill matching features
    for col in X_test_dense.columns:
        if isinstance(col, int) and col < len(feature_names):
            aligned_df[feature_names[col]] = X_test_dense[col]
    
    # Fill missing features with 0
    aligned_df.fillna(0, inplace=True)
    
    logging.info(f"Feature alignment complete. Matrix shape: {aligned_df.shape}")
    return aligned_df.values


def plot_confusion_matrix(cm, classes, output_path):
    """
    Plots a confusion matrix.
    
    Args:
        cm: Confusion matrix
        classes: Class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(fpr, tpr, auc, output_path):
    """
    Plots the ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc: Area under the curve
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall_curve(precision, recall, avg_precision, output_path):
    """
    Plots the precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        avg_precision: Average precision
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(feature_names, importance, top_n, output_path):
    """
    Plots feature importance.
    
    Args:
        feature_names: Names of features
        importance: Importance values
        top_n: Number of top features to plot
        output_path: Path to save the plot
    """
    if len(feature_names) == 0 or len(importance) == 0:
        logging.warning("No feature names or importance values provided. Cannot plot feature importance.")
        return
    
    # Convert to numpy arrays for easier handling
    feature_names = np.array(feature_names)
    importance = np.array(importance)
    
    # If there are too many features, limit to top N
    if len(feature_names) > top_n:
        # Get indices of top N features by importance
        indices = np.argsort(importance)[-top_n:]
        feature_names = feature_names[indices]
        importance = importance[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_names)), importance, align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {len(feature_names)} Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained XGBoost model')
    
    # Required parameters
    parser.add_argument('--model_name', type=str, default="xgb", help='Name of the model')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the CSV test data file')
    parser.add_argument('--text_column', type=str, default="Note_Column", help='Name of the column containing text data')
    parser.add_argument('--label_column', type=str, default="Malnutrition_Label", help='Name of the column containing labels')
    parser.add_argument('--id_column', type=str, default="Patient_ID", help='Name of the column containing IDs')
    
    # Optional parameters
    parser.add_argument('--model_dir', type=str, default='./xgb_models', help='Directory containing model artifacts')
    parser.add_argument('--output_dir', type=str, default='./xgb_evaluation', help='Directory to save evaluation results')
    parser.add_argument('--num_shap_samples', type=int, default=100, help='Number of samples for SHAP explanation')
    parser.add_argument('--top_n_features', type=int, default=20, help='Number of top features to plot')
    parser.add_argument('--debug', action='store_true', help='Enable extra debug logging')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting model evaluation with arguments: {args}")
    
    try:
        # Load artifacts: model, label encoder, pipeline, and feature names
        xgb_model, label_encoder, pipeline, feature_names = load_artifacts(args.model_dir, args.model_name)
        
        # Process the test data
        logger.info(f"Loading test data from {args.data_file}...")
        df = pd.read_csv(args.data_file)
        
        # Verify required columns exist
        required_columns = [args.text_column, args.label_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in test data: {missing_columns}. Available columns: {list(df.columns)}")
        
        logger.info(f"Applying pipeline to transform text data...")
        X_test = pipeline.transform(df[args.text_column])
        
        # Ensure features match the training data
        X_test_aligned = ensure_features_match(X_test, feature_names)
        
        # Log shape information for debugging
        logger.info(f"Original test data shape: {X_test.shape if hasattr(X_test, 'shape') else 'unknown'}")
        logger.info(f"Aligned test data shape: {X_test_aligned.shape}")
        logger.info(f"Feature names count: {len(feature_names) if feature_names else 'unknown'}")
        
        # Get labels
        y_test = df[args.label_column].values
        logger.info(f"Test data labels: {np.unique(y_test)}")
        logger.info(f"Label encoder classes: {label_encoder.classes_}")
        
        # Map labels if necessary (for example, converting 'yes'/'no' to numeric values)
        if hasattr(y_test, 'dtype') and y_test.dtype == object:
            y_test = label_encoder.transform(y_test)
        
        # Generate predictions
        logger.info("Generating predictions...")
        y_pred_proba = xgb_model.predict_proba(X_test_aligned)[:, 1]  # Positive class probabilities
        y_pred = xgb_model.predict(X_test_aligned)  # Class predictions
        
        # --- Evaluation Metrics ---
        logger.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # Calculate ROC AUC and average precision
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
        except Exception as e:
            logger.warning(f"Error calculating AUC or average precision: {str(e)}")
            auc = 0.0
            avg_precision = 0.0
        
        # Print classification report
        logger.info("\nClassification Report:")
        class_names = label_encoder.classes_
        cls_report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        logger.info("\n" + cls_report)
        
        # Log metrics
        logger.info(f"\nAccuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC-ROC: {auc:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        cm_plot_path = os.path.join(args.output_dir, f"{args.model_name}_confusion_matrix.png")
        plot_confusion_matrix(cm, class_names, cm_plot_path)
        logger.info(f"Confusion matrix saved to {cm_plot_path}")
        
        # Plot ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_plot_path = os.path.join(args.output_dir, f"{args.model_name}_roc_curve.png")
            plot_roc_curve(fpr, tpr, auc, roc_plot_path)
            logger.info(f"ROC curve saved to {roc_plot_path}")
        except Exception as e:
            logger.warning(f"Error plotting ROC curve: {str(e)}")
            fpr, tpr = np.array([]), np.array([])
        
        # Plot precision-recall curve
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_plot_path = os.path.join(args.output_dir, f"{args.model_name}_precision_recall_curve.png")
            plot_precision_recall_curve(precision_curve, recall_curve, avg_precision, pr_plot_path)
            logger.info(f"Precision-recall curve saved to {pr_plot_path}")
        except Exception as e:
            logger.warning(f"Error plotting precision-recall curve: {str(e)}")
            precision_curve, recall_curve = np.array([]), np.array([])
        
        # --- Feature Importance (if available) ---
        try:
            logger.info("Extracting feature importance...")
            if hasattr(xgb_model, 'feature_importances_'):
                importance = xgb_model.feature_importances_
                feature_plot_path = os.path.join(args.output_dir, f"{args.model_name}_feature_importance.png")
                plot_feature_importance(
                    feature_names,
                    importance, 
                    args.top_n_features, 
                    feature_plot_path
                )
                logger.info(f"Feature importance plot saved to {feature_plot_path}")
            elif hasattr(xgb_model, 'get_score'):
                # Get feature importance scores
                try:
                    score_dict = xgb_model.get_score(importance_type='gain')
                    
                    if score_dict and feature_names:
                        # Parse feature indices from score_dict keys and map to actual feature names
                        importance = []
                        selected_feature_names = []
                        
                        for feat_key, score in score_dict.items():
                            # XGBoost feature keys are in format "f123" where 123 is the index
                            if feat_key.startswith('f'):
                                try:
                                    feat_idx = int(feat_key[1:])
                                    if feat_idx < len(feature_names):
                                        importance.append(score)
                                        selected_feature_names.append(feature_names[feat_idx])
                                except:
                                    logger.warning(f"Could not parse feature index from {feat_key}")
                        
                        if importance and selected_feature_names:
                            feature_plot_path = os.path.join(args.output_dir, f"{args.model_name}_feature_importance.png")
                            plot_feature_importance(
                                selected_feature_names, 
                                importance, 
                                args.top_n_features, 
                                feature_plot_path
                            )
                            logger.info(f"Feature importance plot saved to {feature_plot_path}")
                    else:
                        logger.warning("No feature importance scores found or no feature names available")
                except Exception as e:
                    logger.warning(f"Error getting feature scores: {str(e)}")
            else:
                logger.warning("Model does not support feature importance extraction")
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {str(e)}")
        
        # Save evaluation results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'avg_precision': float(avg_precision),
            'confusion_matrix': cm.tolist(),  # Convert to list for JSON serialization
            'classification_report': cls_report,
            'fpr': fpr.tolist() if len(fpr) > 0 else [],  # Convert to list for JSON serialization
            'tpr': tpr.tolist() if len(tpr) > 0 else [],  # Convert to list for JSON serialization
            'precision_curve': precision_curve.tolist() if len(precision_curve) > 0 else [],  # Convert to list for JSON serialization
            'recall_curve': recall_curve.tolist() if len(recall_curve) > 0 else []  # Convert to list for JSON serialization
        }
        
        # Save as JSON for better readability
        results_json_path = os.path.join(args.output_dir, f"{args.model_name}_results.json")
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results (JSON) saved to {results_json_path}")
        
        # Also save as joblib for backward compatibility
        results_joblib_path = os.path.join(args.output_dir, f"{args.model_name}_results.joblib")
        joblib.dump(results, results_joblib_path)
        logger.info(f"Evaluation results (joblib) saved to {results_joblib_path}")
        
        # Save predictions (optional)
        predictions_df = pd.DataFrame({
            args.id_column: df[args.id_column] if args.id_column in df.columns else np.arange(len(y_test)),
            'true_label': y_test,
            'pred_label': y_pred,
            'pred_probability': y_pred_proba
        })
        predictions_path = os.path.join(args.output_dir, f"{args.model_name}_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()