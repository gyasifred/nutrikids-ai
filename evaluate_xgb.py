#!/usr/bin/env python3
"""
XGBoost Evaluation Script: Evaluates a trained XGBoost model on a test dataset.
Generates and saves comprehensive evaluation metrics and visualizations.
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
    average_precision_score
)
from utils import (
    load_xgbartifacts, ensure_xgbfeatures_match,
    plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance
)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained XGBoost model')

    # Required parameters
    parser.add_argument('--model_name', type=str,
                        default="xgboost", help='Name of the model')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the CSV test data file')
    parser.add_argument('--text_column', type=str, default="txt",
                        help='Name of the column containing text data')
    parser.add_argument('--label_column', type=str, default="label",
                        help='Name of the column containing labels')
    parser.add_argument('--id_column', type=str, default="DEID",
                        help='Name of the column containing IDs')

    # Optional parameters
    parser.add_argument('--model_dir', type=str, default='./xgboost',
                        help='Directory containing model artifacts')
    parser.add_argument('--output_dir', type=str, default='./xgboost_ouput',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_shap_samples', type=int, default=100,
                        help='Number of samples for SHAP explanation')
    parser.add_argument('--top_n_features', type=int,
                        default=20, help='Number of top features to plot')
    parser.add_argument('--debug', action='store_true',
                        help='Enable extra debug logging')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(
                args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting model evaluation with arguments: {args}")

    try:
        # Load artifacts: model, label encoder, pipeline, and feature names
        xgb_model, label_encoder, pipeline, feature_names = load_xgbartifacts(
            args.model_dir, args.model_name)

        # Process the test data
        logger.info(f"Loading test data from {args.data_file}...")
        df = pd.read_csv(args.data_file)

        # Verify required columns exist
        required_columns = [args.text_column, args.label_column]
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in test data: {missing_columns}.\
                      Available columns: {list(df.columns)}")

        logger.info("Applying pipeline to transform text data...")
        X_test = pipeline.transform(df[args.text_column])

        # Ensure features match the training data
        X_test_aligned = ensure_xgbfeatures_match(X_test, feature_names)

        # Log shape information for debugging
        logger.info(
            f"Original test data shape:\
            {X_test.shape if hasattr(X_test, 'shape') else 'unknown'}")
        logger.info(f"Aligned test data shape: {X_test_aligned.shape}")
        logger.info(
            f"Feature names count:\
                {len(feature_names) if feature_names else 'unknown'}")

        # Get labels
        y_test = df[args.label_column].values
        logger.info(f"Test data labels: {np.unique(y_test)}")
        logger.info(f"Label encoder classes: {label_encoder.classes_}")

        # Map labels if necessary (for example,
        #  converting 'yes'/'no' to numeric values)
        if hasattr(y_test, 'dtype') and y_test.dtype == object:
            y_test = label_encoder.transform(y_test)

        # Generate predictions
        logger.info("Generating predictions...")
        y_pred_proba = xgb_model.predict_proba(
            X_test_aligned)[:, 1]
        y_pred = xgb_model.predict(X_test_aligned)

        # --- Evaluation Metrics ---
        logger.info("Calculating evaluation metrics...")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(
            y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

        # Calculate ROC AUC and average precision
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
        except Exception as e:
            logger.warning(
                f"Error calculating AUC or average precision: {str(e)}")
            auc = 0.0
            avg_precision = 0.0

        # Print classification report
        logger.info("\nClassification Report:")
        class_names = label_encoder.classes_
        cls_report = classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0)
        logger.info("\n" + cls_report)

        # Log metrics
        logger.info(f"\nAccuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC-ROC: {auc:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f}")

        # Plot confusion matrix
        cm_plot_path = os.path.join(
            args.output_dir, f"{args.model_name}_confusion_matrix.png")
        cm = plot_confusion_matrix(y_test, y_pred, cm_plot_path)
        logger.info(f"Confusion matrix saved to {cm_plot_path}")

        # Plot ROC curve
        try:
            roc_plot_path = os.path.join(
                args.output_dir, f"{args.model_name}_roc_curve.png")
            fpr, tpr, _ = plot_roc_curve(y_test, y_pred_proba, roc_plot_path)
            logger.info(f"ROC curve saved to {roc_plot_path}")
        except Exception as e:
            logger.warning(f"Error plotting ROC curve: {str(e)}")

        # Plot precision-recall curve
        try:
            pr_plot_path = os.path.join(
                args.output_dir,
                f"{args.model_name}_precision_recall_curve.png")
            precision_curve, recall_curve, _ = plot_precision_recall_curve(
                y_test, y_pred_proba, pr_plot_path)
            logger.info(f"Precision-recall curve saved to {pr_plot_path}")
        except Exception as e:
            logger.warning(f"Error plotting precision-recall curve: {str(e)}")
        # --- Feature Importance (if available) ---
        try:
            logger.info("Extracting feature importance...")
            if hasattr(xgb_model, 'feature_importances_'):
                importance = xgb_model.feature_importances_
                feature_plot_path = os.path.join(
                    args.output_dir,
                    f"{args.model_name}_feature_importance.png")
                plot_feature_importance(
                    feature_names,
                    importance,
                    args.top_n_features,
                    feature_plot_path
                )
                logger.info(
                    f"Feature importance plot saved to {feature_plot_path}")
            elif hasattr(xgb_model, 'get_score'):
                # Get feature importance scores
                try:
                    score_dict = xgb_model.get_score(importance_type='gain')

                    if score_dict and feature_names:
                        # Parse feature indices from score_dict keys
                        #  and map to actual feature names
                        importance = []
                        selected_feature_names = []

                        for feat_key, score in score_dict.items():
                            # XGBoost feature keys are in format "f123"
                            #  where 123 is the index
                            if feat_key.startswith('f'):
                                try:
                                    feat_idx = int(feat_key[1:])
                                    if feat_idx < len(feature_names):
                                        importance.append(score)
                                        selected_feature_names.append(
                                            feature_names[feat_idx])
                                except KeyError:
                                    logger.warning(
                                        f"Could not parse feature \
                                        index from {feat_key}")

                        if importance and selected_feature_names:
                            feature_plot_path = os.path.join(
                                args.output_dir,
                                f"{args.model_name}_feature_importance.png")
                            plot_feature_importance(
                                selected_feature_names,
                                importance,
                                args.top_n_features,
                                feature_plot_path
                            )
                            logger.info(
                                f"Feature importance plot\
                                    saved to {feature_plot_path}")
                    else:
                        logger.warning(
                            "No feature importance scores found \
                                  no feature names available")
                except Exception as e:
                    logger.warning(f"Error getting feature scores: {str(e)}")
            else:
                logger.warning(
                    "Model does not support feature importance extraction")
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
            'confusion_matrix': cm.tolist(),
            'classification_report': cls_report,
            'fpr': fpr.tolist() if len(fpr) > 0 else [],
            'tpr': tpr.tolist() if len(tpr) > 0 else [],
            'precision_curve': precision_curve.tolist()
            if len(precision_curve) > 0 else [],
            'recall_curve': recall_curve.tolist()
            if len(recall_curve) > 0 else []
        }

        # Save as JSON for better readability
        results_json_path = os.path.join(
            args.output_dir,
            f"{args.model_name}_results.json")
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results (JSON) saved to {results_json_path}")

        # Also save as joblib for backward compatibility
        results_joblib_path = os.path.join(
            args.output_dir, f"{args.model_name}_results.joblib")
        joblib.dump(results, results_joblib_path)
        logger.info(
            f"Evaluation results (joblib) saved to {results_joblib_path}")

        # Save predictions (optional)
        predictions_df = pd.DataFrame({
            args.id_column: df[args.id_column]
            if args.id_column in df.columns else np.arange(len(y_test)),
            'true_label': y_test,
            'pred_label': y_pred,
            'pred_probability': y_pred_proba
        })
        predictions_path = os.path.join(
            args.output_dir, f"{args.model_name}_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
