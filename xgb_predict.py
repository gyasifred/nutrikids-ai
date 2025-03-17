#!/usr/bin/env python3
"""
XGBoost Prediction Script: Makes predictions using a trained XGBoost model.
Can process both raw text input and CSV files,
providing explainability for predictions.
"""

import os
import argparse
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import shap
from utils import load_xgbartifact, ensure_xgbfeatures_match


def get_prediction_explanation(model, X, feature_names, top_n=5):
    """
    Generate explanation for a prediction using feature importance.
    Args:
        model: Trained XGBoost model
        X: Feature matrix for a single sample
        feature_names: List of feature names
        top_n: Number of top features to include in explanation
    Returns:
        explanation: Dictionary with top features and their contribution
    """
    try:
        # Try using SHAP for explanation
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        # For a single prediction
        if X.shape[0] == 1:
            # Get the SHAP values for the first (and only) sample
            values = shap_values.values[0]
            # Get the feature indices sorted by absolute SHAP value
            indices = np.argsort(np.abs(values))[-top_n:][::-1]
            # Create explanation dictionary
            explanation = {
                "top_features": [(feature_names[i],
                                  float(values[i])) for i in indices],
                "base_value": float(shap_values.base_values[0]),
                "method": "shap"
            }  
            return explanation
    except Exception as e:
        logging.warning(f"SHAP explanation failed: {str(e)}.\
                         Falling back to feature importance.")
    # Fallback to feature importance if SHAP fails
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            # Get the top N features
            indices = np.argsort(importance)[-top_n:][::-1]
            # Get the feature values for this sample
            if isinstance(X, (np.ndarray, np.generic)):
                feature_values = X[0]
            else:
                feature_values = X.iloc[0].values
            # Create explanation dictionary
            explanation = {
                "top_features": [
                    (feature_names[i], float(feature_values[i]))
                    for i in indices if i < len(feature_names)
                    and i < len(feature_values)
                    ],
                "method": "feature_importance"
            }
            return explanation
    except Exception as e:
        logging.warning(f"Feature importance explanation failed: {str(e)}")
    # If all methods fail, return a simple message
    return {"error": "Could not generate explanation for this prediction."}


def format_explanation(explanation, class_names):
    """
    Format the explanation dictionary into a human-readable string.
    Args:
        explanation: Dictionary with explanation details
        class_names: List of class names
    Returns:
        formatted_explanation: String with formatted explanation
    """
    if "error" in explanation:
        return explanation["error"]
    lines = []
    if explanation["method"] == "shap":
        lines.append("PREDICTION EXPLANATION (SHAP VALUES):")
        lines.append(f"Base value: {explanation['base_value']:.4f}")
        lines.append("Top contributing features:")
        for i, (feature, value) in enumerate(explanation["top_features"]):
            direction = "+" if value > 0 else "-"
            lines.append(f"{i+1}. {feature}: {direction}{abs(value):.4f}\
                          ({'increases' if value > 0 else 'decreases'}\
                            prediction)")
    elif explanation["method"] == "feature_importance":
        lines.append("PREDICTION EXPLANATION (FEATURE IMPORTANCE):")
        lines.append("Top important features:")
        for i, (feature, value) in enumerate(explanation["top_features"]):
            lines.append(f"{i+1}. {feature}: {value:.4f}")
    return "\n".join(lines)


def plot_explanation(explanation, output_path):
    """
    Plot the explanation as a horizontal bar chart and save to file.
    Args:
        explanation: Dictionary with explanation details
        output_path: Path to save the plot
    """
    if "error" in explanation or not explanation.get("top_features"):
        return
    # Extract feature names and values
    features = [f[0] for f in explanation["top_features"]]
    values = [f[1] for f in explanation["top_features"]]
    # Create the plot
    plt.figure(figsize=(10, 6))
    colors = ['#ff9999' if v < 0 else '#66b3ff' for v in values]
    # Plot horizontal bars
    y_pos = range(len(features))
    plt.barh(y_pos, values, color=colors)
    plt.yticks(y_pos, features)
    plt.xlabel('Contribution to prediction'
               if explanation["method"] == "shap" else 'Feature importance')
    plt.title('Feature Contributions to Prediction'
              if explanation["method"] == "shap" else 'Top Important Features')
    # Add a vertical line at x=0 for SHAP values
    if explanation["method"] == "shap":
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def predict_text(text,
                 xgb_model,
                 pipeline,
                 label_encoder,
                 feature_names,
                 explain=True,
                 top_n=20):
    """
    Make a prediction for a single text input.
    Args:
        text: Raw text input
        xgb_model: Trained XGBoost model
        pipeline: Preprocessing pipeline
        label_encoder: Label encoder
        feature_names: List of feature names
        explain: Whether to generate explanation
        top_n: Number of top features for explanation
    Returns:
        result: Dictionary with prediction results
    """
    # Convert text to a list (required by pipeline)
    text_list = [text]
    # Transform the text using the pipeline
    X_transformed = pipeline.transform(text_list)
    # Align features
    X_aligned = ensure_xgbfeatures_match(X_transformed, feature_names)
    # Get predictions
    y_pred_proba = xgb_model.predict_proba(X_aligned)[:, 1]
    y_pred = xgb_model.predict(X_aligned)
    # Get class names
    class_names = label_encoder.classes_
    predicted_class = class_names[y_pred[0]]
    result = {
        "text": text,
        "predicted_class": predicted_class,
        "probability": float(y_pred_proba[0]),
        "class_names": class_names.tolist()
    }
    # Generate explanation if requested
    if explain:
        explanation = get_prediction_explanation(xgb_model,
                                                 X_aligned,
                                                 feature_names,
                                                 top_n)
        result["explanation"] = explanation
        result["explanation_text"] = format_explanation(explanation,
                                                        class_names)
    return result


def process_csv_file(data_file,
                     text_column,
                     id_column,
                     xgb_model,
                     pipeline,
                     label_encoder,
                     feature_names,
                     output_dir,
                     explain=True,
                     top_n=10):
    """
    Process a CSV file and make predictions for each row.
    Args:
        data_file: Path to CSV file
        text_column: Name of column containing text data
        id_column: Name of column containing IDs
        xgb_model: Trained XGBoost model
        pipeline: Preprocessing pipeline
        label_encoder: Label encoder
        feature_names: List of feature names
        output_dir: Directory to save results
        explain: Whether to generate explanations
        top_n: Number of top features for explanation
    Returns:
        results_df: DataFrame with prediction results
    """
    # Load the CSV file
    df = pd.read_csv(data_file)
    # Verify required columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}'\
            not found in CSV file. Available columns: {list(df.columns)}")
    # Initialize results list
    results = []
    explanations = []
    # Process each row
    for idx, row in df.iterrows():
        text = row[text_column]
        id_value = row[id_column] if id_column in df.columns else idx
        # Skip empty texts
        if pd.isna(text) or text == "":
            logging.warning(f"Skipping empty text at row {idx}")
            continue
        # Make prediction
        pred_result = predict_text(
            text, xgb_model, pipeline,
            label_encoder, feature_names, explain, top_n
        )
        # Add ID to result
        result_row = {
            "id": id_value,
            "predicted_class": pred_result["predicted_class"],
            "probability": pred_result["probability"]
        }
        results.append(result_row)
        # Store explanation separately
        if explain and "explanation" in pred_result:
            explanation = {
                "id": id_value,
                "explanation": pred_result["explanation_text"]
            }
            explanations.append(explanation)
            # Generate explanation plot if directory exists
            if output_dir:
                os.makedirs(os.path.join(output_dir,
                                         "explanation_plots"), exist_ok=True)
                plot_path = os.path.join(output_dir, "explanation_plots",
                                         f"explanation_{id_value}.png")
                plot_explanation(pred_result["explanation"], plot_path)
        # Log progress for large files
        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx + 1} rows...")
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    # Save results to CSV file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "prediction_results.csv")
        results_df.to_csv(results_path, index=False)
        logging.info(f"Prediction results saved to {results_path}")
        # Save explanations to text file
        if explanations:
            explanations_path = os.path.join(output_dir,
                                             "prediction_explanations.txt")
            with open(explanations_path, 'w') as f:
                for expl in explanations:
                    f.write(f"--- ID: {expl['id']} ---\n")
                    f.write(expl["explanation"])
                    f.write("\n\n" + "-" * 80 + "\n\n")
            logging.info(
                f"Prediction explanations saved to {explanations_path}"
                )
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using a trained XGBoost model')
    # Required parameters
    parser.add_argument('--model_name', type=str, default="xgboost",
                        help='Name of the model')
    # Data input options (either file or text)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data_file', type=str,
                            help='Path to the CSV test data file')
    data_group.add_argument('--text', type=str,
                            help='Raw text input for prediction')
    # CSV-specific parameters
    parser.add_argument('--text_column', type=str, default="txt",
                        help='Name of the column containing text data')
    parser.add_argument('--id_column', type=str, default="DEID",
                        help='Name of the column containing IDs')
    # Optional parameters
    parser.add_argument('--model_dir', type=str, default='./xgboost',
                        help='Directory containing model artifacts')
    parser.add_argument('--output_dir', type=str,
                        default='./xgboost_predictions',
                        help='Directory to save prediction results')
    parser.add_argument('--explain', action='store_true', default=False,
                        help='Generate explanations for predictions')
    parser.add_argument('--top_n_features', type=int, default=5,
                        help='Number of top features to include in explanation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable extra debug logging')
    args = parser.parse_args()
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(args.output_dir,
                                                 'prediction.log')),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=log_level, format=log_format)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting prediction with arguments: {args}")
    try:
        # Load artifacts
        logger.info("Loading model artifacts...")
        xgb_model, label_encoder, pipeline, feature_names = load_xgbartifact(
            args.model_dir, args.model_name)
        # Process based on input type
        if args.text:
            # Single text prediction
            logger.info("Processing single text input...")
            result = predict_text(
                args.text, xgb_model, pipeline, label_encoder, feature_names,
                args.explain, args.top_n_features
            )   
            # Print results to console
            print("\n" + "=" * 80)
            print("PREDICTION RESULT:")
            print(f"Text: {args.text[:100]}..." if len(args.text) > 100
                  else f"Text: {args.text}")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Probability: {result['probability']:.4f}")
            print("=" * 80) 
            if args.explain and "explanation_text" in result:
                print("\n" + result["explanation_text"])
                print("=" * 80 + "\n")
            # Save to file if output directory specified
            if args.output_dir:
                # Save prediction result
                result_path = os.path.join(args.output_dir,
                                           "single_prediction_result.txt")
                with open(result_path, 'w') as f:
                    f.write("PREDICTION RESULT:\n")
                    f.write(f"Text: {args.text}\n")
                    f.write(f"Predicted class: {result['predicted_class']}\n")
                    f.write(f"Probability: {result['probability']:.4f}\n\n")
                    if args.explain and "explanation_text" in result:
                        f.write(result["explanation_text"])
                logger.info(f"Prediction result saved to {result_path}")
                # Save explanation plot
                if args.explain and "explanation" in result:
                    plot_path = os.path.join(
                        args.output_dir, "single_prediction_explanation.png")
                    plot_explanation(result["explanation"], plot_path)
                    logger.info(f"Explanation plot saved to {plot_path}")
        elif args.data_file:
            # Batch prediction from CSV file
            logger.info(f"Processing CSV file: {args.data_file}...")
            results = process_csv_file(
                args.data_file, args.text_column, args.id_column,
                xgb_model, pipeline, label_encoder, feature_names,
                args.output_dir, args.explain, args.top_n_features
            )
            # Print summary to console
            print("\n" + "=" * 80)
            print("BATCH PREDICTION SUMMARY:")
            print(f"Total predictions: {len(results)}")
            if len(results) > 0:
                class_counts = results['predicted_class'].value_counts()
                print("\nClass distribution:")
                for cls, count in class_counts.items():
                    print(f"  {cls}: {count} ({count/len(results)*100:.1f}%)")
            print("\nSample predictions:")
            print(results.head(5))
            print("=" * 80 + "\n")
            print(f"Full results saved to: \
                  {os.path.join(args.output_dir, 'prediction_results.csv')}")
            if args.explain:
                print(f"Explanations saved to:\
                      {os.path.join(args.output_dir,
                                    'prediction_explanations.txt')}")
                print(f"Explanation plots saved to:\
                      {os.path.join(args.output_dir, 'explanation_plots/')}")
        logger.info("Prediction completed successfully!")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()