#!/usr/bin/env python3
"""
TabPFN Prediction Script: Uses a trained TabPFN model to generate predictions on new data.
Processes input data, applies the model, and outputs predictions with feature importance.
All artifacts are saved to a single, organized output folder.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from typing import Optional
from models.tabpfn import get_feature_importance
from utils import load_tabfnartifacts


def predict_tabpfn(
    model_dir: str,
    model_name: str,
    input_source: Optional[str] = None,
    text_column: Optional[str] = None,
    id_column: Optional[str] = None,
    output_dir: Optional[str] = None,
    text_input: Optional[str] = None,
    include_features: bool = False,
    calculate_importance: bool = False,
    n_top_features: int = 20,
    run_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Make predictions using a trained TabPFN model.

    Args:
        model_dir: Directory containing the trained model and preprocessing pipeline
        input_source: Path to CSV file with data to predict on
        text_column: Name of column containing text (required if input_source is provided)
        id_column: Name of column containing IDs (optional)
        output_dir: Directory to save all prediction artifacts
        text_input: Single text input to predict on (alternative to input_source)
        include_features: Whether to include features in output
        calculate_importance: Whether to calculate feature importance
        n_top_features: Number of top features to display in importance plot
        run_name: Name for this prediction run (default: timestamp)

    Returns:
        DataFrame with predictions
    """
    # Set up output directory and run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not run_name:
        run_name = f"prediction_{timestamp}"

    if output_dir:
        # Create organized output structure
        full_output_dir = os.path.join(output_dir, run_name)
        os.makedirs(full_output_dir, exist_ok=True)
        print(f"Saving all prediction artifacts to: {full_output_dir}")
    else:
        full_output_dir = None

    print(f"Loading TabPFN model from {model_dir}...")

    # Load model artifacts
    model, label_encoder, pipeline = load_tabfnartifacts(model_dir, model_name)

    if pipeline is None:
        raise ValueError(f"Preprocessing pipeline not found in {model_dir}")

    # Log run information
    run_info = {
        "timestamp": timestamp,
        "model_dir": model_dir,
        "input_source": input_source if input_source else "text_input",
        "text_column": text_column,
        "id_column": id_column,
        "include_features": include_features,
        "calculate_importance": calculate_importance
    }

    # Process the input
    if text_input is not None:
        print("Processing single text input...")
        matrix = pipeline.transform([text_input])
        feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out(
        )
        X = pd.DataFrame(matrix.toarray(), columns=feature_names)
        text_ids = ['input_text']
        run_info["text_input"] = text_input
    elif input_source is not None:
        if text_column is None:
            raise ValueError(
                "text_column must be provided when using an input_source file")

        print(f"Processing input file: {input_source}")
        input_df = pd.read_csv(input_source)

        if text_column not in input_df.columns:
            raise ValueError(
                f"Text column '{text_column}' not found in input file. Available columns: {list(input_df.columns)}")

        # Get the IDs if available
        if id_column is not None and id_column in input_df.columns:
            text_ids = input_df[id_column].tolist()
        else:
            text_ids = [f"text_{i}" for i in range(len(input_df))]

        # Process each text through the pipeline
        texts = input_df[text_column].fillna("").tolist()
        matrix = pipeline.transform(texts)
        feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out(
        )
        X = pd.DataFrame(matrix.toarray(), columns=feature_names)

        # Save input data statistics
        run_info["input_samples"] = len(texts)
        run_info["input_features"] = X.shape[1]
    else:
        raise ValueError("Either input_source or text_input must be provided")

    # Generate predictions
    print("Generating predictions...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Handle label encoder if available
    if label_encoder is not None:
        print("Decoding class labels...")
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        # Save label encoder information
        run_info["labels"] = label_encoder.classes_.tolist()
    else:
        y_pred_decoded = y_pred

    # Extract probability for positive class (if binary classification)
    if y_pred_proba.shape[1] > 1:
        pos_proba = y_pred_proba[:, 1]
    else:
        pos_proba = y_pred_proba.ravel()

    # Create predictions DataFrame
    results = {
        'id': text_ids,
        'prediction': y_pred_decoded,
        'raw_prediction': y_pred
    }

    # Add probabilities
    for i in range(y_pred_proba.shape[1]):
        class_name = str(i)
        if label_encoder is not None:
            try:
                class_name = label_encoder.inverse_transform([i])[0]
            except:
                pass
        results[f'prob_{class_name}'] = y_pred_proba[:, i]

    # Add features if requested
    if include_features:
        for col in X.columns:
            safe_col_name = f"feature_{col.replace(' ', '_')}"
            results[safe_col_name] = X[col].tolist()

    predictions_df = pd.DataFrame(results)

    # Save results to output directory if specified
    if full_output_dir:
        # Save predictions
        predictions_path = os.path.join(full_output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")

        # Save processed features
        if include_features:
            features_path = os.path.join(full_output_dir, "features.csv")
            X.to_csv(features_path, index=False)
            print(f"Features saved to: {features_path}")

        # Save run metadata
        run_info_path = os.path.join(full_output_dir, "run_info.json")
        with open(run_info_path, 'w') as f:
            json.dump(run_info, f, indent=4)
        print(f"Run information saved to: {run_info_path}")

        # Calculate and save prediction statistics
        stats = {}

        # Get prediction distribution
        value_counts = pd.Series(y_pred_decoded).value_counts().to_dict()
        stats["prediction_counts"] = value_counts

        # Calculate overall probability stats
        stats["probability_mean"] = float(np.mean(pos_proba))
        stats["probability_std"] = float(np.std(pos_proba))
        stats["probability_min"] = float(np.min(pos_proba))
        stats["probability_max"] = float(np.max(pos_proba))

        # Save statistics
        stats_path = os.path.join(full_output_dir, "prediction_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Prediction statistics saved to: {stats_path}")

        # Create and save probability distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(pos_proba, kde=True, bins=20)
        plt.title("Prediction Probability Distribution")
        plt.xlabel("Probability")
        plt.ylabel("Count")
        prob_plot_path = os.path.join(
            full_output_dir, "probability_distribution.png")
        plt.savefig(prob_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Probability distribution plot saved to: {prob_plot_path}")

        # Calculate feature importance if requested
        # Need more than one sample for meaningful importance
        if calculate_importance and len(X) > 1:
            importance_df, importance_fig = get_feature_importance(
                model=model,
                X=X,
                y=y_pred,  # Using predictions as we don't have true labels
                feature_names=X.columns.tolist(),
                n_top_features=n_top_features
            )

            if not importance_df.empty:
                # Save importance data
                importance_path = os.path.join(
                    full_output_dir, "feature_importance.csv")
                importance_df.to_csv(importance_path, index=False)
                print(f"Feature importance data saved to: {importance_path}")

                # Save importance plot
                if importance_fig:
                    importance_plot_path = os.path.join(
                        full_output_dir, "feature_importance.png")
                    importance_fig.savefig(
                        importance_plot_path, bbox_inches='tight', dpi=300)
                    plt.close(importance_fig)
                    print(
                        f"Feature importance plot saved to: {importance_plot_path}")

                # Print top features to console
                print(
                    f"\nTop {min(n_top_features, len(importance_df))} important features:")
                for i, (feature, importance) in enumerate(
                    zip(importance_df['Feature'].head(n_top_features),
                        importance_df['Importance'].head(n_top_features))
                ):
                    print(f"{i+1}. {feature}: {importance:.4f}")
    else:
        # If no output directory, print predictions to console
        print("\nPredictions:")
        for i, (text_id, pred, prob) in enumerate(zip(text_ids, y_pred_decoded, pos_proba)):
            print(f"Text {text_id}: Prediction={pred}, Probability={prob:.4f}")

        # Print feature importance if calculated
        if calculate_importance and len(X) > 1:
            importance_df, _ = get_feature_importance(
                model=model,
                X=X,
                y=y_pred,
                feature_names=X.columns.tolist(),
                n_top_features=n_top_features
            )

            if not importance_df.empty:
                print(
                    f"\nTop {min(n_top_features, len(importance_df))} important features:")
                for i, (feature, importance) in enumerate(
                    zip(importance_df['Feature'].head(n_top_features),
                        importance_df['Importance'].head(n_top_features))
                ):
                    print(f"{i+1}. {feature}: {importance:.4f}")

    return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions using a trained TabPFN model')

    # Required parameter
    parser.add_argument('--model_path', type=str, default="TABPFN",
                        help='Path to the directory containing model artifacts')

    # Input options (one of these is required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--data_file', type=str, help='Path to the CSV file with data to predict on')
    input_group.add_argument(
        '--text', type=str, help='Single text input to predict on')
    parser.add_argument('--model_name', type=str, default="tabpfn",
                        help='Name of the type of Model being trained')

    # Optional parameters for CSV input
    parser.add_argument('--text_column', type=str, default='txt',
                        help='Name of the column containing text data')
    parser.add_argument('--id_column', type=str, default='DEID',
                        help='Name of the column containing IDs')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='TABPFN/prediction',
                        help='Directory to save all prediction artifacts')
    parser.add_argument('--run_name', type=str,
                        help='Name for this prediction run (default: timestamp-based)')
    parser.add_argument('--include_features',
                        action='store_true', help='Include features in output')

    # Feature importance parameters
    parser.add_argument('--calculate_importance',
                        action='store_true', help='Calculate feature importance')
    parser.add_argument('--top_features', type=int, default=20,
                        help='Number of top features to display in importance analysis')

    args = parser.parse_args()

    # Validate text_column is provided when using data_file
    if args.data_file and not args.text_column:
        parser.error("--text_column is required when using --data_file")

    # Call prediction function
    predict_tabpfn(
        model_dir=args.model_path,
        model_name=args.model_name,
        input_source=args.data_file,
        text_column=args.text_column,
        id_column=args.id_column,
        output_dir=args.output_dir,
        text_input=args.text,
        include_features=args.include_features,
        calculate_importance=args.calculate_importance,
        n_top_features=args.top_features,
        run_name=args.run_name
    )

    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()
