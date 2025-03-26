#!/usr/bin/env python3
"""
TabPFN Evaluation Script: Evaluates a trained TabPFN model on a test dataset.
Generates and saves comprehensive evaluation metrics and visualizations.
"""

import os
import argparse
import pandas as pd
import json
import numpy as np
from datetime import datetime
from models.tabpfn import evaluate_model
from utils import load_tabfnartifacts
from utils import process_labels, detect_label_type


def convert_to_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        return obj.item()
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained TabPFN model')

    # Required parameter
    parser.add_argument('--model_path', type=str, default="TABPFN",
                        help='Path to the directory containing model artifacts')
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the CSV test data file')
    parser.add_argument('--text_column', type=str, default="txt",
                        help='Name of the column containing text data')
    parser.add_argument('--label_column', type=str, default="label",
                        help='Name of the column containing labels')
    parser.add_argument('--id_column', type=str, default="DEID",
                        help='Name of the column containing IDs')

    # Optional parameters
    parser.add_argument('--output_dir', type=str, default='TABPFN/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--model_name', type=str, default='tabpfn',
                        help='Name to use for saved artifacts')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load artifacts
    model, label_encoder, pipeline = load_tabfnartifacts(
        args.model_path, args.model_name)

    # Process the test data
    print(f"Processing test data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    X_test = pipeline.transform(df[args.text_column])

    # Detect and process labels 
    test_labels = df[args.label_column].tolist()
    label_type = detect_label_type(test_labels)
    print(f"Detected label type: {label_type}")

    # Process labels
    if label_encoder is not None:
        # If label encoder exists (for text categories), use it
        y_test = label_encoder.transform(test_labels)
        print(f"Transforming labels using saved label encoder. Classes: {label_encoder.classes_}")
    else:
        # If no label encoder, direct processing (for numeric or already numeric labels)
        y_test, _ = process_labels(test_labels)
        print("No label encoder used. Labels processed directly.")

    # Ensure X_test is a DataFrame
    if hasattr(X_test, 'toarray'):
        feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out() if 'vectorizer' in pipeline.named_steps else [f"feature_{i}" for i in range(X_test.shape[1])]
        X_test = pd.DataFrame(X_test.toarray(), columns=feature_names)
    else:
        X_test = pd.DataFrame(X_test)

    # Evaluate model
    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=X_test.columns.tolist(),
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_encoder=label_encoder
    )

    # Add label type to results for reference
    results['label_type'] = label_type

    # Convert results to JSON-safe format
    results_serializable = {key: convert_to_serializable(value) for key, value in results.items()}

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_results_filename = os.path.join(
        args.output_dir, f"{args.model_name}_evaluation_results_{timestamp}.json")

    with open(evaluation_results_filename, "w") as f:
        json.dump(results_serializable, f, indent=4)

    print(f"Evaluation results saved to: {evaluation_results_filename}")
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()