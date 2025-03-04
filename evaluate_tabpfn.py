#!/usr/bin/env python3
"""
TabPFN Evaluation Script: Evaluates a trained TabPFN model on a test dataset.
Generates and saves comprehensive evaluation metrics and visualizations.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional
from utils import process_csv
from models.tabpfn import evaluate_model 
from utils import load_artifacts

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained TabPFN model')
    
    # Required parameter
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing model artifacts')
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True, help='Path to the CSV test data file')
    parser.add_argument('--text_column', type=str, default="Note_Column", help='Name of the column containing text data')
    parser.add_argument('--label_column', type=str, default="Malnutrition_Label", help='Name of the column containing labels')
    parser.add_argument('--id_column', type=str, default="Patient_ID", help='Name of the column containing IDs')
    
    # Optional parameters
    parser.add_argument('--output_dir', type=str, default='model_output/tabpfn', help='Directory to save evaluation results')
    parser.add_argument('--model_name', type=str, default='tabpfn', help='Name to use for saved artifacts')
    
    args = parser.parse_args()
    
    # Load artifacts
    model, label_encoder, pipeline = load_artifacts(args.model_path, args.model_name)
    
    # Process the test data
    print(f"Processing test data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    X_test = pipeline.transform(df[args.text_column])
    y_test = df[args.label_column]
    
    # Handle label transformation correctly
    if label_encoder is not None:
        # First check if the labels match the encoder's expected format
        unique_labels = y_test.unique()
        expected_classes = label_encoder.classes_
        
        print(f"Test data labels: {unique_labels}")
        print(f"Label encoder classes: {expected_classes}")
        
        # If test data has string values like 'yes'/'no' but encoder expects 0/1,
        # we need to map them first
        if set(unique_labels) != set(expected_classes):
            # Check if we need to convert from yes/no to numeric
            if ('yes' in unique_labels or 'no' in unique_labels) and (0 in expected_classes or 1 in expected_classes):
                print("Converting string labels to numeric format...")
                y_test = y_test.map({'yes': 1, 'no': 0})
            # Check if we need to handle other label formats
            elif any(isinstance(label, str) for label in unique_labels) and all(isinstance(cls, (int, float)) for cls in expected_classes):
                print("WARNING: String labels in test data don't match encoder's numeric classes.")
                print("Attempting to fit a new label encoder for evaluation purposes...")
                # Save the original classes mapping for reference
                original_classes = dict(enumerate(expected_classes))
                
                # Create a temporary mapping for evaluation
                temp_mapping = {label: i for i, label in enumerate(sorted(unique_labels))}
                y_test = y_test.map(temp_mapping)
                
                # Save this mapping to help interpret results later
                mapping_filename = os.path.join(args.output_dir, f"temp_label_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(mapping_filename, 'w') as f:
                    import json
                    json.dump({
                        "original_encoder_classes": {str(k): str(v) for k, v in original_classes.items()},
                        "test_data_mapping": {str(k): str(v) for k, v in temp_mapping.items()}
                    }, f, indent=2)
                print(f"Label mapping saved to: {mapping_filename}")
            else:
                # Try to use the encoder directly if format seems compatible
                try:
                    y_test = label_encoder.transform(y_test)
                except ValueError as e:
                    print(f"WARNING: Label transformation error: {e}")
                    print("Creating a new label encoder for evaluation...")
                    from sklearn.preprocessing import LabelEncoder
                    new_encoder = LabelEncoder()
                    y_test = new_encoder.fit_transform(y_test)
                    
                    # Save mapping between original and new encoders
                    mapping_filename = os.path.join(args.output_dir, f"label_encoder_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    with open(mapping_filename, 'w') as f:
                        import json
                        json.dump({
                            "original_encoder_classes": [str(cls) for cls in expected_classes],
                            "new_encoder_classes": [str(cls) for cls in new_encoder.classes_]
                        }, f, indent=2)
                    print(f"Label encoder mapping saved to: {mapping_filename}")
        else:
            # If the label formats match, apply the transformation directly
            y_test = label_encoder.transform(y_test)
    
    # Convert X_test to DataFrame if it's a sparse matrix
    if hasattr(X_test, 'toarray'):
        feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out() if 'vectorizer' in pipeline.named_steps else [f"feature_{i}" for i in range(X_test.shape[1])]
        X_test = pd.DataFrame(X_test.toarray(), columns=feature_names)
    else:
        X_test = pd.DataFrame(X_test)
    
    # Evaluate the model
    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=X_test.columns.tolist(),
        output_dir=args.output_dir,
        model_name=args.model_name,
        label_encoder=label_encoder
    )
    
    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_results_filename = os.path.join(args.output_dir, f"{args.model_name}_evaluation_results_{timestamp}.joblib")
    joblib.dump(results, evaluation_results_filename)
    print(f"Evaluation results saved to: {evaluation_results_filename}")
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()