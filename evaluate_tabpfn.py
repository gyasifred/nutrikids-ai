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
from models.tabpfn import load_artifacts, evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained TabPFN model')
    
    # Required parameter
    parser.add_argument('--model', type=str, required=True, help='Path to the directory containing model artifacts')
    
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
    model, pipeline, label_encoder = load_artifacts(args.model)
    
    # Process the test data
    print(f"Processing test data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    X_test = pipeline.transform(df[args.text_column])
    y_test = df[args.label_column]
    
    # Encode labels if we have a label encoder
    if label_encoder is not None:
        y_test = label_encoder.transform(y_test)
    
    # Convert X_test to DataFrame if it's a sparse matrix
    if hasattr(X_test, 'toarray'):
        X_test = pd.DataFrame(X_test.toarray())
    
    # Evaluate the model
    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=None,  # Adjust based on your pipeline
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
