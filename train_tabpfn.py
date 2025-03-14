#!/usr/bin/env python3 
"""
TabPFN Training Script: Processes text data and trains a TabPFN classifier.
Saves the model, text vectorization pipeline, and label encoder to disk.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier
from models.tabpfn import train_tabpfn
from utils import process_csv


def main():
    parser = argparse.ArgumentParser(description='Train a TabPFN classifier on text data')
    
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--text_column', type=str, default="Note_Column", help='Column containing text data')
    parser.add_argument('--label_column', type=str, default="Malnutrition_Label", help='Column containing labels')
    parser.add_argument('--id_column', type=str, default="Patient_ID", help='Column containing IDs')
    
    # Text processing parameters
    parser.add_argument('--max_features', type=int, default=8000, help='Max number of features to extract')
    parser.add_argument('--remove_stop_words', action='store_true',default = False, help='Remove stop words')
    parser.add_argument('--apply_stemming', action='store_true',default = False, help='Apply stemming')
    parser.add_argument('--vectorization_mode', type=str, default='tfidf', choices=['count', 'tfidf'], help='Vectorization mode')
    parser.add_argument('--ngram_min', type=int, default=1, help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=1, help='Maximum n-gram size')
    
    # Model parameters
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--model_name', type=str, default="tabpfn", help='Name of the type of Model being trained')
        
    # Output parameters
    parser.add_argument('--model_dir', type=str, default='tabpfn_model', help='Directory to save all models and artifacts')
    
    args = parser.parse_args()
    
    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Process CSV data
    print(f"Processing CSV data from {args.data_file}...")
    ngram_range = (args.ngram_min, args.ngram_max)
    
    X_df, complete_df, y, pipeline, feature_dict, label_encoder = process_csv(
        file_path=args.data_file,
        text_column=args.text_column,
        label_column=args.label_column,
        model_name=args.model_name,
        id_column=args.id_column,
        max_features=args.max_features,
        remove_stop_words=args.remove_stop_words,
        apply_stemming=args.apply_stemming,
        vectorization_mode=args.vectorization_mode,
        ngram_range=ngram_range,
        save_path=args.model_dir
    )
    
    # Save feature dictionary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_dict_path = os.path.join(args.model_dir, f"{args.model_name}_nutrikidai_classifier_feature_dict_{timestamp}.joblib")
    joblib.dump(feature_dict, feature_dict_path)
    print(f"Feature dictionary saved to: {feature_dict_path}")
    
    label_encoder = None
    if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or len(np.unique(y)) < len(y) * 0.5:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        label_encoder_path = os.path.join(args.model_dir, f"{args.model_name}_nutrikidai_classifier_label_encoder_{timestamp}.joblib")
        joblib.dump(label_encoder, label_encoder_path)
        
        # Train model
    results = train_tabpfn(
        X_train=X_df, 
        y_train=y,
        model_dir=args.model_dir,
        device=args.device,
        model_name=f"{args.model_name}_nutrikidai_model"
    )
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {results['model_file']}")

if __name__ == "__main__":
    main()
