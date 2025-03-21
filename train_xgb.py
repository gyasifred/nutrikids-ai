#!/usr/bin/env python3

import argparse
import os
import joblib
import logging
import xgboost as xgb
import pandas as pd
import numpy as np
import ray
from utils import process_csv
from models.xgboost import get_scaling_config_and_tree_method


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train an XGBoost model.')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to training data CSV file')
    parser.add_argument('--text_column', type=str, default="txt",
                        help='Name of the text column in the CSV')
    parser.add_argument('--label_column', type=str, default="label",
                        help='Name of the label column in the CSV')
    parser.add_argument('--id_column', type=str, default="DEID",
                        help='Name of the ID column in the CSV')
    parser.add_argument("--config_dir", default="xgboost", type=str,
                        help='Path to best hyperparameter directory')
    parser.add_argument('--max_features', type=int, default=10000,
                        help='Maximum number of features for vectorization')
    parser.add_argument('--remove_stop_words', action='store_true',
                        default=False,
                        help='Remove stop words during preprocessing')
    parser.add_argument('--apply_stemming', action='store_true',
                        default=False,
                        help='Apply stemming during preprocessing')
    parser.add_argument('--vectorization_mode', type=str, default='tfidf',
                        choices=['tfidf', 'count', 'binary'],
                        help='Vectorization mode')
    parser.add_argument('--ngram_min', type=int, default=1,
                        help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=1,
                        help='Maximum n-gram size')
    parser.add_argument('--model_dir', type=str, default='./xgboost',
                        help='Directory to save/load the model')
    parser.add_argument('--model_name', type=str, default="xgboost",
                        help='Name of the type of Model being trained')
    parser.add_argument('--chunk_size', type=int, default=5000,
                        help='Chunk size for processing large datasets')
    
    # XGBoost parameters
    parser.add_argument('--eta',
                        type=float, default=0.1, help='Learning rate')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Maximum depth of trees')
    parser.add_argument('--min_child_weight', type=float, default=1,
                        help='Minimum sum of instance weight needed in a child')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='Subsample ratio of the training instances')
    parser.add_argument('--colsample_bytree', type=float, default=0.8,
                        help='Subsample ratio of columns when constructing\
                              each tree')
    
    return parser.parse_args()


@ray.remote
def process_chunk(chunk_df, pipeline, label_encoder, text_column, label_column):
    """Process a chunk of data using the fitted pipeline and label encoder."""
    # Transform text using the pipeline
    features_sparse = pipeline.transform(chunk_df[text_column])
    feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
    features_df = pd.DataFrame(
        features_sparse.toarray(),
        columns=feature_names,
        index=chunk_df.index
    )
    
    # Transform labels based on whether a label_encoder exists
    if label_encoder is not None:
        # Using a label encoder for text labels (like yes/no)
        # Make sure all values are strings for consistent encoding
        labels = chunk_df[label_column].astype(str).str.lower().str.strip()
        labels_encoded = label_encoder.transform(labels)
    else:
        # No label encoder needed (labels are already 0/1)
        # Just ensure they're integers
        labels_encoded = chunk_df[label_column].astype(int).values
    
    # Create a DataFrame for the labels
    labels_df = pd.DataFrame({label_column: labels_encoded}, index=chunk_df.index)
    
    # Return the processed chunk with features and labels
    return pd.concat([features_df, labels_df], axis=1)


def process_large_dataset(df,
                          pipeline,
                          label_encoder,
                          text_column,
                          label_column,
                          chunk_size=5000,
                          logger=None):
    """Process a large dataset in chunks using Ray."""
    if logger:
        logger.info(f"Processing large dataset in chunks of size {chunk_size}...")
    
    # Split the dataframe into chunks
    chunks = np.array_split(df, max(1, len(df) // chunk_size))
    
    if logger:
        logger.info(f"Dataset split into {len(chunks)} chunks")
    
    # Store references to Ray objects
    refs = []
    
    # Process each chunk in parallel
    for i, chunk in enumerate(chunks):
        if logger and i % 10 == 0:
            logger.info(f"Submitting chunk {i+1}/{len(chunks)} for processing")
        
        ref = process_chunk.remote(
            chunk, pipeline, label_encoder, text_column, label_column
        )
        refs.append(ref)
    
    # Retrieve results in batches to manage memory
    if logger:
        logger.info("Processing chunks in Ray...")
    
    processed_chunks = []
    batch_size = 5  # Number of chunks to process at once
    
    for i in range(0, len(refs), batch_size):
        if logger:
            logger.info(f"Getting results for chunks {i+1}-{min(i+batch_size, len(refs))}/{len(refs)}")
        
        batch_refs = refs[i:i+batch_size]
        batch_results = ray.get(batch_refs)
        processed_chunks.extend(batch_results)
    
    # Combine results
    if logger:
        logger.info("Combining processed chunks...")
    
    combined_df = pd.concat(processed_chunks, axis=0)
    
    if logger:
        logger.info(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df


@ray.remote
def train_xgboost_model(X_train, y_train, params):
    """Train XGBoost model in a Ray task."""
    # Check the unique values of y_train to confirm it's properly processed
    unique_values = np.unique(y_train)
    print(f"Training with label values: {unique_values}")
    
    if len(unique_values) != 2:
        print(f"Warning: Expected 2 classes for binary classification, got {len(unique_values)}")
    
    # Ensure params has the correct objective for binary classification
    if 'objective' not in params or params['objective'] != 'binary:logistic':
        print("Setting objective to binary:logistic for binary classification")
        params['objective'] = 'binary:logistic'
    
    # Create and train the model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

# Update to the main() function in train_xgb.py to add additional validation
def main():
    try:
        args = parse_arguments()
        ngram_range = (args.ngram_min, args.ngram_max)
        
        # Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(
                    args.model_dir,
                    f"{args.model_name}_nutrikidai_training.log")),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Starting XGBoost training with arguments: {args}")   
        
        # Process CSV data to get pipeline and label encoder
        # The process_csv function will automatically determine if label encoding is needed
        logger.info("Creating text processing pipeline...")
        X_df, _, y, pipeline, feature_dict, label_encoder = process_csv(
            file_path=args.data_file,
            text_column=args.text_column,
            label_column=args.label_column,
            id_column=args.id_column,
            model_name=args.model_name,
            max_features=args.max_features,
            remove_stop_words=args.remove_stop_words,
            apply_stemming=args.apply_stemming,
            vectorization_mode=args.vectorization_mode,
            ngram_range=ngram_range,
            save_path=args.model_dir
        )
        
        # Log whether label encoding was used
        if label_encoder is None:
            logger.info("Labels were already numeric (0/1) - no encoding was needed")
        else:
            logger.info(f"Label encoding applied. Classes: {label_encoder.classes_} -> {np.unique(y)}")
        
        # Define feature columns from the processed features
        feature_columns = list(X_df.columns)
        logger.info(f"Number of features: {len(feature_columns)}")
        
        # Read the original data in chunks
        logger.info(f"Reading dataset from {args.data_file}...")
        df = pd.read_csv(args.data_file)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        
        # Process the entire dataset in chunks
        processed_df = process_large_dataset(
            df, pipeline, label_encoder, 
            args.text_column, args.label_column, 
            args.chunk_size, logger
        )
        
        # Verify the processed labels to ensure they're binary
        unique_processed = processed_df[args.label_column].unique()
        logger.info(f"Unique values in processed labels: {unique_processed}")
        if set(unique_processed) != {0, 1}:
            logger.warning(f"Expected binary labels (0/1), but got: {unique_processed}")
        
        # Get the tree method and scaling configuration for Ray
        scaling_config, tree_method = get_scaling_config_and_tree_method()
        
        # Try to load best parameters if config_dir is provided
        if args.config_dir and os.path.exists(args.config_dir):
            try:
                best_params_path = os.path.join(
                    args.config_dir,
                    f"{args.model_name}_nutrikidai_config.joblib")
                if os.path.exists(best_params_path):
                    best_params = joblib.load(best_params_path)
                    logger.info(f"Loaded hyperparameters from {best_params_path}")
                else:
                    raise FileNotFoundError(f"Config file not found {best_params_path}")
            except Exception as e:
                logger.warning(f"Failed to load hyperparameters: {str(e)}. Using command line parameters.")
                best_params = None
        else:
            logger.info("No config directory provided or it doesn't exist. Using command line parameters.")
            best_params = None
        
        # If best_params is not loaded, use command line arguments and save defaults
        if best_params is None:
            best_params = {
                "objective": "binary:logistic",  # Ensure this is set for binary classification
                "tree_method": tree_method,
                "eval_metric": ["logloss", "error"],
                "eta": args.eta,
                "max_depth": args.max_depth,
                "min_child_weight": args.min_child_weight,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree
            }
            default_params_path = os.path.join(
                args.model_dir,
                f"{args.model_name}_nutrikidai_config.joblib")
            joblib.dump(best_params, default_params_path)
            logger.info(f"Saved default hyperparameters to {default_params_path}")
        
        # Ensure objective is set to binary:logistic
        if 'objective' not in best_params or best_params['objective'] != 'binary:logistic':
            logger.info("Setting objective to binary:logistic for binary classification")
            best_params['objective'] = 'binary:logistic'
        
        logger.info(f"Training with parameters: {best_params}")
        
        # Extract features and labels from processed data
        logger.info("Preparing data for model training...")
        X_train = processed_df[feature_columns]
        y_train = processed_df[args.label_column]
        
        # Double-check y_train is proper format for XGBoost
        logger.info(f"Label dtype: {y_train.dtype}, unique values: {y_train.unique()}")
        
        # Put training data in Ray object store
        logger.info("Putting training data in Ray object store...")
        X_train_ref = ray.put(X_train)
        y_train_ref = ray.put(y_train)
        
        # Train model using Ray 
        logger.info("Starting model training in Ray...")
        # Pass the Ray object references to the remote function
        model_ref = train_xgboost_model.remote(X_train_ref, y_train_ref, best_params)
        
        # Get trained model
        logger.info("Waiting for model training to complete...")
        model = ray.get(model_ref)
        
        # Save model
        model_path = os.path.join(args.model_dir,
                                 f"{args.model_name}_nutrikidai_model.json")
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Clear references to free memory
        del X_train_ref
        del y_train_ref
        del model_ref
        
        logger.info("Training completed successfully.")
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
        else:
            logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True) 
    main()
