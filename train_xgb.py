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


# def compute_scale_pos_weight(y):
#     """Compute scale_pos_weight for handling class imbalance."""
#     num_neg = (y == 0).sum()
#     num_pos = (y == 1).sum()
#     return num_neg / num_pos if num_pos > 0 else 1


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
                        help='Subsample ratio of columns when constructing each tree')
    parser.add_argument('--num_boost_round', type=int, default= 100, help="Number of boost round")
    
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
    
    # Check if labels are already numeric (0/1)
    if chunk_df[label_column].dtype in ['int64', 'int32', 'float64', 'float32']:
        # Check if values are already binary (only 0 and 1)
        unique_values = set(chunk_df[label_column].unique())
        if unique_values.issubset({0, 1}) or unique_values.issubset({0.0, 1.0}):
            # Already binary numeric, just ensure it's int
            labels_encoded = chunk_df[label_column].astype(int)
        else:
            # Numeric but not binary, use encoder
            if label_encoder is not None:
                labels = chunk_df[label_column].astype(str)
                labels_encoded = pd.Series(
                    label_encoder.transform(labels),
                    index=chunk_df.index
                )
            else:
                # Fallback to as-is if no encoder provided
                labels_encoded = chunk_df[label_column]
    else:
        # Handle string labels
        if label_encoder is not None:
            # Try to detect if it's already '0'/'1' strings
            unique_values = set(chunk_df[label_column].astype(str).str.strip().unique())
            if unique_values.issubset({'0', '1'}):
                # Already binary strings, convert directly to int
                labels_encoded = chunk_df[label_column].astype(str).str.strip().astype(int)
            else:
                # Not binary strings, use encoder
                labels = chunk_df[label_column].astype(str).str.lower().str.strip()
                labels_encoded = pd.Series(
                    label_encoder.transform(labels),
                    index=chunk_df.index
                )
        else:
            # No encoder, try direct conversion (will fail if not convertible)
            labels_encoded = chunk_df[label_column].astype(int)
    
    # Ensure labels_encoded is a Series, not a DataFrame
    if isinstance(labels_encoded, pd.DataFrame):
        labels_encoded = labels_encoded.iloc[:, 0]
    
    # Create a new DataFrame with features
    result_df = features_df.copy()
    
    # Add label column as a Series (not a DataFrame)
    result_df[label_column] = labels_encoded
    
    return result_df


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
        # Check if the label column is already binary numeric
        if df[label_column].dtype in ['int64', 'int32', 'float64', 'float32']:
            unique_values = set(df[label_column].unique())
            if unique_values.issubset({0, 1}) or unique_values.issubset({0.0, 1.0}):
                logger.info(f"Label column '{label_column}' is already binary numeric (0/1). No encoding needed.")
            else:
                logger.info(f"Label column '{label_column}' is numeric but not binary. Encoding may be needed.")
                if label_encoder is not None:
                    logger.info(f"Using label encoder with classes: {label_encoder.classes_}")
        else:
            # Check if strings are already '0'/'1'
            unique_values = set(df[label_column].astype(str).str.strip().unique())
            if unique_values.issubset({'0', '1'}):
                logger.info(f"Label column '{label_column}' contains binary strings ('0'/'1'). Will convert to integers.")
            else:
                logger.info(f"Label column '{label_column}' contains non-binary strings. Encoding will be applied.")
                if label_encoder is not None:
                    logger.info(f"Using label encoder with classes: {label_encoder.classes_}")
    
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
        
        # Check if combined_df[label_column] is a DataFrame instead of Series
        if isinstance(combined_df[label_column], pd.DataFrame):
            logger.warning(f"Label column '{label_column}' returned a DataFrame instead of a Series. Using first column.")
            combined_df[label_column] = combined_df[label_column].iloc[:, 0]
            logger.info(f"Label column dtype: {combined_df[label_column].dtype}")
        else:
            logger.info(f"Label column dtype: {combined_df[label_column].dtype}")
        
        # Ensure label column is 1-dimensional before calling value_counts
        try:
            label_counts = combined_df[label_column].value_counts()
            logger.info(f"Label distribution in processed data: {label_counts}")
        except Exception as e:
            logger.warning(f"Could not compute value_counts: {str(e)}")
            logger.info(f"First few label values: {combined_df[label_column].head()}")
        
        # Verify data types
        if not pd.api.types.is_integer_dtype(combined_df[label_column]):
            # Force conversion to int if somehow it wasn't done correctly
            logger.warning(f"Converting {label_column} to int64 - was {combined_df[label_column].dtype}")
            combined_df[label_column] = combined_df[label_column].astype(int)
    
    return combined_df


@ray.remote
def train_xgboost_model(X_train, y_train, params):
    """
    Train an XGBoost model in a Ray remote function.
    
    Args:
        X_train: Training features dataframe
        y_train: Training labels series
        params: Dictionary of XGBoost parameters
    
    Returns:
        The trained XGBoost model
    """
    # Compute scale_pos_weight to handle class imbalance
    # scale_pos_weight = compute_scale_pos_weight(y_train)
    
    # # Update params with computed scale_pos_weight
    # params['scale_pos_weight'] = scale_pos_weight
    
    # Create a DMatrix for XGBoost (no need to call ray.get() anymore)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Train the model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train")],
        verbose_eval=10  # Print evaluation metrics every 10 rounds
    )
    
    return model


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
        
        # Verify label type returned from process_csv
        logger.info(f"Labels from process_csv have dtype: {y.dtype}")
        if label_encoder is not None:
            logger.info(f"Label encoder classes: {label_encoder.classes_} -> {np.unique(y)}")
        else:
            logger.info("No label encoder needed - labels are already binary")
        
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
                "objective": "binary:logistic",
                "tree_method": tree_method,
                "eval_metric": ["logloss", "error"],
                "eta": args.eta,
                "max_depth": args.max_depth,
                "min_child_weight": args.min_child_weight,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "num_boost_round": args.num_boost_round
            }
            default_params_path = os.path.join(
                args.model_dir,
                f"{args.model_name}_nutrikidai_config.joblib")
            joblib.dump(best_params, default_params_path)
            logger.info(f"Saved default hyperparameters to {default_params_path}")
        
        # Extract features and labels from processed data
        logger.info("Preparing data for model training...")
        X_train = processed_df[feature_columns]
        y_train = processed_df[args.label_column]
        
        # Final verification that labels are integers
        if not pd.api.types.is_integer_dtype(y_train):
            logger.warning(f"Converting y_train to int64 - was {y_train.dtype}")
            y_train = y_train.astype(int)
        
        # Verify label distribution before training
        label_counts = y_train.value_counts()
        logger.info(f"Final label distribution before training: {label_counts}")
        
        # # Compute scale_pos_weight 
        # scale_pos_weight = compute_scale_pos_weight(y_train)
        # logger.info(f"Computed scale_pos_weight: {scale_pos_weight}")
        
        # Log parameters before training
        final_params = best_params.copy()
        # final_params['scale_pos_weight'] = scale_pos_weight
        logger.info(f"Final training parameters: {final_params}")
        
        # Train model using Ray - pass data directly without ray.put()
        logger.info("Starting model training in Ray...")
        model_ref = train_xgboost_model.remote(X_train, y_train, final_params)
        
        # Get trained model
        logger.info("Waiting for model training to complete...")
        model, used_scale_pos_weight = ray.get(model_ref)
        
        # Save model
        model_path = os.path.join(args.model_dir,
                                  f"{args.model_name}_nutrikidai_model.json")
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # # Save scale_pos_weight for potential later use
        # scale_weight_path = os.path.join(args.model_dir,
        #                                  f"{args.model_name}_scale_pos_weight.joblib")
        # joblib.dump(used_scale_pos_weight, scale_weight_path)
        # logger.info(f"scale_pos_weight saved to {scale_weight_path}")
        
        logger.info("Training completed successfully.")
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
        else:
            logging.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True) 
    main()
