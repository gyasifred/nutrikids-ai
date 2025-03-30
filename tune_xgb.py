#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from models.xgboost import get_scaling_config_and_tree_method
from utils import process_csv
import ray
from ray import tune
from ray.train.xgboost import XGBoostTrainer
from ray.train import RunConfig
from ray.tune.tuner import Tuner


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Train an XGBoost model with hyperparameter tuning.')
    parser.add_argument('--train_data_file', type=str, required=True,
                        help='Path to training data CSV file')
    parser.add_argument('--valid_data_file', type=str, required=True,
                        help='Path to validation data CSV file')
    parser.add_argument('--text_column', type=str, default="txt",
                        help='Name of the text column in the CSV')
    parser.add_argument('--label_column', type=str, default="label",
                        help='Name of the label column in the CSV')
    parser.add_argument('--id_column', type=str, default="DEID",
                        help='Name of the ID column in the CSV')
    parser.add_argument('--max_features', type=int, default=10000,
                        help='Maximum number of features for vectorization')
    parser.add_argument('--remove_stop_words', action='store_true',
                        default=False,
                        help='Remove stop words during preprocessing')
    parser.add_argument('--apply_stemming', action='store_false',
                        default=False,
                        help='Apply stemming during preprocessing')
    parser.add_argument('--vectorization_mode', type=str, default='tfidf',
                        choices=['tfidf', 'count', 'binary'],
                        help='Vectorization mode')
    parser.add_argument('--ngram_min', type=int, default=1,
                        help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=1,
                        help='Maximum n-gram size')
    parser.add_argument('--model_name', type=str, default="xgboost",
                        help='Name of the type of Model being trained')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of parameter settings that are sampled \
                            (default: 20)')
    parser.add_argument('--model_dir', type=str, default='./xgboost',
                        help='Directory to save the model')
    parser.add_argument('--chunk_size', type=int, default=5000,
                        help='Chunk size for processing large datasets')
    return parser.parse_args()


# Define a Ray task for processing chunks of data
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
    
    # Transform labels
    if label_encoder is not None:
        # Ensure consistent formatting
        labels = chunk_df[label_column].astype(str).str.strip()
        labels_encoded = pd.Series(
            label_encoder.transform(labels), 
            index=chunk_df.index
        )
    else:
        # For binary labels that don't need encoding
        labels_encoded = chunk_df[label_column].astype(int)
    
    # Return the processed chunk
    result_df = features_df.copy()
    result_df[label_column] = labels_encoded
    return result_df

def process_large_dataset(df, pipeline, label_encoder, text_column, label_column, chunk_size=5000):
    """Process a large dataset in chunks using Ray."""
    # Split the dataframe into chunks
    chunks = np.array_split(df, max(1, len(df) // chunk_size))
    
    # Store references to Ray objects
    refs = []
    
    # Process each chunk in parallel
    for chunk in chunks:
        ref = process_chunk.remote(
            chunk, pipeline, label_encoder, text_column, label_column
        )
        refs.append(ref)
    
    # Retrieve results
    processed_chunks = ray.get(refs)
    
    # Combine results
    return pd.concat(processed_chunks, axis=0)


def create_ray_datasets_in_chunks(train_df, valid_df, pipeline, label_encoder, 
                                 text_column, label_column, chunk_size=5000):
    """Create Ray datasets from large DataFrames by processing in chunks."""
    # Process training data in chunks
    print(f"Processing training data in chunks of size {chunk_size}...")
    train_processed_df = process_large_dataset(
        train_df, pipeline, label_encoder, text_column, label_column, chunk_size
    )
    
    # Process validation data in chunks
    print(f"Processing validation data in chunks of size {chunk_size}...")
    valid_processed_df = process_large_dataset(
        valid_df, pipeline, label_encoder, text_column, label_column, chunk_size
    )
    
    # Create Ray datasets directly
    print("Creating Ray datasets...")
    train_ds = ray.data.from_pandas(train_processed_df)
    valid_ds = ray.data.from_pandas(valid_processed_df)
    
    return train_ds, valid_ds


def main():
    args = parse_arguments()
    ngram_range = (args.ngram_min, args.ngram_max)
    
    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    
    try:
        # Process the training CSV to get the pipeline and label encoder
        X_df, _, y, pipeline, feature_dict, label_encoder = process_csv(
            file_path=args.train_data_file,
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
        print("Pipeline and encoder created. Feature DataFrame shape:", X_df.shape)
        
        # Read the raw DataFrames
        train_df = pd.read_csv(args.train_data_file)
        valid_df = pd.read_csv(args.valid_data_file)
        
        # FIXED: Create Ray datasets directly without using Ray object store intermediates
        train_ds, valid_ds = create_ray_datasets_in_chunks(
            train_df, valid_df, pipeline, label_encoder,
            args.text_column, args.label_column, args.chunk_size
        )
        
        # Get scaling config and tree method
        scaling_config, tree_method = get_scaling_config_and_tree_method()
        
        # Define hyperparameter search space
        param_space = {
            "scaling_config": scaling_config,
            "params": {
                "objective": "binary:logistic",
                "tree_method": tree_method,
                "eval_metric": ["logloss", "error"],
                "eta": tune.loguniform(1e-4, 1e-1),
                "subsample": tune.uniform(0.5, 1.0),
                "max_depth": tune.randint(3, 10),
                "min_child_weight": tune.randint(1, 10),
                "gamma": tune.uniform(0, 5),
                "colsample_bytree": tune.uniform(0.3, 1.0),
                "reg_alpha": tune.loguniform(1e-4, 1e-1),
                "reg_lambda": tune.loguniform(1e-4, 1e-1),
                "max_bin": tune.randint(100, 300),
                "num_boost_round": tune.choice([100, 150, 200, 250, 300, 350, 400, 450, 500]),
                "scale_pos_weight": tune.choice([1,2,3])
            },
        }
        
        # Initialize XGBoostTrainer for hyperparameter tuning
        trainer = XGBoostTrainer(
            label_column=args.label_column,
            params={}, 
            datasets={"train": train_ds, "validation": valid_ds},
        )  
        
        tuner = Tuner(
            trainable=trainer,
            param_space=param_space,
            tune_config=tune.TuneConfig(num_samples=args.num_samples),
            run_config=RunConfig(name="xgboost_gpu_tune_nutrikidai")
        )
        
        print("Starting hyperparameter tuning...")
        results = tuner.fit()
        
        # Save hyperparameter tuning results summary
        results_df = results.get_dataframe()
        results_path = os.path.join(
            args.model_dir, f"{args.model_name}_nutrikidai_tuning_results.csv")
        results_df.to_csv(results_path)
        print(f"Tuning results saved to {results_path}")
        
        # Extract best hyperparameters
        if not results_df.empty:
            try:
                best_result = results.get_best_result(
                    metric="validation-logloss", mode="min")
                print("Best trial config:", best_result.config)
                print("Best trial final evaluation logloss:",
                      best_result.metrics["validation-logloss"])
            except KeyError:
                available_metrics = list(results_df.columns)
                metric_cols = [
                    col for col in available_metrics if "validation-" in col
                ]
                if metric_cols:
                    best_metric = metric_cols[0]
                    best_result = results.get_best_result(
                        metric=best_metric, mode="min")
                    print(f"Best trial config based on {best_metric}:",
                          best_result.config)
                    print(f"Best trial final evaluation {best_metric}:",
                          best_result.metrics[best_metric])
                else:
                    print("No validation metrics found in results. Using the first result as best.")
                    best_result = results.get_best_result()
        else:
            print("No valid results from tuning. Using default parameters.")
            best_result = None
            
        if best_result:
            best_params = best_result.config["params"]
            # Save best hyperparameters
            hyperparams_path = os.path.join(
                args.model_dir, f"{args.model_name}_nutrikidai_config.joblib")
            joblib.dump(best_result.config, hyperparams_path)
            print(f"Best hyperparameters saved to {hyperparams_path}")
            
            # Save best model metrics
            metrics_path = os.path.join(
                args.model_dir, f"{args.model_name}_nutrikidai_metrics.joblib")
            joblib.dump(best_result.metrics, metrics_path)
            print(f"Best model metrics saved to {metrics_path}")
        else:
            best_params = {
                "objective": "binary:logistic",
                "tree_method": tree_method,
                "eval_metric": ["logloss", "error"],
                "eta": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            }
            default_params_path = os.path.join(
                args.model_dir, f"{args.model_name}_nutrikidai_configs.joblib")
            joblib.dump(best_params, default_params_path)
            print(f"Using default parameters. Saved to {default_params_path}")
            
        return best_params, pipeline, feature_dict
    
    except Exception as e:
        print(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)  
    best_params, pipeline, feature_dict = main()
