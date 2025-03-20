#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import joblib
from models.xgboost import get_scaling_config_and_tree_method
from utils import process_csv
import ray
from ray import tune
from ray.data import from_pandas
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
    return parser.parse_args()


def main():
    args = parse_arguments()
    ngram_range = (args.ngram_min, args.ngram_max)
    
    # Ensure model directory exists.
    os.makedirs(args.model_dir, exist_ok=True)
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Process the training CSV.
        X_df, complete_xdf, y, pipeline, feature_dict, label_encoder = process_csv(
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
        print("Training CSV processed. Feature DataFrame shape:", X_df.shape)
        
        # Store pipeline and label_encoder in Ray's object store
        pipeline_ref = ray.put(pipeline)
        label_encoder_ref = ray.put(label_encoder)
        
        # Process the validation CSV within a function to avoid large object capture
        def process_validation_data():
            # Get validation data
            valid_df = pd.read_csv(args.valid_data_file)
            
            # Retrieve pipeline and label_encoder from Ray's object store
            pipeline = ray.get(pipeline_ref)
            label_encoder = ray.get(label_encoder_ref)
            
            # Transform the text column using the pipeline
            valid_features_sparse = pipeline.transform(valid_df[args.text_column])
            feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
            
            valid_features_df = pd.DataFrame(valid_features_sparse.toarray(),
                                             columns=feature_names,
                                             index=valid_df.index)
            
            # Transform the validation labels using the fitted LabelEncoder.
            if label_encoder is not None:
                valid_labels = valid_df[args.label_column].astype(str).str.strip()
                valid_labels_encoded = pd.Series(
                    label_encoder.transform(valid_labels), index=valid_df.index)
            else:
                valid_labels_encoded = valid_df[args.label_column]
            
            # Concatenate the encoded column back to the transformed features.
            valid_complete_df = pd.concat([valid_features_df, valid_labels_encoded.rename(args.label_column)], axis=1)
            
            return valid_complete_df
        
        # Execute validation data processing 
        valid_complete_df = process_validation_data()
        
        # Convert training and validation dataframes to Ray Datasets
        # Using Ray's from_pandas() function already handles placing data in Ray's object store
        train_ds = from_pandas(complete_xdf)
        valid_ds = from_pandas(valid_complete_df)
        
        # Get scaling config and tree method (e.g., based on GPU availability).
        scaling_config, tree_method = get_scaling_config_and_tree_method()
        
        # Define hyperparameter search space.
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
            },
        }
        
        # Initialize XGBoostTrainer for hyperparameter tuning.
        # By design, Ray already stores the datasets in Ray's object store
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
        
        results = tuner.fit()
        
        # Save hyperparameter tuning results summary.
        results_df = results.get_dataframe()
        results_path = os.path.join(args.model_dir, f"{args.model_name}_nutrikidai_tuning_results.csv")
        results_df.to_csv(results_path)
        print(f"Tuning results saved to {results_path}")
        
        # Extract best hyperparameters.
        if not results_df.empty:
            try:
                best_result = results.get_best_result(metric="validation-logloss", mode="min")
                print("Best trial config:", best_result.config)
                print("Best trial final evaluation logloss:", best_result.metrics["validation-logloss"])
            except KeyError:
                available_metrics = list(results_df.columns)
                metric_cols = [col for col in available_metrics if "validation-" in col]
                if metric_cols:
                    best_metric = metric_cols[0]
                    best_result = results.get_best_result(metric=best_metric, mode="min")
                    print(f"Best trial config based on {best_metric}:", best_result.config)
                    print(f"Best trial final evaluation {best_metric}:", best_result.metrics[best_metric])
                else:
                    print("No validation metrics found in results. Using the first result as best.")
                    best_result = results.get_best_result()
        else:
            print("No valid results from tuning. Using default parameters.")
            best_result = None
        
        if best_result:
            best_params = best_result.config["params"]
            # Save best hyperparameters.
            hyperparams_path = os.path.join(args.model_dir, f"{args.model_name}_nutrikidai_config.joblib")
            joblib.dump(best_result.config, hyperparams_path)
            print(f"Best hyperparameters saved to {hyperparams_path}")
            # Save best model metrics.
            metrics_path = os.path.join(args.model_dir, f"{args.model_name}_nutrikidai_metrics.joblib")
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
            default_params_path = os.path.join(args.model_dir, f"{args.model_name}_nutrikidai_configs.joblib")
            joblib.dump(best_params, default_params_path)
            print(f"Using default parameters. Saved to {default_params_path}")
        
        # Get pipeline and feature dict from Ray's object store
        pipeline = ray.get(pipeline_ref)
        
        return best_params, pipeline, feature_dict
    except Exception as e:
        print(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    best_params, pipeline, feature_dict = main()
