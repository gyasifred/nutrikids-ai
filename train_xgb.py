#!/usr/bin/env python3

import argparse
import os
import joblib
import logging
import json
import pandas as pd
import xgboost as xgb
import ray
from xgboost.core import XGBoostError

from utils import process_csv
from models.xgboost import get_scaling_config_and_tree_method

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train an XGBoost model.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to training data CSV file')
    parser.add_argument('--text_column', type=str, default="Note_Column", help='Name of the text column in the CSV')
    parser.add_argument('--label_column', type=str, default="Malnutrition_Label", help='Name of the label column in the CSV')
    parser.add_argument('--id_column', type=str, default="Patient_ID", help='Name of the ID column in the CSV')
    parser.add_argument("--config_dir", default="xgb_models", type=str, help='Path to best hyperparameter directory')
    parser.add_argument('--max_features', type=int, default=10000, help='Maximum number of features for vectorization')
    parser.add_argument('--remove_stop_words', action='store_true',default= False, help='Remove stop words during preprocessing')
    parser.add_argument('--apply_stemming', action='store_true',default= False, help='Apply stemming during preprocessing')
    parser.add_argument('--vectorization_mode', type=str, default='tfidf', choices=['tfidf', 'count', 'binary'], help='Vectorization mode')
    parser.add_argument('--ngram_min', type=int, default=1, help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=1, help='Maximum n-gram size')
    parser.add_argument('--model_dir', type=str, default='./xgb_models', help='Directory to save/load the model')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'], default='train', help='Operation mode')
    parser.add_argument('--input_text', type=str, help='Single text string for prediction (when mode is predict)')
    parser.add_argument('--input_file', type=str, help='CSV file for batch prediction (when mode is predict)')
    parser.add_argument('--model_name', type=str, default="xgb", help='Name of the type of Model being trained')
    
    # XGBoost parameters
    parser.add_argument('--eta', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--max_depth', type=int, default=6, help='Maximum depth of trees')
    parser.add_argument('--min_child_weight', type=float, default=1, help='Minimum sum of instance weight needed in a child')
    parser.add_argument('--subsample', type=float, default=0.8, help='Subsample ratio of the training instances')
    parser.add_argument('--colsample_bytree', type=float, default=0.8, help='Subsample ratio of columns when constructing each tree')
    
    return parser.parse_args()

def main():
    try:
        ray.init(ignore_reinit_error=True)
        args = parse_arguments()
        ngram_range = (args.ngram_min, args.ngram_max)
        
        # Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(args.model_dir, f"{args.model_name}_nutrikidai_training.log")),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Starting XGBoost training with arguments: {args}")
        
        # Process CSV data
        X_df, complete_xdf, y, pipeline, feature_dict, label_encoder = process_csv(
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
        
        # Define feature columns from the processed features
        feature_columns = list(X_df.columns)
        
        # Get the tree method and scaling configuration for Ray
        scaling_config, tree_method = get_scaling_config_and_tree_method()
        
        # Try to load best parameters if config_dir is provided
        if args.config_dir and os.path.exists(args.config_dir):
            try:
                best_params_path = os.path.join(args.config_dir, f"{args.model_name}_nutrikidai_config.joblib")
                if os.path.exists(best_params_path):
                    best_params = joblib.load(best_params_path)
                    logger.info(f"Loaded hyperparameters from {best_params_path}")
                else:
                    raise FileNotFoundError(f"Config file not found at {best_params_path}")
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
                "colsample_bytree": args.colsample_bytree
            }
            default_params_path = os.path.join(args.model_dir, f"{args.model_name}_nutrikidai_config.joblib")
            joblib.dump(best_params, default_params_path)
            logger.info(f"Saved default hyperparameters to {default_params_path}")
        
        logger.info(f"Training with parameters: {best_params}")
        
        # Training
        combined_df = complete_xdf[feature_columns + [args.label_column]]
        X_train = combined_df[feature_columns]
        y_train = combined_df[args.label_column]
        
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # Save model
        model_path = os.path.join(args.model_dir, f"{args.model_name}_nutrikidai_model.json")
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    main()
