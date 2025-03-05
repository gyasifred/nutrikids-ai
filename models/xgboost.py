#!/usr/bin/env python3

import glob
import os
import joblib
import ray
from ray import tune
from ray.data import from_pandas
from ray.train import RunConfig, ScalingConfig
from ray.train.xgboost import XGBoostTrainer
from ray.tune.tuner import Tuner
import xgboost as xgb
############################################################
# GPU Detection: Configure Resources and Tree Method
############################################################


def get_scaling_config_and_tree_method():
    """
    Checks Ray's cluster resources to see if a GPU is available.
    Returns a ScalingConfig for training and the appropriate XGBoost tree_method.
    """
    resources = ray.cluster_resources()
    if resources.get("GPU", 0) > 0:
        print("GPU detected. Configuring to use GPU.")
        scaling_config = ScalingConfig(
            num_workers=1,
            resources_per_worker={"CPU": 1, "GPU": 1},
            use_gpu=True,
        )
        tree_method = "gpu_hist"
    else:
        print("GPU not detected. Configuring to use CPU.")
        scaling_config = ScalingConfig(
            num_workers=1,
            resources_per_worker={"CPU": 1},
            use_gpu=False,
        )
        tree_method = "auto" 
    return scaling_config, tree_method


def load_model_artifacts(model_dir):
    """Load all model artifacts from the specified directory."""
    try:
        # Use glob to find files
        pipeline_path = glob.glob(os.path.join(model_dir, "*pipeline*.joblib"))[0]
        feature_columns_path = glob.glob(os.path.join(model_dir, "*feature_columns*.joblib"))[0]
        model_path = glob.glob(os.path.join(model_dir, "*xgboost_model*.json"))[0]
        
        # Load artifacts
        pipeline = joblib.load(pipeline_path)
        feature_columns = joblib.load(feature_columns_path)
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Try to load label encoder if it exists
        label_encoder = None
        label_encoder_paths = glob.glob(os.path.join(model_dir, "*label_encoder*.joblib"))
        if label_encoder_paths:
            label_encoder = joblib.load(label_encoder_paths[0])
        
        return model, pipeline, feature_columns, label_encoder
    
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
