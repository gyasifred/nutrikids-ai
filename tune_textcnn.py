#!/usr/bin/env python3
import os
import joblib
import numpy as np
import pandas as pd
import argparse
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
import torch
from utils import process_labels
from models.text_cnn import train_textcnn, TextTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Tune TextCNN hyperparameters with Ray Tune')
    
    # Data paths and columns
    parser.add_argument('--train_data', type=str, required=True, 
                        help='Path to training CSV file')
    parser.add_argument('--val_data', type=str, required=True, 
                        help='Path to validation CSV file')
    parser.add_argument('--text_column', type=str, default='txt',
                        help='Name of the text column in CSV (default: txt)')
    parser.add_argument('--label_column', type=str, default='label',
                        help='Name of the label column in CSV (default: label)')
    
    # Tokenizer parameters
    parser.add_argument('--max_vocab_size', type=int, default=8000)
    parser.add_argument('--min_frequency', type=int, default=2)
    parser.add_argument('--pad_token', type=str, default='<PAD>')
    parser.add_argument('--unk_token', type=str, default='<UNK>')
    parser.add_argument('--max_length', 
                        type=lambda x: None if x.lower() == 'none' else int(x),
                        default=None)
    parser.add_argument('--padding', type=str, default='post',
                        choices=['pre', 'post'])
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--pretrained_embeddings', 
                        type=lambda x: None if x.lower() == 'none' else str(x),
                        default=None) 
    parser.add_argument('--positive_weight', type=float, default=1.0,
                        help='Weight multiplier for positive class (default: 2.0)')
    
    # Ray Tune parameters
    parser.add_argument('--output_dir', type=str, default='CNN')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--grace_period', type=int, default=10)
    
    return parser.parse_args()


def calculate_class_weights(labels, positive_weight=1.0):
    """Calculate pos_weight for BCEWithLogitsLoss"""
    counts = np.bincount(labels)
    num_neg = counts[0]
    num_pos = counts[1]
    pos_weight = (num_neg / num_pos) * positive_weight
    return torch.tensor(pos_weight, dtype=torch.float)
                        
def main():
    # Parse command line arguments
    args = parse_args()

    # Load data
    print(f"Loading Training Data from {args.train_data}...")
    train_df = pd.read_csv(args.train_data)

    print(f"Loading Validation Data from {args.val_data}...")
    val_df = pd.read_csv(args.val_data)

    train_texts = train_df[args.text_column].tolist()
    train_labels_raw = train_df[args.label_column].tolist()
    val_texts = val_df[args.text_column].tolist()
    val_labels_raw = val_df[args.label_column].tolist()

    # Process labels
    train_labels, label_encoder = process_labels(train_labels_raw)
    val_labels, _ = process_labels(val_labels_raw) if label_encoder is None else (
        label_encoder.transform(val_labels_raw), None)

    print(f"Training data: {len(train_texts)} examples")
    print(f"Validation data: {len(val_texts)} examples")
    print(f"Label type: {'text (with encoding)' if label_encoder else 'numeric (no encoding needed)'}")

    # Load pretrained embeddings if provided
    pretrained_embeddings_dict = None
    if args.pretrained_embeddings:
        print(f"Loading pretrained embeddings from {args.pretrained_embeddings}...")
        pretrained_embeddings_dict = TextTokenizer.load_pretrained_embeddings(
            args.pretrained_embeddings)
        print(f"Loaded {len(pretrained_embeddings_dict)} pretrained word vectors")

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_labels, args.positive_weight)

    # Put the large objects in Ray's object store
    train_texts_ref = ray.put(train_texts)
    train_labels_ref = ray.put(train_labels)
    val_texts_ref = ray.put(val_texts)
    val_labels_ref = ray.put(val_labels)
    embeddings_ref = ray.put(pretrained_embeddings_dict)
    label_encoder_ref = ray.put(label_encoder)
    class_weights_ref = ray.put(class_weights)
    
    # Store fixed args in Ray's object store too
    fixed_args_ref = ray.put({
        "max_vocab_size": args.max_vocab_size,
        "min_frequency": args.min_frequency,
        "pad_token": args.pad_token,
        "unk_token": args.unk_token,
        "max_length": args.max_length,
        "padding": args.padding,
        "embed_dim": args.embedding_dim,
        "freeze_embeddings": False,
        "max_epochs": args.max_epochs
    })

    # Define trainable function for Ray that retrieves data from object store
    def train_func(config):
        # Retrieve data from Ray's object store
        train_texts = ray.get(train_texts_ref)
        train_labels = ray.get(train_labels_ref)
        val_texts = ray.get(val_texts_ref)
        val_labels = ray.get(val_labels_ref)
        pretrained_embeddings_dict = ray.get(embeddings_ref)
        label_encoder = ray.get(label_encoder_ref)
        class_weights = ray.get(class_weights_ref)
        fixed_args = ray.get(fixed_args_ref)
        
        # Merge provided args with tunable config
        full_config = fixed_args.copy()
        full_config.update(config)

        # Train model - Pass class_weights as a parameter in the full_config
        full_config['class_weights'] = class_weights
        model, tokenizer, trained_label_encoder, metrics = train_textcnn(
            train_texts, train_labels, val_texts, val_labels,
            full_config, fixed_args["max_epochs"], pretrained_embeddings_dict,
            provided_label_encoder=label_encoder
        )

        # Report final metrics to Ray Tune
        tune.report({
            "val_loss": metrics["val_loss"][-1],
            "val_accuracy": metrics["val_accuracy"][-1],
            "train_accuracy": metrics["train_accuracy"][-1],
            "train_loss": metrics["train_loss"][-1],
            "val_f1": metrics['val_f1'][-1]
        })
    
    # Hyperparameter space for Ray Tune
    param_space = {
        "embed_dim": tune.choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500]),
        "num_filters": tune.choice([50, 100, 150, 200, 250, 300, 350, 400]),
        "dropout_rate": tune.uniform(0.2, 0.6),
        "lr": tune.loguniform(1e-4, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "max_vocab_size": tune.choice([3000,5000,8000,10000, 15000]),
        "kernel_sizes": tune.choice([[3, 4, 5], [2, 3, 4], [4, 5, 6], [2, 3, 4]])
    }

    # Set resources based on GPU availability
    gpu_per_trial = 1 if torch.cuda.is_available() else 0
    cpu_per_trial = 1

    # Configure trainer with resources
    train_textcnn_with_resources = tune.with_resources(
        train_func, 
        {"gpu": gpu_per_trial, "cpu": cpu_per_trial}
    )

    # Create tuner instance
    tuner = tune.Tuner(
        train_textcnn_with_resources,
        tune_config=tune.TuneConfig(
            metric="val_f1",
            mode="max", 
            scheduler=ASHAScheduler(
                max_t=args.max_epochs,
                grace_period=args.grace_period,
                reduction_factor=2
            ),
            num_samples=args.num_samples
        ),
        param_space=param_space,
    )

    # Run hyperparameter search
    print("Starting hyperparameter tuning...")
    results = tuner.fit()

    # Get and print best results based on val_accuracy
    best_result = results.get_best_result(metric="val_accuracy", mode="max")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation accuracy: {best_result.metrics['val_accuracy']}")
    print(f"Best trial final training accuracy: {best_result.metrics['train_accuracy']}")
    print(f"Best trial final validation loss: {best_result.metrics['val_loss']}")
    print(f"Best trial final training loss: {best_result.metrics['train_loss']}")

    # Apply the best config (using the best configuration directly for further training or testing)
    best_config = best_result.config
    print(f"Applying best configuration: {best_config}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the best config using joblib
    best_config_path = os.path.join(args.output_dir, "best_config.joblib")
    joblib.dump(best_config, best_config_path)
    print(f"Best configuration saved to {best_config_path}")
    
    # Save label encoder if it was used
    if label_encoder is not None:
        label_encoder_path = os.path.join(args.output_dir, "label_encoder.joblib")
        joblib.dump(label_encoder, label_encoder_path)
        print(f"Label encoder saved to {label_encoder_path}")

if __name__ == "__main__":
    main()
