#!/usr/bin/env python3
import os
import joblib
import pandas as pd
import argparse
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
import torch

from models.text_cnn import train_textcnn, TextTokenizer
from utils import encode_labels

def parse_args():
    parser = argparse.ArgumentParser(description='Tune TextCNN hyperparameters with Ray Tune')
    
    # Data paths and columns
    parser.add_argument('--train', type=str, required=True, 
                        help='Path to training CSV file')
    parser.add_argument('--val', type=str, required=True, 
                        help='Path to validation CSV file')
    parser.add_argument('--text_column', type=str, default='Note_Column',
                        help='Name of the text column in CSV (default: Note_Column)')
    parser.add_argument('--label_column', type=str, default='Malnutrition_Label',
                        help='Name of the label column in CSV (default: Malnutrition_Label)')
    
    # Tokenizer parameters
    parser.add_argument('--max_vocab_size', type=int, default=20000,
                        help='Maximum vocabulary size (default: 20000)')
    parser.add_argument('--min_frequency', type=int, default=2,
                        help='Minimum word frequency to include in vocabulary (default: 1)')
    parser.add_argument('--pad_token', type=str, default='<PAD>',
                        help='Token used for padding (default: <PAD>)')
    parser.add_argument('--unk_token', type=str, default='<UNK>',
                        help='Token used for unknown words (default: <UNK>)')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum sequence length (default: None, will use longest sequence)')
    parser.add_argument('--padding', type=str, default='post', choices=['pre', 'post'],
                        help='Padding type: pre or post (default: post)')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Dimension of word embeddings (default: 100)')
    parser.add_argument('--pretrained_embeddings', type=str, default=None,
                        help='Path to pretrained word embeddings file (default: None)')
    
    # Ray Tune parameters
    parser.add_argument('--output_dir', type=str, default='textcnn_model',
                        help='Directory to save model and artifacts (default: cnn_model)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of parameter settings that are sampled (default: 10)')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum number of epochs for hyperparameter search (default: 10)')
    parser.add_argument('--grace_period', type=int, default=3,
                        help='Minimum number of epochs for each trial (default: 3)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load data
    print(f"Loading Training Data from {args.train}...")
    train_df = pd.read_csv(args.train)

    print(f"Loading Validation Data from {args.val}...")
    val_df = pd.read_csv(args.val)

    train_texts = train_df[args.text_column].tolist()
    train_labels = train_df[args.label_column].tolist()
    val_texts = val_df[args.text_column].tolist()
    val_labels = val_df[args.label_column].tolist()
    
    print(f"Training data: {len(train_texts)} examples")
    print(f"Validation data: {len(val_texts)} examples")
    
    # Load pretrained embeddings if provided
    pretrained_embeddings_dict = None
    if args.pretrained_embeddings:
        print(f"Loading pretrained embeddings from {args.pretrained_embeddings}...")
        pretrained_embeddings_dict = TextTokenizer.load_pretrained_embeddings(args.pretrained_embeddings)
        print(f"Loaded {len(pretrained_embeddings_dict)} pretrained word vectors")
    
    # Define a trainable function for Ray Tune
    def train_func(config):
        # Merge provided args with tunable config
        full_config = {
            "max_vocab_size": args.max_vocab_size,
            "min_frequency": args.min_frequency,
            "pad_token": args.pad_token,
            "unk_token": args.unk_token,
            "max_length": args.max_length,
            "padding": args.padding,
            "embed_dim": args.embedding_dim,
            "freeze_embeddings": False
        }
        # Update with tunable parameters
        full_config.update(config)
        
        # Train the model with the given config
        model, tokenizer, label_encoder, metrics = train_textcnn(
            train_texts, train_labels, val_texts, val_labels, 
            full_config, args.max_epochs, pretrained_embeddings_dict
        )
        
        # Report final metrics to Ray Tune
        tune.report({
            "val_loss": metrics["val_loss"][-1],
            "val_accuracy": metrics["val_accuracy"][-1],
            "train_accuracy": metrics["train_accuracy"][-1],
            "train_loss": metrics["train_loss"][-1]
        })
    
    # Hyperparameter space for Ray Tune
    param_space = {
        "embed_dim": tune.choice([50, 100, 150, 200, 250, 300]),
        "num_filters": tune.choice([50, 100, 150, 200, 250, 300]),
        "dropout_rate": tune.uniform(0.2, 0.6),
        "lr": tune.loguniform(1e-4, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "max_vocab_size": tune.choice([5000, 10000, 15000]),
        "kernel_sizes": tune.choice([[3, 4, 5], [2, 3, 4], [4, 5, 6], [2, 3, 4]])
    }
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    
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
            metric="val_loss",
            mode="min",
            scheduler=ASHAScheduler(
                max_t=args.max_epochs,
                grace_period=args.grace_period,
                reduction_factor=2
            ),
            num_samples=args.num_samples
        ),
        param_space=param_space
    )
    
    # Run hyperparameter search
    print("Starting hyperparameter tuning...")
    results = tuner.fit()
    
    # Get and print best results
    best_result = results.get_best_result(metric="val_loss", mode="min") 
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['val_loss']}")
    print(f"Best trial final validation accuracy: {best_result.metrics['val_accuracy']}")
    print(f"Best trial final training accuracy: {best_result.metrics['train_accuracy']}")
    print(f"Best trial final training loss: {best_result.metrics['train_loss']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the best config using joblib
    best_config_path = os.path.join(args.output_dir, "best_config.joblib")
    joblib.dump(best_result.config, best_config_path)
    print(f"Best configuration saved to {best_config_path}")

if __name__ == "__main__":
    main()