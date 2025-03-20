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
    parser.add_argument('--max_vocab_size', type=int, default=20000)
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
    
    # Ray Tune parameters
    parser.add_argument('--output_dir', type=str, default='CNN')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--grace_period', type=int, default=5)
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Load data
    print(f"Loading Training Data from {args.train_data}...")
    train_df = pd.read_csv(args.train_data)

    print(f"Loading Validation Data from {args.val_data}...")
    val_df = pd.read_csv(args.val_data)

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
        pretrained_embeddings_dict = TextTokenizer.load_pretrained_embeddings(
            args.pretrained_embeddings)
        print(f"Loaded {len(pretrained_embeddings_dict)} pretrained word vectors")

    # Store large objects in Ray Object Store
    train_texts_ref = ray.put(train_texts)
    train_labels_ref = ray.put(train_labels)
    val_texts_ref = ray.put(val_texts)
    val_labels_ref = ray.put(val_labels)
    embeddings_ref = ray.put(pretrained_embeddings_dict) if pretrained_embeddings_dict else None

    # Define trainable function for Ray
    def train_func(config):
        # Retrieve stored objects inside function
        train_texts = ray.get(train_texts_ref)
        train_labels = ray.get(train_labels_ref)
        val_texts = ray.get(val_texts_ref)
        val_labels = ray.get(val_labels_ref)
        pretrained_embeddings = ray.get(embeddings_ref) if embeddings_ref else None

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
        full_config.update(config)

        # Train model
        model, tokenizer, label_encoder, metrics = train_textcnn(
            train_texts, train_labels, val_texts, val_labels,
            full_config, args.max_epochs, pretrained_embeddings
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
        "embed_dim": tune.choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500]),
        "num_filters": tune.choice([50, 100, 150, 200, 250, 300, 350, 400]),
        "dropout_rate": tune.uniform(0.2, 0.6),
        "lr": tune.loguniform(1e-4, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "max_vocab_size": tune.choice([5000, 10000, 15000, 20000, 30000]),
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
