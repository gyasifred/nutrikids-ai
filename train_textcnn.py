#!/usr/bin/env python3

import pandas as pd
import torch
import os
import joblib
import json
import argparse
import numpy as np
from models.text_cnn import train_textcnn, TextTokenizer
from utils import process_labels

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train TextCNN with best hyperparameters and class weights')
    parser.add_argument('--train_data', type=str, required=True, 
                        help='Path to training CSV file')
    parser.add_argument('--val_data', type=str, required=True, 
                        help='Path to validation CSV file')
    parser.add_argument('--text_column', type=str, default='txt',
                        help='Name of the text column in CSV (default: txt)')
    parser.add_argument('--label_column', type=str,
                        default='label',
                        help='Name of the label column in CSV (default: label)')
    parser.add_argument("--config_dir", default="CNN", type=str,
                        help='Path to best hyperparameter directory')
    parser.add_argument('--output_dir', type=str, default='CNN',
                        help='Directory to save model and artifacts (default: CNN)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for final training (default: 10)')
    parser.add_argument('--pretrained_embeddings',
                        type=lambda x: None if x.lower() == 'none' else str(x),
                        default=None,
                        help='Path to pretrained word embeddings file (default: None)')
    parser.add_argument('--freeze_embeddings', action='store_true',
                        help='Whether to freeze embeddings during training (default: False)')
    parser.add_argument('--positive_weight', type=float, default=2.0,
                        help='Weight multiplier for positive class (default: 2.0)')
    return parser.parse_args()

def calculate_class_weights(labels, positive_weight=2.0):
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
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_labels_raw, args.positive_weight)
    
    # Load best config
    best_config_path = os.path.join(args.config_dir, "best_config.joblib")
    print(f"Loading best configuration from {best_config_path}...")
    best_config = joblib.load(best_config_path)
    print(f"Best configuration: {best_config}")
    
    # Load pretrained embeddings if provided
    pretrained_embeddings_dict = None
    if args.pretrained_embeddings:
        print(f"Loading pretrained embeddings from {args.pretrained_embeddings}...")
        pretrained_embeddings_dict = TextTokenizer.load_pretrained_embeddings(
            args.pretrained_embeddings)
        print(f"Loaded {len(pretrained_embeddings_dict)} pretrained word vectors")
    
    # Add freeze_embeddings parameter to config
    best_config['freeze_embeddings'] = args.freeze_embeddings
    
    # Add class weights to config
    best_config['class_weights'] = class_weights
    
    # Train final model with best configuration
    print("Training final model with best configuration...")
    final_model, final_tokenizer, final_metrics = train_textcnn(
        train_texts, train_labels_raw, val_texts, val_labels_raw,
        best_config, num_epochs=args.epochs,
        pretrained_embeddings_dict=pretrained_embeddings_dict
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save final model, tokenizer, and label encoder
    torch.save(final_model.state_dict(), os.path.join(
        args.output_dir, "nutrikidaitextcnn_model.pt"))
    
    # Save tokenizer and training metrics using joblib
    joblib.dump(final_tokenizer,
                os.path.join(args.output_dir, "tokenizer.joblib"))
    
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in final_metrics.items()}, f, indent=4)
   
    # Save model configuration
    model_config = {
        "vocab_size": final_tokenizer.vocab_size_,
        "embed_dim": best_config.get("embed_dim", 100),
        "num_filters": best_config.get("num_filters", 100),
        "kernel_sizes": best_config.get("kernel_sizes", [3, 4, 5]),
        "dropout_rate": best_config.get("dropout_rate", 0.5),
        "freeze_embeddings": best_config.get("freeze_embeddings", False),
        "positive_weight": args.positive_weight
    }
    joblib.dump(model_config, os.path.join(args.output_dir,
                                           "model_config.joblib"))

    print(f"Model and artifacts saved to {args.output_dir}")
    print("Final model metrics:")
    print(f"  Train Loss: {final_metrics['train_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {final_metrics['train_accuracy'][-1]:.4f}")
    print(f"  Validation Loss: {final_metrics['val_loss'][-1]:.4f}")
    print(f"  Validation Accuracy: {final_metrics['val_accuracy'][-1]:.4f}")

    # Plot training curves if possible
    try:
        import matplotlib.pyplot as plt
        # Plot loss curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(final_metrics['train_loss'], label='Train Loss')
        plt.plot(final_metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(final_metrics['train_accuracy'], label='Train Accuracy')
        plt.plot(final_metrics['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy') 
        # Save the plots
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
        print(f"Training curves saved to {os.path.join(args.output_dir, 'training_curves.png')}")
    except Exception as e:
        print(f"Could not generate training curves: {e}")


if __name__ == "__main__":
    main()
