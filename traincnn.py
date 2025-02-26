#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import os
import joblib
import argparse
from models.text_cnn import train_textcnn

def parse_args():
    parser = argparse.ArgumentParser(description='Train TextCNN with best hyperparameters')
    parser.add_argument('--train', type=str, required=True, 
                        help='Path to training CSV file')
    parser.add_argument('--val', type=str, required=True, 
                        help='Path to validation CSV file')
    parser.add_argument('--text-column', type=str, default='Note_Column',
                        help='Name of the text column in CSV (default: Note_Column)')
    parser.add_argument('--label-column', type=str, default='Malnutrition_Label',
                        help='Name of the label column in CSV (default: Malnutrition_Label)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to best configuration joblib file')
    parser.add_argument('--output-dir', type=str, default='model_output',
                        help='Directory to save model and artifacts (default: model_output)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for final training (default: 10)')
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
    
    # Load best configuration
    print(f"Loading best configuration from {args.config}...")
    best_config = joblib.load(args.config)
    print(f"Best configuration: {best_config}")
    
    # Train final model with best configuration
    print("Training final model with best configuration...")
    final_model, final_tokenizer, final_label_encoder, final_metrics = train_textcnn(
        train_texts, train_labels, val_texts, val_labels, 
        best_config, num_epochs=args.epochs
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save final model, tokenizer, and label encoder
    torch.save(final_model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    # Save tokenizer, label encoder, and training metrics using joblib
    joblib.dump(final_tokenizer, os.path.join(args.output_dir, "tokenizer.joblib"))
    joblib.dump(final_label_encoder, os.path.join(args.output_dir, "label_encoder.joblib"))
    joblib.dump(final_metrics, os.path.join(args.output_dir, "training_metrics.joblib"))

    print(f"Model and artifacts saved to {args.output_dir}")
    print("Final model metrics:")
    print(f"  Train Loss: {final_metrics['train_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {final_metrics['train_accuracy'][-1]:.4f}")
    print(f"  Validation Loss: {final_metrics['val_loss'][-1]:.4f}")
    print(f"  Validation Accuracy: {final_metrics['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
