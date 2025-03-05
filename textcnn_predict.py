#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import pandas as pd
import os
import json
from sklearn.preprocessing import LabelEncoder

# Import the necessary classes and functions from your project modules
from models.text_cnn import TextCNN, load_model_artifacts, predict_batch,\
    generate_integrated_gradients, generate_occlusion_importance,\
          generate_permutation_importance, generate_prediction_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained TextCNN model for prediction')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file or a text string for prediction')
    parser.add_argument('--text_column', type=str, default='Note_Column',
                        help='Name of the text column in CSV (default: Note_Column)')
    parser.add_argument('--label_column', type=str, default=None,
                        help='Name of the label column in CSV (optional)')
    parser.add_argument('--model_dir', type=str, default="textcnn_model",
                        help='Directory containing saved model and artifacts')
    parser.add_argument('--output_dir', type=str, default='text_cnn_predictions',
                        help='Output directory for predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction (default: 32)')
    parser.add_argument('--explain', action='store_true',
                        help='Generate explanations for predictions')
    parser.add_argument('--explain_method', type=str, choices=['integrated', 'permutation', 'occlusion', 'all'],
                        default='all', help='Explanation method to use (default: all)')
    parser.add_argument('--explanation_dir', type=str, default='text_cnn_predictions',
                        help='Directory to save explanation visualizations (default: explanations)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to explain for batch methods (default: 10)')
    parser.add_argument('--summary', action='store_true',
                        help='Generate a summary report of predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for binary classification (default: 0.5)')
    return parser.parse_args()


def convert_to_serializable(obj):
    """
    Convert numpy/torch types to native Python types for JSON serialization.
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def print_top_influential_words(results, method_name):
    """
    Print top influential words for a given explanation method.
    """
    if not results:
        print(f"No {method_name} results available.")
        return

    print(f"\nTop influential words ({method_name}):")
    for i, sample_result in enumerate(results[:min(10, len(results))]):
        words = sample_result.get("words", [])
        scores = sample_result.get("importance_scores", [])
        
        if not words or not scores:
            print(f"  Sample {i+1}: No word importance data")
            continue

        # Combine words and scores, sort by absolute score in descending order
        word_scores = list(zip(words, scores))
        word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"  Sample {i+1}:")
        for word, score in word_scores[:10]:
            direction = "+" if score > 0 else "-"
            print(f"    {direction} {word}: {abs(score):.4f}")


def main():
    args = parse_args()

    # Load model and artifacts (including the custom tokenizer)
    print(f"Loading model and artifacts from {args.model_dir}...")
    model, tokenizer, label_encoder, config = load_model_artifacts(args.model_dir)

    # Create necessary directories
    if args.explain:
        os.makedirs(args.explanation_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if input is a file or a direct text input
    if os.path.isfile(args.input):
        print(f"Loading input data from {args.input}...")
        input_df = pd.read_csv(args.input)
        
        if args.text_column not in input_df.columns:
            raise ValueError(f"Text column '{args.text_column}' not found in input CSV.")
        
        texts = input_df[args.text_column].tolist()

        true_labels = None
        true_labels_encoded = None
        if args.label_column and args.label_column in input_df.columns:
            try:
                true_labels = input_df[args.label_column].tolist()
                true_labels_encoded = label_encoder.transform(true_labels).ravel()
                print("Labels found and processed for evaluation.")
            except Exception as e:
                print(f"Warning: Could not process labels. Continuing without label evaluation. Error: {e}")

        print(f"Making predictions on {len(texts)} texts...")
        all_predictions = []
        all_probabilities = []

        # Process predictions in batches
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i+args.batch_size]
            batch_preds, batch_probs = predict_batch(model, tokenizer, batch_texts)
            all_predictions.extend(batch_preds)
            all_probabilities.extend(batch_probs)

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Convert numerical predictions back to original labels
        predicted_labels = label_encoder.inverse_transform(all_predictions)
        output_df = input_df.copy()
        output_df['predicted_label'] = predicted_labels

        # Add class probabilities for each class
        for i, class_name in enumerate(label_encoder.classes_):
            output_df[f'prob_{class_name}'] = all_probabilities[:, i]

        # Add confidence (probability of predicted class)
        output_df['confidence'] = all_probabilities[np.arange(len(all_predictions)), all_predictions]

        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        output_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

        # Generate summary if requested and if labels are available
        if args.summary and true_labels is not None:
            summary_path = os.path.join(args.output_dir, 'prediction_summary.csv')
            summary = generate_prediction_summary(
                all_predictions, 
                all_probabilities, 
                true_labels_encoded, 
                summary_path
            )
            print("\nPrediction Summary:")
            for label, count in summary["label_counts"].items():
                percentage = summary["label_percentages"][label]
                avg_prob = summary["average_probabilities"][label]
                print(f"  {label}: {count} ({percentage:.2f}%) - Avg. Probability: {avg_prob:.4f}")
        elif args.summary and true_labels is None:
            print("Warning: Cannot generate summary - no ground truth labels provided.")

        # Generate explanations if requested
        if args.explain:
            explanation_results = {}
            if args.explain_method in ['integrated', 'all']:
                print("Generating integrated gradients explanations...")
                integrated_dir = os.path.join(args.explanation_dir, 'integrated_gradients')
                os.makedirs(integrated_dir, exist_ok=True)
                integrated_results = generate_integrated_gradients(
                    model, tokenizer, texts, integrated_dir, min(args.num_samples, len(texts)))
                explanation_results['integrated_gradients'] = integrated_results
                print(f"Integrated gradients explanations saved to {integrated_dir}")
                print_top_influential_words(integrated_results, "integrated gradients")

            if args.explain_method in ['permutation', 'all']:
                print("Generating permutation importance explanations...")
                permutation_dir = os.path.join(args.explanation_dir, 'permutation')
                os.makedirs(permutation_dir, exist_ok=True)
                permutation_results = generate_permutation_importance(
                    model, tokenizer, texts, permutation_dir, min(args.num_samples, len(texts)))
                explanation_results['permutation'] = permutation_results
                print(f"Permutation importance explanations saved to {permutation_dir}")
                print_top_influential_words(permutation_results, "permutation")

            if args.explain_method in ['occlusion', 'all']:
                print("Generating occlusion importance explanations...")
                occlusion_dir = os.path.join(args.explanation_dir, 'occlusion')
                os.makedirs(occlusion_dir, exist_ok=True)
                occlusion_results = generate_occlusion_importance(
                    model, tokenizer, texts, occlusion_dir, min(args.num_samples, len(texts)))
                explanation_results['occlusion'] = occlusion_results
                print(f"Occlusion importance explanations saved to {occlusion_dir}")
                print_top_influential_words(occlusion_results, "occlusion")

            explanation_path = os.path.join(args.explanation_dir, 'explanation_results.json')
            with open(explanation_path, 'w') as f:
                json.dump(explanation_results, f, default=convert_to_serializable, indent=2)

    else:
        # Single text prediction
        text = args.input
        print(f"Making prediction for: '{text}'")
        prediction, probability = predict_batch(model, tokenizer, [text])
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {probability[0, prediction[0]]:.4f}")

        print("\nClass probabilities:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"{class_name}: {probability[0, i]:.4f}")

        if args.explain:
            explanation_results = {}
            if args.explain_method in ['integrated', 'all']:
                print("\nGenerating integrated gradients explanation...")
                integrated_dir = os.path.join(args.explanation_dir, 'integrated_gradients')
                os.makedirs(integrated_dir, exist_ok=True)
                integrated_results = generate_integrated_gradients(
                    model, tokenizer, [text], integrated_dir, 1)
                explanation_results['integrated_gradients'] = integrated_results
                print(f"Integrated gradients explanation saved to {integrated_dir}")
                print_top_influential_words(integrated_results, "integrated gradients")

            if args.explain_method in ['permutation', 'all']:
                print("\nGenerating permutation importance explanation...")
                permutation_dir = os.path.join(args.explanation_dir, 'permutation')
                os.makedirs(permutation_dir, exist_ok=True)
                permutation_results = generate_permutation_importance(
                    model, tokenizer, [text], permutation_dir, 1)
                explanation_results['permutation'] = permutation_results
                print(f"Permutation importance explanation saved to {permutation_dir}")
                print_top_influential_words(permutation_results, "permutation")

            if args.explain_method in ['occlusion', 'all']:
                print("\nGenerating occlusion importance explanation...")
                occlusion_dir = os.path.join(args.explanation_dir, 'occlusion')
                os.makedirs(occlusion_dir, exist_ok=True)
                occlusion_results = generate_occlusion_importance(
                    model, tokenizer, [text], occlusion_dir, 1)
                explanation_results['occlusion'] = occlusion_results
                print(f"Occlusion importance explanation saved to {occlusion_dir}")
                print_top_influential_words(occlusion_results, "occlusion")

            explanation_path = os.path.join(args.explanation_dir, 'explanation_results.json')
            with open(explanation_path, 'w') as f:
                json.dump(explanation_results, f, default=convert_to_serializable, indent=2)

if __name__ == "__main__":
    main()
