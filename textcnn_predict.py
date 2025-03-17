#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import pandas as pd
import os
import json
from models.text_cnn import (
    load_model_artifacts,
    predict_batch,
    generate_integrated_gradients,
    generate_occlusion_importance,
    generate_permutation_importance,
    generate_prediction_summary
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained TextCNN model for prediction')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file or a text string for prediction')
    parser.add_argument('--text_column', type=str, default='txt',
                        help='Name of the text column in CSV (default: txt)')
    parser.add_argument('--label_column', type=str, default=None,
                        help='Name of the label column in CSV (optional)')
    parser.add_argument('--id_column', type=str, default="DEID",
                        help='Name of the column containing IDs')
    parser.add_argument('--model_dir', type=str, default="CNN",
                        help='Directory containing saved model and artifacts')
    parser.add_argument('--output_dir', type=str, default='CNN/prediction',
                        help='Output directory for predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction (default: 32)')
    parser.add_argument('--explain', action='store_true',
                        help='Generate explanations for predictions')
    parser.add_argument('--explain_method', type=str,
                        choices=['integrated', 'permutation',
                                 'occlusion', 'all'],
                        default='all',
                        help='Explanation method to use (default: all)')
    parser.add_argument('--explanation_dir', type=str,
                        default='CNN/pred_explanation',
                        help='Directory to save explanation visualizations\
                              (default: CNN/pred_explanation)')
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

    print(f"Loading model and artifacts from {args.model_dir}...")
    model, tokenizer, label_encoder, config = load_model_artifacts(
        args.model_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.explain:
        os.makedirs(args.explanation_dir, exist_ok=True)

    if os.path.isfile(args.input):
        print(f"Loading input data from {args.input}...")
        input_df = pd.read_csv(args.input)
        if args.text_column not in input_df.columns:
            raise ValueError(
                f"Text column '{args.text_column}' not found in input CSV.")

        texts = input_df[args.text_column].tolist()
        true_labels_encoded = None

        if args.label_column and args.label_column in input_df.columns:
            try:
                true_labels = input_df[args.label_column].tolist()
                true_labels_encoded = label_encoder.transform(
                    true_labels).ravel()
                print("Labels found and processed for evaluation.")
            except Exception as e:
                print(f"Warning: Could not process labels.\
                       Continuing without label evaluation. Error: {e}")
                true_labels = None

        print(f"Making predictions on {len(texts)} texts...")
        all_predictions, all_probabilities = [], []

        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            batch_preds, batch_probs = predict_batch(model,
                                                     tokenizer,
                                                     batch_texts)
            if len(batch_preds) == len(batch_texts) and batch_probs.shape[0] == len(batch_texts):
                all_predictions.extend(batch_preds)
                all_probabilities.extend(batch_probs)
            else:
                print(f"Warning: Batch prediction size mismatch at index {i}.\
                       Skipping batch.")

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        predicted_labels = label_encoder.inverse_transform(all_predictions)

        output_df = input_df[[args.id_column]].copy()
        if args.label_column and args.label_column in input_df.columns:
            output_df[args.label_column] = input_df[args.label_column]
        output_df['predicted_label'] = predicted_labels

        for i, class_name in enumerate(label_encoder.classes_):
            output_df[f'prob_{class_name}'] = all_probabilities[:, i]

        if all_probabilities.shape[1] == len(label_encoder.classes_):
            output_df['confidence'] = all_probabilities[np.arange(
                len(all_predictions)), all_predictions]
        else:
            print("Warning: Probability shape mismatch.\
                   Skipping confidence calculation.")

        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        output_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

        if args.summary and true_labels_encoded is not None:
            summary_path = os.path.join(args.output_dir,
                                        'prediction_summary.csv')
            summary = generate_prediction_summary(all_predictions,
                                                  all_probabilities,
                                                  true_labels_encoded,
                                                  summary_path)
            print("\nPrediction Summary:")
            for label, count in summary["label_counts"].items():
                percentage = summary["label_percentages"][label]
                avg_prob = summary["average_probabilities"][label]
                print(f"{label}: {count} ({percentage:.2f}%) - Avg.\
                       Probability: {avg_prob:.4f}")

        if args.explain:
            explanation_results = {}
            explain_methods = ['integrated', 'permutation',
                               'occlusion'] if args.explain_method == 'all' else [args.explain_method]

            for method in explain_methods:
                print(f"Generating {method} explanations...")
                explain_dir = os.path.join(args.explanation_dir, method)
                os.makedirs(explain_dir, exist_ok=True)

                explain_func = {
                    'integrated': generate_integrated_gradients,
                    'permutation': generate_permutation_importance,
                    'occlusion': generate_occlusion_importance
                }[method]

                explanation_results[method] = explain_func(
                    model,
                    tokenizer, texts,
                    explain_dir,
                    min(args.num_samples, len(texts)))
                print(f"{method.capitalize()} explanations saved to\
                       {explain_dir}")
                print_top_influential_words(explanation_results[method],
                                            method)

            explanation_path = os.path.join(args.explanation_dir,
                                            'explanation_results.json')
            with open(explanation_path, 'w') as f:
                json.dump(explanation_results, f,
                          default=convert_to_serializable, indent=2)
    else:
        text = args.input
        print(f"Making prediction for: '{text}'")
        prediction, probability = predict_batch(model, tokenizer, [text])
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        confidence = probability[0, prediction[0]] if probability.ndim == 2 and probability.shape[1] == len(
            label_encoder.classes_) else None
        print(f"Predicted label: {predicted_label}")
        if confidence is not None:
            print(f"Confidence: {confidence:.4f}")

        print("\nClass probabilities:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"{class_name}: {probability[0, i]:.4f}")


if __name__ == "__main__":
    main()
