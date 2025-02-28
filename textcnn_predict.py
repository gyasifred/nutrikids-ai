#!/usr/bin/env python3

import argparse
import torch
import joblib
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from collections import Counter
from models.text_cnn import TextCNN, predict_batch, load_model_artifacts, generate_integrated_gradients, \
    generate_occlusion_importance, generate_permutation_importance, generate_prediction_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use trained TextCNN model for prediction')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file or a text string for prediction')
    parser.add_argument('--text_column', type=str, default='Note_Column',
                        help='Name of the text column in CSV (default: Note_Column)')
    parser.add_argument('--model_dir', type=str, default="textcnn_model",
                        help='Directory containing saved model and artifacts')
    parser.add_argument('--output_dir', type=str, default='text_cnn_predictions',
                        help='Output file path for predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction (default: 32)')
    parser.add_argument('--explain', action='store_true',
                        help='Generate explanations for predictions')
    parser.add_argument('--explain_method', type=str, choices=['integrated', 'permutation', 'occlusion', 'all'],
                        default='all', help='Explanation method to use (default: integrated)')
    parser.add_argument('--explanation_dir', type=str, default='text_cnn_predictions',
                        help='Directory to save explanation visualizations (default: explanations)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to explain for batch methods (default: 10)')
    parser.add_argument('--summary', action='store_true',
                        help='Generate a summary report of predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for binary classification (default: 0.5)')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Load model and artifacts
    print(f"Loading model and artifacts from {args.model_dir}...")
    model, tokenizer, label_encoder, config = load_model_artifacts(
        args.model_dir)

    # Create explanation directory if needed
    if args.explain:
        os.makedirs(args.explanation_dir, exist_ok=True)

    # Check if input is a file or a direct text
    if os.path.isfile(args.input):
        print(f"Loading input data from {args.input}...")
        input_df = pd.read_csv(args.input)
        texts = input_df[args.text_column].tolist()

        print(f"Making predictions on {len(texts)} texts...")
        # Process in batches to handle large datasets
        all_predictions = []
        all_probabilities = []

        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i+args.batch_size]
            batch_preds, batch_probs = predict_batch(
                model, tokenizer, batch_texts)
            all_predictions.extend(batch_preds)
            all_probabilities.extend(batch_probs)

        # Convert numerical predictions back to original labels
        predicted_labels = label_encoder.inverse_transform(all_predictions)

        # Create output DataFrame
        output_df = input_df.copy()
        output_df['predicted_label'] = predicted_labels

        # Add probabilities for each class
        for i, class_name in enumerate(label_encoder.classes_):
            output_df[f'prob_{class_name}'] = [prob[i]
                                               for prob in all_probabilities]

        # Add confidence (probability of predicted class)
        output_df['confidence'] = [all_probabilities[i][pred]
                                   for i, pred in enumerate(all_predictions)]

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        file_path = os.path.join(args.output_dir, 'predictions.csv')
        output_df.to_csv(file_path, index=False)
        print(f"Predictions saved to {args.output_dir}")

        # Generate summary if requested
        if args.summary:
            summary_dir = os.path.dirname(args.output_dir)
            summary_path = os.path.join(summary_dir, 'prediction_summary.json')
            summary = generate_prediction_summary(
                all_predictions, all_probabilities, predicted_labels, summary_path)
            print("\nPrediction Summary:")
            for label, count in summary["label_counts"].items():
                percentage = summary["label_percentages"][label]
                avg_prob = summary["average_probabilities"][label]
                print(
                    f"  {label}: {count} ({percentage:.2f}%) - Avg. Probability: {avg_prob:.4f}")

        # Generate explanations if requested
        if args.explain:
            explanation_results = {}

            if args.explain_method in ['integrated', 'all']:
                print("Generating integrated gradients explanations...")
                integrated_dir = os.path.join(
                    args.explanation_dir, 'integrated_gradients')
                os.makedirs(integrated_dir, exist_ok=True)
                integrated_results = generate_integrated_gradients(
                    model, tokenizer, texts, integrated_dir, args.num_samples)
                explanation_results['integrated_gradients'] = integrated_results
                print(
                    f"Integrated gradients explanations saved to {integrated_dir}")

            if args.explain_method in ['permutation', 'all']:
                print("Generating permutation importance explanations...")
                permutation_dir = os.path.join(
                    args.explanation_dir, 'permutation')
                os.makedirs(permutation_dir, exist_ok=True)
                permutation_results = generate_permutation_importance(
                    model, tokenizer, texts, permutation_dir, args.num_samples)
                explanation_results['permutation'] = permutation_results
                print(
                    f"Permutation importance explanations saved to {permutation_dir}")

            if args.explain_method in ['occlusion', 'all']:
                print("Generating occlusion importance explanations...")
                occlusion_dir = os.path.join(args.explanation_dir, 'occlusion')
                os.makedirs(occlusion_dir, exist_ok=True)
                occlusion_results = generate_occlusion_importance(
                    model, tokenizer, texts, occlusion_dir, args.num_samples)
                explanation_results['occlusion'] = occlusion_results
                print(
                    f"Occlusion importance explanations saved to {occlusion_dir}")

            # Save all explanation results to a JSON file
            with open(os.path.join(args.explanation_dir, 'explanation_results.json'), 'w') as f:
                json.dump(explanation_results, f, indent=2)

    else:
        # Single text prediction
        text = args.input
        print(f"Making prediction for: '{text}'")
        prediction, probability = predict_batch(model, tokenizer, [text])
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {probability[0, prediction[0]].item():.4f}")

        # Show probabilities for all classes
        print("\nClass probabilities:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"{class_name}: {probability[0, i].item():.4f}")

        if args.explain:
            explanation_results = {}

            if args.explain_method in ['integrated', 'all']:
                print("\nGenerating integrated gradients explanation...")
                integrated_dir = os.path.join(
                    args.explanation_dir, 'integrated_gradients')
                os.makedirs(integrated_dir, exist_ok=True)
                integrated_results = generate_integrated_gradients(
                    model, tokenizer, [text], integrated_dir, 1)
                explanation_results['integrated_gradients'] = integrated_results
                print(
                    f"Integrated gradients explanation saved to {integrated_dir}")

                # Print top influential words
                words = integrated_results[0]["words"]
                scores = integrated_results[0]["importance_scores"]
                word_scores = list(zip(words, scores))
                word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

                print("\nTop influential words (integrated gradients):")
                for word, score in word_scores[:10]:  # Show top 10
                    direction = "+" if score > 0 else "-"
                    print(f"{direction} {word}: {abs(score):.4f}")

            if args.explain_method in ['permutation', 'all']:
                print("\nGenerating permutation importance explanation...")
                permutation_dir = os.path.join(
                    args.explanation_dir, 'permutation')
                os.makedirs(permutation_dir, exist_ok=True)
                permutation_results = generate_permutation_importance(
                    model, tokenizer, [text], permutation_dir, 1)
                explanation_results['permutation'] = permutation_results
                print(
                    f"Permutation importance explanation saved to {permutation_dir}")

                # Print top influential words
                if permutation_results:
                    words = permutation_results[0]["words"]
                    scores = permutation_results[0]["importance_scores"]
                    word_scores = list(zip(words, scores))
                    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

                    print("\nTop influential words (permutation):")
                    for word, score in word_scores[:10]:  # Show top 10
                        print(f"{word}: {score:.4f}")

            if args.explain_method in ['occlusion', 'all']:
                print("\nGenerating occlusion importance explanation...")
                occlusion_dir = os.path.join(args.explanation_dir, 'occlusion')
                os.makedirs(occlusion_dir, exist_ok=True)
                occlusion_results = generate_occlusion_importance(
                    model, tokenizer, [text], occlusion_dir, 1)
                explanation_results['occlusion'] = occlusion_results
                print(
                    f"Occlusion importance explanation saved to {occlusion_dir}")

                # Print top influential words
                if occlusion_results:
                    words = occlusion_results[0]["words"]
                    scores = occlusion_results[0]["importance_scores"]
                    word_scores = list(zip(words, scores))
                    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

                    print("\nTop influential words (occlusion):")
                    for word, score in word_scores[:10]:
                        print(f"{word}: {score:.4f}")

            # Save all explanation results to a JSON file
            with open(os.path.join(args.explanation_dir, 'explanation_results.json'), 'w') as f:
                json.dump(explanation_results, f, indent=2)


if __name__ == "__main__":
    main()
